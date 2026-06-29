from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import BatteryConfig
from .simulator import simulate_policy, summarize_details


def run_no_battery(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    details = data[["time", "load_kwh", "pv_kwh", "price"]].copy()
    details["soc_current"] = np.nan
    details["soc_next"] = np.nan
    details["requested_command_kw"] = 0.0
    details["command_kw"] = 0.0
    details["charge_kw"] = 0.0
    details["discharge_kw"] = 0.0
    details["grid_import_kwh"] = details["load_kwh"] - details["pv_kwh"]
    details["electricity_cost"] = details["price"] * details["grid_import_kwh"].clip(lower=0)
    details["degradation_cost"] = 0.0
    details["reward"] = -details["electricity_cost"]

    summary = summarize_details(details.assign(soc_next=0.0))
    summary.update({"min_soc": np.nan, "max_soc": np.nan, "final_soc": np.nan})
    return details, summary


def run_rule_based_tou(
    data: pd.DataFrame,
    config: BatteryConfig | None = None,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, float]]:
    cfg = config or BatteryConfig()
    low_price = float(data["price"].quantile(low_quantile))
    high_price = float(data["price"].quantile(high_quantile))

    def policy(_: int, row: pd.Series, soc: float) -> float:
        if row["price"] <= low_price and soc < cfg.soc_max:
            return cfg.max_charge_power_kw
        if row["price"] >= high_price and soc > cfg.soc_min:
            return -cfg.max_discharge_power_kw
        if row["pv_kwh"] > row["load_kwh"] and soc < cfg.soc_max:
            return min(cfg.max_charge_power_kw, float(row["pv_kwh"] - row["load_kwh"]))
        return 0.0

    return simulate_policy(data, policy, cfg)


def run_perfect_forecast_optimization(
    data: pd.DataFrame,
    config: BatteryConfig | None = None,
    degradation_cost_per_cycle: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    cfg = config or BatteryConfig()
    if degradation_cost_per_cycle is not None:
        cfg = replace(cfg, degradation_cost_per_cycle=degradation_cost_per_cycle)
    cfg.validate()

    n = len(data)
    if n == 0:
        raise ValueError("Cannot optimize an empty dataset")

    charge_slice = slice(0, n)
    discharge_slice = slice(n, 2 * n)
    soc_slice = slice(2 * n, 3 * n)
    grid_slice = slice(3 * n, 4 * n)
    variables = 4 * n

    objective = np.zeros(variables)
    objective[grid_slice] = data["price"].to_numpy(dtype=float)
    objective[charge_slice] += (
        cfg.degradation_cost_per_cycle * cfg.charge_efficiency / (2 * cfg.battery_capacity_kwh)
    )
    objective[discharge_slice] += cfg.degradation_cost_per_cycle / (
        2 * cfg.discharge_efficiency * cfg.battery_capacity_kwh
    )

    bounds = []
    bounds.extend((0.0, cfg.max_charge_power_kw) for _ in range(n))
    bounds.extend((0.0, cfg.max_discharge_power_kw) for _ in range(n))
    bounds.extend((cfg.soc_min, cfg.soc_max) for _ in range(n))
    bounds.extend((0.0, None) for _ in range(n))

    a_eq = np.zeros((n, variables))
    b_eq = np.zeros(n)
    for t in range(n):
        a_eq[t, soc_slice.start + t] = 1.0
        a_eq[t, charge_slice.start + t] = -cfg.charge_efficiency / cfg.battery_capacity_kwh
        a_eq[t, discharge_slice.start + t] = 1.0 / (
            cfg.discharge_efficiency * cfg.battery_capacity_kwh
        )
        if t == 0:
            b_eq[t] = cfg.initial_soc
        else:
            a_eq[t, soc_slice.start + t - 1] = -1.0

    base_grid = data["load_kwh"].to_numpy(dtype=float) - data["pv_kwh"].to_numpy(dtype=float)
    a_ub = np.zeros((n, variables))
    b_ub = -base_grid
    for t in range(n):
        a_ub[t, charge_slice.start + t] = 1.0
        a_ub[t, discharge_slice.start + t] = -1.0
        a_ub[t, grid_slice.start + t] = -1.0

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        details, summary = run_rule_based_tou(data, cfg)
        summary["lp_status"] = f"fallback_rule_based: {result.message}"
        return details, summary

    charge = result.x[charge_slice]
    discharge = result.x[discharge_slice]
    commands = charge - discharge

    def policy(step_index: int, _: pd.Series, __: float) -> float:
        return float(commands[step_index])

    details, summary = simulate_policy(data, policy, cfg)
    summary["lp_status"] = "optimal"
    return details, summary
