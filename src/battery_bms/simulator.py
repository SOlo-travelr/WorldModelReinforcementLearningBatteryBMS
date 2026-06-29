from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import BatteryConfig


@dataclass(frozen=True)
class StepResult:
    soc_current: float
    soc_next: float
    requested_command_kw: float
    command_kw: float
    charge_kw: float
    discharge_kw: float
    grid_import_kwh: float
    electricity_cost: float
    degradation_cost: float
    reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class BatterySimulator:
    def __init__(self, config: BatteryConfig | None = None):
        self.config = config or BatteryConfig()
        self.config.validate()

    def _clip_command_for_physics(self, soc: float, command_kw: float) -> float:
        cfg = self.config
        command_kw = float(
            np.clip(command_kw, -cfg.max_discharge_power_kw, cfg.max_charge_power_kw)
        )

        max_charge_by_soc = (
            (cfg.soc_max - soc)
            * cfg.battery_capacity_kwh
            / (cfg.charge_efficiency * cfg.dt_hours)
        )
        max_discharge_by_soc = (
            (soc - cfg.soc_min)
            * cfg.battery_capacity_kwh
            * cfg.discharge_efficiency
            / cfg.dt_hours
        )

        if command_kw >= 0:
            return float(min(command_kw, max(0.0, max_charge_by_soc)))
        return float(-min(-command_kw, max(0.0, max_discharge_by_soc)))

    def step(
        self,
        soc: float,
        load_kwh: float,
        pv_kwh: float,
        price: float,
        command_kw: float,
    ) -> StepResult:
        cfg = self.config
        command_limited = self._clip_command_for_physics(soc, command_kw)
        charge_kw = max(command_limited, 0.0)
        discharge_kw = max(-command_limited, 0.0)

        soc_next = (
            soc
            + cfg.charge_efficiency * charge_kw * cfg.dt_hours / cfg.battery_capacity_kwh
            - discharge_kw * cfg.dt_hours / (cfg.discharge_efficiency * cfg.battery_capacity_kwh)
        )
        soc_next = float(np.clip(soc_next, cfg.soc_min, cfg.soc_max))

        grid_import_kwh = load_kwh - pv_kwh - discharge_kw * cfg.dt_hours + charge_kw * cfg.dt_hours
        electricity_cost = float(price * max(grid_import_kwh, 0.0))
        equivalent_cycle_loss = abs(soc_next - soc) / 2.0
        degradation_cost = float(equivalent_cycle_loss * cfg.degradation_cost_per_cycle)
        reward = -electricity_cost - degradation_cost

        return StepResult(
            soc_current=float(soc),
            soc_next=soc_next,
            requested_command_kw=float(command_kw),
            command_kw=command_limited,
            charge_kw=float(charge_kw),
            discharge_kw=float(discharge_kw),
            grid_import_kwh=float(grid_import_kwh),
            electricity_cost=electricity_cost,
            degradation_cost=degradation_cost,
            reward=float(reward),
        )


PolicyFn = Callable[[int, pd.Series, float], float]


def summarize_details(details: pd.DataFrame) -> dict[str, float]:
    return {
        "electricity_cost": float(details["electricity_cost"].sum()),
        "degradation_cost": float(details["degradation_cost"].sum()),
        "total_cost": float(
            details["electricity_cost"].sum() + details["degradation_cost"].sum()
        ),
        "grid_import_kwh": float(details["grid_import_kwh"].clip(lower=0).sum()),
        "grid_export_kwh": float((-details["grid_import_kwh"].clip(upper=0)).sum()),
        "battery_throughput_kwh": float(
            (details["charge_kw"] + details["discharge_kw"]).sum()
        ),
        "min_soc": float(details["soc_next"].min()),
        "max_soc": float(details["soc_next"].max()),
        "final_soc": float(details["soc_next"].iloc[-1]),
    }


def simulate_policy(
    data: pd.DataFrame,
    policy: PolicyFn,
    config: BatteryConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    simulator = BatterySimulator(config)
    cfg = simulator.config
    soc = cfg.initial_soc
    records: list[dict[str, float | str | pd.Timestamp]] = []

    for step_index, (_, row) in enumerate(data.iterrows()):
        command_kw = float(policy(step_index, row, soc))
        result = simulator.step(
            soc=soc,
            load_kwh=float(row["load_kwh"]),
            pv_kwh=float(row["pv_kwh"]),
            price=float(row["price"]),
            command_kw=command_kw,
        )
        record = {
            "time": row["time"],
            "load_kwh": float(row["load_kwh"]),
            "pv_kwh": float(row["pv_kwh"]),
            "price": float(row["price"]),
            **result.to_dict(),
        }
        records.append(record)
        soc = result.soc_next

    details = pd.DataFrame.from_records(records)
    return details, summarize_details(details)
