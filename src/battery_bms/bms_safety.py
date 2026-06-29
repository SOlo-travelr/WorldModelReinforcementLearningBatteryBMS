from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BatteryConfig, SafetyConfig


@dataclass(frozen=True)
class SafetyDecision:
    requested_kw: float
    filtered_kw: float
    lower_bound_kw: float
    upper_bound_kw: float
    adjusted: bool


class BMSSafetyFilter:
    def __init__(
        self,
        battery_config: BatteryConfig | None = None,
        safety_config: SafetyConfig | None = None,
    ):
        self.battery_config = battery_config or BatteryConfig()
        self.safety_config = safety_config or SafetyConfig()
        self.battery_config.validate()
        self.safety_config.validate()

    def filter_command(
        self,
        soc: float,
        load_kwh: float,
        pv_kwh: float,
        requested_kw: float,
    ) -> SafetyDecision:
        battery = self.battery_config
        safety = self.safety_config

        c_rate_power_kw = safety.max_c_rate * battery.battery_capacity_kwh
        max_charge_kw = min(battery.max_charge_power_kw, c_rate_power_kw)
        max_discharge_kw = min(battery.max_discharge_power_kw, c_rate_power_kw)

        soc_charge_limit_kw = (
            (battery.soc_max - soc)
            * battery.battery_capacity_kwh
            / (battery.charge_efficiency * battery.dt_hours)
        )
        soc_discharge_limit_kw = (
            (soc - battery.soc_min)
            * battery.battery_capacity_kwh
            * battery.discharge_efficiency
            / battery.dt_hours
        )

        degradation_charge_limit_kw = (
            safety.max_soc_delta_per_step
            * battery.battery_capacity_kwh
            / (battery.charge_efficiency * battery.dt_hours)
        )
        degradation_discharge_limit_kw = (
            safety.max_soc_delta_per_step
            * battery.battery_capacity_kwh
            * battery.discharge_efficiency
            / battery.dt_hours
        )

        upper = min(max_charge_kw, soc_charge_limit_kw, degradation_charge_limit_kw)
        lower = -min(max_discharge_kw, soc_discharge_limit_kw, degradation_discharge_limit_kw)

        base_grid_kw = load_kwh - pv_kwh
        lower = max(lower, -safety.max_grid_export_kw - base_grid_kw)
        upper = min(upper, safety.max_grid_import_kw - base_grid_kw)

        if lower > upper:
            midpoint = (lower + upper) / 2.0
            lower = midpoint
            upper = midpoint

        filtered = float(np.clip(requested_kw, lower, upper))
        return SafetyDecision(
            requested_kw=float(requested_kw),
            filtered_kw=filtered,
            lower_bound_kw=float(lower),
            upper_bound_kw=float(upper),
            adjusted=not np.isclose(float(requested_kw), filtered),
        )
