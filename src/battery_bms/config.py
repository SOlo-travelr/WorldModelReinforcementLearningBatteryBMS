from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    battery_capacity_kwh: float = 500.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    max_charge_power_kw: float = 250.0
    max_discharge_power_kw: float = 250.0
    dt_hours: float = 1.0
    degradation_cost_per_cycle: float = 40.0

    def validate(self) -> None:
        if self.battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be positive")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("SoC bounds must satisfy 0 <= soc_min < soc_max <= 1")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must be inside SoC bounds")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if self.max_charge_power_kw < 0 or self.max_discharge_power_kw < 0:
            raise ValueError("Power limits must be non-negative")
        if self.dt_hours <= 0:
            raise ValueError("dt_hours must be positive")


@dataclass(frozen=True)
class SafetyConfig:
    max_c_rate: float = 0.50
    max_grid_import_kw: float = 500.0
    max_grid_export_kw: float = 500.0
    max_soc_delta_per_step: float = 0.10

    def validate(self) -> None:
        if self.max_c_rate <= 0:
            raise ValueError("max_c_rate must be positive")
        if self.max_grid_import_kw < 0 or self.max_grid_export_kw < 0:
            raise ValueError("Grid import/export limits must be non-negative")
        if not 0 < self.max_soc_delta_per_step <= 1:
            raise ValueError("max_soc_delta_per_step must be in (0, 1]")
