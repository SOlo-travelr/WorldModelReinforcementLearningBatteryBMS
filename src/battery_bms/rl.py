from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .bms_safety import BMSSafetyFilter
from .config import BatteryConfig, SafetyConfig
from .simulator import BatterySimulator, simulate_policy


ACTION_NAMES = (
    "full_charge",
    "half_charge",
    "idle",
    "half_discharge",
    "full_discharge",
)


@dataclass
class QLearningController:
    q_table: dict[tuple[int, int, int, int], np.ndarray]
    actions_kw: tuple[float, ...]
    soc_bins: np.ndarray
    price_bins: np.ndarray
    net_load_bins: np.ndarray

    def _state_key(self, row: pd.Series, soc: float) -> tuple[int, int, int, int]:
        predicted_net = np.mean(
            [
                float(row.get(f"predicted_load_{horizon}h", row["load_kwh"]))
                - float(row.get(f"predicted_pv_{horizon}h", row["pv_kwh"]))
                for horizon in (1, 2, 3)
            ]
        )
        soc_bin = int(np.digitize([soc], self.soc_bins)[0])
        price_bin = int(np.digitize([float(row["price"])], self.price_bins)[0])
        net_bin = int(np.digitize([predicted_net], self.net_load_bins)[0])
        hour = int(row["hour"])
        return soc_bin, hour, price_bin, net_bin

    def q_values(self, row: pd.Series, soc: float) -> np.ndarray:
        key = self._state_key(row, soc)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.actions_kw), dtype=float)
        return self.q_table[key]

    def act(
        self,
        row: pd.Series,
        soc: float,
        rng: np.random.Generator | None = None,
        epsilon: float = 0.0,
    ) -> tuple[int, float]:
        if rng is not None and rng.random() < epsilon:
            action_index = int(rng.integers(0, len(self.actions_kw)))
        else:
            action_index = int(np.argmax(self.q_values(row, soc)))
        return action_index, float(self.actions_kw[action_index])


def make_actions(config: BatteryConfig) -> tuple[float, ...]:
    return (
        config.max_charge_power_kw,
        config.max_charge_power_kw / 2.0,
        0.0,
        -config.max_discharge_power_kw / 2.0,
        -config.max_discharge_power_kw,
    )


def _net_forecast_series(data: pd.DataFrame) -> pd.Series:
    values = []
    for _, row in data.iterrows():
        values.append(
            np.mean(
                [
                    float(row.get(f"predicted_load_{horizon}h", row["load_kwh"]))
                    - float(row.get(f"predicted_pv_{horizon}h", row["pv_kwh"]))
                    for horizon in (1, 2, 3)
                ]
            )
        )
    return pd.Series(values, index=data.index)


def train_q_learning_controller(
    data: pd.DataFrame,
    config: BatteryConfig | None = None,
    degradation_cost_per_cycle: float = 0.0,
    episodes: int = 16,
    alpha: float = 0.08,
    gamma: float = 0.985,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.03,
    use_safety_filter: bool = False,
    safety_config: SafetyConfig | None = None,
    random_state: int = 42,
) -> QLearningController:
    base_config = config or BatteryConfig()
    cfg = replace(base_config, degradation_cost_per_cycle=degradation_cost_per_cycle)
    cfg.validate()

    price_bins = np.unique(data["price"].quantile([0.33, 0.66]).to_numpy(dtype=float))
    net_bins = np.unique(_net_forecast_series(data).quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(dtype=float))
    soc_bins = np.linspace(cfg.soc_min, cfg.soc_max, 9)[1:-1]
    controller = QLearningController(
        q_table={},
        actions_kw=make_actions(cfg),
        soc_bins=soc_bins,
        price_bins=price_bins,
        net_load_bins=net_bins,
    )

    simulator = BatterySimulator(cfg)
    safety = BMSSafetyFilter(cfg, safety_config) if use_safety_filter else None
    rng = np.random.default_rng(random_state)
    rows = [row for _, row in data.iterrows()]

    for episode in range(max(1, episodes)):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
            1.0 - episode / max(1, episodes - 1)
        )
        soc = cfg.initial_soc
        for index, row in enumerate(rows):
            key = controller._state_key(row, soc)
            if key not in controller.q_table:
                controller.q_table[key] = np.zeros(len(controller.actions_kw), dtype=float)

            action_index, command_kw = controller.act(row, soc, rng=rng, epsilon=epsilon)
            if safety is not None:
                command_kw = safety.filter_command(
                    soc=soc,
                    load_kwh=float(row["load_kwh"]),
                    pv_kwh=float(row["pv_kwh"]),
                    requested_kw=command_kw,
                ).filtered_kw

            result = simulator.step(
                soc=soc,
                load_kwh=float(row["load_kwh"]),
                pv_kwh=float(row["pv_kwh"]),
                price=float(row["price"]),
                command_kw=command_kw,
            )
            next_row = rows[min(index + 1, len(rows) - 1)]
            next_key = controller._state_key(next_row, result.soc_next)
            if next_key not in controller.q_table:
                controller.q_table[next_key] = np.zeros(len(controller.actions_kw), dtype=float)

            current = controller.q_table[key][action_index]
            target = result.reward + gamma * float(np.max(controller.q_table[next_key]))
            controller.q_table[key][action_index] = current + alpha * (target - current)
            soc = result.soc_next

    return controller


def evaluate_q_learning_controller(
    data: pd.DataFrame,
    controller: QLearningController,
    config: BatteryConfig | None = None,
    degradation_cost_per_cycle: float = 0.0,
    use_safety_filter: bool = False,
    safety_config: SafetyConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    base_config = config or BatteryConfig()
    cfg = replace(base_config, degradation_cost_per_cycle=degradation_cost_per_cycle)
    safety = BMSSafetyFilter(cfg, safety_config) if use_safety_filter else None

    def policy(_: int, row: pd.Series, soc: float) -> float:
        _, requested_kw = controller.act(row, soc)
        if safety is None:
            return requested_kw
        return safety.filter_command(
            soc=soc,
            load_kwh=float(row["load_kwh"]),
            pv_kwh=float(row["pv_kwh"]),
            requested_kw=requested_kw,
        ).filtered_kw

    return simulate_policy(data, policy, cfg)


def save_controller(controller: QLearningController, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "q_table": controller.q_table,
            "actions_kw": controller.actions_kw,
            "action_names": ACTION_NAMES,
            "soc_bins": controller.soc_bins,
            "price_bins": controller.price_bins,
            "net_load_bins": controller.net_load_bins,
        },
        path,
    )
