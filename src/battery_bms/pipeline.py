from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .baselines import (
    run_no_battery,
    run_perfect_forecast_optimization,
    run_rule_based_tou,
)
from .config import BatteryConfig, SafetyConfig
from .data import plot_sign_convention_week, prepare_hourly_dataset
from .forecast import train_forecast_model
from .rl import (
    evaluate_q_learning_controller,
    save_controller,
    train_q_learning_controller,
)


@dataclass(frozen=True)
class PipelineResult:
    hourly_rows: int
    train_rows: int
    test_rows: int
    controller_results: pd.DataFrame
    forecast_metrics: dict[str, dict[str, float]]
    output_dir: Path


def _controller_row(name: str, summary: dict[str, float]) -> dict[str, float | str]:
    row: dict[str, float | str] = {"controller": name}
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            row[key] = float(value)
        else:
            row[key] = str(value)
    return row


def _plot_controller_cumulative_cost(
    details_by_name: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, details in details_by_name.items():
        cumulative = (details["electricity_cost"] + details["degradation_cost"]).cumsum()
        ax.plot(details["time"], cumulative, label=name, linewidth=1.4)

    ax.set_title("Controller cumulative cost on test horizon")
    ax.set_ylabel("Cumulative cost")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_pipeline(
    root: Path,
    episodes: int = 16,
    test_fraction: float = 0.30,
    random_state: int = 42,
) -> PipelineResult:
    output_dir = root / "outputs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    for directory in [tables_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    battery_config = BatteryConfig()
    safety_config = SafetyConfig()
    battery_config.validate()
    safety_config.validate()

    prepared = prepare_hourly_dataset(root, output_dir=tables_dir)
    plot_sign_convention_week(
        prepared.raw_power,
        figures_dir / "sign_convention_week.png",
    )
    with (tables_dir / "sign_convention.json").open("w", encoding="utf-8") as file:
        json.dump(prepared.convention.to_dict(), file, indent=2)

    forecast = train_forecast_model(
        prepared.hourly,
        output_dir=models_dir,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    hourly = forecast.data
    hourly.to_csv(tables_dir / "hourly_microgrid.csv", index=False)

    split_index = int(len(hourly) * (1 - test_fraction))
    split_index = max(24, min(split_index, len(hourly) - 24))
    train = hourly.iloc[:split_index].reset_index(drop=True)
    test = hourly.iloc[split_index:].reset_index(drop=True)

    details_by_name: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, float | str]] = []

    no_battery_details, no_battery_summary = run_no_battery(test)
    details_by_name["No battery"] = no_battery_details
    rows.append(_controller_row("No battery", no_battery_summary))

    tou_details, tou_summary = run_rule_based_tou(test, battery_config)
    details_by_name["Rule-based ToU"] = tou_details
    rows.append(_controller_row("Rule-based ToU", tou_summary))

    lp_details, lp_summary = run_perfect_forecast_optimization(test, battery_config)
    details_by_name["Perfect-forecast LP"] = lp_details
    rows.append(_controller_row("Perfect-forecast LP", lp_summary))

    rl_no_deg = train_q_learning_controller(
        train,
        battery_config,
        degradation_cost_per_cycle=0.0,
        episodes=episodes,
        use_safety_filter=False,
        random_state=random_state,
    )
    save_controller(rl_no_deg, models_dir / "q_learning_without_degradation.joblib")
    rl_no_deg_details, rl_no_deg_summary = evaluate_q_learning_controller(
        test,
        rl_no_deg,
        battery_config,
        degradation_cost_per_cycle=0.0,
        use_safety_filter=False,
    )
    details_by_name["RL without degradation"] = rl_no_deg_details
    rows.append(_controller_row("RL without degradation", rl_no_deg_summary))

    rl_with_deg = train_q_learning_controller(
        train,
        battery_config,
        degradation_cost_per_cycle=battery_config.degradation_cost_per_cycle,
        episodes=episodes,
        use_safety_filter=False,
        random_state=random_state + 1,
    )
    save_controller(rl_with_deg, models_dir / "q_learning_with_degradation.joblib")
    rl_with_deg_details, rl_with_deg_summary = evaluate_q_learning_controller(
        test,
        rl_with_deg,
        battery_config,
        degradation_cost_per_cycle=battery_config.degradation_cost_per_cycle,
        use_safety_filter=False,
    )
    details_by_name["RL with degradation"] = rl_with_deg_details
    rows.append(_controller_row("RL with degradation", rl_with_deg_summary))

    rl_bms = train_q_learning_controller(
        train,
        battery_config,
        degradation_cost_per_cycle=battery_config.degradation_cost_per_cycle,
        episodes=episodes,
        use_safety_filter=True,
        safety_config=safety_config,
        random_state=random_state + 2,
    )
    save_controller(rl_bms, models_dir / "q_learning_bms_safety.joblib")
    rl_bms_details, rl_bms_summary = evaluate_q_learning_controller(
        test,
        rl_bms,
        battery_config,
        degradation_cost_per_cycle=battery_config.degradation_cost_per_cycle,
        use_safety_filter=True,
        safety_config=safety_config,
    )
    details_by_name["RL + BMS safety layer"] = rl_bms_details
    rows.append(_controller_row("RL + BMS safety layer", rl_bms_summary))

    controller_results = pd.DataFrame(rows)
    controller_results.to_csv(tables_dir / "controller_results.csv", index=False)

    for name, details in details_by_name.items():
        safe_name = (
            name.lower()
            .replace(" + ", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        details.to_csv(tables_dir / f"details_{safe_name}.csv", index=False)

    _plot_controller_cumulative_cost(
        details_by_name,
        figures_dir / "controller_cumulative_cost.png",
    )

    return PipelineResult(
        hourly_rows=len(hourly),
        train_rows=len(train),
        test_rows=len(test),
        controller_results=controller_results,
        forecast_metrics=forecast.metrics,
        output_dir=output_dir,
    )
