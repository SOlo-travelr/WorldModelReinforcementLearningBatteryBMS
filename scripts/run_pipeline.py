from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from battery_bms.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the battery BMS modeling pipeline.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root path.")
    parser.add_argument("--episodes", type=int, default=16, help="Q-learning episodes.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.30,
        help="Chronological fraction reserved for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        root=args.root.resolve(),
        episodes=args.episodes,
        test_fraction=args.test_fraction,
        random_state=args.seed,
    )

    print("\nPipeline complete")
    print(f"Hourly rows: {result.hourly_rows}")
    print(f"Train rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"Outputs: {result.output_dir}")
    print("\nController summary:")
    columns = [
        "controller",
        "electricity_cost",
        "degradation_cost",
        "total_cost",
        "grid_import_kwh",
        "grid_export_kwh",
        "battery_throughput_kwh",
        "min_soc",
        "max_soc",
        "final_soc",
    ]
    available_columns = [column for column in columns if column in result.controller_results.columns]
    print(result.controller_results[available_columns].to_string(index=False))

    print("\nForecast metrics:")
    for target, metrics in result.forecast_metrics.items():
        print(f"{target}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")


if __name__ == "__main__":
    main()
