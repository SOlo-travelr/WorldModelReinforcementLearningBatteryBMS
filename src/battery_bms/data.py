from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


POWER_FILE = "Power Data.xlsx"
EXCEL_ORIGIN = "1899-12-30"


@dataclass(frozen=True)
class SignConvention:
    battery_positive_means: str
    demand_identity: str
    plus_residual_abs_max: float
    minus_residual_abs_max: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedDataset:
    raw_power: pd.DataFrame
    hourly: pd.DataFrame
    convention: SignConvention


def parse_excel_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask].astype(float),
            unit="D",
            origin=EXCEL_ORIGIN,
        )

    text_mask = parsed.isna()
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(series.loc[text_mask], errors="coerce")

    return parsed


def load_power_data(root: Path) -> pd.DataFrame:
    path = root / "Data" / POWER_FILE
    if not path.exists():
        raise FileNotFoundError(f"Expected power data at {path}")

    df = pd.read_excel(path)
    column_map = {
        "Time Stamp": "time_stamp",
        "Battery": "battery_kw",
        "Net Demand": "net_grid_kw",
        "Solar": "pv_kw",
        "Demand": "demand_kw",
    }
    missing = sorted(set(column_map) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    df = df.rename(columns=column_map)
    df["time"] = parse_excel_time(df["time_stamp"])
    for column in ["battery_kw", "net_grid_kw", "pv_kw", "demand_kw"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "battery_kw", "net_grid_kw", "pv_kw", "demand_kw"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return df.reset_index(drop=True)


def infer_sign_convention(df: pd.DataFrame) -> SignConvention:
    plus_residual = df["demand_kw"] - (
        df["net_grid_kw"] + df["pv_kw"] + df["battery_kw"]
    )
    minus_residual = df["demand_kw"] - (
        df["net_grid_kw"] + df["pv_kw"] - df["battery_kw"]
    )
    plus_abs_max = float(plus_residual.abs().max())
    minus_abs_max = float(minus_residual.abs().max())

    if plus_abs_max <= minus_abs_max:
        meaning = "discharge"
        identity = "Demand = Net Demand + Solar + Battery"
    else:
        meaning = "charge"
        identity = "Demand = Net Demand + Solar - Battery"

    return SignConvention(
        battery_positive_means=meaning,
        demand_identity=identity,
        plus_residual_abs_max=plus_abs_max,
        minus_residual_abs_max=minus_abs_max,
    )


def default_tou_price(timestamp: pd.Timestamp) -> float:
    hour = int(timestamp.hour)
    is_weekend = timestamp.dayofweek >= 5
    if is_weekend:
        return 0.12
    if 16 <= hour <= 21:
        return 0.32
    if 7 <= hour <= 15:
        return 0.18
    return 0.10


def _fill_by_group_median(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
    fallback_column: str | None = None,
) -> pd.Series:
    filled = frame[value_column].copy()
    group_median = frame.groupby(group_columns, dropna=False)[value_column].transform("median")
    filled = filled.fillna(group_median)

    if fallback_column is not None:
        fallback_median = frame.groupby(fallback_column, dropna=False)[value_column].transform("median")
        filled = filled.fillna(fallback_median)

    return filled.fillna(float(frame[value_column].median())).fillna(0.0)


def prepare_hourly_dataset(root: Path, output_dir: Path | None = None) -> PreparedDataset:
    raw = load_power_data(root)
    convention = infer_sign_convention(raw)

    if convention.battery_positive_means == "discharge":
        raw["reconstructed_load_kw"] = raw["net_grid_kw"] + raw["pv_kw"] + raw["battery_kw"]
    else:
        raw["reconstructed_load_kw"] = raw["net_grid_kw"] + raw["pv_kw"] - raw["battery_kw"]

    raw["load_kw"] = raw["demand_kw"].fillna(raw["reconstructed_load_kw"])
    hourly_power = (
        raw.set_index("time")
        [["pv_kw", "load_kw", "net_grid_kw", "battery_kw", "reconstructed_load_kw"]]
        .resample("1h")
        .mean()
    )

    full_index = pd.date_range(
        hourly_power.index.min(),
        hourly_power.index.max(),
        freq="1h",
    )
    hourly_power = hourly_power.reindex(full_index)
    hourly_power.index.name = "time"

    observed_hour = hourly_power[["pv_kw", "load_kw"]].notna().all(axis=1)
    hourly_power["hour"] = hourly_power.index.hour
    hourly_power["day_of_week"] = hourly_power.index.dayofweek
    hourly_power["month"] = hourly_power.index.month

    hourly_power["pv_kw"] = _fill_by_group_median(
        hourly_power,
        "pv_kw",
        ["month", "hour"],
        fallback_column="hour",
    ).clip(lower=0)
    hourly_power["load_kw"] = _fill_by_group_median(
        hourly_power,
        "load_kw",
        ["day_of_week", "hour"],
        fallback_column="hour",
    ).clip(lower=0)
    hourly_power["battery_kw"] = hourly_power["battery_kw"].fillna(0.0)
    hourly_power["reconstructed_load_kw"] = hourly_power["reconstructed_load_kw"].fillna(
        hourly_power["load_kw"]
    )
    hourly_power["net_grid_kw"] = hourly_power["net_grid_kw"].fillna(
        hourly_power["load_kw"] - hourly_power["pv_kw"] - hourly_power["battery_kw"]
    )

    hourly = pd.DataFrame(
        {
            "time": hourly_power.index,
            "pv_kwh": hourly_power["pv_kw"].clip(lower=0).to_numpy(),
            "load_kwh": hourly_power["load_kw"].clip(lower=0).to_numpy(),
            "net_grid_kwh": hourly_power["net_grid_kw"].to_numpy(),
            "source_battery_kwh": hourly_power["battery_kw"].to_numpy(),
            "reconstructed_load_kwh": hourly_power["reconstructed_load_kw"].clip(lower=0).to_numpy(),
            "is_observed_hour": observed_hour.to_numpy(dtype=bool),
        }
    )
    hourly["price"] = hourly["time"].map(default_tou_price).astype(float)
    hourly["hour"] = hourly["time"].dt.hour.astype(int)
    hourly["day"] = np.where(hourly["time"].dt.dayofweek < 5, "weekday", "weekend")
    hourly["day_of_week"] = hourly["time"].dt.dayofweek.astype(int)
    hourly["month"] = hourly["time"].dt.month.astype(int)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(output_dir / "hourly_microgrid.csv", index=False)

    return PreparedDataset(raw_power=raw, hourly=hourly, convention=convention)


def plot_sign_convention_week(raw: pd.DataFrame, output_path: Path, days: int = 7) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = raw["time"].min().normalize()
    end = start + pd.Timedelta(days=days)
    week = raw[(raw["time"] >= start) & (raw["time"] < end)].copy()
    if week.empty:
        raise ValueError("No data available for sign-convention plot")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(week["time"], week["pv_kw"], color="#D79A00", linewidth=1.1)
    axes[0].set_ylabel("PV kW")
    axes[0].grid(alpha=0.25)

    axes[1].plot(week["time"], week["battery_kw"], color="#2E7D6F", linewidth=1.1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Battery kW")
    axes[1].grid(alpha=0.25)

    axes[2].plot(week["time"], week["net_grid_kw"], color="#B33A3A", linewidth=1.1)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Net demand kW")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Second-Life dataset sign-convention check")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
