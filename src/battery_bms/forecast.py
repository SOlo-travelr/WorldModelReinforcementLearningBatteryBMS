from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


HORIZONS = (1, 2, 3)


@dataclass(frozen=True)
class ForecastResult:
    data: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    model: MultiOutputRegressor
    feature_columns: list[str]
    target_columns: list[str]
    split_row: int


def _cyclical(value: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * value / period
    return np.sin(angle), np.cos(angle)


def make_supervised_frame(
    hourly: pd.DataFrame,
    history_hours: int = 24,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_horizon = max(horizons)
    rows: list[dict[str, float]] = []
    targets: list[dict[str, float]] = []
    indices: list[int] = []

    for index in range(history_hours, len(hourly) - max_horizon):
        row = hourly.iloc[index]
        features: dict[str, float] = {}
        for lag in range(1, history_hours + 1):
            features[f"pv_lag_{lag}h"] = float(hourly.iloc[index - lag]["pv_kwh"])
            features[f"load_lag_{lag}h"] = float(hourly.iloc[index - lag]["load_kwh"])

        hour_sin, hour_cos = _cyclical(pd.Series([row["hour"]]), 24)
        day_sin, day_cos = _cyclical(pd.Series([row["day_of_week"]]), 7)
        month_sin, month_cos = _cyclical(pd.Series([row["month"] - 1]), 12)
        features.update(
            {
                "hour_sin": float(hour_sin.iloc[0]),
                "hour_cos": float(hour_cos.iloc[0]),
                "day_sin": float(day_sin.iloc[0]),
                "day_cos": float(day_cos.iloc[0]),
                "month_sin": float(month_sin.iloc[0]),
                "month_cos": float(month_cos.iloc[0]),
            }
        )

        target: dict[str, float] = {}
        for horizon in horizons:
            future = hourly.iloc[index + horizon]
            target[f"predicted_pv_{horizon}h"] = float(future["pv_kwh"])
            target[f"predicted_load_{horizon}h"] = float(future["load_kwh"])

        rows.append(features)
        targets.append(target)
        indices.append(index)

    features_df = pd.DataFrame(rows, index=indices)
    targets_df = pd.DataFrame(targets, index=indices)
    return features_df, targets_df


def train_forecast_model(
    hourly: pd.DataFrame,
    output_dir: Path | None = None,
    history_hours: int = 24,
    horizons: tuple[int, ...] = HORIZONS,
    test_fraction: float = 0.30,
    random_state: int = 42,
) -> ForecastResult:
    features, targets = make_supervised_frame(hourly, history_hours, horizons)
    if features.empty:
        raise ValueError("Not enough rows to build forecast features")

    split = int(len(features) * (1 - test_fraction))
    split = max(1, min(split, len(features) - 1))
    x_train, y_train = features.iloc[:split], targets.iloc[:split]
    x_test, y_test = features.iloc[split:], targets.iloc[split:]

    base_model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=random_state,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)

    predictions = model.predict(features)
    prediction_df = pd.DataFrame(predictions, index=features.index, columns=targets.columns)
    prediction_df = prediction_df.clip(lower=0)

    enriched = hourly.copy()
    for horizon in horizons:
        enriched[f"predicted_pv_{horizon}h"] = enriched["pv_kwh"]
        enriched[f"predicted_load_{horizon}h"] = enriched["load_kwh"]
    for column in targets.columns:
        enriched.loc[prediction_df.index, column] = prediction_df[column].to_numpy()

    test_predictions = prediction_df.loc[x_test.index]
    metrics: dict[str, dict[str, float]] = {}
    for column in targets.columns:
        mse = mean_squared_error(y_test[column], test_predictions[column])
        metrics[column] = {
            "mae": float(mean_absolute_error(y_test[column], test_predictions[column])),
            "rmse": float(np.sqrt(mse)),
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_columns": list(features.columns),
                "target_columns": list(targets.columns),
                "history_hours": history_hours,
                "horizons": horizons,
            },
            output_dir / "forecast_model.joblib",
        )
        table_dir = output_dir.parent / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        with (table_dir / "forecast_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

    return ForecastResult(
        data=enriched,
        metrics=metrics,
        model=model,
        feature_columns=list(features.columns),
        target_columns=list(targets.columns),
        split_row=int(features.index[split]),
    )
