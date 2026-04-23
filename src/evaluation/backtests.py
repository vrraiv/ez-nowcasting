from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    name: str
    window_scheme: str
    forecast_horizon: str
    metrics: tuple[str, ...]
    notes: str


def default_backtest_spec() -> BacktestSpec:
    return BacktestSpec(
        name="expanding_window_nowcast",
        window_scheme="expanding",
        forecast_horizon="current_quarter_nowcast",
        metrics=("rmse", "mae", "directional_accuracy"),
        notes="Expanding-window rolling-origin evaluation by within-quarter information set.",
    )


def rolling_origin_splits(n_samples: int, min_train_size: int) -> list[tuple[slice, int]]:
    if min_train_size < 1:
        raise ValueError("min_train_size must be at least 1.")
    if n_samples <= min_train_size:
        return []
    return [(slice(0, evaluation_index), evaluation_index) for evaluation_index in range(min_train_size, n_samples)]


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_values, predicted_values = _coerce_metric_inputs(actual, predicted)
    if len(actual_values) == 0:
        return math.nan
    return float(np.sqrt(np.mean((actual_values - predicted_values) ** 2)))


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_values, predicted_values = _coerce_metric_inputs(actual, predicted)
    if len(actual_values) == 0:
        return math.nan
    return float(np.mean(np.abs(actual_values - predicted_values)))


def directional_accuracy(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_values, predicted_values = _coerce_metric_inputs(actual, predicted)
    if len(actual_values) == 0:
        return math.nan
    return float(np.mean(np.sign(actual_values) == np.sign(predicted_values)))


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    actual_column: str = "actual",
    prediction_column: str = "prediction",
    group_columns: tuple[str, ...] = ("model_name", "information_set", "target_column"),
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[*group_columns, "rmse", "mae", "directional_accuracy", "evaluation_count"]
        )

    metric_rows = []
    for keys, group in predictions.groupby(list(group_columns), sort=False):
        actual = group[actual_column]
        predicted = group[prediction_column]
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: keys[index] for index, column in enumerate(group_columns)}
        row.update(
            {
                "rmse": rmse(actual, predicted),
                "mae": mae(actual, predicted),
                "directional_accuracy": directional_accuracy(actual, predicted),
                "evaluation_count": int(group[[actual_column, prediction_column]].dropna().shape[0]),
            }
        )
        metric_rows.append(row)

    return pd.DataFrame.from_records(metric_rows).sort_values(list(group_columns), kind="stable").reset_index(drop=True)


def _coerce_metric_inputs(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual_series = pd.to_numeric(pd.Series(actual), errors="coerce")
    predicted_series = pd.to_numeric(pd.Series(predicted), errors="coerce")
    valid = actual_series.notna() & predicted_series.notna()
    return actual_series.loc[valid].to_numpy(dtype="float64"), predicted_series.loc[valid].to_numpy(dtype="float64")
