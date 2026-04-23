from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

MONTH_END_FREQUENCY = "ME"
TRAILING_ZSCORE_WINDOW = 60
TRAILING_ZSCORE_MIN_PERIODS = 24
DEFAULT_OUTLIER_THRESHOLD = 6.0
RATE_LIKE_UNITS = {"PC_ACT", "INX", "BAL", "RT1-ABS-SA"}
RATE_LIKE_TRANSFORMS = {"level", "diffusion_style_standardized_level"}


def month_end_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=MONTH_END_FREQUENCY)


def align_month_end(values: pd.Series | Sequence[object]) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return pd.to_datetime(series, errors="coerce") + pd.offsets.MonthEnd(0)


def infer_change_style(unit: str | None, transformation: str | None) -> str:
    if unit in RATE_LIKE_UNITS or transformation in RATE_LIKE_TRANSFORMS:
        return "difference"
    return "ratio"


def apply_named_transformation(
    values: pd.Series,
    transformation: str,
    unit: str | None = None,
) -> pd.Series:
    style = infer_change_style(unit, transformation)
    if transformation == "level":
        return values.astype("float64")
    if transformation == "diffusion_style_standardized_level":
        return expanding_zscore(values)
    if transformation == "y_y_percent":
        return year_over_year_change(values, style=style)
    if transformation == "3m_3m_saar":
        return three_month_over_three_month_annualized(values, style=style)
    if transformation == "m_m_percent":
        if style == "difference" or unit == "RT1-ABS-SA":
            return values.astype("float64")
        return one_month_change(values, style=style)
    raise ValueError(f"Unsupported transformation: {transformation}")


def one_month_change(values: pd.Series, style: str = "ratio") -> pd.Series:
    series = values.astype("float64")
    if style == "difference":
        return series - series.shift(1)
    return _percent_change(series, periods=1)


def three_month_over_three_month_annualized(values: pd.Series, style: str = "ratio") -> pd.Series:
    series = values.astype("float64")
    rolling_mean = series.rolling(window=3, min_periods=3).mean()
    if style == "difference":
        return 4.0 * (rolling_mean - rolling_mean.shift(3))

    prior = rolling_mean.shift(3)
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = (rolling_mean > 0) & (prior > 0)
    result.loc[valid] = 100.0 * ((rolling_mean.loc[valid] / prior.loc[valid]) ** 4.0 - 1.0)
    return result


def year_over_year_change(values: pd.Series, style: str = "ratio") -> pd.Series:
    series = values.astype("float64")
    if style == "difference":
        return series - series.shift(12)
    return _percent_change(series, periods=12)


def trailing_zscore(
    values: pd.Series,
    window: int = TRAILING_ZSCORE_WINDOW,
    min_periods: int = TRAILING_ZSCORE_MIN_PERIODS,
) -> pd.Series:
    series = values.astype("float64")
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    result = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return result.astype("float64")


def expanding_zscore(values: pd.Series, min_periods: int = 12) -> pd.Series:
    series = values.astype("float64")
    expanding_mean = series.expanding(min_periods=min_periods).mean()
    expanding_std = series.expanding(min_periods=min_periods).std(ddof=0)
    return ((series - expanding_mean) / expanding_std.replace(0.0, np.nan)).astype("float64")


def detect_outliers(values: pd.Series, threshold: float = DEFAULT_OUTLIER_THRESHOLD) -> pd.Series:
    series = values.astype("float64")
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index, dtype="bool")

    median = valid.median()
    mad = (valid - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=series.index, dtype="bool")

    robust_z = (series - median) / (1.4826 * mad)
    return robust_z.abs().gt(threshold).fillna(False)


def available_month_end(
    month_end: pd.Series,
    release_lag_months: int | None = None,
    release_lag_days: int | None = None,
) -> pd.Series:
    lag_months = int(release_lag_months or 0)
    lag_days = int(release_lag_days or 0)
    available = pd.to_datetime(month_end, errors="coerce")
    if lag_months or lag_days:
        available = available + pd.DateOffset(months=lag_months, days=lag_days)
    return available + pd.offsets.MonthEnd(0)


def _percent_change(values: pd.Series, periods: int) -> pd.Series:
    prior = values.shift(periods)
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = (values > 0) & (prior > 0)
    result.loc[valid] = 100.0 * (values.loc[valid] / prior.loc[valid] - 1.0)
    return result
