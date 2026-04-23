from __future__ import annotations

from dataclasses import dataclass


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
        notes="Placeholder evaluation design for vintage-aware nowcast comparisons.",
    )
