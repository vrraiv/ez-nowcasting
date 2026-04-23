from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaselineModelSpec:
    name: str
    objective: str
    inputs: str
    status: str = "placeholder"


def baseline_model_specs() -> tuple[BaselineModelSpec, ...]:
    return (
        BaselineModelSpec(
            name="bridge_ols",
            objective="Map monthly indicators into a quarterly GDP nowcast with a simple bridge regression.",
            inputs="Quarterly-aggregated monthly features and the quarterly GDP target.",
        ),
        BaselineModelSpec(
            name="ridge_bridge",
            objective="Provide a regularized baseline when the monthly feature set expands.",
            inputs="Standardized monthly features with quarter-level alignment.",
        ),
        BaselineModelSpec(
            name="dynamic_factor_baseline",
            objective="Reserve a slot for a lightweight factor-style baseline once the data pipeline is stable.",
            inputs="Common factors extracted from standardized monthly indicators.",
        ),
    )
