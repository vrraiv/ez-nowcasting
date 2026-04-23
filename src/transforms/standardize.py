from __future__ import annotations

from dataclasses import dataclass

from config import ProjectSettings


@dataclass(frozen=True, slots=True)
class TransformationSpec:
    alias: str
    target_frequency: str
    fill_method: str
    seasonal_treatment: str
    transform: str
    notes: str


def default_transform_specs(settings: ProjectSettings) -> tuple[TransformationSpec, ...]:
    del settings
    return (
        TransformationSpec(
            alias="industrial_production",
            target_frequency="M",
            fill_method="no_fill",
            seasonal_treatment="prefer pre-adjusted Eurostat series where available",
            transform="month_over_month_log_difference",
            notes="High-frequency activity indicator often used in bridge regressions.",
        ),
        TransformationSpec(
            alias="retail_trade",
            target_frequency="M",
            fill_method="no_fill",
            seasonal_treatment="prefer pre-adjusted Eurostat series where available",
            transform="month_over_month_log_difference",
            notes="Consumption proxy aligned to the monthly bridge target.",
        ),
        TransformationSpec(
            alias="unemployment",
            target_frequency="M",
            fill_method="forward_fill_short_gaps_only",
            seasonal_treatment="use standardized monthly rate as reported",
            transform="level_or_first_difference",
            notes="Slack indicator that may enter in levels rather than growth rates.",
        ),
        TransformationSpec(
            alias="hicp",
            target_frequency="M",
            fill_method="no_fill",
            seasonal_treatment="keep index definition explicit",
            transform="year_over_year_growth",
            notes="Useful for real-activity normalization and energy-shock context.",
        ),
        TransformationSpec(
            alias="oil_balance",
            target_frequency="M",
            fill_method="no_fill",
            seasonal_treatment="review seasonality before standardizing",
            transform="rolling_z_score_of_shocks",
            notes="Primary placeholder input to the oil supply stress indicator.",
        ),
    )
