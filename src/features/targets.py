from __future__ import annotations

from dataclasses import dataclass

from config import ProjectSettings


@dataclass(frozen=True, slots=True)
class TargetBlueprint:
    name: str
    source_alias: str
    target_frequency: str
    construction_rule: str
    notes: str


def quarterly_gdp_target_blueprint(settings: ProjectSettings) -> TargetBlueprint:
    return TargetBlueprint(
        name=f"{settings.geography.aggregate.lower()}_quarterly_gdp",
        source_alias="quarterly_gdp",
        target_frequency="Q",
        construction_rule="Use the configured quarterly GDP series as the headline target.",
        notes="Benchmark target for model training, evaluation, and forecast vintage analysis.",
    )


def monthly_bridge_target_blueprint(settings: ProjectSettings) -> TargetBlueprint:
    return TargetBlueprint(
        name=f"{settings.geography.aggregate.lower()}_monthly_gdp_bridge",
        source_alias="quarterly_gdp",
        target_frequency="M",
        construction_rule=(
            "Map the quarterly GDP target onto constituent months so monthly indicators can be aligned "
            "to a ragged-edge bridge target."
        ),
        notes="Placeholder for Mariano-Murasawa style or simpler within-quarter target design.",
    )
