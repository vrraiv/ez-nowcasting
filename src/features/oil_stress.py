from __future__ import annotations

from dataclasses import dataclass

from config import ProjectSettings


@dataclass(frozen=True, slots=True)
class OilStressBlueprint:
    name: str
    source_aliases: tuple[str, ...]
    transform_steps: tuple[str, ...]
    aggregation_window: str
    notes: str


def oil_supply_stress_blueprint(settings: ProjectSettings) -> OilStressBlueprint:
    return OilStressBlueprint(
        name=f"{settings.geography.aggregate.lower()}_oil_supply_stress",
        source_aliases=("oil_balance", "hicp"),
        transform_steps=(
            "standardize the monthly oil balance series into comparable shocks",
            "combine oil-balance shocks with an energy-price pass-through proxy",
            "scale the composite indicator as a rolling z-score",
        ),
        aggregation_window="3M",
        notes="Design placeholder for a stress indicator that can enter baseline nowcasts as an exogenous feature.",
    )
