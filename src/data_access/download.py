from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from config import ProjectSettings


class SeriesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    dataset_code: str
    frequency: str
    geo: str
    start_period: date
    end_period: date
    unit: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)


def build_monthly_download_plan(settings: ProjectSettings) -> tuple[SeriesRequest, ...]:
    return (
        SeriesRequest(
            alias="industrial_production",
            dataset_code=settings.datasets.industrial_production,
            frequency="M",
            geo=settings.geography.aggregate,
            start_period=settings.date_range.start,
            end_period=settings.date_range.end,
            unit=settings.preferred_units.industrial_production,
        ),
        SeriesRequest(
            alias="retail_trade",
            dataset_code=settings.datasets.retail_trade,
            frequency="M",
            geo=settings.geography.aggregate,
            start_period=settings.date_range.start,
            end_period=settings.date_range.end,
            unit=settings.preferred_units.retail_trade,
        ),
        SeriesRequest(
            alias="unemployment",
            dataset_code=settings.datasets.unemployment,
            frequency="M",
            geo=settings.geography.aggregate,
            start_period=settings.date_range.start,
            end_period=settings.date_range.end,
            unit=settings.preferred_units.unemployment,
        ),
        SeriesRequest(
            alias="hicp",
            dataset_code=settings.datasets.hicp,
            frequency="M",
            geo=settings.geography.aggregate,
            start_period=settings.date_range.start,
            end_period=settings.date_range.end,
            unit=settings.preferred_units.hicp,
        ),
        SeriesRequest(
            alias="oil_balance",
            dataset_code=settings.datasets.oil_balance,
            frequency="M",
            geo=settings.geography.aggregate,
            start_period=settings.date_range.start,
            end_period=settings.date_range.end,
            unit=settings.preferred_units.oil_balance,
        ),
    )


def build_quarterly_target_request(settings: ProjectSettings) -> SeriesRequest:
    return SeriesRequest(
        alias="quarterly_gdp",
        dataset_code=settings.datasets.quarterly_gdp,
        frequency="Q",
        geo=settings.geography.aggregate,
        start_period=settings.date_range.start,
        end_period=settings.date_range.end,
        unit=settings.preferred_units.quarterly_gdp,
    )
