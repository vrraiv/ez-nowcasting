from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
import tomllib

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("project_config.toml")


class GeographyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregate: str = Field(..., description="Euro-area aggregate code used for headline nowcasts.")
    members: list[str] = Field(
        default_factory=list,
        description="Optional member-state panel used for cross-checks or richer factor models.",
    )


class DateRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date


class PreferredUnitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quarterly_gdp: str
    industrial_production: str
    retail_trade: str
    unemployment: str
    hicp: str
    oil_balance: str


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quarterly_gdp: str
    industrial_production: str
    retail_trade: str
    unemployment: str
    hicp: str
    oil_balance: str


class DownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = 30.0
    user_agent: str = "eurozone-nowcasting/0.1.0"
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    cache_dir: Path = Path("data_raw/cache/http")


class EurostatApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0"
    statistics_base_url: str = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    agency_id: str = "ESTAT"
    dataflow_version: str = "1.0"
    language: str = "en"
    structure_reports_dir: Path = Path("outputs/structure_reports")
    search_results_dir: Path = Path("outputs/discovery")
    default_search_limit: int = 10


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_data_dir: Path = Path("data_raw")
    processed_data_dir: Path = Path("data_processed")
    outputs_dir: Path = Path("outputs")


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geography: GeographyConfig
    date_range: DateRangeConfig
    preferred_units: PreferredUnitsConfig
    datasets: DatasetConfig
    download: DownloadConfig = DownloadConfig()
    eurostat_api: EurostatApiConfig = EurostatApiConfig()
    paths: PathConfig = PathConfig()

    def resolve_path(self, candidate: Path, root: Path = PROJECT_ROOT) -> Path:
        return candidate if candidate.is_absolute() else root / candidate

    def resolved_paths(self, root: Path = PROJECT_ROOT) -> dict[str, Path]:
        return {
            "raw_data_dir": self.resolve_path(self.paths.raw_data_dir, root),
            "processed_data_dir": self.resolve_path(self.paths.processed_data_dir, root),
            "outputs_dir": self.resolve_path(self.paths.outputs_dir, root),
            "cache_dir": self.resolve_path(self.download.cache_dir, root),
            "structure_reports_dir": self.resolve_path(self.eurostat_api.structure_reports_dir, root),
            "search_results_dir": self.resolve_path(self.eurostat_api.search_results_dir, root),
        }


def load_settings(path: Path | None = None) -> ProjectSettings:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("rb") as handle:
        raw_settings = tomllib.load(handle)
    return ProjectSettings.model_validate(raw_settings)


@lru_cache(maxsize=1)
def get_settings() -> ProjectSettings:
    return load_settings(DEFAULT_CONFIG_PATH)
