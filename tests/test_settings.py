from __future__ import annotations

from config import DEFAULT_CONFIG_PATH, load_settings


def test_default_settings_load() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.geography.aggregate == "EA20"
    assert settings.date_range.start.isoformat() == "2000-01-01"
    assert settings.datasets.quarterly_gdp == "namq_10_gdp"
    assert settings.eurostat_api.agency_id == "ESTAT"
    assert settings.eurostat_api.statistics_base_url.endswith("/statistics/1.0/data")
    assert settings.download.max_retries == 3


def test_resolved_paths_are_project_relative() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)
    resolved = settings.resolved_paths()

    assert resolved["raw_data_dir"].name == "data_raw"
    assert resolved["processed_data_dir"].name == "data_processed"
    assert resolved["outputs_dir"].name == "outputs"
    assert resolved["cache_dir"].parts[-3:] == ("data_raw", "cache", "http")
    assert resolved["structure_reports_dir"].parts[-2:] == ("outputs", "structure_reports")
    assert resolved["search_results_dir"].parts[-2:] == ("outputs", "discovery")
