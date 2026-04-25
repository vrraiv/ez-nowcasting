from __future__ import annotations

from pathlib import Path

from config import settings as settings_module
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


def test_load_settings_uses_toml_reader_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "project_config.toml"
    config_path.write_text(
        """
[geography]
aggregate = "EA20"
members = ["DE", "FR"]

[date_range]
start = "2000-01-01"
end = "2026-03-31"

[preferred_units]
quarterly_gdp = "CLV10_MEUR"
industrial_production = "I15"
retail_trade = "I15"
unemployment = "PC_ACT"
hicp = "RCH_A"
oil_balance = "THS_T"

[datasets]
quarterly_gdp = "namq_10_gdp"
industrial_production = "sts_inpr_m"
retail_trade = "sts_trtu_m"
unemployment = "une_rt_m"
hicp = "prc_hicp_midx"
oil_balance = "nrg_cb_oilm"
""".strip(),
        encoding="utf-8",
    )

    class StubTomlReader:
        @staticmethod
        def load(handle):
            text = handle.read().decode("utf-8")
            assert 'aggregate = "EA20"' in text
            return {
                "geography": {"aggregate": "EA20", "members": ["DE", "FR"]},
                "date_range": {"start": "2000-01-01", "end": "2026-03-31"},
                "preferred_units": {
                    "quarterly_gdp": "CLV10_MEUR",
                    "industrial_production": "I15",
                    "retail_trade": "I15",
                    "unemployment": "PC_ACT",
                    "hicp": "RCH_A",
                    "oil_balance": "THS_T",
                },
                "datasets": {
                    "quarterly_gdp": "namq_10_gdp",
                    "industrial_production": "sts_inpr_m",
                    "retail_trade": "sts_trtu_m",
                    "unemployment": "une_rt_m",
                    "hicp": "prc_hicp_midx",
                    "oil_balance": "nrg_cb_oilm",
                },
            }

    original_reader = settings_module.toml_reader
    settings_module.toml_reader = StubTomlReader()
    try:
        settings = settings_module.load_settings(config_path)
    finally:
        settings_module.toml_reader = original_reader

    assert settings.geography.members == ["DE", "FR"]
