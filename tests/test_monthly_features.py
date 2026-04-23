from __future__ import annotations

import pandas as pd

from data_access.pull_eurostat import SelectedSeriesConfig
from features.monthly_features import build_monthly_feature_pipeline


def test_monthly_feature_pipeline_builds_configured_and_rolling_features() -> None:
    months = pd.date_range("2020-01-01", periods=18, freq="MS")
    observations = pd.DataFrame(
        {
            "date": months,
            "geo": ["EA20"] * len(months),
            "indicator_code": ["industrial_production"] * len(months),
            "value": [100 + index for index in range(len(months))],
            "unit": ["I15"] * len(months),
            "seasonal_adjustment": ["SCA"] * len(months),
            "source_dataset": ["STS_INPR_M"] * len(months),
        }
    )
    config = SelectedSeriesConfig.model_validate(
        {
            "version": 1,
            "geo_panels": {"ea20_plus_large_members": ["EA20", "DE", "FR"]},
            "selected_series": {
                "industrial_production": {
                    "concept": "Industrial production",
                    "dataset_id": "STS_INPR_M",
                    "dimensions": {"freq": "M", "geo": ["EA20"], "unit": "I15", "s_adj": "SCA"},
                    "transformation": "3m_3m_saar",
                }
            },
        }
    )

    artifacts = build_monthly_feature_pipeline(observations, config, start_period="2020-01", end_period="2021-06")
    long_frame = artifacts.observation_long

    final_row = long_frame.loc[
        (long_frame["indicator_code"] == "industrial_production") & (long_frame["geo"] == "EA20")
    ].iloc[-1]
    expected_yoy = 100.0 * ((117.0 / 105.0) - 1.0)

    assert "industrial_production__EA20__configured_value" in artifacts.observation_wide.columns
    assert final_row["change_1m"] > 0
    assert round(final_row["change_yoy"], 6) == round(expected_yoy, 6)
    assert pd.isna(long_frame["configured_value"].iloc[0])
    assert long_frame["is_missing_observation"].sum() == 0


def test_monthly_feature_pipeline_builds_panel_aggregate_and_release_availability() -> None:
    months = pd.date_range("2020-01-01", periods=4, freq="MS")
    observations = pd.DataFrame(
        {
            "date": list(months) * 2,
            "geo": ["DE"] * 4 + ["FR"] * 4,
            "indicator_code": ["energy_import_bill"] * 8,
            "value": [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
            "unit": ["VAL_THS_USD"] * 8,
            "seasonal_adjustment": [""] * 8,
            "source_dataset": ["NRG_TI_COIFPM"] * 8,
        }
    )
    config = SelectedSeriesConfig.model_validate(
        {
            "version": 1,
            "geo_panels": {"large_members_no_ea20": ["DE", "FR", "IT"]},
            "selected_series": {
                "energy_import_bill": {
                    "concept": "Energy import bill",
                    "dataset_id": "NRG_TI_COIFPM",
                    "dimensions": {"freq": "M", "geo": ["DE", "FR"], "indic_nrg": "VAL_THS_USD"},
                    "transformation": "y_y_percent",
                    "release_lag_months": 1,
                    "aggregate_from_panel": "large_members_no_ea20",
                }
            },
        }
    )

    artifacts = build_monthly_feature_pipeline(observations, config, start_period="2020-01", end_period="2020-04")
    aggregate_rows = artifacts.observation_long.loc[
        (artifacts.observation_long["indicator_code"] == "energy_import_bill")
        & (artifacts.observation_long["geo"] == "AGG_LARGE_MEMBERS_NO_EA20")
    ]
    availability = artifacts.feature_availability.loc[
        artifacts.feature_availability["geo"] == "AGG_LARGE_MEMBERS_NO_EA20"
    ].iloc[0]

    assert not aggregate_rows.empty
    assert aggregate_rows["aggregation_method"].iloc[0] == "simple_mean"
    assert aggregate_rows["raw_value"].iloc[0] == 15.0
    assert availability["last_observation_month_end"].strftime("%Y-%m-%d") == "2020-04-30"
    assert availability["last_available_month_end"].strftime("%Y-%m-%d") == "2020-05-31"
