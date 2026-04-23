from __future__ import annotations

import pandas as pd

from models.baselines import build_quarter_stage_aggregates, run_baseline_nowcast_backtests


def _synthetic_feature_long() -> pd.DataFrame:
    rows = []
    quarter_starts = pd.period_range("2015Q1", "2021Q4", freq="Q")
    for quarter_index, quarter in enumerate(quarter_starts):
        months = pd.period_range(quarter.asfreq("M", "start"), quarter.asfreq("M", "end"), freq="M")
        latent_level = 100.0 + quarter_index * 1.5
        for month_position, month in enumerate(months, start=1):
            month_end = month.to_timestamp(how="end").normalize()
            rows.extend(
                [
                    {
                        "month_end": month_end,
                        "available_month_end": month_end,
                        "indicator_code": "industrial_production",
                        "geo": "EA20",
                        "configured_value": latent_level + month_position,
                        "change_1m": 0.5,
                        "change_3m_3m_saar": 1.0,
                        "change_yoy": quarter_index * 0.2,
                        "zscore_5y": 0.1 * quarter_index,
                    },
                    {
                        "month_end": month_end,
                        "available_month_end": month_end,
                        "indicator_code": "retail_trade_volume",
                        "geo": "EA20",
                        "configured_value": latent_level * 0.6 + month_position,
                        "change_1m": 0.3,
                        "change_3m_3m_saar": 0.8,
                        "change_yoy": quarter_index * 0.15,
                        "zscore_5y": 0.05 * quarter_index,
                    },
                ]
            )
    return pd.DataFrame.from_records(rows)


def _synthetic_bridge_targets() -> pd.DataFrame:
    rows = []
    quarter_starts = pd.period_range("2015Q1", "2021Q4", freq="Q")
    for quarter_index, quarter in enumerate(quarter_starts):
        quarter_start = quarter.to_timestamp(how="start")
        quarter_end = quarter.to_timestamp(how="end").normalize()
        months = pd.period_range(quarter.asfreq("M", "start"), quarter.asfreq("M", "end"), freq="M")
        qoq = 0.5 + 0.1 * quarter_index
        yoy = 1.0 + 0.2 * quarter_index
        for month_position, month in enumerate(months, start=1):
            month_end = month.to_timestamp(how="end").normalize()
            rows.append(
                {
                    "month_end": month_end,
                    "quarter": str(quarter),
                    "quarter_end": quarter_end,
                    "quarter_start": quarter_start,
                    "month_in_quarter": month_position,
                    "nowcast_stage": f"month_{month_position}",
                    "geo": "EA20",
                    "qoq_real_gdp_growth": qoq,
                    "yoy_real_gdp_growth": yoy,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_build_quarter_stage_aggregates_returns_stage_level_rows() -> None:
    feature_long = _synthetic_feature_long()
    targets = _synthetic_bridge_targets()
    stage_targets = targets.loc[targets["month_in_quarter"] == 2].copy()
    stage_targets["information_set"] = "month_2"
    stage_targets["target"] = stage_targets["qoq_real_gdp_growth"]
    stage_targets["target_geo"] = "EA20"
    stage_targets["target_column"] = "qoq_real_gdp_growth"

    design = build_quarter_stage_aggregates(
        feature_long=feature_long,
        stage_targets=stage_targets,
        value_columns=["configured_value"],
    )

    assert len(design) == len(stage_targets)
    assert "industrial_production__EA20__configured_value" in design.columns
    assert "retail_trade_volume__EA20__configured_value" in design.columns


def test_bridge_backtest_pipeline_runs_on_synthetic_data() -> None:
    feature_long = _synthetic_feature_long()
    targets = _synthetic_bridge_targets()

    artifacts = run_baseline_nowcast_backtests(
        feature_long=feature_long,
        bridge_targets=targets,
        target_geo="EA20",
        target_column="qoq_real_gdp_growth",
        min_train_quarters=8,
        n_dynamic_factors=1,
    )

    assert not artifacts.predictions.empty
    assert set(artifacts.metrics["model_name"]) >= {"bridge_ols"}
    assert "month_1" in set(artifacts.metrics["information_set"])
