from __future__ import annotations

import pandas as pd

from features.oil_stress import OilStressConfig, build_oil_supply_stress_artifacts


def test_oil_stress_pipeline_aggregates_country_flows_by_sum_and_builds_indices() -> None:
    months = pd.date_range("2020-01-01", periods=36, freq="MS")
    observations = pd.DataFrame(
        {
            "date": list(months) * 3,
            "geo": ["DE"] * 36 + ["FR"] * 36 + ["EA20"] * 36,
            "indicator_code": ["petroleum_imports_volume"] * 72 + ["energy_producer_prices"] * 36,
            "value": [200.0 - index for index in range(36)]
            + [400.0 - (2 * index) for index in range(36)]
            + [100.0 + (1.5 * index) for index in range(36)],
            "unit": ["THS_T"] * 72 + ["I15"] * 36,
            "seasonal_adjustment": [""] * 72 + ["NSA"] * 36,
            "source_dataset": ["NRG_TI_OILM"] * 72 + ["STS_INPPD_M"] * 36,
        }
    )
    config = OilStressConfig.model_validate(
        {
            "version": 1,
            "target_geo": "EA20",
            "minimum_component_count": 2,
            "geo_panels": {"large_members_no_ea20": ["DE", "FR"]},
            "components": {
                "petroleum_imports_volume": {
                    "label": "Oil import volume",
                    "bucket": "direct_supply",
                    "dataset_id": "NRG_TI_OILM",
                    "dimensions": {"freq": "M", "geo": ["DE", "FR"], "unit": "THS_T"},
                    "signal_transformation": "y_y_percent",
                    "stress_direction": "negative",
                    "aggregate_from_panel": "large_members_no_ea20",
                    "panel_aggregation": "sum",
                    "structural_weight": 0.6,
                    "interpretation": "Lower imports should raise stress.",
                    "cyclical_sensitivity": "medium",
                },
                "energy_producer_prices": {
                    "label": "Energy producer prices",
                    "bucket": "pass_through_prices",
                    "dataset_id": "STS_INPPD_M",
                    "dimensions": {"freq": "M", "geo": "EA20", "unit": "I15", "s_adj": "NSA"},
                    "signal_transformation": "y_y_percent",
                    "stress_direction": "positive",
                    "structural_weight": 0.4,
                    "interpretation": "Higher prices should raise stress.",
                    "cyclical_sensitivity": "low",
                },
            },
        }
    )

    artifacts = build_oil_supply_stress_artifacts(observations, config, start_period="2020-01", end_period="2022-12")
    flow_panel = artifacts.component_panel_long.loc[
        artifacts.component_panel_long["component_code"].eq("petroleum_imports_volume")
    ]
    latest_flow = flow_panel.loc[flow_panel["standardized_component"].notna()].iloc[-1]
    latest_index = artifacts.index_history.loc[
        artifacts.index_history["structural_index_standardized"].notna()
    ].iloc[-1]

    assert flow_panel["source_geo"].iloc[0] == "AGG_LARGE_MEMBERS_NO_EA20"
    assert flow_panel["aggregation_method"].iloc[0] == "sum"
    assert flow_panel["raw_value"].iloc[0] == 600.0
    assert latest_flow["signed_signal"] > 0
    assert latest_flow["standardized_component"] > 0
    assert latest_index["component_count"] == 2
    assert pd.notna(latest_index["structural_index_standardized"])


def test_oil_stress_component_table_exposes_weights_and_pca_loadings() -> None:
    months = pd.date_range("2020-01-01", periods=36, freq="MS")
    observations = pd.DataFrame(
        {
            "date": list(months) * 2,
            "geo": ["EA20"] * 72,
            "indicator_code": ["refined_petroleum_output"] * 36 + ["energy_producer_prices"] * 36,
            "value": [100.0 - (0.8 * index) for index in range(36)] + [100.0 + (1.2 * index) for index in range(36)],
            "unit": ["I15"] * 72,
            "seasonal_adjustment": ["SCA"] * 36 + ["NSA"] * 36,
            "source_dataset": ["STS_INPR_M"] * 36 + ["STS_INPPD_M"] * 36,
        }
    )
    config = OilStressConfig.model_validate(
        {
            "version": 1,
            "target_geo": "EA20",
            "minimum_component_count": 2,
            "components": {
                "refined_petroleum_output": {
                    "label": "Refined petroleum output",
                    "bucket": "refining_and_downstream",
                    "dataset_id": "STS_INPR_M",
                    "dimensions": {"freq": "M", "geo": "EA20", "unit": "I15", "s_adj": "SCA"},
                    "signal_transformation": "3m_3m_saar",
                    "stress_direction": "negative",
                    "structural_weight": 0.55,
                    "interpretation": "Lower output should raise stress.",
                    "cyclical_sensitivity": "medium",
                },
                "energy_producer_prices": {
                    "label": "Energy producer prices",
                    "bucket": "pass_through_prices",
                    "dataset_id": "STS_INPPD_M",
                    "dimensions": {"freq": "M", "geo": "EA20", "unit": "I15", "s_adj": "NSA"},
                    "signal_transformation": "y_y_percent",
                    "stress_direction": "positive",
                    "structural_weight": 0.45,
                    "interpretation": "Higher prices should raise stress.",
                    "cyclical_sensitivity": "low",
                },
            },
        }
    )

    artifacts = build_oil_supply_stress_artifacts(observations, config, start_period="2020-01", end_period="2022-12")
    component_table = artifacts.component_table.set_index("component_code")

    assert round(float(component_table["structural_weight"].sum()), 6) == 1.0
    assert "pca_loading" in component_table.columns
    assert component_table.loc["refined_petroleum_output", "source_geo"] == "EA20"
    assert component_table.loc["energy_producer_prices", "stress_direction"] == "positive"
    assert "Euro-Area Oil Supply Stress Indicator" in artifacts.narrative_markdown
    assert artifacts.index_history_chart_svg.lstrip().startswith("<svg")
