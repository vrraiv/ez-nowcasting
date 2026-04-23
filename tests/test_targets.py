from __future__ import annotations

import pandas as pd

from features.targets import (
    build_monthly_bridge_targets,
    build_quarterly_gdp_targets,
    normalize_quarterly_gdp_sdmx_csv,
)


SDMX_GDP_SAMPLE = """STRUCTURE,STRUCTURE_ID,freq,unit,s_adj,na_item,geo,TIME_PERIOD,OBS_VALUE
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,EA20,2020-Q1,100.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,EA20,2020-Q2,102.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,EA20,2020-Q3,105.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,EA20,2020-Q4,108.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,EA20,2021-Q1,110.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,DE,2020-Q1,50.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,DE,2020-Q2,51.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,DE,2020-Q3,52.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,DE,2020-Q4,53.0
dataflow,ESTAT:namq_10_gdp(1.0),Q,CLV10_MEUR,SCA,B1GQ,DE,2021-Q1,54.0
"""


def test_normalize_quarterly_gdp_sdmx_csv_parses_levels() -> None:
    levels = normalize_quarterly_gdp_sdmx_csv(SDMX_GDP_SAMPLE)

    assert len(levels) == 10
    assert levels["source_dataset"].tolist()[0] == "namq_10_gdp"
    assert levels["quarter"].tolist()[0] == "2020-Q1"
    assert levels["quarter_end"].dt.strftime("%Y-%m-%d").tolist()[0] == "2020-03-31"
    assert levels["gdp_real_level"].tolist()[0] == 50.0


def test_build_quarterly_and_monthly_bridge_targets() -> None:
    levels = normalize_quarterly_gdp_sdmx_csv(SDMX_GDP_SAMPLE)
    targets = build_quarterly_gdp_targets(levels, aggregate_geo="EA20")
    bridge = build_monthly_bridge_targets(targets)

    ea20_q1_2021 = targets.loc[(targets["geo"] == "EA20") & (targets["quarter"] == "2021-Q1")].iloc[0]
    ea20_bridge_q1_2021 = bridge.loc[(bridge["geo"] == "EA20") & (bridge["quarter"] == "2021-Q1")]

    assert round(float(ea20_q1_2021["qoq_real_gdp_growth"]), 6) == round(100.0 * (110.0 / 108.0 - 1.0), 6)
    assert round(float(ea20_q1_2021["yoy_real_gdp_growth"]), 6) == 10.0
    assert ea20_q1_2021["is_aggregate"] == True
    assert len(ea20_bridge_q1_2021) == 3
    assert ea20_bridge_q1_2021["month_in_quarter"].tolist() == [1, 2, 3]
    assert ea20_bridge_q1_2021["month_end"].dt.strftime("%Y-%m-%d").tolist() == [
        "2021-01-31",
        "2021-02-28",
        "2021-03-31",
    ]
    assert ea20_bridge_q1_2021["qoq_real_gdp_growth"].nunique() == 1
