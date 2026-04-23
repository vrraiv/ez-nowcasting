from __future__ import annotations

from data_access.pull_eurostat import (
    SeriesSelection,
    normalize_jsonstat_dataset,
    normalize_sdmx_csv_dataset,
)


JSONSTAT_SAMPLE = {
    "id": ["freq", "unit", "coicop", "geo", "time"],
    "size": [1, 1, 1, 2, 2],
    "dimension": {
        "freq": {"category": {"index": {"M": 0}}},
        "unit": {"category": {"index": {"I15": 0}}},
        "coicop": {"category": {"index": {"CP00": 0}}},
        "geo": {"category": {"index": {"EA20": 0, "DE": 1}}},
        "time": {"category": {"index": {"2024-01": 0, "2024-02": 1}}},
    },
    "value": {
        "0": 123.62,
        "1": 124.38,
        "2": 126.40,
        "3": 127.20,
    },
}

SDMX_CSV_SAMPLE = """STRUCTURE,STRUCTURE_ID,freq,sex,age,s_adj,unit,geo,TIME_PERIOD,OBS_VALUE
dataflow,ESTAT:UNE_RT_M(1.0),M,T,TOTAL,SA,PC_ACT,EA20,2024-01,6.4
dataflow,ESTAT:UNE_RT_M(1.0),M,T,TOTAL,SA,PC_ACT,DE,2024-01,3.2
dataflow,ESTAT:UNE_RT_M(1.0),M,T,TOTAL,SA,PC_ACT,EA20,2024-02,6.5
dataflow,ESTAT:UNE_RT_M(1.0),M,T,TOTAL,SA,PC_ACT,DE,2024-02,3.3
"""


def test_normalize_jsonstat_dataset_returns_tidy_rows() -> None:
    selection = SeriesSelection(
        alias="hicp_headline",
        concept="HICP headline",
        dataset_id="PRC_HICP_MIDX",
        dimensions={
            "freq": "M",
            "coicop": "CP00",
            "unit": "I15",
            "geo": ["EA20", "DE"],
        },
        transformation="y_y_percent",
    )

    frame = normalize_jsonstat_dataset(JSONSTAT_SAMPLE, selection)

    assert frame.columns.tolist() == [
        "date",
        "geo",
        "indicator_code",
        "value",
        "unit",
        "seasonal_adjustment",
        "source_dataset",
    ]
    assert len(frame) == 4
    assert frame["geo"].tolist() == ["DE", "DE", "EA20", "EA20"]
    assert frame["unit"].tolist() == ["I15", "I15", "I15", "I15"]
    assert frame["seasonal_adjustment"].isna().all()
    assert frame["source_dataset"].tolist() == ["PRC_HICP_MIDX"] * 4
    assert frame["date"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-02", "2024-01", "2024-02"]


def test_normalize_sdmx_csv_dataset_returns_tidy_rows() -> None:
    selection = SeriesSelection(
        alias="unemployment_rate",
        concept="Unemployment rate",
        dataset_id="UNE_RT_M",
        dimensions={
            "freq": "M",
            "sex": "T",
            "age": "TOTAL",
            "s_adj": "SA",
            "unit": "PC_ACT",
            "geo": ["EA20", "DE"],
        },
        transformation="level",
    )

    frame = normalize_sdmx_csv_dataset(SDMX_CSV_SAMPLE, selection)

    assert len(frame) == 4
    assert frame["indicator_code"].tolist() == ["unemployment_rate"] * 4
    assert frame["geo"].tolist() == ["DE", "DE", "EA20", "EA20"]
    assert frame["seasonal_adjustment"].tolist() == ["SA", "SA", "SA", "SA"]
    assert frame["unit"].tolist() == ["PC_ACT", "PC_ACT", "PC_ACT", "PC_ACT"]
    assert frame["value"].tolist() == [3.2, 3.3, 6.4, 6.5]
    assert frame["date"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-02", "2024-01", "2024-02"]
