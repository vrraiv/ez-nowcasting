from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import argparse
import logging

import pandas as pd

from config import PROJECT_ROOT, ProjectSettings, get_settings
from data_access.ingestion import (
    FileResponseCache,
    HttpRequestSpec,
    RetryingHttpClient,
    write_dataframe_parquet,
    write_raw_response,
)
from data_access.pull_eurostat import DEFAULT_SELECTED_SERIES_PATH, load_selected_series_config


@dataclass(frozen=True, slots=True)
class TargetBlueprint:
    name: str
    source_alias: str
    target_frequency: str
    construction_rule: str
    notes: str


def quarterly_gdp_target_blueprint(settings: ProjectSettings) -> TargetBlueprint:
    return TargetBlueprint(
        name=f"{settings.geography.aggregate.lower()}_quarterly_gdp",
        source_alias="quarterly_gdp",
        target_frequency="Q",
        construction_rule="Use the configured quarterly GDP series as the headline target.",
        notes="Benchmark target for model training, evaluation, and forecast vintage analysis.",
    )


def monthly_bridge_target_blueprint(settings: ProjectSettings) -> TargetBlueprint:
    return TargetBlueprint(
        name=f"{settings.geography.aggregate.lower()}_monthly_gdp_bridge",
        source_alias="quarterly_gdp",
        target_frequency="M",
        construction_rule=(
            "Map the quarterly GDP target onto constituent months so monthly indicators can be aligned "
            "to a ragged-edge bridge target."
        ),
        notes="Placeholder for Mariano-Murasawa style or simpler within-quarter target design.",
    )


@dataclass(frozen=True, slots=True)
class GdpTargetArtifacts:
    quarterly_targets: pd.DataFrame
    monthly_bridge_targets: pd.DataFrame
    stage_1_targets: pd.DataFrame
    stage_2_targets: pd.DataFrame
    stage_3_targets: pd.DataFrame
    alignment_markdown: str


DEFAULT_TARGET_INPUT_DATASET = "namq_10_gdp"
DEFAULT_QUARTERLY_GDP_GEO_PANEL = ("EA20", "DE", "FR", "IT", "ES", "NL", "BE", "AT")
TARGET_OUTPUT_SUBDIR = "targets"
SDMX_CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0"


def default_quarterly_gdp_geo_panel(
    selected_series_path: Path = DEFAULT_SELECTED_SERIES_PATH,
) -> tuple[str, ...]:
    try:
        selected_config = load_selected_series_config(selected_series_path)
        panel = selected_config.geo_panels.get("ea20_plus_large_members", [])
        if panel:
            return tuple(panel)
    except Exception:
        pass
    return DEFAULT_QUARTERLY_GDP_GEO_PANEL


def build_quarterly_real_gdp_request(
    geos: list[str],
    start_quarter: str,
    end_quarter: str | None = None,
    dataset_id: str | None = None,
    unit: str | None = None,
    seasonal_adjustment: str = "SCA",
    na_item: str = "B1GQ",
) -> HttpRequestSpec:
    settings = get_settings()
    effective_dataset = dataset_id or settings.datasets.quarterly_gdp or DEFAULT_TARGET_INPUT_DATASET
    effective_unit = unit or settings.preferred_units.quarterly_gdp

    params: dict[str, str] = {
        "attributes": "none",
        "measures": "all",
        "c[FREQ]": "Q",
        "c[NA_ITEM]": na_item,
        "c[UNIT]": effective_unit,
        "c[S_ADJ]": seasonal_adjustment,
        "c[GEO]": ",".join(geos),
        "c[TIME_PERIOD]": f"ge:{validate_quarter_period(start_quarter)}",
    }
    if end_quarter:
        params["c[TIME_PERIOD]"] = params["c[TIME_PERIOD]"] + "+" + f"le:{validate_quarter_period(end_quarter)}"

    return HttpRequestSpec(
        url=(
            f"{settings.eurostat_api.base_url.rstrip('/')}/data/dataflow/"
            f"{settings.eurostat_api.agency_id}/{effective_dataset}/{settings.eurostat_api.dataflow_version}"
        ),
        params=params,
        headers={"Accept": SDMX_CSV_ACCEPT},
        response_format="sdmx-csv",
        provider="eurostat",
    )


def pull_real_gdp_levels(
    start_quarter: str,
    geos: list[str] | None = None,
    end_quarter: str | None = None,
    dataset_id: str | None = None,
    unit: str | None = None,
    seasonal_adjustment: str = "SCA",
    na_item: str = "B1GQ",
    force_refresh: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    settings = get_settings()
    target_geos = geos or list(default_quarterly_gdp_geo_panel())
    active_logger = logger or logging.getLogger(__name__)

    request_spec = build_quarterly_real_gdp_request(
        geos=target_geos,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
        dataset_id=dataset_id,
        unit=unit,
        seasonal_adjustment=seasonal_adjustment,
        na_item=na_item,
    )
    cache = FileResponseCache(settings.resolve_path(settings.download.cache_dir))
    client = RetryingHttpClient(
        timeout_seconds=settings.download.timeout_seconds,
        max_retries=settings.download.max_retries,
        retry_backoff_seconds=settings.download.retry_backoff_seconds,
        user_agent=settings.download.user_agent,
    )

    try:
        payload = client.fetch(
            request_spec,
            cache=cache,
            force_refresh=force_refresh,
            logger=active_logger,
        )
    finally:
        client.close()

    raw_dir = settings.resolve_path(settings.paths.raw_data_dir) / "eurostat" / TARGET_OUTPUT_SUBDIR
    raw_path = raw_dir / f"quarterly_real_gdp__{start_quarter.replace('-', '')}__sdmx-csv.csv"
    write_raw_response(payload, raw_path)
    levels = normalize_quarterly_gdp_sdmx_csv(payload.text)
    active_logger.info("Pulled %s quarterly GDP rows for %s geos", len(levels), len(target_geos))
    return levels


def normalize_quarterly_gdp_sdmx_csv(csv_text: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(csv_text))
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "quarter",
                "quarter_end",
                "quarter_start",
                "geo",
                "gdp_real_level",
                "unit",
                "seasonal_adjustment",
                "na_item",
                "source_dataset",
            ]
        )

    normalized_columns = {column.casefold(): column for column in raw.columns}
    quarter_column = normalized_columns.get("time_period")
    geo_column = normalized_columns.get("geo")
    value_column = normalized_columns.get("obs_value")
    unit_column = normalized_columns.get("unit")
    seasonal_column = normalized_columns.get("s_adj")
    na_item_column = normalized_columns.get("na_item")
    dataset_column = normalized_columns.get("structure_id")

    quarter_period = raw[quarter_column].astype("string")
    quarter_start = pd.PeriodIndex(quarter_period, freq="Q").to_timestamp(how="start")
    quarter_end = pd.PeriodIndex(quarter_period, freq="Q").to_timestamp(how="end").normalize()

    frame = pd.DataFrame(
        {
            "quarter": quarter_period,
            "quarter_end": quarter_end,
            "quarter_start": quarter_start,
            "geo": raw[geo_column].astype("string"),
            "gdp_real_level": pd.to_numeric(raw[value_column], errors="coerce"),
            "unit": raw[unit_column].astype("string") if unit_column is not None else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "seasonal_adjustment": (
                raw[seasonal_column].astype("string")
                if seasonal_column is not None
                else pd.Series(pd.NA, index=raw.index, dtype="string")
            ),
            "na_item": raw[na_item_column].astype("string") if na_item_column is not None else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "source_dataset": (
                raw[dataset_column].astype("string")
                if dataset_column is not None
                else pd.Series(DEFAULT_TARGET_INPUT_DATASET, index=raw.index, dtype="string")
            ),
        }
    )
    frame["source_dataset"] = frame["source_dataset"].str.extract(r"ESTAT:(?P<dataset>[^()]+)", expand=True).fillna(frame["source_dataset"])
    return frame.sort_values(["geo", "quarter_end"], kind="stable").reset_index(drop=True)


def build_quarterly_gdp_targets(levels: pd.DataFrame, aggregate_geo: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    headline_geo = aggregate_geo or settings.geography.aggregate
    target_frames: list[pd.DataFrame] = []

    for geo, group in levels.groupby("geo", sort=False):
        working = group.sort_values("quarter_end", kind="stable").copy()
        working["qoq_real_gdp_growth"] = _percent_change(working["gdp_real_level"], periods=1)
        working["yoy_real_gdp_growth"] = _percent_change(working["gdp_real_level"], periods=4)
        working["is_aggregate"] = working["geo"].eq(headline_geo)
        working["target_frequency"] = "Q"
        working["target_name"] = working["geo"].str.lower() + "_real_gdp_growth"
        target_frames.append(working)

    targets = pd.concat(target_frames, ignore_index=True)
    targets["quarter_index"] = pd.PeriodIndex(targets["quarter"], freq="Q").astype("string")
    return targets.sort_values(["geo", "quarter_end"], kind="stable").reset_index(drop=True)


def build_monthly_bridge_targets(quarterly_targets: pd.DataFrame) -> pd.DataFrame:
    bridge_frames: list[pd.DataFrame] = []

    for _, row in quarterly_targets.iterrows():
        quarter = pd.Period(str(row["quarter"]), freq="Q")
        months = pd.period_range(start=quarter.asfreq("M", "start"), end=quarter.asfreq("M", "end"), freq="M")
        month_rows = []
        for month_in_quarter, month in enumerate(months, start=1):
            month_end = month.to_timestamp(how="end").normalize()
            month_rows.append(
                {
                    "month_end": month_end,
                    "quarter": str(row["quarter"]),
                    "quarter_end": row["quarter_end"],
                    "quarter_start": row["quarter_start"],
                    "month_in_quarter": month_in_quarter,
                    "nowcast_stage": f"month_{month_in_quarter}",
                    "geo": row["geo"],
                    "is_aggregate": bool(row["is_aggregate"]),
                    "gdp_real_level": row["gdp_real_level"],
                    "qoq_real_gdp_growth": row["qoq_real_gdp_growth"],
                    "yoy_real_gdp_growth": row["yoy_real_gdp_growth"],
                    "unit": row["unit"],
                    "seasonal_adjustment": row["seasonal_adjustment"],
                    "na_item": row["na_item"],
                    "source_dataset": row["source_dataset"],
                    "feature_snapshot_month_end": month_end,
                    "feature_alignment_key": f"{row['geo']}::{month_end.strftime('%Y-%m-%d')}",
                    "feature_alignment_notes": (
                        "Monthly features observed or available by this month-end map to the quarter shown here. "
                        "Filter nowcast_stage to month_1, month_2, or month_3 to emulate incomplete-quarter nowcasts."
                    ),
                }
            )
        bridge_frames.append(pd.DataFrame.from_records(month_rows))

    bridge = pd.concat(bridge_frames, ignore_index=True)
    return bridge.sort_values(["geo", "month_end"], kind="stable").reset_index(drop=True)


def build_gdp_target_pipeline(
    start_quarter: str,
    geos: list[str] | None = None,
    end_quarter: str | None = None,
    dataset_id: str | None = None,
    unit: str | None = None,
    seasonal_adjustment: str = "SCA",
    na_item: str = "B1GQ",
    force_refresh: bool = False,
    logger: logging.Logger | None = None,
) -> GdpTargetArtifacts:
    active_logger = logger or logging.getLogger(__name__)
    levels = pull_real_gdp_levels(
        start_quarter=start_quarter,
        geos=geos,
        end_quarter=end_quarter,
        dataset_id=dataset_id,
        unit=unit,
        seasonal_adjustment=seasonal_adjustment,
        na_item=na_item,
        force_refresh=force_refresh,
        logger=active_logger,
    )
    quarterly_targets = build_quarterly_gdp_targets(levels)
    monthly_bridge = build_monthly_bridge_targets(quarterly_targets)
    alignment_markdown = render_target_alignment_report(quarterly_targets, monthly_bridge)

    return GdpTargetArtifacts(
        quarterly_targets=quarterly_targets,
        monthly_bridge_targets=monthly_bridge,
        stage_1_targets=monthly_bridge.loc[monthly_bridge["month_in_quarter"] == 1].reset_index(drop=True),
        stage_2_targets=monthly_bridge.loc[monthly_bridge["month_in_quarter"] == 2].reset_index(drop=True),
        stage_3_targets=monthly_bridge.loc[monthly_bridge["month_in_quarter"] == 3].reset_index(drop=True),
        alignment_markdown=alignment_markdown,
    )


def save_gdp_target_outputs(
    artifacts: GdpTargetArtifacts,
    output_root: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    active_logger = logger or logging.getLogger(__name__)
    root = output_root or settings.resolve_path(settings.paths.processed_data_dir) / TARGET_OUTPUT_SUBDIR
    root.mkdir(parents=True, exist_ok=True)

    saved_paths = {
        "quarterly_targets": _write_table_with_csv_fallback(artifacts.quarterly_targets, root / "quarterly_real_gdp_targets", active_logger),
        "monthly_bridge_targets": _write_table_with_csv_fallback(artifacts.monthly_bridge_targets, root / "monthly_bridge_targets", active_logger),
        "month_1_targets": _write_table_with_csv_fallback(artifacts.stage_1_targets, root / "monthly_bridge_targets_month_1", active_logger),
        "month_2_targets": _write_table_with_csv_fallback(artifacts.stage_2_targets, root / "monthly_bridge_targets_month_2", active_logger),
        "month_3_targets": _write_table_with_csv_fallback(artifacts.stage_3_targets, root / "monthly_bridge_targets_month_3", active_logger),
    }

    report_path = settings.resolve_path(settings.paths.outputs_dir) / "gdp_target_alignment.md"
    report_path.write_text(artifacts.alignment_markdown, encoding="utf-8")
    saved_paths["alignment_report"] = report_path
    return saved_paths


def render_target_alignment_report(
    quarterly_targets: pd.DataFrame,
    monthly_bridge_targets: pd.DataFrame,
) -> str:
    if quarterly_targets.empty:
        return "# GDP Target Alignment\n\n_No quarterly GDP targets were built._\n"

    geos = ", ".join(sorted(quarterly_targets["geo"].dropna().astype(str).unique()))
    unit = quarterly_targets["unit"].dropna().astype(str).iloc[0]
    seasonal_adjustment = quarterly_targets["seasonal_adjustment"].dropna().astype(str).iloc[0]
    source_dataset = quarterly_targets["source_dataset"].dropna().astype(str).iloc[0]
    quarter_min = quarterly_targets["quarter"].min()
    quarter_max = quarterly_targets["quarter"].max()

    stage_counts = (
        monthly_bridge_targets.groupby("month_in_quarter")["month_end"].count().sort_index().to_dict()
        if not monthly_bridge_targets.empty
        else {}
    )

    lines = [
        "# GDP Target Alignment",
        "",
        "## Source Series",
        "",
        f"- Dataset: `{source_dataset}`",
        f"- National accounts item: `B1GQ`",
        f"- Real-volume unit: `{unit}`",
        f"- Seasonal adjustment: `{seasonal_adjustment}`",
        f"- Quarterly coverage: `{quarter_min}` to `{quarter_max}`",
        f"- Geographies: `{geos}`",
        "",
        "## Target Definitions",
        "",
        "- `qoq_real_gdp_growth` is computed as `100 * (GDP_t / GDP_{t-1} - 1)` from quarterly real GDP levels.",
        "- `yoy_real_gdp_growth` is computed as `100 * (GDP_t / GDP_{t-4} - 1)` from quarterly real GDP levels.",
        "",
        "## Monthly Bridge Mapping",
        "",
        "- Each quarter is expanded to three month-end rows: `month_1`, `month_2`, and `month_3`.",
        "- January/February/March map to `Q1`, April/May/June map to `Q2`, July/August/September map to `Q3`, and October/November/December map to `Q4`.",
        "- The quarterly GDP target is repeated across the three months of its quarter so incomplete-quarter nowcasts can be trained or evaluated by filtering on `month_in_quarter`.",
        f"- Bridge rows by stage: `{stage_counts}`",
        "",
        "## Feature Alignment",
        "",
        "- Observation-date monthly features should join to `monthly_bridge_targets.month_end` using the same month-end timestamp.",
        "- Availability-date monthly features from the lag-aware feature pipeline should also join on month-end, but using the feature matrix built on `available_month_end`.",
        "- `month_in_quarter = 1` represents the information set after the first month of the quarter, `2` after the second month, and `3` after the third month.",
        "- The target quarter is never shifted: all three within-quarter monthly snapshots point to the same quarter-end GDP outcome.",
        "",
    ]
    return "\n".join(lines)


def configure_logging(level: str = "INFO") -> Path:
    settings = get_settings()
    log_path = settings.resolve_path(settings.paths.outputs_dir) / "logs" / "gdp_targets.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build quarterly real-GDP target datasets and monthly bridge targets.")
    parser.add_argument("--start", required=True, help="Quarterly start period in YYYY-Q# format.")
    parser.add_argument("--end", help="Optional quarterly end period in YYYY-Q# format.")
    parser.add_argument(
        "--geo",
        action="append",
        default=[],
        help="Optional geography code filter. Repeat the flag to specify multiple geographies.",
    )
    parser.add_argument("--unit", help="Optional Eurostat quarterly GDP unit override.")
    parser.add_argument("--dataset", help="Optional Eurostat dataset override.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore the HTTP cache and re-download the GDP series.")
    parser.add_argument("--log-level", default="INFO", help="Standard Python logging level.")
    args = parser.parse_args(argv)

    log_path = configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    target_geos = args.geo or list(default_quarterly_gdp_geo_panel())
    artifacts = build_gdp_target_pipeline(
        start_quarter=args.start,
        geos=target_geos,
        end_quarter=args.end,
        dataset_id=args.dataset,
        unit=args.unit,
        force_refresh=args.force_refresh,
        logger=logger,
    )
    saved_paths = save_gdp_target_outputs(artifacts, logger=logger)
    logger.info("GDP target pipeline completed. Log file: %s", log_path)
    print(
        f"Saved {len(artifacts.quarterly_targets)} quarterly target rows and "
        f"{len(artifacts.monthly_bridge_targets)} monthly bridge rows. "
        f"Alignment report: {saved_paths['alignment_report']}"
    )
    return 0


def validate_quarter_period(value: str) -> str:
    cleaned = (value or "").strip().upper()
    if not cleaned or len(cleaned) != 7 or cleaned[4] != "-" or cleaned[5] != "Q" or cleaned[6] not in {"1", "2", "3", "4"}:
        raise ValueError(f"Invalid quarterly period: {value}")
    return cleaned


def _percent_change(values: pd.Series, periods: int) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    prior = series.shift(periods)
    result = pd.Series(float("nan"), index=series.index, dtype="float64")
    valid = series.gt(0) & prior.gt(0)
    result.loc[valid] = 100.0 * (series.loc[valid] / prior.loc[valid] - 1.0)
    return result


def _write_table_with_csv_fallback(frame: pd.DataFrame, stem: Path, logger: logging.Logger) -> Path:
    csv_path = stem.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    try:
        write_dataframe_parquet(frame, stem.with_suffix(".parquet"))
    except RuntimeError as exc:
        logger.warning("Skipping parquet output for %s: %s", stem.name, exc)
    return csv_path


if __name__ == "__main__":
    raise SystemExit(main())
