from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import argparse
import logging

import pandas as pd

from config import PROJECT_ROOT, get_settings
from data_access.ingestion import write_dataframe_parquet
from data_access.pull_eurostat import (
    DEFAULT_SELECTED_SERIES_PATH,
    SelectedSeriesConfig,
    load_selected_series_config,
)
from transforms import (
    align_month_end,
    apply_named_transformation,
    available_month_end,
    detect_outliers,
    infer_change_style,
    month_end_index,
    one_month_change,
    three_month_over_three_month_annualized,
    trailing_zscore,
    year_over_year_change,
)

DEFAULT_INPUT_PARQUET = PROJECT_ROOT / "data_processed" / "eurostat" / "selected_series_monthly.parquet"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data_processed" / "eurostat" / "selected_series_monthly.csv"
OUTPUT_DIRNAME = "features"
LONG_NUMERIC_COLUMNS = [
    "raw_value",
    "configured_value",
    "change_1m",
    "change_3m_3m_saar",
    "change_yoy",
    "zscore_5y",
    "is_missing_observation",
    "is_outlier_raw",
    "is_outlier_configured",
]


@dataclass(frozen=True, slots=True)
class FeaturePipelineArtifacts:
    observation_long: pd.DataFrame
    availability_long: pd.DataFrame
    observation_wide: pd.DataFrame
    availability_wide: pd.DataFrame
    feature_availability: pd.DataFrame
    coverage_report_markdown: str


def load_normalized_monthly_observations(path: Path | None = None) -> pd.DataFrame:
    candidate = path or DEFAULT_INPUT_PARQUET
    if candidate.suffix.casefold() == ".csv":
        return pd.read_csv(candidate, parse_dates=["date"])

    if candidate.exists():
        try:
            return pd.read_parquet(candidate)
        except (ImportError, ValueError) as exc:
            csv_candidate = candidate.with_suffix(".csv")
            if csv_candidate.exists():
                return pd.read_csv(csv_candidate, parse_dates=["date"])
            raise RuntimeError(
                f"Unable to read parquet input at {candidate}. Install pyarrow or provide a CSV input."
            ) from exc

    csv_candidate = candidate.with_suffix(".csv")
    if csv_candidate.exists():
        return pd.read_csv(csv_candidate, parse_dates=["date"])

    raise FileNotFoundError(f"No normalized monthly observation file found at {candidate} or {csv_candidate}.")


def build_monthly_feature_pipeline(
    observations: pd.DataFrame,
    selected_config: SelectedSeriesConfig,
    start_period: str | None = None,
    end_period: str | None = None,
) -> FeaturePipelineArtifacts:
    settings = get_settings()
    monthly = _prepare_observation_frame(observations, selected_config)
    monthly = _append_country_aggregates(monthly, selected_config)

    start_month_end = _resolve_month_end(start_period or settings.date_range.start.strftime("%Y-%m"))
    end_month_end = _resolve_month_end(end_period or settings.date_range.end.strftime("%Y-%m"))

    standardized = _complete_monthly_grid(monthly, start_month_end=start_month_end, end_month_end=end_month_end)
    engineered = _engineer_group_features(standardized)
    availability_long = _build_availability_long(engineered)
    observation_wide = _pivot_feature_matrix(engineered)
    availability_wide = _pivot_feature_matrix(availability_long)
    feature_availability = _build_feature_availability(engineered)
    coverage_report_markdown = _render_coverage_report(
        engineered=engineered,
        feature_availability=feature_availability,
        start_month_end=start_month_end,
        end_month_end=end_month_end,
    )

    return FeaturePipelineArtifacts(
        observation_long=engineered,
        availability_long=availability_long,
        observation_wide=observation_wide,
        availability_wide=availability_wide,
        feature_availability=feature_availability,
        coverage_report_markdown=coverage_report_markdown,
    )


def save_feature_pipeline_outputs(
    artifacts: FeaturePipelineArtifacts,
    output_root: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    root = output_root or settings.resolve_path(settings.paths.processed_data_dir) / OUTPUT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger(__name__)

    saved_paths = {
        "observation_long_csv": _write_table_with_csv_fallback(artifacts.observation_long, root / "monthly_features_long", active_logger),
        "availability_long_csv": _write_table_with_csv_fallback(artifacts.availability_long, root / "monthly_features_available_long", active_logger),
        "observation_wide_csv": _write_table_with_csv_fallback(artifacts.observation_wide.reset_index(), root / "monthly_feature_matrix_wide", active_logger),
        "availability_wide_csv": _write_table_with_csv_fallback(artifacts.availability_wide.reset_index(), root / "monthly_feature_matrix_available_wide", active_logger),
        "feature_availability_csv": _write_table_with_csv_fallback(artifacts.feature_availability, root / "feature_availability", active_logger),
    }

    report_path = settings.resolve_path(settings.paths.outputs_dir) / "monthly_feature_coverage_report.md"
    report_path.write_text(artifacts.coverage_report_markdown, encoding="utf-8")
    saved_paths["coverage_report"] = report_path
    return saved_paths


def configure_logging(level: str = "INFO") -> Path:
    settings = get_settings()
    log_path = settings.resolve_path(settings.paths.outputs_dir) / "logs" / "monthly_features.log"
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
    parser = argparse.ArgumentParser(description="Build monthly macro nowcasting features from normalized monthly observations.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PARQUET), help="Path to the normalized monthly observation file.")
    parser.add_argument("--config", default=str(DEFAULT_SELECTED_SERIES_PATH), help="Path to config/selected_series.yml.")
    parser.add_argument("--start", help="Optional feature-matrix start month in YYYY-MM format.")
    parser.add_argument("--end", help="Optional feature-matrix end month in YYYY-MM format.")
    parser.add_argument("--log-level", default="INFO", help="Standard Python logging level.")
    args = parser.parse_args(argv)

    log_path = configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    observations = load_normalized_monthly_observations(Path(args.input))
    selected_config = load_selected_series_config(Path(args.config))
    artifacts = build_monthly_feature_pipeline(
        observations=observations,
        selected_config=selected_config,
        start_period=args.start,
        end_period=args.end,
    )
    saved_paths = save_feature_pipeline_outputs(artifacts, logger=logger)
    logger.info("Feature pipeline completed. Log file: %s", log_path)
    print(
        f"Saved {len(artifacts.observation_long)} long rows and "
        f"{artifacts.observation_wide.shape[1]} wide feature columns. "
        f"Coverage report: {saved_paths['coverage_report']}"
    )
    return 0


def _prepare_observation_frame(observations: pd.DataFrame, selected_config: SelectedSeriesConfig) -> pd.DataFrame:
    required_columns = {"date", "geo", "indicator_code", "value", "unit", "seasonal_adjustment", "source_dataset"}
    missing_columns = required_columns - set(observations.columns)
    if missing_columns:
        raise ValueError(f"Observation frame is missing required columns: {sorted(missing_columns)}")

    metadata_rows = []
    for alias, definition in selected_config.selected_series.items():
        metadata_rows.append(
            {
                "indicator_code": alias,
                "concept": definition.concept,
                "configured_transformation": definition.transformation,
                "preferred_unit": _dimension_scalar(definition.dimensions.get("unit")),
                "preferred_seasonal_adjustment": _dimension_scalar(definition.dimensions.get("s_adj")),
                "release_lag_months": definition.release_lag_months or 0,
                "release_lag_days": definition.release_lag_days or 0,
                "aggregate_from_panel": definition.aggregate_from_panel,
                "aggregate_method_preference": definition.aggregate_method or "simple_mean",
                "configured_geos": tuple(_dimension_list(definition.dimensions.get("geo"))),
            }
        )

    metadata = pd.DataFrame.from_records(metadata_rows)
    monthly = observations.copy()
    monthly["month_end"] = align_month_end(monthly["date"])
    monthly["geo"] = monthly["geo"].astype("string")
    monthly["indicator_code"] = monthly["indicator_code"].astype("string")
    monthly["raw_value"] = pd.to_numeric(monthly["value"], errors="coerce")
    monthly["unit"] = monthly["unit"].fillna("").astype("string")
    monthly["seasonal_adjustment"] = monthly["seasonal_adjustment"].fillna("").astype("string")
    monthly["source_dataset"] = monthly["source_dataset"].astype("string")
    monthly = monthly.merge(metadata, how="inner", on="indicator_code")
    monthly["unit"] = monthly["unit"].mask(monthly["unit"].eq(""), monthly["preferred_unit"])
    monthly["seasonal_adjustment"] = monthly["seasonal_adjustment"].mask(
        monthly["seasonal_adjustment"].eq(""),
        monthly["preferred_seasonal_adjustment"],
    )
    monthly["frequency"] = "M"
    monthly["aggregation_method"] = "official"
    monthly["aggregate_source_panel"] = pd.Series(pd.NA, index=monthly.index, dtype="string")
    keep_columns = [
        "month_end",
        "geo",
        "indicator_code",
        "concept",
        "configured_transformation",
        "raw_value",
        "unit",
        "seasonal_adjustment",
        "source_dataset",
        "frequency",
        "release_lag_months",
        "release_lag_days",
        "aggregate_from_panel",
        "aggregate_method_preference",
        "configured_geos",
        "aggregation_method",
        "aggregate_source_panel",
    ]
    return monthly.loc[:, keep_columns].drop_duplicates(subset=["month_end", "geo", "indicator_code"], keep="last")


def _append_country_aggregates(monthly: pd.DataFrame, selected_config: SelectedSeriesConfig) -> pd.DataFrame:
    settings = get_settings()
    aggregate_geo = settings.geography.aggregate
    aggregate_rows: list[pd.DataFrame] = []

    for indicator_code, group in monthly.groupby("indicator_code", sort=False):
        observed_geos = set(group["geo"].dropna().astype(str))
        if aggregate_geo in observed_geos:
            continue

        indicator_definition = selected_config.selected_series[str(indicator_code)]
        candidate_panel = indicator_definition.aggregate_from_panel
        aggregate_method = indicator_definition.aggregate_method or "simple_mean"
        if candidate_panel is None:
            candidate_panel = _infer_panel_name(observed_geos, selected_config.geo_panels)
        if candidate_panel is None:
            continue

        panel_members = selected_config.geo_panels.get(candidate_panel, [])
        if not panel_members:
            continue

        eligible = group.loc[group["geo"].isin(panel_members)].copy()
        if eligible["geo"].nunique() < 2:
            continue

        if aggregate_method == "sum":
            raw_aggregation = "sum"
        elif aggregate_method == "simple_mean":
            raw_aggregation = "mean"
        else:
            raise ValueError(f"Unsupported aggregate method for {indicator_code}: {aggregate_method}")

        aggregate = (
            eligible.groupby("month_end", as_index=False)
            .agg(
                {
                    "raw_value": raw_aggregation,
                    "concept": "first",
                    "configured_transformation": "first",
                    "unit": "first",
                    "seasonal_adjustment": "first",
                    "source_dataset": "first",
                    "frequency": "first",
                    "release_lag_months": "first",
                    "release_lag_days": "first",
                    "aggregate_from_panel": "first",
                    "aggregate_method_preference": "first",
                }
            )
            .assign(
                geo=f"AGG_{candidate_panel.upper()}",
                indicator_code=indicator_code,
                aggregation_method=aggregate_method,
                aggregate_source_panel=candidate_panel,
            )
        )
        aggregate["configured_geos"] = [tuple(panel_members)] * len(aggregate)
        aggregate_rows.append(aggregate)

    if not aggregate_rows:
        return monthly

    combined = pd.concat([monthly, *aggregate_rows], ignore_index=True)
    combined["aggregate_source_panel"] = combined["aggregate_source_panel"].astype("string")
    return combined


def _complete_monthly_grid(monthly: pd.DataFrame, start_month_end: pd.Timestamp, end_month_end: pd.Timestamp) -> pd.DataFrame:
    group_columns = ["indicator_code", "geo"]
    grid_frames: list[pd.DataFrame] = []

    for (_, _), group in monthly.groupby(group_columns, sort=False):
        group = group.sort_values("month_end", kind="stable").reset_index(drop=True)
        full_index = month_end_index(start=start_month_end, end=end_month_end)
        reindexed = group.set_index("month_end").reindex(full_index)
        reindexed.index.name = "month_end"
        reindexed = reindexed.reset_index()
        for column in [
            "indicator_code",
            "geo",
            "concept",
            "configured_transformation",
            "unit",
            "seasonal_adjustment",
            "source_dataset",
            "frequency",
            "release_lag_months",
            "release_lag_days",
            "aggregate_from_panel",
            "aggregate_method_preference",
            "configured_geos",
            "aggregation_method",
            "aggregate_source_panel",
        ]:
            if column in reindexed.columns:
                reindexed[column] = reindexed[column].ffill().bfill()
        reindexed["is_missing_observation"] = reindexed["raw_value"].isna()
        grid_frames.append(reindexed)

    standardized = pd.concat(grid_frames, ignore_index=True)
    standardized["month_end"] = pd.to_datetime(standardized["month_end"], errors="coerce")
    standardized["geo"] = standardized["geo"].astype("string")
    standardized["indicator_code"] = standardized["indicator_code"].astype("string")
    standardized["concept"] = standardized["concept"].astype("string")
    standardized["configured_transformation"] = standardized["configured_transformation"].astype("string")
    standardized["unit"] = standardized["unit"].astype("string")
    standardized["seasonal_adjustment"] = standardized["seasonal_adjustment"].astype("string")
    standardized["source_dataset"] = standardized["source_dataset"].astype("string")
    standardized["frequency"] = standardized["frequency"].astype("string")
    standardized["aggregation_method"] = standardized["aggregation_method"].astype("string")
    standardized["aggregate_source_panel"] = standardized["aggregate_source_panel"].astype("string")
    standardized["raw_value"] = pd.to_numeric(standardized["raw_value"], errors="coerce")
    return standardized.sort_values(["indicator_code", "geo", "month_end"], kind="stable").reset_index(drop=True)


def _engineer_group_features(standardized: pd.DataFrame) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []

    for (_, _), group in standardized.groupby(["indicator_code", "geo"], sort=False):
        working = group.sort_values("month_end", kind="stable").copy()
        raw = working["raw_value"]
        transform_name = str(working["configured_transformation"].iloc[0])
        unit = _clean_string(working["unit"].iloc[0])
        change_style = infer_change_style(unit=unit, transformation=transform_name)

        working["configured_value"] = apply_named_transformation(raw, transformation=transform_name, unit=unit)
        working["change_1m"] = one_month_change(raw, style=change_style)
        working["change_3m_3m_saar"] = three_month_over_three_month_annualized(raw, style=change_style)
        working["change_yoy"] = year_over_year_change(raw, style=change_style)
        working["zscore_5y"] = trailing_zscore(raw)
        working["is_outlier_raw"] = detect_outliers(raw).astype("int8")
        working["is_outlier_configured"] = detect_outliers(working["configured_value"]).astype("int8")
        working["is_missing_observation"] = working["is_missing_observation"].astype("int8")
        working["available_month_end"] = available_month_end(
            working["month_end"],
            release_lag_months=int(working["release_lag_months"].iloc[0] or 0),
            release_lag_days=int(working["release_lag_days"].iloc[0] or 0),
        )
        feature_frames.append(working)

    engineered = pd.concat(feature_frames, ignore_index=True)
    return engineered.sort_values(["indicator_code", "geo", "month_end"], kind="stable").reset_index(drop=True)


def _build_availability_long(engineered: pd.DataFrame) -> pd.DataFrame:
    available = engineered.copy()
    available["observation_month_end"] = available["month_end"]
    available["month_end"] = pd.to_datetime(available["available_month_end"], errors="coerce")
    return available.sort_values(["indicator_code", "geo", "month_end"], kind="stable").reset_index(drop=True)


def _pivot_feature_matrix(engineered: pd.DataFrame) -> pd.DataFrame:
    melted = engineered.loc[:, ["month_end", "indicator_code", "geo", *LONG_NUMERIC_COLUMNS]].melt(
        id_vars=["month_end", "indicator_code", "geo"],
        value_vars=LONG_NUMERIC_COLUMNS,
        var_name="feature_name",
        value_name="feature_value",
    )
    melted["column_name"] = (
        melted["indicator_code"].astype(str)
        + "__"
        + melted["geo"].astype(str)
        + "__"
        + melted["feature_name"].astype(str)
    )
    wide = melted.pivot(index="month_end", columns="column_name", values="feature_value")
    wide = wide.sort_index().sort_index(axis=1)
    return wide


def _build_feature_availability(engineered: pd.DataFrame) -> pd.DataFrame:
    availability_rows = []
    for (_, _), group in engineered.groupby(["indicator_code", "geo"], sort=False):
        observed = group.loc[group["raw_value"].notna()].sort_values("month_end", kind="stable")
        configured = group.loc[group["configured_value"].notna()].sort_values("month_end", kind="stable")

        availability_rows.append(
            {
                "indicator_code": group["indicator_code"].iloc[0],
                "geo": group["geo"].iloc[0],
                "concept": group["concept"].iloc[0],
                "configured_transformation": group["configured_transformation"].iloc[0],
                "unit": group["unit"].iloc[0],
                "seasonal_adjustment": group["seasonal_adjustment"].iloc[0],
                "aggregation_method": group["aggregation_method"].iloc[0],
                "aggregate_source_panel": group["aggregate_source_panel"].iloc[0],
                "release_lag_months": int(group["release_lag_months"].iloc[0] or 0),
                "release_lag_days": int(group["release_lag_days"].iloc[0] or 0),
                "first_observation_month_end": observed["month_end"].min() if not observed.empty else pd.NaT,
                "last_observation_month_end": observed["month_end"].max() if not observed.empty else pd.NaT,
                "last_available_month_end": observed["available_month_end"].max() if not observed.empty else pd.NaT,
                "observation_count": int(observed.shape[0]),
                "missing_count": int(group["is_missing_observation"].sum()),
                "missing_share": float(group["is_missing_observation"].mean()),
                "outlier_raw_count": int(group["is_outlier_raw"].sum()),
                "outlier_configured_count": int(group["is_outlier_configured"].sum()),
                "latest_raw_value": observed["raw_value"].iloc[-1] if not observed.empty else pd.NA,
                "latest_configured_value": configured["configured_value"].iloc[-1] if not configured.empty else pd.NA,
            }
        )

    availability = pd.DataFrame.from_records(availability_rows)
    date_columns = [
        "first_observation_month_end",
        "last_observation_month_end",
        "last_available_month_end",
    ]
    for column in date_columns:
        availability[column] = pd.to_datetime(availability[column], errors="coerce")
    return availability.sort_values(["indicator_code", "geo"], kind="stable").reset_index(drop=True)


def _render_coverage_report(
    engineered: pd.DataFrame,
    feature_availability: pd.DataFrame,
    start_month_end: pd.Timestamp,
    end_month_end: pd.Timestamp,
) -> str:
    latest_month = engineered["month_end"].max()
    latest_month_available = feature_availability["last_observation_month_end"].eq(latest_month).sum()
    overall_missing_share = engineered["is_missing_observation"].mean()
    total_outliers = int(engineered["is_outlier_raw"].sum())

    summary_lines = [
        "# Monthly Feature Coverage Report",
        "",
        "## Summary",
        "",
        f"- Feature window: `{start_month_end.strftime('%Y-%m-%d')}` to `{end_month_end.strftime('%Y-%m-%d')}`",
        f"- Indicator-geo series: `{feature_availability.shape[0]}`",
        f"- Long rows: `{engineered.shape[0]}`",
        f"- Latest matrix month-end: `{latest_month.strftime('%Y-%m-%d')}`" if pd.notna(latest_month) else "- Latest matrix month-end: unavailable",
        f"- Series with an observation in the latest month: `{int(latest_month_available)}`",
        f"- Overall missing share: `{overall_missing_share:.2%}`",
        f"- Raw outlier flags: `{total_outliers}`",
        "",
        "## Coverage By Series",
        "",
        _markdown_table(
            headers=(
                "indicator_code",
                "geo",
                "first_obs",
                "last_obs",
                "last_available",
                "obs_count",
                "missing_share",
                "outliers",
            ),
            rows=[
                (
                    str(row.indicator_code),
                    str(row.geo),
                    _format_date(row.first_observation_month_end),
                    _format_date(row.last_observation_month_end),
                    _format_date(row.last_available_month_end),
                    str(int(row.observation_count)),
                    f"{float(row.missing_share):.1%}",
                    str(int(row.outlier_raw_count)),
                )
                for row in feature_availability.itertuples(index=False)
            ],
        ),
        "",
    ]
    return "\n".join(summary_lines)


def _write_table_with_csv_fallback(frame: pd.DataFrame, stem: Path, logger: logging.Logger) -> Path:
    csv_path = stem.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    try:
        write_dataframe_parquet(frame, stem.with_suffix(".parquet"))
    except RuntimeError as exc:
        logger.warning("Skipping parquet output for %s: %s", stem.name, exc)
    return csv_path


def _resolve_month_end(period: str) -> pd.Timestamp:
    return pd.Period(period, freq="M").to_timestamp() + pd.offsets.MonthEnd(0)


def _dimension_scalar(value: str | list[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value


def _dimension_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _infer_panel_name(observed_geos: set[str], geo_panels: dict[str, list[str]]) -> str | None:
    best_name: str | None = None
    best_overlap = 0
    for panel_name, members in geo_panels.items():
        overlap = len(observed_geos & set(members))
        if overlap >= 2 and overlap > best_overlap:
            best_name = panel_name
            best_overlap = overlap
    return best_name


def _clean_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "_No rows available._"
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


if __name__ == "__main__":
    raise SystemExit(main())
