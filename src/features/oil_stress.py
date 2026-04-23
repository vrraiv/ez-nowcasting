from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config import PROJECT_ROOT, get_settings
from config.yaml_utils import load_yaml_document
from data_access.ingestion import FileResponseCache, RetryingHttpClient, write_dataframe_parquet
from data_access.pull_eurostat import EurostatPuller, SelectedSeriesConfig, iter_selected_series
from features.monthly_features import build_monthly_feature_pipeline
from transforms.monthly import expanding_zscore, trailing_zscore

DEFAULT_COMPONENT_CONFIG_PATH = PROJECT_ROOT / "config" / "oil_stress_components.yml"
OUTPUT_SUBDIRECTORY = Path("indicators") / "oil_supply_stress"
RAW_SUBDIRECTORY = Path("eurostat") / "oil_supply_stress"

BUCKET_LABELS = {
    "direct_supply": "Direct supply",
    "pass_through_prices": "Pass-through prices",
    "refining_and_downstream": "Refining and downstream",
    "energy_intensive_activity": "Energy-intensive activity",
    "external_dependence": "External dependence",
    "logistics_and_freight": "Logistics and freight",
}
BUCKET_COLORS = {
    "direct_supply": "#8a1c1c",
    "pass_through_prices": "#c96f1a",
    "refining_and_downstream": "#d9a441",
    "energy_intensive_activity": "#6b8e23",
    "external_dependence": "#2f6b7c",
    "logistics_and_freight": "#465f9d",
}
INDEX_COLORS = {
    "simple_average_index_standardized": "#8a1c1c",
    "pca_index_standardized": "#2f6b7c",
    "structural_index_standardized": "#111111",
}


class OilStressComponentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    bucket: str
    dataset_id: str
    dimensions: dict[str, str | list[str]]
    signal_transformation: str
    stress_direction: Literal["positive", "negative"]
    aggregate_from_panel: str | None = None
    panel_aggregation: Literal["simple_mean", "sum", "mean"] | None = None
    structural_weight: float
    interpretation: str
    cyclical_sensitivity: Literal["low", "medium", "high"]


class OilStressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    target_geo: str
    minimum_component_count: int = 3
    geo_panels: dict[str, list[str]] = Field(default_factory=dict)
    components: dict[str, OilStressComponentDefinition]


@dataclass(frozen=True, slots=True)
class OilStressArtifacts:
    normalized_observations: pd.DataFrame
    component_panel_long: pd.DataFrame
    component_panel_wide: pd.DataFrame
    component_table: pd.DataFrame
    index_history: pd.DataFrame
    structural_component_contributions: pd.DataFrame
    structural_bucket_contributions: pd.DataFrame
    pca_loadings: pd.DataFrame
    narrative_markdown: str
    component_table_markdown: str
    index_history_chart_svg: str
    bucket_decomposition_chart_svg: str
    latest_component_chart_svg: str


def load_oil_stress_config(path: Path = DEFAULT_COMPONENT_CONFIG_PATH) -> OilStressConfig:
    raw_document = load_yaml_document(path)
    config = OilStressConfig.model_validate(raw_document)
    total_weight = sum(component.structural_weight for component in config.components.values())
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError(f"Oil-stress structural weights must sum to 1.0, found {total_weight:.6f}.")
    if config.minimum_component_count < 1:
        raise ValueError("minimum_component_count must be at least 1.")
    return config


def pull_oil_stress_observations(
    oil_config: OilStressConfig,
    start_period: str,
    end_period: str | None = None,
    response_format: str = "sdmx-csv",
    force_refresh: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    settings = get_settings()
    selected_config = _build_selected_series_config(oil_config)
    selections = iter_selected_series(selected_config)
    cache = FileResponseCache(settings.resolve_path(settings.download.cache_dir))
    client = RetryingHttpClient(
        timeout_seconds=settings.download.timeout_seconds,
        max_retries=settings.download.max_retries,
        retry_backoff_seconds=settings.download.retry_backoff_seconds,
        user_agent=settings.download.user_agent,
    )
    active_logger = logger or logging.getLogger(__name__)
    try:
        puller = EurostatPuller(
            client=client,
            cache=cache,
            raw_root=settings.resolve_path(settings.paths.raw_data_dir) / RAW_SUBDIRECTORY,
            processed_root=settings.resolve_path(settings.paths.processed_data_dir) / OUTPUT_SUBDIRECTORY,
            logger=active_logger,
        )
        return puller.pull(
            selected_series=selections,
            start_period=start_period,
            response_format=response_format,
            end_period=end_period,
            force_refresh=force_refresh,
            combined_output_name="oil_supply_stress_observations",
        )
    finally:
        client.close()


def load_component_observations(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError) as exc:
        csv_candidate = path.with_suffix(".csv")
        if csv_candidate.exists():
            return pd.read_csv(csv_candidate, parse_dates=["date"])
        raise RuntimeError(
            f"Unable to read component observations from {path}. Install pyarrow or pass a CSV file."
        ) from exc


def build_oil_supply_stress_artifacts(
    observations: pd.DataFrame,
    oil_config: OilStressConfig,
    start_period: str | None = None,
    end_period: str | None = None,
) -> OilStressArtifacts:
    selected_config = _build_selected_series_config(oil_config)
    feature_artifacts = build_monthly_feature_pipeline(
        observations=observations,
        selected_config=selected_config,
        start_period=start_period,
        end_period=end_period,
    )
    component_panel = _build_component_panel(feature_artifacts.observation_long, oil_config)
    component_wide = (
        component_panel.pivot(index="month_end", columns="component_code", values="standardized_component")
        .sort_index()
        .sort_index(axis=1)
    )
    index_history, component_contributions, bucket_contributions, pca_loadings = _build_index_history(
        component_panel,
        oil_config,
    )
    component_table = _build_component_table(component_panel, oil_config, pca_loadings)
    component_table_markdown = _render_component_table_markdown(component_table)
    narrative_markdown = _render_narrative(
        oil_config=oil_config,
        component_table=component_table,
        index_history=index_history,
        bucket_contributions=bucket_contributions,
        component_contributions=component_contributions,
    )
    index_history_chart_svg = _render_index_history_chart(index_history)
    bucket_decomposition_chart_svg = _render_bucket_decomposition_chart(bucket_contributions)
    latest_component_chart_svg = _render_latest_component_chart(component_contributions)

    return OilStressArtifacts(
        normalized_observations=observations,
        component_panel_long=component_panel,
        component_panel_wide=component_wide.reset_index(),
        component_table=component_table,
        index_history=index_history,
        structural_component_contributions=component_contributions,
        structural_bucket_contributions=bucket_contributions,
        pca_loadings=pca_loadings,
        narrative_markdown=narrative_markdown,
        component_table_markdown=component_table_markdown,
        index_history_chart_svg=index_history_chart_svg,
        bucket_decomposition_chart_svg=bucket_decomposition_chart_svg,
        latest_component_chart_svg=latest_component_chart_svg,
    )


def save_oil_supply_stress_outputs(
    artifacts: OilStressArtifacts,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    active_logger = logger or logging.getLogger(__name__)
    processed_root = settings.resolve_path(settings.paths.processed_data_dir) / OUTPUT_SUBDIRECTORY
    outputs_root = settings.resolve_path(settings.paths.outputs_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    saved_paths = {
        "component_panel_long": _write_table_with_csv_fallback(
            artifacts.component_panel_long,
            processed_root / "oil_supply_stress_component_panel_long",
            active_logger,
        ),
        "component_panel_wide": _write_table_with_csv_fallback(
            artifacts.component_panel_wide,
            processed_root / "oil_supply_stress_component_panel_wide",
            active_logger,
        ),
        "index_history": _write_table_with_csv_fallback(
            artifacts.index_history,
            processed_root / "oil_supply_stress_index_history",
            active_logger,
        ),
        "component_contributions": _write_table_with_csv_fallback(
            artifacts.structural_component_contributions,
            processed_root / "oil_supply_stress_structural_component_contributions",
            active_logger,
        ),
        "bucket_contributions": _write_table_with_csv_fallback(
            artifacts.structural_bucket_contributions.reset_index(),
            processed_root / "oil_supply_stress_structural_bucket_contributions",
            active_logger,
        ),
        "component_table": _write_table_with_csv_fallback(
            artifacts.component_table,
            outputs_root / "oil_supply_stress_component_table",
            active_logger,
        ),
        "pca_loadings": _write_table_with_csv_fallback(
            artifacts.pca_loadings,
            outputs_root / "oil_supply_stress_pca_loadings",
            active_logger,
        ),
    }

    component_table_md = outputs_root / "oil_supply_stress_component_table.md"
    component_table_md.write_text(artifacts.component_table_markdown, encoding="utf-8")
    saved_paths["component_table_markdown"] = component_table_md

    narrative_md = outputs_root / "oil_supply_stress_narrative.md"
    narrative_md.write_text(artifacts.narrative_markdown, encoding="utf-8")
    saved_paths["narrative_markdown"] = narrative_md

    index_svg = outputs_root / "oil_supply_stress_index_history.svg"
    index_svg.write_text(artifacts.index_history_chart_svg, encoding="utf-8")
    saved_paths["index_history_chart"] = index_svg

    decomposition_svg = outputs_root / "oil_supply_stress_bucket_decomposition.svg"
    decomposition_svg.write_text(artifacts.bucket_decomposition_chart_svg, encoding="utf-8")
    saved_paths["bucket_decomposition_chart"] = decomposition_svg

    latest_component_svg = outputs_root / "oil_supply_stress_latest_component_contributions.svg"
    latest_component_svg.write_text(artifacts.latest_component_chart_svg, encoding="utf-8")
    saved_paths["latest_component_chart"] = latest_component_svg

    return saved_paths


def configure_logging(level: str = "INFO") -> Path:
    settings = get_settings()
    log_path = settings.resolve_path(settings.paths.outputs_dir) / "logs" / "oil_supply_stress.log"
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
    parser = argparse.ArgumentParser(
        description="Build a euro-area oil supply stress indicator from filtered Eurostat monthly series."
    )
    parser.add_argument("--start", default="2000-01", help="Monthly lower bound in YYYY-MM format.")
    parser.add_argument("--end", help="Optional monthly upper bound in YYYY-MM format.")
    parser.add_argument("--config", default=str(DEFAULT_COMPONENT_CONFIG_PATH), help="Path to oil_stress_components.yml.")
    parser.add_argument(
        "--format",
        default="sdmx-csv",
        choices=("jsonstat", "sdmx-csv"),
        help="Preferred Eurostat response format for live pulls.",
    )
    parser.add_argument(
        "--input",
        help=(
            "Optional path to a pre-existing normalized component observation file. "
            "If omitted, the command pulls live Eurostat data."
        ),
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached HTTP responses for live pulls.")
    parser.add_argument("--log-level", default="INFO", help="Standard Python logging level.")
    args = parser.parse_args(argv)

    log_path = configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    oil_config = load_oil_stress_config(Path(args.config))

    if args.input:
        observations = load_component_observations(Path(args.input))
    else:
        observations = pull_oil_stress_observations(
            oil_config=oil_config,
            start_period=args.start,
            end_period=args.end,
            response_format=args.format,
            force_refresh=args.force_refresh,
            logger=logger,
        )

    artifacts = build_oil_supply_stress_artifacts(
        observations=observations,
        oil_config=oil_config,
        start_period=args.start,
        end_period=args.end,
    )
    saved_paths = save_oil_supply_stress_outputs(artifacts, logger=logger)
    latest_structural = artifacts.index_history.loc[
        artifacts.index_history["structural_index_standardized"].notna(),
        ["month_end", "structural_index_standardized"],
    ]
    latest_text = "unavailable"
    if not latest_structural.empty:
        latest_row = latest_structural.iloc[-1]
        latest_text = (
            f"{pd.Timestamp(latest_row['month_end']).strftime('%Y-%m-%d')} "
            f"({float(latest_row['structural_index_standardized']):+.2f} z)"
        )
    logger.info("Oil supply stress build completed. Log file: %s", log_path)
    print(
        f"Saved oil-stress outputs for {artifacts.component_table.shape[0]} components. "
        f"Latest structural reading: {latest_text}. Narrative: {saved_paths['narrative_markdown']}"
    )
    return 0


def _build_selected_series_config(oil_config: OilStressConfig) -> SelectedSeriesConfig:
    payload = {
        "version": oil_config.version,
        "geo_panels": oil_config.geo_panels,
        "selected_series": {},
    }
    for component_code, definition in oil_config.components.items():
        aggregate_method = definition.panel_aggregation
        if aggregate_method == "mean":
            aggregate_method = "simple_mean"
        payload["selected_series"][component_code] = {
            "concept": definition.label,
            "dataset_id": definition.dataset_id,
            "dimensions": definition.dimensions,
            "transformation": definition.signal_transformation,
            "aggregate_from_panel": definition.aggregate_from_panel,
            "aggregate_method": aggregate_method,
        }
    return SelectedSeriesConfig.model_validate(payload)


def _build_component_panel(engineered: pd.DataFrame, oil_config: OilStressConfig) -> pd.DataFrame:
    component_frames: list[pd.DataFrame] = []

    for component_code, definition in oil_config.components.items():
        group = engineered.loc[engineered["indicator_code"].eq(component_code)].copy()
        if group.empty:
            continue
        selected_geo, source_geo_type = _preferred_component_geo(group, definition, oil_config.target_geo)
        selected = group.loc[group["geo"].eq(selected_geo)].copy()
        if selected.empty:
            continue

        sign_multiplier = 1.0 if definition.stress_direction == "positive" else -1.0
        selected = selected.rename(columns={"geo": "source_geo"})
        selected["component_code"] = component_code
        selected["component_label"] = definition.label
        selected["bucket"] = definition.bucket
        selected["bucket_label"] = BUCKET_LABELS.get(definition.bucket, definition.bucket.replace("_", " ").title())
        selected["signal_transformation"] = definition.signal_transformation
        selected["stress_direction"] = definition.stress_direction
        selected["sign_multiplier"] = sign_multiplier
        selected["structural_weight"] = definition.structural_weight
        selected["component_dataset_id"] = definition.dataset_id
        selected["panel_aggregation"] = definition.panel_aggregation or ""
        selected["interpretation"] = definition.interpretation
        selected["cyclical_sensitivity"] = definition.cyclical_sensitivity
        selected["source_geo_type"] = source_geo_type
        selected["target_geo"] = oil_config.target_geo
        selected["signal_value"] = pd.to_numeric(selected["configured_value"], errors="coerce")
        selected["signed_signal"] = sign_multiplier * selected["signal_value"]
        selected["standardized_component"] = _standardize_signal(selected["signed_signal"])
        selected["component_available"] = selected["standardized_component"].notna().astype("int8")
        component_frames.append(selected)

    if not component_frames:
        raise ValueError("No oil-stress components could be built from the supplied observations.")

    combined = pd.concat(component_frames, ignore_index=True)
    keep_columns = [
        "month_end",
        "available_month_end",
        "component_code",
        "component_label",
        "bucket",
        "bucket_label",
        "target_geo",
        "source_geo",
        "source_geo_type",
        "component_dataset_id",
        "source_dataset",
        "signal_transformation",
        "stress_direction",
        "sign_multiplier",
        "structural_weight",
        "panel_aggregation",
        "concept",
        "unit",
        "seasonal_adjustment",
        "raw_value",
        "signal_value",
        "signed_signal",
        "standardized_component",
        "is_missing_observation",
        "is_outlier_raw",
        "is_outlier_configured",
        "aggregation_method",
        "aggregate_source_panel",
        "interpretation",
        "cyclical_sensitivity",
    ]
    return combined.loc[:, keep_columns].sort_values(["component_code", "month_end"], kind="stable").reset_index(drop=True)


def _build_index_history(
    component_panel: pd.DataFrame,
    oil_config: OilStressConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standardized_wide = (
        component_panel.pivot(index="month_end", columns="component_code", values="standardized_component")
        .sort_index()
        .sort_index(axis=1)
    )
    standardized_wide = standardized_wide.dropna(axis=1, how="all")
    if standardized_wide.empty:
        raise ValueError("The standardized oil-stress component matrix is empty.")

    weights = pd.Series(
        {
            component_code: definition.structural_weight
            for component_code, definition in oil_config.components.items()
            if component_code in standardized_wide.columns
        },
        dtype="float64",
    )
    standardized_wide = standardized_wide.loc[:, weights.index]
    minimum_component_count = min(oil_config.minimum_component_count, len(weights))
    observed_count = standardized_wide.notna().sum(axis=1).astype("int64")
    available_weight = standardized_wide.notna().mul(weights, axis=1).sum(axis=1)

    simple_average_index = standardized_wide.mean(axis=1, skipna=True).where(observed_count >= minimum_component_count)
    structural_index = (
        standardized_wide.mul(weights, axis=1).sum(axis=1, skipna=True) / available_weight.replace(0.0, np.nan)
    ).where(observed_count >= minimum_component_count)
    pca_index, pca_loadings = _compute_pca_index(
        standardized_wide,
        reference_series=structural_index,
        minimum_component_count=minimum_component_count,
    )

    normalized_contributions = standardized_wide.fillna(0.0).mul(weights, axis=1).div(
        available_weight.replace(0.0, np.nan),
        axis=0,
    )
    normalized_contributions = normalized_contributions.where(observed_count >= minimum_component_count)

    component_metadata = (
        component_panel.loc[:, ["component_code", "component_label", "bucket", "bucket_label", "structural_weight"]]
        .drop_duplicates(subset=["component_code"])
        .set_index("component_code")
    )

    component_contributions = (
        normalized_contributions.reset_index()
        .melt(id_vars=["month_end"], var_name="component_code", value_name="structural_contribution")
        .merge(component_metadata.reset_index(), how="left", on="component_code")
        .sort_values(["month_end", "component_code"], kind="stable")
        .reset_index(drop=True)
    )

    bucket_map = component_metadata["bucket"]
    bucket_contributions = normalized_contributions.T.groupby(bucket_map).sum().T
    bucket_contributions = bucket_contributions.sort_index().sort_index(axis=1)

    index_history = pd.DataFrame(
        {
            "month_end": standardized_wide.index,
            "component_count": observed_count.values,
            "available_weight": available_weight.values,
            "simple_average_index": simple_average_index.values,
            "pca_index": pca_index.reindex(standardized_wide.index).values,
            "structural_index": structural_index.values,
        }
    )
    for column in ["simple_average_index", "pca_index", "structural_index"]:
        index_history[f"{column}_standardized"] = _sample_standardize(index_history[column])
    return (
        index_history,
        component_contributions,
        bucket_contributions,
        pca_loadings.merge(component_metadata.reset_index(), how="left", on="component_code")
        if not pca_loadings.empty
        else pca_loadings,
    )


def _build_component_table(
    component_panel: pd.DataFrame,
    oil_config: OilStressConfig,
    pca_loadings: pd.DataFrame,
) -> pd.DataFrame:
    pca_loading_map = (
        pca_loadings.set_index("component_code")["pca_loading"] if not pca_loadings.empty else pd.Series(dtype="float64")
    )
    rows = []
    for component_code, definition in oil_config.components.items():
        panel = component_panel.loc[component_panel["component_code"].eq(component_code)].sort_values("month_end")
        raw_observed = panel.loc[panel["raw_value"].notna()] if not panel.empty else pd.DataFrame()
        standardized = panel.loc[panel["standardized_component"].notna()] if not panel.empty else pd.DataFrame()
        latest_standardized = standardized.iloc[-1] if not standardized.empty else None
        latest_raw = raw_observed.iloc[-1] if not raw_observed.empty else None
        rows.append(
            {
                "component_code": component_code,
                "component_label": definition.label,
                "bucket": definition.bucket,
                "bucket_label": BUCKET_LABELS.get(definition.bucket, definition.bucket.replace("_", " ").title()),
                "dataset_id": definition.dataset_id,
                "source_geo": panel["source_geo"].iloc[0] if not panel.empty else "",
                "source_geo_type": panel["source_geo_type"].iloc[0] if not panel.empty else "no_live_data",
                "unit": panel["unit"].iloc[0] if not panel.empty else "",
                "seasonal_adjustment": panel["seasonal_adjustment"].iloc[0] if not panel.empty else "",
                "signal_transformation": definition.signal_transformation,
                "stress_direction": definition.stress_direction,
                "panel_aggregation": definition.panel_aggregation or "",
                "structural_weight": definition.structural_weight,
                "pca_loading": float(pca_loading_map.get(component_code)) if component_code in pca_loading_map.index else np.nan,
                "first_observation_month_end": raw_observed["month_end"].min() if not raw_observed.empty else pd.NaT,
                "last_observation_month_end": raw_observed["month_end"].max() if not raw_observed.empty else pd.NaT,
                "last_standardized_month_end": standardized["month_end"].max() if not standardized.empty else pd.NaT,
                "observation_count": int(raw_observed.shape[0]),
                "standardized_count": int(standardized.shape[0]),
                "coverage_share": float(panel["standardized_component"].notna().mean()),
                "latest_raw_value": latest_raw["raw_value"] if latest_raw is not None else np.nan,
                "latest_signal_value": latest_standardized["signal_value"] if latest_standardized is not None else np.nan,
                "latest_standardized_component": (
                    latest_standardized["standardized_component"] if latest_standardized is not None else np.nan
                ),
                "interpretation": definition.interpretation,
                "cyclical_sensitivity": definition.cyclical_sensitivity,
            }
        )
    component_table = pd.DataFrame.from_records(rows)
    for column in ["first_observation_month_end", "last_observation_month_end", "last_standardized_month_end"]:
        component_table[column] = pd.to_datetime(component_table[column], errors="coerce")
    return component_table.sort_values(["structural_weight", "component_code"], ascending=[False, True]).reset_index(
        drop=True
    )


def _render_component_table_markdown(component_table: pd.DataFrame) -> str:
    if component_table.empty:
        return "# Oil Supply Stress Component Table\n\n_No components available._\n"

    rows = []
    for row in component_table.itertuples(index=False):
        rows.append(
            (
                str(row.component_code),
                str(row.component_label),
                str(row.dataset_id),
                str(row.source_geo),
                str(row.signal_transformation),
                str(row.stress_direction),
                f"{float(row.structural_weight):.2f}",
                _format_date(row.last_observation_month_end),
            )
        )
    return "\n".join(
        [
            "# Oil Supply Stress Component Table",
            "",
            _markdown_table(
                headers=(
                    "component_code",
                    "label",
                    "dataset_id",
                    "geo_used",
                    "signal",
                    "stress_sign",
                    "weight",
                    "last_obs",
                ),
                rows=rows,
            ),
            "",
        ]
    )


def _render_narrative(
    oil_config: OilStressConfig,
    component_table: pd.DataFrame,
    index_history: pd.DataFrame,
    bucket_contributions: pd.DataFrame,
    component_contributions: pd.DataFrame,
) -> str:
    structural_history = index_history.loc[index_history["structural_index_standardized"].notna()].copy()
    latest_date = structural_history["month_end"].max() if not structural_history.empty else pd.NaT
    common_start = structural_history["month_end"].min() if not structural_history.empty else pd.NaT

    if pd.notna(latest_date):
        latest_row = structural_history.loc[structural_history["month_end"].eq(latest_date)].iloc[-1]
        latest_structural = float(latest_row["structural_index_standardized"])
        latest_simple = float(latest_row["simple_average_index_standardized"])
        latest_pca = float(latest_row["pca_index_standardized"]) if pd.notna(latest_row["pca_index_standardized"]) else np.nan
        latest_component_rows = component_contributions.loc[
            component_contributions["month_end"].eq(latest_date) & component_contributions["structural_contribution"].notna()
        ].copy()
        top_stress = latest_component_rows.sort_values("structural_contribution", ascending=False).head(3)
        top_easing = latest_component_rows.sort_values("structural_contribution", ascending=True).head(3)
        latest_bucket = bucket_contributions.loc[latest_date] if latest_date in bucket_contributions.index else pd.Series(dtype="float64")
    else:
        latest_structural = np.nan
        latest_simple = np.nan
        latest_pca = np.nan
        top_stress = pd.DataFrame()
        top_easing = pd.DataFrame()
        latest_bucket = pd.Series(dtype="float64")

    direct_supply_components = component_table.loc[
        component_table["bucket"].isin(["direct_supply", "pass_through_prices"]),
        "component_label",
    ].tolist()
    cyclical_components = component_table.loc[
        component_table["cyclical_sensitivity"].eq("high"),
        "component_label",
    ].tolist()
    unavailable_components = component_table.loc[
        component_table["observation_count"].eq(0),
        "component_code",
    ].tolist()

    lines = [
        "# Euro-Area Oil Supply Stress Indicator",
        "",
        "## Method",
        "",
        f"- Candidate panel size: `{component_table.shape[0]}` monthly Eurostat components.",
        (
            "- Components with no live observations under the current filter set: "
            + ", ".join(f"`{component}`" for component in unavailable_components)
            if unavailable_components
            else "- All configured components returned live data."
        ),
        (
            "- Each component is transformed using its configured monthly signal, signed so that higher always "
            "means more oil-supply stress, then standardized with a trailing 5-year z-score where available "
            "and an expanding z-score fallback early in the sample."
        ),
        (
            "- `simple_average_index` is the equal-weight average of standardized components, `pca_index` is the "
            "first principal component of the standardized panel, and `structural_index` uses the transparent "
            "hand weights from `config/oil_stress_components.yml`."
        ),
        "",
        "## Interpreting Supply Stress Vs Demand",
        "",
        (
            "- The cleanest supply-stress evidence comes from the direct import and pass-through block: "
            + ", ".join(f"`{label}`" for label in direct_supply_components)
            + "."
        ),
        (
            "- The more cyclical or demand-sensitive block includes "
            + ", ".join(f"`{label}`" for label in cyclical_components)
            + ". Weakness there can reflect a euro-area slowdown even when physical oil supply is not especially tight."
        ),
        (
            "- A genuine oil-supply shock should therefore show up as a combination of higher direct-supply and "
            "pass-through stress, ideally reinforced by refining strain. A move driven only by freight or "
            "energy-intensive activity is more ambiguous and can be cyclical demand weakness."
        ),
        "",
        "## Latest Reading",
        "",
        (
            f"- Latest month with a structural reading: `{_format_date(latest_date)}`"
            if pd.notna(latest_date)
            else "- Latest month with a structural reading: unavailable"
        ),
        (
            f"- Structural index: `{latest_structural:+.2f}` z | "
            f"Simple average: `{latest_simple:+.2f}` z | "
            f"PCA index: `{latest_pca:+.2f}` z"
            if pd.notna(latest_date)
            else "- Composite readings unavailable."
        ),
        (
            "- Largest positive latest contributions: "
            + _format_component_contribution_list(top_stress)
            if not top_stress.empty
            else "- Largest positive latest contributions: unavailable"
        ),
        (
            "- Largest negative latest contributions: "
            + _format_component_contribution_list(top_easing)
            if not top_easing.empty
            else "- Largest negative latest contributions: unavailable"
        ),
        (
            "- Latest bucket decomposition: "
            + ", ".join(
                f"`{BUCKET_LABELS.get(bucket, bucket)}` {value:+.2f}"
                for bucket, value in latest_bucket.sort_values(ascending=False).items()
                if pd.notna(value)
            )
            if not latest_bucket.empty
            else "- Latest bucket decomposition: unavailable"
        ),
        "",
        "## Coverage",
        "",
        (
            f"- Composite history starts in `{_format_date(common_start)}` once at least "
            f"`{min(oil_config.minimum_component_count, component_table.shape[0])}` components are simultaneously available."
            if pd.notna(common_start)
            else "- Composite history start is unavailable."
        ),
        (
            f"- Lowest-frequency binding datasets are the monthly oil import tables, which is why the useful common history is later than the global project start date."
        ),
        (
            f"- The latest composite reading stops at `{_format_date(latest_date)}` because the EA20 short-term statistics blocks used for prices and refinery-linked activity currently end there, even though some import tables continue later."
            if pd.notna(latest_date)
            else "- The latest composite endpoint is unavailable."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_index_history_chart(index_history: pd.DataFrame) -> str:
    chart_frame = index_history.loc[
        :,
        [
            "month_end",
            "simple_average_index_standardized",
            "pca_index_standardized",
            "structural_index_standardized",
        ],
    ].dropna(how="all", subset=[
        "simple_average_index_standardized",
        "pca_index_standardized",
        "structural_index_standardized",
    ])
    label_map = {
        "simple_average_index_standardized": "Simple average",
        "pca_index_standardized": "PCA",
        "structural_index_standardized": "Structural",
    }
    return _render_line_chart_svg(
        frame=chart_frame,
        x_column="month_end",
        series_columns=list(label_map),
        title="Euro-area oil supply stress indices",
        subtitle="All three lines are rescaled to z-score units for comparability.",
        label_map=label_map,
        color_map=INDEX_COLORS,
    )


def _render_bucket_decomposition_chart(bucket_contributions: pd.DataFrame) -> str:
    if bucket_contributions.empty:
        return _empty_svg("Oil supply stress bucket decomposition", "No structural contributions available.")
    recent = bucket_contributions.tail(60)
    label_map = {column: BUCKET_LABELS.get(column, column.replace("_", " ").title()) for column in recent.columns}
    return _render_stacked_bar_chart_svg(
        frame=recent,
        title="Structural index decomposition by bucket",
        subtitle="Recent 60 months of weighted standardized-point contributions.",
        label_map=label_map,
        color_map=BUCKET_COLORS,
    )


def _render_latest_component_chart(component_contributions: pd.DataFrame) -> str:
    if component_contributions.empty:
        return _empty_svg("Latest component contributions", "No component contributions available.")
    latest_date = component_contributions.loc[
        component_contributions["structural_contribution"].notna(),
        "month_end",
    ].max()
    if pd.isna(latest_date):
        return _empty_svg("Latest component contributions", "No component contributions available.")
    latest = component_contributions.loc[
        component_contributions["month_end"].eq(latest_date) & component_contributions["structural_contribution"].notna(),
        ["component_label", "structural_contribution"],
    ].copy()
    return _render_horizontal_bar_chart_svg(
        series=latest.set_index("component_label")["structural_contribution"].sort_values(),
        title=f"Latest structural contributions ({pd.Timestamp(latest_date).strftime('%Y-%m')})",
        subtitle="Positive bars raise stress; negative bars offset it.",
    )


def _preferred_component_geo(
    group: pd.DataFrame,
    definition: OilStressComponentDefinition,
    target_geo: str,
) -> tuple[str, str]:
    available_geos = set(group["geo"].dropna().astype(str))
    if target_geo in available_geos:
        return target_geo, "official_target_geo"

    if definition.aggregate_from_panel:
        aggregate_geo = f"AGG_{definition.aggregate_from_panel.upper()}"
        if aggregate_geo in available_geos:
            return aggregate_geo, "panel_aggregate"

    requested_geos = _dimension_list(definition.dimensions.get("geo"))
    if len(requested_geos) == 1 and requested_geos[0] in available_geos:
        return requested_geos[0], "single_requested_geo"

    if requested_geos:
        for geo in requested_geos:
            if geo in available_geos:
                return geo, "first_requested_geo"

    fallback_geo = sorted(available_geos)[0]
    return fallback_geo, "fallback_available_geo"


def _standardize_signal(values: pd.Series) -> pd.Series:
    trailing = trailing_zscore(values)
    expanding = expanding_zscore(values, min_periods=12)
    standardized = trailing.where(trailing.notna(), expanding)
    if standardized.notna().sum() >= 1:
        return standardized.astype("float64")
    return _sample_standardize(values).astype("float64")


def _compute_pca_index(
    wide: pd.DataFrame,
    reference_series: pd.Series,
    minimum_component_count: int,
) -> tuple[pd.Series, pd.DataFrame]:
    eligible = wide.loc[wide.notna().sum(axis=1) >= minimum_component_count].copy()
    eligible = eligible.dropna(axis=1, how="all")
    if eligible.shape[0] < 3 or eligible.shape[1] < 2:
        return pd.Series(np.nan, index=wide.index, dtype="float64"), pd.DataFrame(
            columns=["component_code", "pca_loading", "abs_pca_loading"]
        )

    fit_matrix = eligible.fillna(0.0).to_numpy(dtype="float64")
    column_means = fit_matrix.mean(axis=0)
    centered_fit = fit_matrix - column_means
    if np.allclose(centered_fit.std(axis=0), 0.0):
        return pd.Series(np.nan, index=wide.index, dtype="float64"), pd.DataFrame(
            columns=["component_code", "pca_loading", "abs_pca_loading"]
        )

    _, singular_values, vt = np.linalg.svd(centered_fit, full_matrices=False)
    if singular_values.size == 0:
        return pd.Series(np.nan, index=wide.index, dtype="float64"), pd.DataFrame(
            columns=["component_code", "pca_loading", "abs_pca_loading"]
        )

    loadings = vt[0]
    fit_scores = centered_fit @ loadings
    orientation = 1.0
    reference = reference_series.reindex(eligible.index)
    common = pd.DataFrame({"fit_scores": fit_scores, "reference": reference}).dropna()
    if len(common) >= 2:
        correlation = common["fit_scores"].corr(common["reference"])
        if pd.notna(correlation) and correlation < 0:
            orientation = -1.0
    loadings = loadings * orientation

    full_matrix = wide.loc[:, eligible.columns].fillna(0.0).to_numpy(dtype="float64")
    full_scores = (full_matrix - column_means) @ loadings
    full_series = pd.Series(full_scores, index=wide.index, dtype="float64")
    full_series = full_series.where(wide.notna().sum(axis=1) >= minimum_component_count)

    loading_frame = pd.DataFrame(
        {
            "component_code": eligible.columns,
            "pca_loading": loadings,
            "abs_pca_loading": np.abs(loadings),
        }
    ).sort_values(["abs_pca_loading", "component_code"], ascending=[False, True], kind="stable")
    return full_series, loading_frame.reset_index(drop=True)


def _write_table_with_csv_fallback(frame: pd.DataFrame, stem: Path, logger: logging.Logger) -> Path:
    csv_path = stem.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    try:
        write_dataframe_parquet(frame, stem.with_suffix(".parquet"))
    except RuntimeError as exc:
        logger.warning("Skipping parquet output for %s: %s", stem.name, exc)
    return csv_path


def _sample_standardize(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").astype("float64")
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    std = float(valid.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(0.0, index=series.index, dtype="float64")
    mean = float(valid.mean())
    return ((series - mean) / std).astype("float64")


def _render_line_chart_svg(
    frame: pd.DataFrame,
    x_column: str,
    series_columns: list[str],
    title: str,
    subtitle: str,
    label_map: dict[str, str],
    color_map: dict[str, str],
) -> str:
    working = frame.copy()
    if working.empty:
        return _empty_svg(title, "No data available.")

    width = 1100
    height = 520
    margin_left = 72
    margin_right = 24
    margin_top = 70
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    y_values = working.loc[:, series_columns].to_numpy(dtype="float64")
    finite_values = y_values[np.isfinite(y_values)]
    if finite_values.size == 0:
        return _empty_svg(title, "No finite values available.")
    y_min = float(finite_values.min())
    y_max = float(finite_values.max())
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = 0.1 * max(abs(y_min), abs(y_max), 1.0)
    y_min -= padding
    y_max += padding

    x_positions = _evenly_spaced_positions(len(working), margin_left, margin_left + plot_width)

    def x_to_svg(position: int) -> float:
        return x_positions[position]

    def y_to_svg(value: float) -> float:
        return margin_top + plot_height * (1.0 - ((value - y_min) / (y_max - y_min)))

    grid_lines = []
    for tick in _linear_ticks(y_min, y_max, tick_count=5):
        y = y_to_svg(tick)
        grid_lines.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{margin_left + plot_width}' y2='{y:.2f}' "
            f"stroke='#dddddd' stroke-width='1' />"
        )
        grid_lines.append(
            f"<text x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='12' fill='#444'>{tick:.1f}</text>"
        )

    x_labels = []
    step = max(1, len(working) // 8)
    for position in range(0, len(working), step):
        label = pd.Timestamp(working.iloc[position][x_column]).strftime("%Y-%m")
        x = x_to_svg(position)
        x_labels.append(
            f"<text x='{x:.2f}' y='{height - 20}' text-anchor='middle' font-size='12' fill='#444'>{label}</text>"
        )

    series_paths = []
    legend_items = []
    for index, column in enumerate(series_columns):
        color = color_map.get(column, "#333333")
        label = label_map.get(column, column)
        series = pd.to_numeric(working[column], errors="coerce")
        for segment in _contiguous_segments(series):
            if not segment:
                continue
            points = " ".join(f"{x_to_svg(position):.2f},{y_to_svg(value):.2f}" for position, value in segment)
            series_paths.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{points}' />"
            )
        legend_x = margin_left + index * 170
        legend_items.append(
            f"<line x1='{legend_x}' y1='32' x2='{legend_x + 28}' y2='32' stroke='{color}' stroke-width='3' />"
            f"<text x='{legend_x + 36}' y='36' font-size='12' fill='#333'>{escape(label)}</text>"
        )

    return "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect x='0' y='0' width='100%' height='100%' fill='white' />",
            f"<text x='{margin_left}' y='24' font-size='18' font-weight='700' fill='#111'>{escape(title)}</text>",
            f"<text x='{margin_left}' y='48' font-size='12' fill='#555'>{escape(subtitle)}</text>",
            *legend_items,
            *grid_lines,
            f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{margin_left + plot_width}' y2='{margin_top + plot_height}' stroke='#666' stroke-width='1.2' />",
            f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#666' stroke-width='1.2' />",
            *series_paths,
            *x_labels,
            "</svg>",
        ]
    )


def _render_stacked_bar_chart_svg(
    frame: pd.DataFrame,
    title: str,
    subtitle: str,
    label_map: dict[str, str],
    color_map: dict[str, str],
) -> str:
    if frame.empty:
        return _empty_svg(title, "No data available.")

    width = 1100
    height = 560
    margin_left = 72
    margin_right = 24
    margin_top = 70
    margin_bottom = 86
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    positive_stack = frame.clip(lower=0).sum(axis=1)
    negative_stack = frame.clip(upper=0).sum(axis=1)
    y_min = float(min(negative_stack.min(), 0.0))
    y_max = float(max(positive_stack.max(), 0.0))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = 0.1 * max(abs(y_min), abs(y_max), 1.0)
    y_min -= padding
    y_max += padding

    def y_to_svg(value: float) -> float:
        return margin_top + plot_height * (1.0 - ((value - y_min) / (y_max - y_min)))

    bar_width = max(2.0, plot_width / max(len(frame), 1) * 0.9)
    spacing = plot_width / max(len(frame), 1)

    grid_lines = []
    for tick in _linear_ticks(y_min, y_max, tick_count=5):
        y = y_to_svg(tick)
        grid_lines.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{margin_left + plot_width}' y2='{y:.2f}' stroke='#e0e0e0' stroke-width='1' />"
        )
        grid_lines.append(
            f"<text x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='12' fill='#444'>{tick:.1f}</text>"
        )

    bars = []
    x_labels = []
    for position, (month_end, row) in enumerate(frame.iterrows()):
        x_center = margin_left + spacing * position + spacing / 2.0
        x_left = x_center - bar_width / 2.0
        positive_base = 0.0
        negative_base = 0.0
        for column in frame.columns:
            value = float(row[column]) if pd.notna(row[column]) else 0.0
            color = color_map.get(column, "#666666")
            if value >= 0:
                y1 = y_to_svg(positive_base)
                y2 = y_to_svg(positive_base + value)
                positive_base += value
                rect_y = min(y1, y2)
                rect_height = abs(y2 - y1)
            else:
                y1 = y_to_svg(negative_base)
                y2 = y_to_svg(negative_base + value)
                negative_base += value
                rect_y = min(y1, y2)
                rect_height = abs(y2 - y1)
            if rect_height <= 0:
                continue
            bars.append(
                f"<rect x='{x_left:.2f}' y='{rect_y:.2f}' width='{bar_width:.2f}' height='{rect_height:.2f}' fill='{color}' opacity='0.92' />"
            )

        if position % max(1, len(frame) // 8) == 0:
            x_labels.append(
                f"<text x='{x_center:.2f}' y='{height - 20}' text-anchor='middle' font-size='12' fill='#444'>{pd.Timestamp(month_end).strftime('%Y-%m')}</text>"
            )

    legend_items = []
    for index, column in enumerate(frame.columns):
        legend_x = margin_left + index * 165
        color = color_map.get(column, "#666666")
        legend_items.append(
            f"<rect x='{legend_x}' y='24' width='16' height='12' fill='{color}' />"
            f"<text x='{legend_x + 24}' y='35' font-size='12' fill='#333'>{escape(label_map.get(column, column))}</text>"
        )

    zero_y = y_to_svg(0.0)
    return "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect x='0' y='0' width='100%' height='100%' fill='white' />",
            f"<text x='{margin_left}' y='24' font-size='18' font-weight='700' fill='#111'>{escape(title)}</text>",
            f"<text x='{margin_left}' y='48' font-size='12' fill='#555'>{escape(subtitle)}</text>",
            *legend_items,
            *grid_lines,
            f"<line x1='{margin_left}' y1='{zero_y:.2f}' x2='{margin_left + plot_width}' y2='{zero_y:.2f}' stroke='#555' stroke-width='1.2' />",
            f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#666' stroke-width='1.2' />",
            *bars,
            *x_labels,
            "</svg>",
        ]
    )


def _render_horizontal_bar_chart_svg(series: pd.Series, title: str, subtitle: str) -> str:
    if series.empty:
        return _empty_svg(title, "No data available.")

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return _empty_svg(title, "No finite values available.")

    width = 1100
    bar_height = 26
    height = 120 + bar_height * len(values) + 40
    margin_left = 280
    margin_right = 30
    margin_top = 72
    margin_bottom = 36
    plot_width = width - margin_left - margin_right

    x_min = float(min(values.min(), 0.0))
    x_max = float(max(values.max(), 0.0))
    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    padding = 0.1 * max(abs(x_min), abs(x_max), 1.0)
    x_min -= padding
    x_max += padding

    def x_to_svg(value: float) -> float:
        return margin_left + plot_width * ((value - x_min) / (x_max - x_min))

    zero_x = x_to_svg(0.0)
    bars = []
    labels = []
    for position, (label, value) in enumerate(values.items()):
        y_top = margin_top + position * bar_height
        x_value = x_to_svg(float(value))
        rect_x = min(zero_x, x_value)
        rect_width = abs(x_value - zero_x)
        fill = "#8a1c1c" if value >= 0 else "#2f6b7c"
        bars.append(
            f"<rect x='{rect_x:.2f}' y='{y_top:.2f}' width='{rect_width:.2f}' height='{bar_height - 6:.2f}' fill='{fill}' opacity='0.9' />"
        )
        labels.append(
            f"<text x='{margin_left - 12}' y='{y_top + 14:.2f}' text-anchor='end' font-size='12' fill='#333'>{escape(str(label))}</text>"
        )
        labels.append(
            f"<text x='{x_value + (6 if value >= 0 else -6):.2f}' y='{y_top + 14:.2f}' text-anchor='{'start' if value >= 0 else 'end'}' font-size='12' fill='#333'>{value:+.2f}</text>"
        )

    tick_labels = []
    for tick in _linear_ticks(x_min, x_max, tick_count=5):
        x = x_to_svg(tick)
        tick_labels.append(
            f"<line x1='{x:.2f}' y1='{margin_top - 8}' x2='{x:.2f}' y2='{height - margin_bottom + 4}' stroke='#e0e0e0' stroke-width='1' />"
        )
        tick_labels.append(
            f"<text x='{x:.2f}' y='{height - 8}' text-anchor='middle' font-size='12' fill='#444'>{tick:.1f}</text>"
        )

    return "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect x='0' y='0' width='100%' height='100%' fill='white' />",
            f"<text x='24' y='24' font-size='18' font-weight='700' fill='#111'>{escape(title)}</text>",
            f"<text x='24' y='48' font-size='12' fill='#555'>{escape(subtitle)}</text>",
            f"<line x1='{zero_x:.2f}' y1='{margin_top - 8}' x2='{zero_x:.2f}' y2='{height - margin_bottom + 4}' stroke='#555' stroke-width='1.3' />",
            *tick_labels,
            *bars,
            *labels,
            "</svg>",
        ]
    )


def _empty_svg(title: str, message: str) -> str:
    return "\n".join(
        [
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='180' viewBox='0 0 900 180'>",
            "<rect x='0' y='0' width='100%' height='100%' fill='white' />",
            f"<text x='24' y='30' font-size='18' font-weight='700' fill='#111'>{escape(title)}</text>",
            f"<text x='24' y='70' font-size='14' fill='#555'>{escape(message)}</text>",
            "</svg>",
        ]
    )


def _format_component_contribution_list(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unavailable"
    return ", ".join(
        f"`{row.component_label}` {float(row.structural_contribution):+.2f}"
        for row in frame.itertuples(index=False)
    )


def _format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _dimension_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "_No rows available._"
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _evenly_spaced_positions(length: int, start: float, end: float) -> list[float]:
    if length <= 1:
        return [(start + end) / 2.0]
    step = (end - start) / (length - 1)
    return [start + index * step for index in range(length)]


def _linear_ticks(lower: float, upper: float, tick_count: int = 5) -> list[float]:
    if tick_count <= 1:
        return [lower, upper]
    return [lower + (upper - lower) * index / (tick_count - 1) for index in range(tick_count)]


def _contiguous_segments(series: pd.Series) -> list[list[tuple[int, float]]]:
    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for position, value in enumerate(series):
        if pd.isna(value):
            if current:
                segments.append(current)
                current = []
            continue
        current.append((position, float(value)))
    if current:
        segments.append(current)
    return segments


if __name__ == "__main__":
    raise SystemExit(main())
