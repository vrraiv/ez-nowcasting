from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from pathlib import Path
import argparse
import json
import logging
import re

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config import PROJECT_ROOT, get_settings
from config.yaml_utils import load_yaml_document
from data_access.ingestion import (
    FileResponseCache,
    HttpRequestSpec,
    RetryingHttpClient,
    build_request_url,
    write_dataframe_parquet,
    write_raw_response,
)

DEFAULT_SELECTED_SERIES_PATH = PROJECT_ROOT / "config" / "selected_series.yml"
JSONSTAT_ACCEPT = "application/json"
SDMX_CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0"
TIDY_COLUMNS = ["date", "geo", "indicator_code", "value", "unit", "seasonal_adjustment", "source_dataset"]
MONTHLY_PERIOD_PATTERN = re.compile(r"^\d{4}-\d{2}$")


class DeferredConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


class SelectedSeriesDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept: str
    dataset_id: str
    dimensions: dict[str, str | list[str]]
    transformation: str
    release_lag_months: int | None = None
    release_lag_days: int | None = None
    aggregate_from_panel: str | None = None
    aggregate_method: str | None = None


class SelectedSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    geo_panels: dict[str, list[str]] = Field(default_factory=dict)
    selected_series: dict[str, SelectedSeriesDefinition]
    deferred_concepts: dict[str, DeferredConcept] = Field(default_factory=dict)


class SeriesSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    concept: str
    dataset_id: str
    dimensions: dict[str, str | list[str]]
    transformation: str
    release_lag_months: int | None = None
    release_lag_days: int | None = None
    aggregate_from_panel: str | None = None
    aggregate_method: str | None = None


class EurostatPuller:
    def __init__(
        self,
        client: RetryingHttpClient,
        cache: FileResponseCache,
        raw_root: Path | None = None,
        processed_root: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = get_settings()
        self.client = client
        self.cache = cache
        self.logger = logger or logging.getLogger(__name__)
        resolved_paths = self.settings.resolved_paths()
        self.raw_root = raw_root or (resolved_paths["raw_data_dir"] / "eurostat")
        self.processed_root = processed_root or (resolved_paths["processed_data_dir"] / "eurostat")

    def pull(
        self,
        selected_series: Iterable[SeriesSelection],
        start_period: str,
        response_format: str = "jsonstat",
        end_period: str | None = None,
        force_refresh: bool = False,
        combined_output_name: str = "selected_series_monthly",
    ) -> pd.DataFrame:
        normalized_frames: list[pd.DataFrame] = []

        for selection in selected_series:
            request_spec = build_eurostat_request_spec(
                selection=selection,
                start_period=start_period,
                response_format=response_format,
                end_period=end_period,
            )
            payload = self.client.fetch(
                request_spec,
                cache=self.cache,
                force_refresh=force_refresh,
                logger=self.logger,
            )
            raw_path = self._raw_response_path(selection, start_period, response_format)
            write_raw_response(payload, raw_path)

            frame = normalize_eurostat_payload(
                payload_text=payload.text,
                selection=selection,
                response_format=response_format,
            )
            normalized_path = _write_table_with_csv_fallback(
                frame,
                self.processed_root / selection.alias,
                self.logger,
            )
            normalized_frames.append(frame)

            self.logger.info(
                "Saved %s rows for %s to %s (cache=%s, url=%s)",
                len(frame),
                selection.alias,
                normalized_path,
                payload.from_cache,
                build_request_url(request_spec.url, request_spec.params),
            )

        combined = (
            pd.concat(normalized_frames, ignore_index=True).sort_values(["indicator_code", "geo", "date"], kind="stable")
            if normalized_frames
            else _empty_tidy_frame()
        )
        combined_path = _write_table_with_csv_fallback(
            combined,
            self.processed_root / combined_output_name,
            self.logger,
        )
        self.logger.info("Saved combined normalized table with %s rows to %s", len(combined), combined_path)
        return combined

    def _raw_response_path(self, selection: SeriesSelection, start_period: str, response_format: str) -> Path:
        extension = "json" if response_format == "jsonstat" else "csv"
        safe_start = start_period.replace("-", "")
        filename = f"{selection.alias}__{safe_start}__{response_format}.{extension}"
        return self.raw_root / selection.dataset_id.lower() / filename


def load_selected_series_config(path: Path = DEFAULT_SELECTED_SERIES_PATH) -> SelectedSeriesConfig:
    raw_document = load_yaml_document(path)
    return SelectedSeriesConfig.model_validate(raw_document)


def iter_selected_series(
    config: SelectedSeriesConfig,
    aliases: Iterable[str] | None = None,
) -> tuple[SeriesSelection, ...]:
    selected_aliases = {alias.strip() for alias in aliases or () if alias.strip()}
    selections = []

    for alias, definition in config.selected_series.items():
        if selected_aliases and alias not in selected_aliases:
            continue
        selections.append(
            SeriesSelection(
                alias=alias,
                concept=definition.concept,
                dataset_id=definition.dataset_id,
                dimensions=definition.dimensions,
                transformation=definition.transformation,
                release_lag_months=definition.release_lag_months,
                release_lag_days=definition.release_lag_days,
                aggregate_from_panel=definition.aggregate_from_panel,
                aggregate_method=definition.aggregate_method,
            )
        )

    return tuple(selections)


def build_eurostat_request_spec(
    selection: SeriesSelection,
    start_period: str,
    response_format: str = "jsonstat",
    end_period: str | None = None,
) -> HttpRequestSpec:
    settings = get_settings()
    if response_format == "jsonstat":
        params: dict[str, str | list[str]] = {
            "lang": settings.eurostat_api.language.upper(),
            "sinceTimePeriod": validate_month_period(start_period),
        }
        if end_period:
            params["untilTimePeriod"] = validate_month_period(end_period)
        for dimension, value in selection.dimensions.items():
            params[dimension] = value
        return HttpRequestSpec(
            url=f"{settings.eurostat_api.statistics_base_url.rstrip('/')}/{selection.dataset_id}",
            params=params,
            headers={"Accept": JSONSTAT_ACCEPT},
            response_format="jsonstat",
            provider="eurostat",
        )

    if response_format != "sdmx-csv":
        raise ValueError(f"Unsupported response format: {response_format}")

    time_filter = [f"ge:{validate_month_period(start_period)}"]
    if end_period:
        time_filter.append(f"le:{validate_month_period(end_period)}")

    params = {
        "attributes": "none",
        "measures": "all",
        "c[TIME_PERIOD]": "+".join(time_filter),
    }
    for dimension, value in selection.dimensions.items():
        params[f"c[{dimension}]"] = _join_dimension_values(value)

    return HttpRequestSpec(
        url=(
            f"{settings.eurostat_api.base_url.rstrip('/')}/data/dataflow/"
            f"{settings.eurostat_api.agency_id}/{selection.dataset_id}/{settings.eurostat_api.dataflow_version}"
        ),
        params=params,
        headers={"Accept": SDMX_CSV_ACCEPT},
        response_format="sdmx-csv",
        provider="eurostat",
    )


def normalize_eurostat_payload(
    payload_text: str,
    selection: SeriesSelection,
    response_format: str = "jsonstat",
) -> pd.DataFrame:
    if response_format == "jsonstat":
        payload = json.loads(payload_text)
        return normalize_jsonstat_dataset(payload, selection)
    if response_format == "sdmx-csv":
        return normalize_sdmx_csv_dataset(payload_text, selection)
    raise ValueError(f"Unsupported response format: {response_format}")


def normalize_jsonstat_dataset(payload: dict[str, object], selection: SeriesSelection) -> pd.DataFrame:
    dimension_ids = [str(value) for value in payload.get("id", [])]
    sizes = [int(value) for value in payload.get("size", [])]
    dimension_block = payload.get("dimension", {})
    value_map = payload.get("value", {})

    if not dimension_ids or not sizes or not isinstance(dimension_block, dict) or not isinstance(value_map, dict):
        return _empty_tidy_frame()

    code_positions = {
        dimension_id: _codes_by_position(dimension_block.get(dimension_id, {}))
        for dimension_id in dimension_ids
    }

    records: list[dict[str, object]] = []
    for position_key, value in value_map.items():
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            continue

        coordinates = _decode_sparse_index(int(position_key), sizes)
        coordinate_map = {
            dimension_id: _value_at_position(code_positions.get(dimension_id, []), coordinates[position])
            for position, dimension_id in enumerate(dimension_ids)
        }
        source_dataset = selection.dataset_id
        extension = payload.get("extension", {})
        if isinstance(extension, dict) and extension.get("id"):
            source_dataset = str(extension["id"])
        records.append(
            {
                "date": _coerce_period_to_timestamp(coordinate_map.get("time") or coordinate_map.get("TIME_PERIOD")),
                "geo": coordinate_map.get("geo") or _dimension_scalar(selection.dimensions.get("geo")),
                "indicator_code": selection.alias,
                "value": float(numeric_value),
                "unit": coordinate_map.get("unit") or _dimension_scalar(selection.dimensions.get("unit")),
                "seasonal_adjustment": coordinate_map.get("s_adj")
                or coordinate_map.get("adj")
                or _dimension_scalar(selection.dimensions.get("s_adj")),
                "source_dataset": source_dataset,
            }
        )

    return _finalize_tidy_frame(pd.DataFrame.from_records(records, columns=TIDY_COLUMNS))


def normalize_sdmx_csv_dataset(csv_text: str, selection: SeriesSelection) -> pd.DataFrame:
    if not csv_text.strip():
        return _empty_tidy_frame()

    raw = pd.read_csv(StringIO(csv_text))
    if raw.empty:
        return _empty_tidy_frame()

    normalized_columns = {column.casefold(): column for column in raw.columns}
    time_column = normalized_columns.get("time_period")
    value_column = normalized_columns.get("obs_value")
    geo_column = normalized_columns.get("geo")
    unit_column = normalized_columns.get("unit")
    seasonal_column = normalized_columns.get("s_adj")

    if value_column is None:
        return _empty_tidy_frame()

    frame = pd.DataFrame(
        {
            "date": (
                raw[time_column].map(_coerce_period_to_timestamp) if time_column is not None else pd.Series(dtype="datetime64[ns]")
            ),
            "geo": raw[geo_column].astype("string") if geo_column is not None else _broadcast_scalar(raw.index, _dimension_scalar(selection.dimensions.get("geo"))),
            "indicator_code": selection.alias,
            "value": pd.to_numeric(raw[value_column], errors="coerce"),
            "unit": raw[unit_column].astype("string") if unit_column is not None else _broadcast_scalar(raw.index, _dimension_scalar(selection.dimensions.get("unit"))),
            "seasonal_adjustment": (
                raw[seasonal_column].astype("string")
                if seasonal_column is not None
                else _broadcast_scalar(raw.index, _dimension_scalar(selection.dimensions.get("s_adj")))
            ),
            "source_dataset": selection.dataset_id,
        }
    )
    return _finalize_tidy_frame(frame)


def pull_selected_series(
    start_period: str,
    selected_series_path: Path = DEFAULT_SELECTED_SERIES_PATH,
    response_format: str = "jsonstat",
    aliases: Iterable[str] | None = None,
    end_period: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    settings = get_settings()
    config = load_selected_series_config(selected_series_path)
    selections = iter_selected_series(config, aliases=aliases)

    cache = FileResponseCache(settings.resolve_path(settings.download.cache_dir))
    client = RetryingHttpClient(
        timeout_seconds=settings.download.timeout_seconds,
        max_retries=settings.download.max_retries,
        retry_backoff_seconds=settings.download.retry_backoff_seconds,
        user_agent=settings.download.user_agent,
    )
    logger = logging.getLogger(__name__)

    try:
        puller = EurostatPuller(client=client, cache=cache, logger=logger)
        return puller.pull(
            selected_series=selections,
            start_period=validate_month_period(start_period),
            response_format=response_format,
            end_period=validate_month_period(end_period) if end_period else _default_end_period(),
            force_refresh=force_refresh,
        )
    finally:
        client.close()


def validate_month_period(value: str) -> str:
    if not value:
        raise ValueError("A monthly period in YYYY-MM format is required.")
    cleaned = value.strip()
    if not MONTHLY_PERIOD_PATTERN.match(cleaned):
        raise ValueError(f"Invalid monthly period: {value}")
    return cleaned


def configure_logging(level: str = "INFO") -> Path:
    settings = get_settings()
    log_path = settings.resolve_path(settings.paths.outputs_dir) / "logs" / "pull_eurostat.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def _write_table_with_csv_fallback(frame: pd.DataFrame, stem: Path, logger: logging.Logger) -> Path:
    csv_path = stem.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    try:
        write_dataframe_parquet(frame, stem.with_suffix(".parquet"))
    except RuntimeError as exc:
        logger.warning("Skipping parquet output for %s: %s", stem.name, exc)
    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download filtered Eurostat monthly series defined in config/selected_series.yml.")
    parser.add_argument("--start", required=True, help="Lower bound for monthly observations, in YYYY-MM format.")
    parser.add_argument("--end", help="Optional upper bound for monthly observations, in YYYY-MM format.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SELECTED_SERIES_PATH),
        help="Path to the selected series YAML configuration.",
    )
    parser.add_argument(
        "--format",
        default="jsonstat",
        choices=("jsonstat", "sdmx-csv"),
        help="Preferred Eurostat response format.",
    )
    parser.add_argument(
        "--indicator",
        action="append",
        default=[],
        help="Optional indicator alias filter. Repeat the flag to limit pulls to a subset.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore the on-disk HTTP cache and re-download matching series.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Standard Python logging level.",
    )
    args = parser.parse_args(argv)

    log_path = configure_logging(args.log_level)
    frame = pull_selected_series(
        start_period=args.start,
        selected_series_path=Path(args.config),
        response_format=args.format,
        aliases=args.indicator,
        end_period=args.end,
        force_refresh=args.force_refresh,
    )
    logging.getLogger(__name__).info("Pull completed. Log file: %s", log_path)
    print(
        f"Saved {len(frame)} normalized rows across {frame['indicator_code'].nunique()} indicators "
        f"using {args.format} responses."
    )
    return 0


def _codes_by_position(dimension_spec: object) -> list[str | None]:
    if not isinstance(dimension_spec, dict):
        return []

    category = dimension_spec.get("category", {})
    if not isinstance(category, dict):
        return []

    index_map = category.get("index", {})
    if not isinstance(index_map, dict):
        return []

    positions: list[str | None] = [None] * len(index_map)
    for code, position in index_map.items():
        positions[int(position)] = str(code)
    return positions


def _decode_sparse_index(index: int, sizes: list[int]) -> list[int]:
    coordinates = [0] * len(sizes)
    remainder = index
    for reverse_position in range(len(sizes) - 1, -1, -1):
        size = sizes[reverse_position]
        coordinates[reverse_position] = remainder % size
        remainder //= size
    return coordinates


def _coerce_period_to_timestamp(value: object) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT
    if MONTHLY_PERIOD_PATTERN.match(text):
        return pd.Period(text, freq="M").to_timestamp()
    if re.match(r"^\d{4}-Q[1-4]$", text):
        return pd.Period(text.replace("-", ""), freq="Q").to_timestamp()
    if re.match(r"^\d{4}$", text):
        return pd.Period(text, freq="Y").to_timestamp()
    return pd.to_datetime(text, errors="coerce")


def _dimension_scalar(value: str | list[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value


def _join_dimension_values(value: str | list[str]) -> str:
    if isinstance(value, list):
        return ",".join(value)
    return value


def _broadcast_scalar(index: pd.Index, value: str | None) -> pd.Series:
    return pd.Series([value] * len(index), index=index, dtype="string")


def _value_at_position(values: list[str | None], position: int) -> str | None:
    if position < 0 or position >= len(values):
        return None
    return values[position]


def _empty_tidy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "geo": pd.Series(dtype="string"),
            "indicator_code": pd.Series(dtype="string"),
            "value": pd.Series(dtype="float64"),
            "unit": pd.Series(dtype="string"),
            "seasonal_adjustment": pd.Series(dtype="string"),
            "source_dataset": pd.Series(dtype="string"),
        }
    )


def _finalize_tidy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_tidy_frame()

    tidy = frame.loc[:, TIDY_COLUMNS].copy()
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy["geo"] = tidy["geo"].astype("string")
    tidy["indicator_code"] = tidy["indicator_code"].astype("string")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy["unit"] = tidy["unit"].astype("string")
    tidy["seasonal_adjustment"] = tidy["seasonal_adjustment"].astype("string")
    tidy["source_dataset"] = tidy["source_dataset"].astype("string")
    tidy = tidy.dropna(subset=["date", "value"]).sort_values(["indicator_code", "geo", "date"], kind="stable")
    return tidy.reset_index(drop=True)


def _default_end_period() -> str:
    settings = get_settings()
    return settings.date_range.end.strftime("%Y-%m")


if __name__ == "__main__":
    raise SystemExit(main())
