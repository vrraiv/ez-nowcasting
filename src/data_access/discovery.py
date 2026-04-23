from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config import ProjectSettings
from config import get_settings

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in limited runtime environments.
    httpx = None

SDMX_XML_NAMESPACES = {
    "m": "http://www.sdmx.org/resources/sdmxml/schemas/v3_0/message",
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v3_0/structure",
    "c": "http://www.sdmx.org/resources/sdmxml/schemas/v3_0/common",
}
XML_LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"
URN_SUFFIX_PATTERN = re.compile(r"=(?P<agency>[^:]+):(?P<identifier>[^()]+)\((?P<version>[^)]+)\)$")
SEARCH_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class DatasetDiscoveryHint:
    alias: str
    expected_frequency: str
    search_terms: tuple[str, ...]
    notes: str


class EurostatApiError(RuntimeError):
    """Raised when the Eurostat SDMX API returns an unexpected response."""


class AllowedValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    label: str | None = None
    description: str | None = None


class DimensionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str | None = None
    role: str
    position: int | None = None
    concept_urn: str | None = None
    codelist_agency_id: str | None = None
    codelist_id: str | None = None
    codelist_version: str | None = None
    codelist_label: str | None = None
    allowed_values: list[AllowedValue] = Field(default_factory=list)


class DatasetStructureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    title: str
    agency_id: str
    version: str
    dataflow_urn: str | None = None
    structure_id: str | None = None
    structure_version: str | None = None
    titles_by_language: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    time_dimension_id: str | None = None
    time_coverage_start: str | None = None
    time_coverage_end: str | None = None
    frequency_dimension_id: str | None = None
    geographic_dimension_id: str | None = None
    unit_dimension_ids: list[str] = Field(default_factory=list)
    adjustment_dimension_ids: list[str] = Field(default_factory=list)
    classification_dimension_ids: list[str] = Field(default_factory=list)
    dimensions: list[DimensionMetadata] = Field(default_factory=list)

    def get_dimension(self, dimension_id: str) -> DimensionMetadata | None:
        dimension_id_normalized = dimension_id.casefold()
        for dimension in self.dimensions:
            if dimension.id.casefold() == dimension_id_normalized:
                return dimension
        return None


def default_search_topics() -> tuple[str, ...]:
    return (
        "quarterly GDP",
        "monthly industrial production",
        "retail trade",
        "wholesale trade",
        "unemployment",
        "inflation",
        "construction",
        "producer prices",
        "external trade",
        "energy",
        "transport",
        "eurocin",
    )


def build_discovery_hints(settings: ProjectSettings) -> tuple[DatasetDiscoveryHint, ...]:
    geo_label = settings.geography.aggregate.replace("_", " ")
    return (
        DatasetDiscoveryHint(
            alias="quarterly_gdp",
            expected_frequency="Q",
            search_terms=("quarterly GDP", geo_label, "national accounts"),
            notes="Primary target series for nowcasting and backtesting.",
        ),
        DatasetDiscoveryHint(
            alias="industrial_production",
            expected_frequency="M",
            search_terms=("industrial production", geo_label, "monthly"),
            notes="Core real-activity indicator for bridge equations.",
        ),
        DatasetDiscoveryHint(
            alias="retail_trade",
            expected_frequency="M",
            search_terms=("retail trade", geo_label, "turnover volume"),
            notes="Consumer demand indicator at monthly frequency.",
        ),
        DatasetDiscoveryHint(
            alias="unemployment",
            expected_frequency="M",
            search_terms=("unemployment rate", geo_label, "monthly"),
            notes="Labor-market indicator for slower-moving activity dynamics.",
        ),
        DatasetDiscoveryHint(
            alias="hicp",
            expected_frequency="M",
            search_terms=("HICP", geo_label, "monthly price index"),
            notes="Price pressure proxy that can also support real-deflation checks.",
        ),
        DatasetDiscoveryHint(
            alias="oil_balance",
            expected_frequency="M",
            search_terms=("oil balance", geo_label, "monthly energy"),
            notes="Candidate energy input for an oil supply stress indicator.",
        ),
    )


def configured_dataset_codes(settings: ProjectSettings) -> dict[str, str]:
    return settings.datasets.model_dump()


class EurostatSdmxClient:
    def __init__(
        self,
        settings: ProjectSettings | None = None,
        http_client: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.eurostat_api.base_url.rstrip("/")
        self.user_agent = self.settings.download.user_agent
        self.timeout_seconds = self.settings.download.timeout_seconds
        self._owns_http_client = http_client is None and httpx is not None
        self.http_client = http_client or self._build_default_http_client()
        self._dataflow_catalog_cache: pd.DataFrame | None = None

    def __enter__(self) -> "EurostatSdmxClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        if self._owns_http_client and self.http_client is not None:
            self.http_client.close()

    def list_dataflows(self, refresh: bool = False) -> pd.DataFrame:
        if self._dataflow_catalog_cache is not None and not refresh:
            return self._dataflow_catalog_cache.copy()

        xml_text = self._get_text(
            f"/structure/dataflow/{self.settings.eurostat_api.agency_id}/*/{self.settings.eurostat_api.dataflow_version}",
            params={"detail": "allstubs"},
        )
        self._dataflow_catalog_cache = _parse_dataflows_xml(
            xml_text,
            preferred_language=self.settings.eurostat_api.language,
        )
        return self._dataflow_catalog_cache.copy()

    def search_dataflows(self, keywords: list[str]) -> pd.DataFrame:
        return _filter_dataflow_catalog(self.list_dataflows(), keywords)

    def get_dataset_structure(self, dataset_id: str) -> DatasetStructureMetadata:
        dataset_key = dataset_id.strip().upper()
        structure_xml = self._get_text(
            f"/structure/dataflow/{self.settings.eurostat_api.agency_id}/{dataset_key}/{self.settings.eurostat_api.dataflow_version}",
            params={"references": "descendants", "detail": "referencepartial"},
        )

        constraint_xml: str | None
        try:
            constraint_xml = self._get_text(
                f"/structure/dataconstraint/{self.settings.eurostat_api.agency_id}/{dataset_key}/{self.settings.eurostat_api.dataflow_version}",
            )
        except EurostatApiError:
            constraint_xml = None

        return _parse_structure_xml(
            structure_xml,
            constraint_xml_text=constraint_xml,
            preferred_language=self.settings.eurostat_api.language,
        )

    def save_structure_report(self, dataset_id: str) -> Path:
        metadata = self.get_dataset_structure(dataset_id)
        report_dir = self.settings.resolve_path(self.settings.eurostat_api.structure_reports_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{metadata.dataset_id.lower()}_structure_report.md"
        report_path.write_text(_render_structure_report(metadata), encoding="utf-8")
        return report_path

    def _get_text(self, path: str, params: dict[str, str] | None = None) -> str:
        if self.http_client is not None:
            response = self.http_client.get(path, params=params)
            try:
                response.raise_for_status()
            except Exception as exc:
                raise EurostatApiError(f"Eurostat request failed for {response.url}: {exc}") from exc
            return response.text

        url = _build_request_url(self.base_url, path, params=params)
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
        except Exception as exc:
            raise EurostatApiError(f"Eurostat request failed for {url}: {exc}") from exc
        return body.decode(charset, errors="replace")

    def _build_default_http_client(self) -> object | None:
        if httpx is None:
            return None
        return httpx.Client(
            base_url=self.base_url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )


def search_dataflows(keywords: list[str]) -> pd.DataFrame:
    return _default_client().search_dataflows(keywords)


def get_dataset_structure(dataset_id: str) -> DatasetStructureMetadata:
    return _default_client().get_dataset_structure(dataset_id)


def save_structure_report(dataset_id: str) -> Path:
    return _default_client().save_structure_report(dataset_id)


@lru_cache(maxsize=1)
def _default_client() -> EurostatSdmxClient:
    return EurostatSdmxClient()


def _build_request_url(base_url: str, path: str, params: dict[str, str] | None = None) -> str:
    url = f"{base_url}/{path.lstrip('/')}"
    if not params:
        return url
    return f"{url}?{urlencode(params)}"


def _parse_dataflows_xml(xml_text: str, preferred_language: str = "en") -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    records: list[dict[str, str]] = []

    for dataflow in root.findall(".//s:Dataflows/s:Dataflow", SDMX_XML_NAMESPACES):
        titles = _read_localized_names(dataflow)
        title = _pick_preferred_text(titles, preferred_language) or dataflow.attrib.get("id", "")
        search_blob = _normalize_search_text(" ".join([dataflow.attrib.get("id", ""), *titles.values()]))
        records.append(
            {
                "dataset_id": dataflow.attrib.get("id", ""),
                "title": title,
                "agency_id": dataflow.attrib.get("agencyID", ""),
                "version": dataflow.attrib.get("version", ""),
                "structure_url": dataflow.attrib.get("structureURL", ""),
                "urn": dataflow.attrib.get("urn", ""),
                "_search_blob": search_blob,
            }
        )

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        return pd.DataFrame(
            columns=["dataset_id", "title", "agency_id", "version", "structure_url", "urn", "_search_blob"]
        )
    return dataframe.sort_values("dataset_id", kind="stable").reset_index(drop=True)


def _filter_dataflow_catalog(catalog: pd.DataFrame, keywords: Sequence[str]) -> pd.DataFrame:
    if catalog.empty:
        return catalog.copy()

    cleaned_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
    working = catalog.copy()
    if not cleaned_keywords:
        return working.drop(columns=["_search_blob"], errors="ignore")

    scores: list[int] = []
    hit_counts: list[int] = []
    matched_keywords: list[str] = []

    for _, row in working.iterrows():
        search_blob = str(row.get("_search_blob", ""))
        row_score = 0
        row_matches: list[str] = []
        for keyword in cleaned_keywords:
            keyword_score = _score_keyword_match(search_blob, keyword)
            if keyword_score > 0:
                row_score += keyword_score
                row_matches.append(keyword)
        scores.append(row_score)
        hit_counts.append(len(row_matches))
        matched_keywords.append(", ".join(row_matches))

    working["match_score"] = scores
    working["keyword_hits"] = hit_counts
    working["matched_keywords"] = matched_keywords
    filtered = working.loc[working["match_score"] > 0].copy()
    filtered = filtered.sort_values(
        by=["match_score", "keyword_hits", "dataset_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    return filtered.drop(columns=["_search_blob"], errors="ignore").reset_index(drop=True)


def _score_keyword_match(search_blob: str, keyword: str) -> int:
    keyword_normalized = _normalize_search_text(keyword)
    if not keyword_normalized:
        return 0
    if keyword_normalized in search_blob:
        return 3

    keyword_tokens = [token for token in keyword_normalized.split() if token]
    if keyword_tokens and all(token in search_blob for token in keyword_tokens):
        return 1
    return 0


def _parse_structure_xml(
    structure_xml_text: str,
    constraint_xml_text: str | None,
    preferred_language: str = "en",
) -> DatasetStructureMetadata:
    structure_root = ET.fromstring(structure_xml_text)
    allowed_values_by_dimension = _parse_constraint_xml(constraint_xml_text) if constraint_xml_text else {}
    concepts = _parse_concepts(structure_root, preferred_language)
    codelists = _parse_codelists(structure_root, preferred_language)

    dataflow = structure_root.find(".//s:Dataflows/s:Dataflow", SDMX_XML_NAMESPACES)
    if dataflow is None:
        raise EurostatApiError("The SDMX structure response did not contain a Dataflow artefact.")

    data_structure = structure_root.find(".//s:DataStructures/s:DataStructure", SDMX_XML_NAMESPACES)
    if data_structure is None:
        raise EurostatApiError("The SDMX structure response did not contain a DataStructure artefact.")

    titles = _read_localized_names(dataflow)
    title = _pick_preferred_text(titles, preferred_language) or dataflow.attrib.get("id", "")
    annotations = _read_annotations(dataflow, preferred_language)

    dimensions = _parse_dimensions(
        data_structure,
        concepts=concepts,
        codelists=codelists,
        allowed_values_by_dimension=allowed_values_by_dimension,
        preferred_language=preferred_language,
    )

    frequency_dimension = next((dimension for dimension in dimensions if dimension.role == "frequency"), None)
    geographic_dimension = next((dimension for dimension in dimensions if dimension.role == "geography"), None)
    unit_dimensions = [dimension.id for dimension in dimensions if dimension.role == "unit"]
    adjustment_dimensions = [dimension.id for dimension in dimensions if dimension.role == "adjustment"]
    classification_dimensions = [dimension.id for dimension in dimensions if dimension.role == "classification"]
    time_dimension = next((dimension for dimension in dimensions if dimension.role == "time"), None)

    return DatasetStructureMetadata(
        dataset_id=dataflow.attrib.get("id", ""),
        title=title,
        agency_id=dataflow.attrib.get("agencyID", ""),
        version=dataflow.attrib.get("version", ""),
        dataflow_urn=dataflow.attrib.get("urn"),
        structure_id=data_structure.attrib.get("id"),
        structure_version=data_structure.attrib.get("version"),
        titles_by_language=titles,
        annotations=annotations,
        time_dimension_id=time_dimension.id if time_dimension else None,
        time_coverage_start=annotations.get("OBS_PERIOD_OVERALL_OLDEST"),
        time_coverage_end=annotations.get("OBS_PERIOD_OVERALL_LATEST"),
        frequency_dimension_id=frequency_dimension.id if frequency_dimension else None,
        geographic_dimension_id=geographic_dimension.id if geographic_dimension else None,
        unit_dimension_ids=unit_dimensions,
        adjustment_dimension_ids=adjustment_dimensions,
        classification_dimension_ids=classification_dimensions,
        dimensions=dimensions,
    )


def _parse_constraint_xml(xml_text: str) -> dict[str, list[str]]:
    root = ET.fromstring(xml_text)
    allowed_values_by_dimension: dict[str, list[str]] = {}

    query = ".//s:DataConstraints/s:DataConstraint/s:CubeRegion[@include='true']/s:KeyValue"
    for key_value in root.findall(query, SDMX_XML_NAMESPACES):
        dimension_id = key_value.attrib.get("id", "").casefold()
        if not dimension_id:
            continue
        values = [_clean_text(value.text) for value in key_value.findall("s:Value", SDMX_XML_NAMESPACES)]
        allowed_values_by_dimension[dimension_id] = [value for value in values if value]

    return allowed_values_by_dimension


def _parse_concepts(root: ET.Element, preferred_language: str) -> dict[str, dict[str, str | dict[str, str]]]:
    concepts: dict[str, dict[str, str | dict[str, str]]] = {}
    for concept in root.findall(".//s:ConceptSchemes/s:ConceptScheme/s:Concept", SDMX_XML_NAMESPACES):
        concept_id = concept.attrib.get("id", "")
        if not concept_id:
            continue
        titles = _read_localized_names(concept)
        concepts[concept_id] = {
            "label": _pick_preferred_text(titles, preferred_language) or concept_id,
            "titles": titles,
        }
    return concepts


def _parse_codelists(
    root: ET.Element,
    preferred_language: str,
) -> dict[tuple[str, str, str], dict[str, str | dict[str, AllowedValue]]]:
    codelists: dict[tuple[str, str, str], dict[str, str | dict[str, AllowedValue]]] = {}

    for codelist in root.findall(".//s:Codelists/s:Codelist", SDMX_XML_NAMESPACES):
        agency_id = codelist.attrib.get("agencyID", "")
        codelist_id = codelist.attrib.get("id", "")
        version = codelist.attrib.get("version", "")
        titles = _read_localized_names(codelist)
        code_map: dict[str, AllowedValue] = {}

        for code in codelist.findall(".//s:Code", SDMX_XML_NAMESPACES):
            code_id = code.attrib.get("id", "")
            if not code_id:
                continue
            code_titles = _read_localized_names(code)
            descriptions = _read_localized_descriptions(code)
            code_map[code_id] = AllowedValue(
                code=code_id,
                label=_pick_preferred_text(code_titles, preferred_language),
                description=_pick_preferred_text(descriptions, preferred_language),
            )

        codelists[(agency_id, codelist_id, version)] = {
            "label": _pick_preferred_text(titles, preferred_language) or codelist_id,
            "codes": code_map,
        }

    return codelists


def _parse_dimensions(
    data_structure: ET.Element,
    concepts: dict[str, dict[str, str | dict[str, str]]],
    codelists: dict[tuple[str, str, str], dict[str, str | dict[str, AllowedValue]]],
    allowed_values_by_dimension: dict[str, list[str]],
    preferred_language: str,
) -> list[DimensionMetadata]:
    del preferred_language
    dimensions: list[DimensionMetadata] = []
    dimension_list = data_structure.find(".//s:DimensionList", SDMX_XML_NAMESPACES)
    if dimension_list is None:
        return dimensions

    for element in list(dimension_list):
        element_name = _strip_namespace(element.tag)
        if element_name not in {"Dimension", "TimeDimension"}:
            continue

        dimension_id = element.attrib.get("id", "")
        if not dimension_id:
            continue

        concept_urn = _find_text(element, "s:ConceptIdentity")
        enumeration_urn = _find_text(element, "s:LocalRepresentation/s:Enumeration")
        codelist_ref = _parse_urn_suffix(enumeration_urn)
        concept_details = concepts.get(dimension_id, {})
        label = concept_details.get("label") if concept_details else None

        codelist_metadata = codelists.get(codelist_ref) if codelist_ref else None
        code_map = codelist_metadata.get("codes", {}) if codelist_metadata else {}
        allowed_codes = allowed_values_by_dimension.get(dimension_id.casefold(), [])
        if not allowed_codes and codelist_metadata:
            allowed_codes = list(code_map.keys())

        allowed_values = []
        for code in allowed_codes:
            code_metadata = code_map.get(code)
            allowed_values.append(
                AllowedValue(
                    code=code,
                    label=code_metadata.label if isinstance(code_metadata, AllowedValue) else None,
                    description=code_metadata.description if isinstance(code_metadata, AllowedValue) else None,
                )
            )

        position_value = element.attrib.get("position")
        position = int(position_value) if position_value and position_value.isdigit() else None

        dimensions.append(
            DimensionMetadata(
                id=dimension_id,
                label=str(label) if label else None,
                role=_classify_dimension(dimension_id, str(label) if label else None, element_name == "TimeDimension"),
                position=position,
                concept_urn=concept_urn,
                codelist_agency_id=codelist_ref[0] if codelist_ref else None,
                codelist_id=codelist_ref[1] if codelist_ref else None,
                codelist_version=codelist_ref[2] if codelist_ref else None,
                codelist_label=str(codelist_metadata["label"]) if codelist_metadata else None,
                allowed_values=allowed_values,
            )
        )

    return sorted(
        dimensions,
        key=lambda dimension: (
            dimension.position is None,
            dimension.position if dimension.position is not None else 10_000,
            dimension.id,
        ),
    )


def _classify_dimension(dimension_id: str, label: str | None, is_time_dimension: bool) -> str:
    if is_time_dimension:
        return "time"

    dimension_key = dimension_id.casefold()
    label_key = (label or "").casefold()

    if dimension_key == "freq" or "frequency" in label_key:
        return "frequency"
    if dimension_key == "geo" or "geopolitical" in label_key:
        return "geography"
    if "unit" in dimension_key or "unit of measure" in label_key:
        return "unit"
    if "adj" in dimension_key or "adjustment" in label_key or "seasonal" in label_key:
        return "adjustment"
    return "classification"


def _render_structure_report(metadata: DatasetStructureMetadata, preview_limit: int = 10) -> str:
    summary_rows = [
        ("Dataset ID", f"`{metadata.dataset_id}`"),
        ("Title", metadata.title),
        ("Time coverage", _format_time_coverage(metadata)),
        ("Frequency dimension", _format_dimension_reference(metadata.get_dimension(metadata.frequency_dimension_id or ""))),
        ("Geographic dimension", _format_dimension_reference(metadata.get_dimension(metadata.geographic_dimension_id or ""))),
        ("Unit dimensions", _format_dimension_group(metadata, metadata.unit_dimension_ids)),
        ("Adjustment dimensions", _format_dimension_group(metadata, metadata.adjustment_dimension_ids)),
        ("Classification dimensions", _format_dimension_group(metadata, metadata.classification_dimension_ids)),
    ]

    detail_rows = []
    detail_dimension_ids = [
        dimension_id
        for dimension_id in [
            metadata.frequency_dimension_id,
            metadata.geographic_dimension_id,
            *metadata.unit_dimension_ids,
            *metadata.adjustment_dimension_ids,
            *metadata.classification_dimension_ids,
        ]
        if dimension_id
    ]
    seen: set[str] = set()
    for dimension_id in detail_dimension_ids:
        if dimension_id in seen:
            continue
        seen.add(dimension_id)
        dimension = metadata.get_dimension(dimension_id)
        if dimension is None:
            continue
        detail_rows.append(
            (
                _format_dimension_reference(dimension),
                dimension.role,
                _format_allowed_values_preview(dimension.allowed_values, preview_limit),
            )
        )

    lines = [
        f"# Eurostat Structure Report: {metadata.dataset_id}",
        "",
        "## Summary",
        "",
        _markdown_table(("Field", "Value"), summary_rows),
        "",
        "## Dimension Details",
        "",
        _markdown_table(("Dimension", "Role", "Allowed values"), detail_rows)
        if detail_rows
        else "_No structural dimensions were parsed from the response._",
        "",
        "_Time coverage is inferred from Eurostat dataflow annotations when available._",
        "",
    ]
    return "\n".join(lines)


def _format_time_coverage(metadata: DatasetStructureMetadata) -> str:
    if metadata.time_coverage_start and metadata.time_coverage_end:
        return f"`{metadata.time_coverage_start}` to `{metadata.time_coverage_end}`"
    if metadata.time_coverage_start:
        return f"Starts at `{metadata.time_coverage_start}`"
    if metadata.time_coverage_end:
        return f"Ends at `{metadata.time_coverage_end}`"
    return "Not inferable from structure annotations"


def _format_dimension_group(metadata: DatasetStructureMetadata, dimension_ids: Sequence[str]) -> str:
    if not dimension_ids:
        return "None identified"
    rendered = [
        _format_dimension_reference(metadata.get_dimension(dimension_id))
        for dimension_id in dimension_ids
        if metadata.get_dimension(dimension_id) is not None
    ]
    return ", ".join(rendered) if rendered else "None identified"


def _format_dimension_reference(dimension: DimensionMetadata | None) -> str:
    if dimension is None:
        return "Not identified"
    if dimension.label:
        return f"`{dimension.id}` ({dimension.label})"
    return f"`{dimension.id}`"


def _format_allowed_values_preview(values: Sequence[AllowedValue], preview_limit: int) -> str:
    if not values:
        return "Not listed"

    rendered_values = []
    for value in values[:preview_limit]:
        if value.label and value.label != value.code:
            rendered_values.append(f"`{value.code}` ({value.label})")
        else:
            rendered_values.append(f"`{value.code}`")

    if len(values) > preview_limit:
        rendered_values.append(f"... (+{len(values) - preview_limit} more)")
    return ", ".join(rendered_values)


def _markdown_table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    if not rows:
        return "_No rows available._"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *data_lines])


def _read_localized_names(element: ET.Element) -> dict[str, str]:
    return _read_localized_children(element, "c:Name")


def _read_localized_descriptions(element: ET.Element) -> dict[str, str]:
    return _read_localized_children(element, "c:Description")


def _read_localized_children(element: ET.Element, path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for child in element.findall(path, SDMX_XML_NAMESPACES):
        lang = child.attrib.get(XML_LANG_ATTR, "")
        text = _clean_text(child.text)
        if text:
            values[lang] = text
    return values


def _read_annotations(element: ET.Element, preferred_language: str) -> dict[str, str]:
    annotations: dict[str, str] = {}
    annotations_root = element.find("c:Annotations", SDMX_XML_NAMESPACES)
    if annotations_root is None:
        return annotations

    for annotation in annotations_root.findall("c:Annotation", SDMX_XML_NAMESPACES):
        annotation_type = _find_text(annotation, "c:AnnotationType")
        if not annotation_type:
            continue
        value = _find_text(annotation, "c:AnnotationTitle")
        if not value:
            texts = {}
            for annotation_text in annotation.findall("c:AnnotationText", SDMX_XML_NAMESPACES):
                lang = annotation_text.attrib.get(XML_LANG_ATTR, "")
                text = _clean_text(annotation_text.text)
                if text:
                    texts[lang] = text
            value = _pick_preferred_text(texts, preferred_language)
        if value:
            annotations[annotation_type] = value
    return annotations


def _pick_preferred_text(values: dict[str, str], preferred_language: str) -> str | None:
    if not values:
        return None
    if preferred_language in values:
        return values[preferred_language]
    if "" in values:
        return values[""]
    return next(iter(values.values()))


def _find_text(element: ET.Element, path: str) -> str | None:
    child = element.find(path, SDMX_XML_NAMESPACES)
    return _clean_text(child.text if child is not None else None)


def _parse_urn_suffix(urn_value: str | None) -> tuple[str, str, str] | None:
    if not urn_value:
        return None
    match = URN_SUFFIX_PATTERN.search(urn_value.strip())
    if not match:
        return None
    return match.group("agency"), match.group("identifier"), match.group("version")


def _normalize_search_text(text: str) -> str:
    return SEARCH_NORMALIZE_PATTERN.sub(" ", text.casefold()).strip()


def _clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    value = text.strip()
    return value or None


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
