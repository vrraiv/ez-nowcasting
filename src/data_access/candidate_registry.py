from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from config import get_settings
from data_access.discovery import DatasetStructureMetadata, EurostatSdmxClient

TARGET_GEOS = ("EA20", "DE", "FR", "IT", "ES", "NL", "BE", "AT")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    bucket: str
    indicator: str
    dataset_id: str
    why_it_could_help: str
    likely_preferred_filters: str
    seasonally_adjusted_note: str
    volume_or_value_note: str


def build_candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            bucket="Real activity",
            indicator="Industrial production",
            dataset_id="STS_INPR_M",
            why_it_could_help="Timely hard-output proxy for the goods cycle and industrial gross value added.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRD; nace_r2=B-D for headline industry; s_adj=SCA; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Optional: NSA, CA, and SCA are available; prefer SCA for nowcasting.",
            volume_or_value_note="Volume-like production index and growth rates.",
        ),
        CandidateSpec(
            bucket="Real activity",
            indicator="Manufacturing output",
            dataset_id="STS_INPR_M",
            why_it_could_help="Manufacturing is a cyclical core of euro-area output and often leads quarterly GDP swings.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRD; nace_r2=C for manufacturing; s_adj=SCA; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Optional: NSA, CA, and SCA are available; prefer SCA.",
            volume_or_value_note="Volume-like production index and growth rates.",
        ),
        CandidateSpec(
            bucket="Real activity",
            indicator="Construction output",
            dataset_id="STS_COPR_M",
            why_it_could_help="Construction is a volatile GDP component that helps track domestic investment and housing cycles.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRD; nace_r2=F or F41/F42/F43; s_adj=SCA; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Optional: NSA, CA, and SCA are available; prefer SCA.",
            volume_or_value_note="Volume-like production index and growth rates.",
        ),
        CandidateSpec(
            bucket="Real activity",
            indicator="Retail trade volume",
            dataset_id="STS_TRTU_M",
            why_it_could_help="Retail sales track household demand at monthly frequency and bridge into private consumption.",
            likely_preferred_filters=(
                "freq=M; indic_bt=VOL_SLS; nace_r2=G47 or G; s_adj=SCA; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Optional: NSA, CA, and SCA are available; prefer SCA.",
            volume_or_value_note="Both: `VOL_SLS` gives volume while `NETTUR` gives nominal turnover.",
        ),
        CandidateSpec(
            bucket="Real activity",
            indicator="Services turnover",
            dataset_id="STS_SETU_M",
            why_it_could_help="Market-services turnover helps fill the largest gap left by industry-heavy monthly hard data.",
            likely_preferred_filters=(
                "freq=M; indic_bt=NETTUR for nominal activity or VOL_SLS where available; "
                "nace_r2=H-N_STS or service branches of interest; s_adj=SCA; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Optional: NSA, CA, and SCA are available; prefer SCA.",
            volume_or_value_note="Both: nominal turnover and some volume-style variants are available.",
        ),
        CandidateSpec(
            bucket="Labour market",
            indicator="Unemployment rate / unemployment level",
            dataset_id="UNE_RT_M",
            why_it_could_help="Labour-market slack moves more slowly than output but is useful for confirming turning points and persistent weakness.",
            likely_preferred_filters=(
                "freq=M; sex=T; age=TOTAL; unit=PC_ACT for the rate or THS_PER for levels; "
                "s_adj=SA; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Optional: NSA, SA, and TC are available; prefer SA for the main unemployment rate.",
            volume_or_value_note="Neither: labor-market rates and headcounts rather than volume or value.",
        ),
        CandidateSpec(
            bucket="Labour market",
            indicator="Employment expectations (monthly survey proxy)",
            dataset_id="EI_BSEE_M_R2",
            why_it_could_help="Forward-looking survey balances can signal labor demand changes before official employment series are published.",
            likely_preferred_filters=(
                "freq=M; indic=employment expectation balances by sector; s_adj=SA; unit=BAL; "
                "geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Yes: SA only.",
            volume_or_value_note="Neither: survey balances or indices.",
        ),
        CandidateSpec(
            bucket="Prices / nominal activity",
            indicator="HICP headline",
            dataset_id="PRC_HICP_MIDX",
            why_it_could_help="Headline consumer prices help distinguish real from nominal swings and capture energy-driven household purchasing-power shocks.",
            likely_preferred_filters=(
                "freq=M; coicop=CP00; unit=I15; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No explicit seasonal-adjustment dimension; the standard HICP monthly index is typically used unadjusted.",
            volume_or_value_note="Neither: consumer price index.",
        ),
        CandidateSpec(
            bucket="Prices / nominal activity",
            indicator="HICP core-related proxies",
            dataset_id="PRC_HICP_MIDX",
            why_it_could_help="Special aggregates such as ex-energy and ex-energy-food measures help isolate underlying inflation pressure relevant for real-activity nowcasts.",
            likely_preferred_filters=(
                "freq=M; coicop=TOT_X_NRG_FOOD_NP or TOT_X_NRG_FOOD; unit=I15; "
                "geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No explicit seasonal-adjustment dimension; special aggregates are typically used as published.",
            volume_or_value_note="Neither: consumer price index special aggregates.",
        ),
        CandidateSpec(
            bucket="Prices / nominal activity",
            indicator="Producer prices (domestic market)",
            dataset_id="STS_INPPD_M",
            why_it_could_help="Domestic producer prices help track factory-gate inflation and nominal value-added pressure in industry.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRC_PRR_DOM; nace_r2=B-D or MIG blocks; unit=I15 or PCH_PRE; "
                "geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No: NSA only.",
            volume_or_value_note="Neither: producer price index.",
        ),
        CandidateSpec(
            bucket="Prices / nominal activity",
            indicator="Producer prices (non-domestic market)",
            dataset_id="STS_INPPND_M",
            why_it_could_help="Non-domestic producer prices add signal on exported tradables, external demand, and foreign-cost pass-through.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRC_PRR_NDOM or PRC_PRR_NDOM_NEU; nace_r2=B-D or MIG blocks; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No: NSA only.",
            volume_or_value_note="Neither: producer price index.",
        ),
        CandidateSpec(
            bucket="Prices / nominal activity",
            indicator="Import prices",
            dataset_id="STS_INPI_M",
            why_it_could_help="Import-price pressure is useful for tracking cost shocks, traded-goods inflation, and external nominal conditions.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRC_IMP or PRC_IMP_NEU; cpa2_1=CPA_B-D or MIG groupings; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No: NSA only.",
            volume_or_value_note="Neither: import price index.",
        ),
        CandidateSpec(
            bucket="Trade / external",
            indicator="Exports",
            dataset_id="EI_ETEA_M",
            why_it_could_help="Monthly exports provide a fast signal on external demand and manufacturing momentum.",
            likely_preferred_filters=(
                "freq=M; stk_flow=EXP; partner=EXT_EA21; indic=ET-T; "
                "unit=MIO-EUR-SA for nominal trade or IVOL-SA for volume; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Mixed: seasonally adjusted and non-adjusted variants are embedded in the `unit` codes.",
            volume_or_value_note="Both: nominal euro values and volume-style indicators are available.",
        ),
        CandidateSpec(
            bucket="Trade / external",
            indicator="Imports",
            dataset_id="EI_ETEA_M",
            why_it_could_help="Monthly imports help capture domestic demand, inventory behavior, and energy-driven trade swings.",
            likely_preferred_filters=(
                "freq=M; stk_flow=IMP; partner=EXT_EA21; indic=ET-T; "
                "unit=MIO-EUR-SA or IVOL-SA; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Mixed: seasonally adjusted and non-adjusted variants are embedded in the `unit` codes.",
            volume_or_value_note="Both: nominal euro values and volume-style indicators are available.",
        ),
        CandidateSpec(
            bucket="Trade / external",
            indicator="Extra-EU trade balance / composition",
            dataset_id="EI_ETEA_M",
            why_it_could_help="The extra-EU trade balance and broad trade groupings help summarize the external contribution to nowcasts.",
            likely_preferred_filters=(
                "freq=M; stk_flow=BAL_RT, EXP, or IMP; partner=EXT_EA21; indic=ET-T, ET-CAP, or ET-INTER; "
                "unit=MIO-EUR-SA, IVOL-SA, or rate units as needed; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="Mixed: seasonally adjusted and non-adjusted variants are embedded in the `unit` codes.",
            volume_or_value_note="Both: nominal, rate, and volume-style views are available.",
        ),
        CandidateSpec(
            bucket="Energy / oil transmission",
            indicator="Oil and petroleum import volumes",
            dataset_id="NRG_TI_OILM",
            why_it_could_help="Monthly petroleum import volumes help trace external energy shocks and physical supply pressures feeding into industry and trade.",
            likely_preferred_filters=(
                "freq=M; siec=O4100_TOT or specific petroleum groups; unit=THS_T; "
                "partner=key non-EU suppliers or totals; geo=DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No explicit seasonal-adjustment dimension.",
            volume_or_value_note="Volume: thousand tonnes.",
        ),
        CandidateSpec(
            bucket="Energy / oil transmission",
            indicator="Crude-oil import values / volumes / prices",
            dataset_id="NRG_TI_COIFPM",
            why_it_could_help="This table is useful for measuring both the quantity and price side of oil-shock transmission into the euro area.",
            likely_preferred_filters=(
                "freq=M; crudeoil=TOTAL or major fields; indic_nrg=VOL_THS_BBL, VAL_THS_USD, or AVGPRC_USD_BBL; "
                "geo=DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No explicit seasonal-adjustment dimension.",
            volume_or_value_note="Both: volumes, values, and average prices are available.",
        ),
        CandidateSpec(
            bucket="Energy / oil transmission",
            indicator="Oil and petroleum supply balance",
            dataset_id="NRG_CB_OILM",
            why_it_could_help="Monthly oil balances add signal on import dependence, stock changes, and refinery-side transmission into activity.",
            likely_preferred_filters=(
                "freq=M; nrg_bal=IMP, STK_CHG, TOS, or IPRD; siec=O4100_TOT or detailed petroleum products; "
                "unit=THS_T; geo=DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No explicit seasonal-adjustment dimension.",
            volume_or_value_note="Volume: thousand tonnes.",
        ),
        CandidateSpec(
            bucket="Energy / oil transmission",
            indicator="Energy producer prices",
            dataset_id="STS_INPPD_M",
            why_it_could_help="Energy-focused producer prices are a direct pass-through channel from commodity shocks into industrial costs and nominal activity.",
            likely_preferred_filters=(
                "freq=M; indic_bt=PRC_PRR_DOM; nace_r2=MIG_NRG or MIG_NRG_X_D_E; "
                "unit=I15 or PCH_PRE; geo=EA20, DE, FR, IT, ES, NL, BE, AT"
            ),
            seasonally_adjusted_note="No: NSA only.",
            volume_or_value_note="Neither: producer price index.",
        ),
        CandidateSpec(
            bucket="Energy / oil transmission",
            indicator="Air-freight transport proxy",
            dataset_id="AVIA_GOOA",
            why_it_could_help="Freight and mail air transport can act as a high-frequency proxy for trade-intensive goods movement when country aggregates are unavailable elsewhere.",
            likely_preferred_filters=(
                "freq=M; tra_meas=FRM_LD or CAF_FRM; schedule=TOT; tra_cov=INTL or INTL_XEU27_2020; "
                "unit=T; aggregate reporting airports to country prefixes"
            ),
            seasonally_adjusted_note="No explicit seasonal-adjustment dimension.",
            volume_or_value_note="Volume: tonnes, with flights as an alternate measure.",
        ),
    )


def main() -> None:
    settings = get_settings()
    output_dir = settings.resolve_path(settings.paths.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "eurostat_candidate_registry.csv"
    markdown_path = output_dir / "eurostat_candidate_registry.md"

    specs = build_candidate_specs()
    rows = build_candidate_registry(specs)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_path, index=False)
    markdown_path.write_text(render_markdown_summary(dataframe), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


def build_candidate_registry(specs: tuple[CandidateSpec, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metadata_cache: dict[str, DatasetStructureMetadata] = {}

    with EurostatSdmxClient() as client:
        catalog = client.list_dataflows()
        for spec in specs:
            dataset_key = spec.dataset_id.upper()
            if dataset_key not in metadata_cache:
                _assert_dataset_in_catalog(dataset_key, catalog)
                metadata_cache[dataset_key] = client.get_dataset_structure(dataset_key)

            metadata = metadata_cache[dataset_key]
            rows.append(
                {
                    "bucket": spec.bucket,
                    "indicator": spec.indicator,
                    "dataset_id": metadata.dataset_id,
                    "dataset_title": metadata.title,
                    "why_it_could_help_gdp_nowcasting": spec.why_it_could_help,
                    "likely_preferred_filters": spec.likely_preferred_filters,
                    "frequency": infer_frequency(metadata),
                    "geographic_coverage": infer_geographic_coverage(metadata),
                    "whether_it_is_monthly": infer_is_monthly(metadata),
                    "whether_it_is_seasonally_adjusted": spec.seasonally_adjusted_note,
                    "whether_it_is_volume_or_value": spec.volume_or_value_note,
                }
            )

    return rows


def render_markdown_summary(dataframe: pd.DataFrame) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Eurostat Candidate Indicator Registry",
        "",
        f"Generated: {generated_at}",
        "",
        "This registry was built from the live Eurostat SDMX 3.0 dataflow catalog and dataset structures.",
        "",
    ]

    for bucket in dataframe["bucket"].drop_duplicates():
        bucket_frame = dataframe.loc[dataframe["bucket"] == bucket].copy()
        lines.append(f"## {bucket}")
        lines.append("")
        lines.append(
            _markdown_table(
                (
                    "indicator",
                    "dataset_id",
                    "dataset title",
                    "why it could help GDP nowcasting",
                    "likely preferred filters",
                    "frequency",
                    "geographic coverage",
                    "monthly",
                    "seasonally adjusted",
                    "volume or value",
                ),
                [
                    (
                        str(row["indicator"]),
                        str(row["dataset_id"]),
                        str(row["dataset_title"]),
                        str(row["why_it_could_help_gdp_nowcasting"]),
                        str(row["likely_preferred_filters"]),
                        str(row["frequency"]),
                        str(row["geographic_coverage"]),
                        str(row["whether_it_is_monthly"]),
                        str(row["whether_it_is_seasonally_adjusted"]),
                        str(row["whether_it_is_volume_or_value"]),
                    )
                    for _, row in bucket_frame.iterrows()
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- No clean monthly Eurostat vacancy table was shortlisted; job vacancy statistics remain mainly quarterly.",
            "- The detailed `EXT_LT_*` external-trade tables inspected during discovery were annual, so the monthly trade bucket relies on `EI_ETEA_M`.",
            "- Several energy and oil tables do not expose an EA20 reporting aggregate even when large member-state coverage is available.",
            "- The transport/freight proxy is airport-level rather than a ready-made country aggregate and would need aggregation in the data pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def infer_frequency(metadata: DatasetStructureMetadata) -> str:
    frequency_dimension = metadata.get_dimension(metadata.frequency_dimension_id or "")
    if frequency_dimension is None or not frequency_dimension.allowed_values:
        return "Unknown"
    labels = []
    for value in frequency_dimension.allowed_values:
        if value.label and value.label != value.code:
            labels.append(f"{value.code} ({value.label})")
        else:
            labels.append(value.code)
    return ", ".join(labels)


def infer_is_monthly(metadata: DatasetStructureMetadata) -> str:
    frequency_dimension = metadata.get_dimension(metadata.frequency_dimension_id or "")
    if frequency_dimension is None:
        return "Unknown"
    frequency_codes = {value.code for value in frequency_dimension.allowed_values}
    return "Yes" if "M" in frequency_codes else "No"


def infer_geographic_coverage(metadata: DatasetStructureMetadata) -> str:
    reporting_dimension = metadata.get_dimension("geo") or metadata.get_dimension(metadata.geographic_dimension_id or "")
    if reporting_dimension is not None:
        available = {value.code for value in reporting_dimension.allowed_values}
        covered = [geo for geo in TARGET_GEOS if geo in available]
        missing = [geo for geo in TARGET_GEOS if geo not in available]

        parts: list[str] = []
        if covered:
            parts.append("available: " + "/".join(covered))
        if missing and covered:
            parts.append("missing: " + "/".join(missing))
        if not covered:
            parts.append("requested coverage not available on reporting dimension")
        return "; ".join(parts)

    airport_dimension = metadata.get_dimension("rep_airp")
    if airport_dimension is not None:
        airport_countries = sorted(
            {
                value.code.split("_", maxsplit=1)[0]
                for value in airport_dimension.allowed_values
                if "_" in value.code
            }
        )
        covered = [geo for geo in TARGET_GEOS[1:] if geo in airport_countries]
        covered_text = "/".join(covered) if covered else "selected countries not obvious from airport codes"
        return f"Airport-level reporting for {covered_text}; no EA20 aggregate"

    return "No reporting-country dimension identified"


def _assert_dataset_in_catalog(dataset_id: str, catalog: pd.DataFrame) -> None:
    if catalog.loc[catalog["dataset_id"].str.upper() == dataset_id].empty:
        raise ValueError(f"Dataset {dataset_id} was not found in the Eurostat dataflow catalog.")


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "_No rows available._"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body])


if __name__ == "__main__":
    main()
