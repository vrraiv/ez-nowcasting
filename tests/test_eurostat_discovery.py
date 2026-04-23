from __future__ import annotations

from data_access.discovery import _filter_dataflow_catalog, _parse_dataflows_xml, _parse_structure_xml


DATAFLOWS_XML = """<?xml version='1.0' encoding='UTF-8'?>
<m:Structure xmlns:m="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/message" xmlns:s="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/structure" xmlns:c="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/common">
  <m:Structures>
    <s:Dataflows>
      <s:Dataflow id="NAMQ_10_GDP" agencyID="ESTAT" version="1.0" structureURL="https://example.test/namq_10_gdp">
        <c:Name xml:lang="en">Gross domestic product (GDP) and main components</c:Name>
      </s:Dataflow>
      <s:Dataflow id="STS_INPR_M" agencyID="ESTAT" version="1.0" structureURL="https://example.test/sts_inpr_m">
        <c:Name xml:lang="en">Industrial production monthly index</c:Name>
      </s:Dataflow>
    </s:Dataflows>
  </m:Structures>
</m:Structure>
"""


STRUCTURE_XML = """<?xml version='1.0' encoding='UTF-8'?>
<m:Structure xmlns:m="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/message" xmlns:s="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/structure" xmlns:c="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/common">
  <m:Structures>
    <s:Dataflows>
      <s:Dataflow id="NAMQ_10_GDP" agencyID="ESTAT" version="1.0" urn="urn:sdmx:org.sdmx.infomodel.datastructure.Dataflow=ESTAT:NAMQ_10_GDP(1.0)">
        <c:Annotations>
          <c:Annotation>
            <c:AnnotationTitle>1975-Q1</c:AnnotationTitle>
            <c:AnnotationType>OBS_PERIOD_OVERALL_OLDEST</c:AnnotationType>
          </c:Annotation>
          <c:Annotation>
            <c:AnnotationTitle>2025-Q4</c:AnnotationTitle>
            <c:AnnotationType>OBS_PERIOD_OVERALL_LATEST</c:AnnotationType>
          </c:Annotation>
        </c:Annotations>
        <c:Name xml:lang="en">Gross domestic product (GDP) and main components</c:Name>
      </s:Dataflow>
    </s:Dataflows>
    <s:DataStructures>
      <s:DataStructure id="NAMQ_10_GDP" version="152.0">
        <s:DataStructureComponents>
          <s:DimensionList id="DimensionDescriptor">
            <s:Dimension id="freq" position="1">
              <s:ConceptIdentity>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=ESTAT:NAMQ_10_GDP(117.0).freq</s:ConceptIdentity>
              <s:LocalRepresentation>
                <s:Enumeration>urn:sdmx:org.sdmx.infomodel.codelist.Codelist=ESTAT:FREQ(3.9)</s:Enumeration>
              </s:LocalRepresentation>
            </s:Dimension>
            <s:Dimension id="unit" position="2">
              <s:ConceptIdentity>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=ESTAT:NAMQ_10_GDP(117.0).unit</s:ConceptIdentity>
              <s:LocalRepresentation>
                <s:Enumeration>urn:sdmx:org.sdmx.infomodel.codelist.Codelist=ESTAT:UNIT(73.0)</s:Enumeration>
              </s:LocalRepresentation>
            </s:Dimension>
            <s:Dimension id="s_adj" position="3">
              <s:ConceptIdentity>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=ESTAT:NAMQ_10_GDP(117.0).s_adj</s:ConceptIdentity>
              <s:LocalRepresentation>
                <s:Enumeration>urn:sdmx:org.sdmx.infomodel.codelist.Codelist=ESTAT:S_ADJ(1.12)</s:Enumeration>
              </s:LocalRepresentation>
            </s:Dimension>
            <s:Dimension id="na_item" position="4">
              <s:ConceptIdentity>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=ESTAT:NAMQ_10_GDP(117.0).na_item</s:ConceptIdentity>
              <s:LocalRepresentation>
                <s:Enumeration>urn:sdmx:org.sdmx.infomodel.codelist.Codelist=ESTAT:NA_ITEM(28.0)</s:Enumeration>
              </s:LocalRepresentation>
            </s:Dimension>
            <s:Dimension id="geo" position="5">
              <s:ConceptIdentity>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=ESTAT:NAMQ_10_GDP(117.0).geo</s:ConceptIdentity>
              <s:LocalRepresentation>
                <s:Enumeration>urn:sdmx:org.sdmx.infomodel.codelist.Codelist=ESTAT:GEO(28.0)</s:Enumeration>
              </s:LocalRepresentation>
            </s:Dimension>
            <s:TimeDimension id="TIME_PERIOD">
              <s:ConceptIdentity>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=ESTAT:NAMQ_10_GDP(117.0).TIME_PERIOD</s:ConceptIdentity>
            </s:TimeDimension>
          </s:DimensionList>
        </s:DataStructureComponents>
      </s:DataStructure>
    </s:DataStructures>
    <s:Codelists>
      <s:Codelist agencyID="ESTAT" id="FREQ" version="3.9">
        <c:Name xml:lang="en">Time frequency</c:Name>
        <s:Code id="Q"><c:Name xml:lang="en">Quarterly</c:Name></s:Code>
      </s:Codelist>
      <s:Codelist agencyID="ESTAT" id="UNIT" version="73.0">
        <c:Name xml:lang="en">Unit of measure</c:Name>
        <s:Code id="CLV10_MEUR"><c:Name xml:lang="en">Chain linked volumes, million euro</c:Name></s:Code>
      </s:Codelist>
      <s:Codelist agencyID="ESTAT" id="S_ADJ" version="1.12">
        <c:Name xml:lang="en">Seasonal adjustment</c:Name>
        <s:Code id="SCA"><c:Name xml:lang="en">Seasonally and calendar adjusted</c:Name></s:Code>
      </s:Codelist>
      <s:Codelist agencyID="ESTAT" id="NA_ITEM" version="28.0">
        <c:Name xml:lang="en">National accounts indicator (ESA 2010)</c:Name>
        <s:Code id="B1GQ"><c:Name xml:lang="en">Gross domestic product at market prices</c:Name></s:Code>
      </s:Codelist>
      <s:Codelist agencyID="ESTAT" id="GEO" version="28.0">
        <c:Name xml:lang="en">Geopolitical entity (reporting)</c:Name>
        <s:Code id="EA20"><c:Name xml:lang="en">Euro area 20</c:Name></s:Code>
      </s:Codelist>
    </s:Codelists>
    <s:ConceptSchemes>
      <s:ConceptScheme agencyID="ESTAT" id="NAMQ_10_GDP" version="117.0">
        <c:Name xml:lang="en">List of concepts for dataset 'NAMQ_10_GDP'</c:Name>
        <s:Concept id="freq"><c:Name xml:lang="en">Time frequency</c:Name></s:Concept>
        <s:Concept id="unit"><c:Name xml:lang="en">Unit of measure</c:Name></s:Concept>
        <s:Concept id="s_adj"><c:Name xml:lang="en">Seasonal adjustment</c:Name></s:Concept>
        <s:Concept id="na_item"><c:Name xml:lang="en">National accounts indicator (ESA 2010)</c:Name></s:Concept>
        <s:Concept id="geo"><c:Name xml:lang="en">Geopolitical entity (reporting)</c:Name></s:Concept>
        <s:Concept id="TIME_PERIOD"><c:Name xml:lang="en">Time</c:Name></s:Concept>
      </s:ConceptScheme>
    </s:ConceptSchemes>
  </m:Structures>
</m:Structure>
"""


CONSTRAINT_XML = """<?xml version='1.0' encoding='UTF-8'?>
<m:Structure xmlns:m="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/message" xmlns:s="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/structure" xmlns:c="http://www.sdmx.org/resources/sdmxml/schemas/v3_0/common">
  <m:Structures>
    <s:DataConstraints>
      <s:DataConstraint agencyID="ESTAT" id="NAMQ_10_GDP" version="1.0" role="Actual">
        <s:CubeRegion include="true">
          <s:KeyValue id="freq"><s:Value>Q</s:Value></s:KeyValue>
          <s:KeyValue id="unit"><s:Value>CLV10_MEUR</s:Value></s:KeyValue>
          <s:KeyValue id="s_adj"><s:Value>SCA</s:Value></s:KeyValue>
          <s:KeyValue id="na_item"><s:Value>B1GQ</s:Value></s:KeyValue>
          <s:KeyValue id="geo"><s:Value>EA20</s:Value></s:KeyValue>
        </s:CubeRegion>
      </s:DataConstraint>
    </s:DataConstraints>
  </m:Structures>
</m:Structure>
"""


def test_parse_dataflows_xml_extracts_titles() -> None:
    catalog = _parse_dataflows_xml(DATAFLOWS_XML)

    assert list(catalog["dataset_id"]) == ["NAMQ_10_GDP", "STS_INPR_M"]
    assert catalog.iloc[0]["title"] == "Gross domestic product (GDP) and main components"


def test_filter_dataflow_catalog_scores_keyword_matches() -> None:
    catalog = _parse_dataflows_xml(DATAFLOWS_XML)

    filtered = _filter_dataflow_catalog(catalog, ["industrial production"])

    assert list(filtered["dataset_id"]) == ["STS_INPR_M"]
    assert filtered.iloc[0]["matched_keywords"] == "industrial production"


def test_parse_structure_xml_maps_dimensions_and_allowed_values() -> None:
    metadata = _parse_structure_xml(STRUCTURE_XML, CONSTRAINT_XML)

    assert metadata.dataset_id == "NAMQ_10_GDP"
    assert metadata.frequency_dimension_id == "freq"
    assert metadata.geographic_dimension_id == "geo"
    assert metadata.unit_dimension_ids == ["unit"]
    assert metadata.adjustment_dimension_ids == ["s_adj"]
    assert metadata.classification_dimension_ids == ["na_item"]
    assert metadata.time_coverage_start == "1975-Q1"
    assert metadata.time_coverage_end == "2025-Q4"

    frequency_dimension = metadata.get_dimension("freq")
    assert frequency_dimension is not None
    assert frequency_dimension.allowed_values[0].code == "Q"
    assert frequency_dimension.allowed_values[0].label == "Quarterly"
