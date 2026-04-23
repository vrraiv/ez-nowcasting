# Euro-Area GDP Nowcasting

This repository is a scaffold for a Python 3.11 nowcasting project built around Eurostat monthly indicators and a quarterly euro-area GDP target. The current version sets up the repository structure, dependency management, configuration patterns, and placeholder modules so that data discovery, download, transformation, feature design, and model implementation can be added incrementally.

## Repository layout

```text
.
|-- data_processed/
|-- data_raw/
|-- outputs/
|-- pyproject.toml
|-- README.md
|-- src/
|   |-- config/
|   |-- data_access/
|   |-- evaluation/
|   |-- features/
|   |-- models/
|   |-- notebooks/
|   `-- transforms/
`-- tests/
```

## Workflow

1. Discover Eurostat datasets
   Use the SDMX discovery helpers in `src/data_access/discovery.py` to search Eurostat dataflows, inspect dataset structures, and generate concise markdown reports before locking dataset IDs into config.

2. Download filtered monthly series
   Use `config/selected_series.yml` together with `src/data_access/pull_eurostat.py` to issue filtered Eurostat API requests, cache responses, save raw payloads, and write normalized parquet files.

3. Standardize and transform series
   Use `src/features/monthly_features.py` together with the reusable utilities in `src/transforms/monthly.py` to align observations to month-end, apply the configured baseline transform, compute rolling features, flag missingness and outliers, and build wide and long monthly feature outputs.

4. Construct quarterly GDP target and monthly bridge target
   Use `src/features/targets.py` to pull quarterly real GDP from Eurostat, compute `q/q` and `y/y` targets, and expand them into `month_1`, `month_2`, and `month_3` bridge tables for incomplete-quarter nowcasts.

5. Run baseline nowcasting models
   Use `src/models/baselines.py` together with `src/evaluation/backtests.py` to run rolling-origin baseline nowcasts by information set for bridge, dynamic-factor, and elastic-net benchmarks.

6. Build an oil supply stress indicator
   Use `config/oil_stress_components.yml` together with `src/features/oil_stress.py` to pull the energy and logistics component panel, sign and standardize the component signals, and build simple-average, PCA, and structural oil-stress indices.

## Configuration

Central project settings live in `src/config/project_config.toml`. That file is the single place to change:

- euro-area geography and optional member-state panel
- global start and end dates
- preferred units for each data family
- candidate Eurostat dataset codes
- local raw, processed, and output directories

The typed loader in `src/config/settings.py` validates that TOML file with Pydantic and exposes a reusable `load_settings()` function for the rest of the repo.

The Eurostat-specific API settings in that same TOML file centralize the SDMX 3.0 base URL, agency, preferred language, and output locations for discovery reports.

The selected monthly indicator registry lives in `config/selected_series.yml`. That file centralizes concept-level dataset IDs, exact dimension filters, and the intended transformation for each baseline nowcasting series.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pip install -e .[polars]
```

If you prefer not to use Polars, the optional extra can be skipped.

## Monthly ingestion

Pull the configured monthly Eurostat panel with:

```powershell
python -m src.data_access.pull_eurostat --start 2000-01
```

Useful optional flags:

- `--format jsonstat` or `--format sdmx-csv`
- `--indicator industrial_production --indicator retail_trade_volume`
- `--end 2026-03`
- `--force-refresh`

The ingestion layer is designed around reusable transport and normalization utilities in `src/data_access/ingestion.py`, so the same cache, retry, raw-response, and tidy-frame patterns can be reused later for non-Eurostat external datasets.

Outputs are written to:

- `data_raw/eurostat/` for raw API responses plus request metadata
- `data_raw/cache/http/` for reusable HTTP cache artifacts
- `data_processed/eurostat/` for per-series parquet files and a combined `selected_series_monthly.parquet`
- `outputs/logs/pull_eurostat.log` for run logs

## Monthly feature engineering

Build the monthly feature set from the normalized Eurostat panel with:

```powershell
python -m src.features.monthly_features --input data_processed/eurostat/selected_series_monthly.parquet
```

The feature pipeline:

- aligns all observations to month-end and completes the monthly grid
- applies the configured series-level transformation from `config/selected_series.yml`
- computes additional rolling features: `1m`, `3m/3m saar`, `y/y`, and trailing `5y` z-scores
- flags missing observations and robust outliers
- builds panel-mean country aggregates when a concept has no official `EA20` aggregate
- writes both observation-date and release-lag-aware availability-date long and wide outputs
- saves a feature availability table and a markdown coverage report

If release-lag metadata becomes available later, `config/selected_series.yml` can optionally add `release_lag_months`, `release_lag_days`, `aggregate_from_panel`, and `aggregate_method` fields per selected series without changing the pipeline code.

## Quarterly GDP targets

Build the quarterly GDP target set and the monthly bridge targets with:

```powershell
python -m src.features.targets --start 2000-Q1
```

The target pipeline:

- pulls quarterly real GDP from Eurostat dataset `namq_10_gdp`
- uses `na_item=B1GQ` with a configurable real-volume unit and `SCA` seasonal adjustment
- computes `q/q` and `y/y` real GDP growth from the quarterly real-GDP level series
- builds one quarterly target table for the euro-area aggregate plus the large-member country panel when available
- expands each quarter into three monthly bridge rows so `month_1`, `month_2`, and `month_3` nowcasts can be trained or evaluated separately
- writes an alignment note explaining how monthly features should join to quarterly targets

Outputs are written to:

- `data_raw/eurostat/targets/` for raw quarterly GDP API responses plus request metadata
- `data_processed/targets/` for quarterly targets, monthly bridge targets, and stage-specific bridge subsets
- `outputs/gdp_target_alignment.md` for the feature-to-target alignment documentation
- `outputs/logs/gdp_targets.log` for run logs

## Baseline nowcasting models

Run the baseline nowcast backtests with:

```powershell
python -m src.models.baselines --features data_processed/features/monthly_features_long.csv --targets data_processed/targets/monthly_bridge_targets.csv
```

The baseline modeling pipeline:

- runs an expanding-window rolling-origin backtest
- evaluates each information set separately: `month_1`, `month_2`, and `month_3`
- reports `RMSE`, `MAE`, and directional accuracy
- fits `bridge_ols`: quarterly GDP on quarter-to-date aggregates of one preferred monthly series per concept
- fits `dynamic_factor_baseline`: latent monthly factors extracted from the standardized configured-value panel, then mapped to quarterly GDP
- fits `elastic_net_baseline`: penalized regression on quarter-stage aggregates of the transformed monthly feature set
- saves prediction tables, metric tables, feature-importance summaries, and actual-vs-nowcast charts

Outputs are written under `outputs/model_backtests/<target_column>/`.

The bridge model runs with the standard library, `numpy`, and `pandas`. The dynamic-factor and elastic-net baselines require the project dependencies from `statsmodels` and `scikit-learn`, and the chart output requires `matplotlib`.

## Oil supply stress indicator

Build the euro-area oil supply stress indicator with:

```powershell
python -m src.features.oil_stress --start 2000-01
```

The oil-stress pipeline:

- reads `config/oil_stress_components.yml` for the exact Eurostat filters, stress signs, country-panel aggregation rules, and structural weights
- pulls the component series with the same cache, retry, and raw-response patterns used elsewhere in the repo
- reuses the monthly feature transforms to create component signals, then signs them so higher always means more stress
- standardizes each component with a trailing 5-year z-score and expanding fallback early in the sample
- builds three composite indices: a simple average, a PCA factor, and a transparent hand-weighted structural index
- saves the component panel, index history, component table, narrative markdown, and SVG charts for the latest decomposition

Outputs are written to:

- `data_raw/eurostat/oil_supply_stress/` for raw Eurostat responses and request metadata
- `data_processed/indicators/oil_supply_stress/` for the normalized component pulls, component panel, index history, and contribution tables
- `outputs/oil_supply_stress_*.{csv,md,svg}` for the component table, narrative report, PCA loadings, and charts

## Suggested development flow

- Update `src/config/project_config.toml` with the exact geography, date span, and dataset codes you want to support.
- Use `search_dataflows()`, `get_dataset_structure()`, and `save_structure_report()` in `src/data_access/discovery.py` to shortlist and document candidate Eurostat tables.
- Add transformation logic in `src/transforms/`.
- Build target-construction utilities in `src/features/`.
- Add model training and evaluation logic only after the data pipeline is stable.

## Dataset discovery helpers

The discovery module now includes:

- `search_dataflows(keywords: list[str]) -> pandas.DataFrame`
- `get_dataset_structure(dataset_id: str) -> DatasetStructureMetadata`
- `save_structure_report(dataset_id: str) -> pathlib.Path`

The implementation uses the Eurostat SDMX 3.0 structure API for:

- dataflow search through the dataflow catalogue
- dataset structure retrieval through the dataflow plus descendants response
- allowed-value inspection through the dataset data-constraint artefact

To search the broader concept list requested for this project, run:

```powershell
eurostat-discovery-search
```

If you want to invoke the module directly from the repo without the console script, use:

```powershell
$env:PYTHONPATH = "src"
python -m data_access.discovery_runner
```

That script searches for:

- quarterly GDP
- monthly industrial production
- retail trade
- wholesale trade
- unemployment
- inflation
- construction
- producer prices
- external trade
- energy
- transport
- eurocin

and writes summary outputs under `outputs/discovery/`.

## Current status

The repository now supports discovery, candidate selection, filtered Eurostat monthly ingestion, monthly feature engineering, quarterly GDP target construction, baseline nowcast backtests, and a euro-area oil supply stress indicator. The main remaining work is deeper model development, richer real-time release-lag metadata, and expanded evaluation/reporting refinements.
