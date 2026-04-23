from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, get_settings
from data_access.ingestion import write_dataframe_parquet
from evaluation import evaluate_prediction_frame, rolling_origin_splits


@dataclass(frozen=True, slots=True)
class BaselineModelSpec:
    name: str
    objective: str
    inputs: str
    status: str = "placeholder"


def baseline_model_specs() -> tuple[BaselineModelSpec, ...]:
    return (
        BaselineModelSpec(
            name="bridge_ols",
            objective="Map monthly indicators into a quarterly GDP nowcast with a simple bridge regression.",
            inputs="Quarter-to-date averages of one preferred monthly series per concept against the quarterly GDP target.",
        ),
        BaselineModelSpec(
            name="dynamic_factor_baseline",
            objective="Reduce the monthly standardized panel to latent factors and map quarterly GDP to the quarter-stage factor summaries.",
            inputs="Monthly configured-value panel across the euro-area and large-member indicators.",
        ),
        BaselineModelSpec(
            name="elastic_net_baseline",
            objective="Provide a penalized regression benchmark on the aggregated monthly feature set.",
            inputs="Quarter-stage aggregates of transformed monthly features across the full panel.",
        ),
    )


DEFAULT_FEATURE_LONG_PARQUET = PROJECT_ROOT / "data_processed" / "features" / "monthly_features_long.parquet"
DEFAULT_FEATURE_LONG_CSV = PROJECT_ROOT / "data_processed" / "features" / "monthly_features_long.csv"
DEFAULT_TARGETS_PARQUET = PROJECT_ROOT / "data_processed" / "targets" / "monthly_bridge_targets.parquet"
DEFAULT_TARGETS_CSV = PROJECT_ROOT / "data_processed" / "targets" / "monthly_bridge_targets.csv"
MODEL_OUTPUT_SUBDIR = "model_backtests"
BRIDGE_VALUE_COLUMNS = ("configured_value",)
PENALIZED_VALUE_COLUMNS = ("configured_value", "change_1m", "change_3m_3m_saar", "change_yoy", "zscore_5y")
IDENTIFIER_COLUMNS = {
    "quarter",
    "quarter_end",
    "quarter_start",
    "month_end",
    "month_in_quarter",
    "information_set",
    "target",
    "target_geo",
    "target_column",
}


@dataclass(frozen=True, slots=True)
class BacktestArtifacts:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    importance: pd.DataFrame
    summary_markdown: str


def load_feature_long(path: Path | None = None) -> pd.DataFrame:
    candidate = path or DEFAULT_FEATURE_LONG_PARQUET
    if candidate.suffix.casefold() == ".csv":
        return pd.read_csv(candidate, parse_dates=["month_end", "available_month_end"])
    if candidate.exists():
        try:
            return pd.read_parquet(candidate)
        except (ImportError, ValueError) as exc:
            csv_candidate = candidate.with_suffix(".csv")
            if csv_candidate.exists():
                return pd.read_csv(csv_candidate, parse_dates=["month_end", "available_month_end"])
            raise RuntimeError(
                f"Unable to read feature file at {candidate}. Install pyarrow or provide a CSV file."
            ) from exc
    csv_candidate = candidate.with_suffix(".csv")
    if csv_candidate.exists():
        return pd.read_csv(csv_candidate, parse_dates=["month_end", "available_month_end"])
    raise FileNotFoundError(f"No feature file found at {candidate} or {csv_candidate}.")


def load_bridge_targets(path: Path | None = None) -> pd.DataFrame:
    candidate = path or DEFAULT_TARGETS_PARQUET
    date_columns = ["month_end", "quarter_end", "quarter_start", "feature_snapshot_month_end"]
    if candidate.suffix.casefold() == ".csv":
        return pd.read_csv(candidate, parse_dates=date_columns)
    if candidate.exists():
        try:
            return pd.read_parquet(candidate)
        except (ImportError, ValueError) as exc:
            csv_candidate = candidate.with_suffix(".csv")
            if csv_candidate.exists():
                return pd.read_csv(csv_candidate, parse_dates=date_columns)
            raise RuntimeError(
                f"Unable to read target file at {candidate}. Install pyarrow or provide a CSV file."
            ) from exc
    csv_candidate = candidate.with_suffix(".csv")
    if csv_candidate.exists():
        return pd.read_csv(csv_candidate, parse_dates=date_columns)
    raise FileNotFoundError(f"No target file found at {candidate} or {csv_candidate}.")


def run_baseline_nowcast_backtests(
    feature_long: pd.DataFrame,
    bridge_targets: pd.DataFrame,
    target_geo: str = "EA20",
    target_column: str = "qoq_real_gdp_growth",
    min_train_quarters: int = 24,
    n_dynamic_factors: int = 2,
    logger: logging.Logger | None = None,
) -> BacktestArtifacts:
    active_logger = logger or logging.getLogger(__name__)
    stage_targets = _build_stage_target_tables(bridge_targets, target_geo=target_geo, target_column=target_column)
    bridge_feature_long = _select_bridge_feature_panel(feature_long, target_geo=target_geo)

    predictions_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for stage, stage_table in stage_targets.items():
        if stage_table.shape[0] <= min_train_quarters:
            active_logger.warning(
                "Skipping information set %s because only %s quarters are available.",
                stage,
                stage_table.shape[0],
            )
            continue

        bridge_design = build_quarter_stage_aggregates(
            feature_long=bridge_feature_long,
            stage_targets=stage_table,
            value_columns=list(BRIDGE_VALUE_COLUMNS),
        )
        bridge_predictions, bridge_importance = _rolling_design_backtest(
            design=bridge_design,
            model_name="bridge_ols",
            stage=stage,
            min_train_quarters=min_train_quarters,
            model_factory=lambda: BridgeEquationModel(),
        )
        predictions_frames.append(bridge_predictions)
        importance_frames.append(bridge_importance)

        try:
            penalized_design = build_quarter_stage_aggregates(
                feature_long=feature_long,
                stage_targets=stage_table,
                value_columns=list(PENALIZED_VALUE_COLUMNS),
            )
            penalized_predictions, penalized_importance = _rolling_design_backtest(
                design=penalized_design,
                model_name="elastic_net_baseline",
                stage=stage,
                min_train_quarters=min_train_quarters,
                model_factory=lambda: PenalizedRegressionModel(),
            )
            predictions_frames.append(penalized_predictions)
            importance_frames.append(penalized_importance)
        except ImportError as exc:
            active_logger.warning("Skipping elastic-net baseline for %s: %s", stage, exc)

        try:
            dynamic_predictions, dynamic_importance = _rolling_dynamic_factor_backtest(
                feature_long=feature_long,
                stage_targets=stage_table,
                min_train_quarters=min_train_quarters,
                n_factors=n_dynamic_factors,
                model_name="dynamic_factor_baseline",
                stage=stage,
                logger=active_logger,
            )
            predictions_frames.append(dynamic_predictions)
            importance_frames.append(dynamic_importance)
        except ImportError as exc:
            active_logger.warning("Skipping dynamic-factor baseline for %s: %s", stage, exc)

    predictions = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
    metrics = evaluate_prediction_frame(predictions) if not predictions.empty else pd.DataFrame()
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    summary_markdown = render_nowcast_summary(metrics)
    return BacktestArtifacts(predictions=predictions, metrics=metrics, importance=importance, summary_markdown=summary_markdown)


def save_backtest_outputs(
    artifacts: BacktestArtifacts,
    target_column: str,
    output_root: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    active_logger = logger or logging.getLogger(__name__)
    root = output_root or settings.resolve_path(settings.paths.outputs_dir) / MODEL_OUTPUT_SUBDIR / target_column
    root.mkdir(parents=True, exist_ok=True)

    saved_paths = {
        "predictions": _write_table_with_csv_fallback(artifacts.predictions, root / "predictions", active_logger),
        "metrics": _write_table_with_csv_fallback(artifacts.metrics, root / "metrics", active_logger),
        "importance": _write_table_with_csv_fallback(artifacts.importance, root / "feature_importance", active_logger),
    }

    report_path = root / "summary.md"
    report_path.write_text(artifacts.summary_markdown, encoding="utf-8")
    saved_paths["summary"] = report_path
    chart_paths = save_actual_vs_nowcast_charts(artifacts.predictions, root / "charts", logger=active_logger)
    if chart_paths:
        saved_paths["charts_dir"] = root / "charts"
    return saved_paths


def configure_logging(level: str = "INFO") -> Path:
    settings = get_settings()
    log_path = settings.resolve_path(settings.paths.outputs_dir) / "logs" / "baseline_models.log"
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
    parser = argparse.ArgumentParser(description="Run baseline rolling-origin GDP nowcast backtests by information set.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_LONG_CSV), help="Path to monthly_features_long parquet or CSV.")
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS_CSV), help="Path to monthly_bridge_targets parquet or CSV.")
    parser.add_argument("--target-geo", default="EA20", help="Target geography code.")
    parser.add_argument(
        "--target-column",
        default="qoq_real_gdp_growth",
        choices=("qoq_real_gdp_growth", "yoy_real_gdp_growth"),
        help="GDP target column to nowcast.",
    )
    parser.add_argument("--min-train-quarters", type=int, default=24, help="Minimum number of historical quarters before evaluation begins.")
    parser.add_argument("--dynamic-factors", type=int, default=2, help="Number of latent factors for the dynamic-factor baseline.")
    parser.add_argument("--log-level", default="INFO", help="Standard Python logging level.")
    args = parser.parse_args(argv)

    log_path = configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    feature_long = load_feature_long(Path(args.features))
    bridge_targets = load_bridge_targets(Path(args.targets))
    artifacts = run_baseline_nowcast_backtests(
        feature_long=feature_long,
        bridge_targets=bridge_targets,
        target_geo=args.target_geo,
        target_column=args.target_column,
        min_train_quarters=args.min_train_quarters,
        n_dynamic_factors=args.dynamic_factors,
        logger=logger,
    )
    saved_paths = save_backtest_outputs(artifacts, target_column=args.target_column, logger=logger)
    logger.info("Baseline backtests completed. Log file: %s", log_path)
    print(
        f"Saved {len(artifacts.predictions)} predictions, {len(artifacts.metrics)} metric rows, "
        f"and importance summaries under {saved_paths['summary'].parent}."
    )
    return 0


class BridgeEquationModel:
    def __init__(self) -> None:
        self.preprocessor = _NumericPreprocessor()
        self.feature_names_: list[str] = []
        self.coefficients_: pd.Series | None = None
        self.intercept_: float = 0.0
        self.pvalues_: pd.Series | None = None
        self.method_: str = "numpy_ols"

    def fit(self, design: pd.DataFrame, target: pd.Series) -> "BridgeEquationModel":
        self.feature_names_ = list(design.columns)
        transformed = self.preprocessor.fit_transform(design)
        y = pd.to_numeric(target, errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(y)
        transformed = transformed.loc[valid, :]
        y = y[valid]
        if transformed.empty:
            raise ValueError("No usable features were available for the bridge regression.")

        coefficients, intercept, pvalues, method = _fit_linear_model_with_optional_statsmodels(transformed, y)
        self.coefficients_ = pd.Series(coefficients, index=self.feature_names_, dtype="float64")
        self.intercept_ = float(intercept)
        self.pvalues_ = pd.Series(pvalues, index=self.feature_names_, dtype="float64")
        self.method_ = method
        return self

    def predict(self, design: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(design.loc[:, self.feature_names_])
        return self.intercept_ + transformed.to_numpy(dtype="float64") @ self.coefficients_.to_numpy(dtype="float64")

    def importance_summary(
        self,
        information_set: str,
        latest_row: pd.Series | None = None,
    ) -> pd.DataFrame:
        latest_contribution = None
        if latest_row is not None:
            latest_scaled = self.preprocessor.transform(latest_row[self.feature_names_].to_frame().T).iloc[0]
            latest_contribution = latest_scaled * self.coefficients_

        summary = pd.DataFrame(
            {
                "feature_name": self.feature_names_,
                "coefficient": self.coefficients_.values,
                "abs_coefficient": self.coefficients_.abs().values,
                "p_value": self.pvalues_.reindex(self.feature_names_).values if self.pvalues_ is not None else np.nan,
                "latest_contribution": (
                    latest_contribution.reindex(self.feature_names_).values if latest_contribution is not None else np.nan
                ),
                "model_fit_method": self.method_,
                "information_set": information_set,
            }
        )
        return summary.sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)


class PenalizedRegressionModel:
    def __init__(self) -> None:
        self.preprocessor = _NumericPreprocessor()
        self.feature_names_: list[str] = []
        self.model_ = None

    def fit(self, design: pd.DataFrame, target: pd.Series) -> "PenalizedRegressionModel":
        try:
            from sklearn.linear_model import ElasticNetCV
        except ImportError as exc:
            raise ImportError("PenalizedRegressionModel requires scikit-learn to be installed.") from exc

        self.feature_names_ = list(design.columns)
        transformed = self.preprocessor.fit_transform(design)
        y = pd.to_numeric(target, errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(y)
        transformed = transformed.loc[valid, :]
        y = y[valid]
        if transformed.empty:
            raise ValueError("No usable features were available for the elastic-net regression.")

        cv_folds = max(2, min(5, transformed.shape[0] // 4)) if transformed.shape[0] >= 8 else 2
        self.model_ = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-3, 1, 25),
            max_iter=20_000,
            cv=cv_folds,
            random_state=0,
        )
        self.model_.fit(transformed.to_numpy(dtype="float64"), y)
        return self

    def predict(self, design: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(design.loc[:, self.feature_names_])
        return self.model_.predict(transformed.to_numpy(dtype="float64"))

    def importance_summary(
        self,
        information_set: str,
        latest_row: pd.Series | None = None,
    ) -> pd.DataFrame:
        coefficients = pd.Series(self.model_.coef_, index=self.feature_names_, dtype="float64")
        latest_contribution = None
        if latest_row is not None:
            latest_scaled = self.preprocessor.transform(latest_row[self.feature_names_].to_frame().T).iloc[0]
            latest_contribution = latest_scaled * coefficients

        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "coefficient": coefficients.values,
                    "abs_coefficient": coefficients.abs().values,
                    "alpha": float(self.model_.alpha_),
                    "l1_ratio": float(self.model_.l1_ratio_),
                    "latest_contribution": (
                        latest_contribution.reindex(self.feature_names_).values if latest_contribution is not None else np.nan
                    ),
                    "information_set": information_set,
                }
            )
            .sort_values("abs_coefficient", ascending=False, kind="stable")
            .reset_index(drop=True)
        )


def build_quarter_stage_aggregates(
    feature_long: pd.DataFrame,
    stage_targets: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    working_features = feature_long.copy()
    working_features["month_end"] = pd.to_datetime(working_features["month_end"], errors="coerce")
    working_features["available_month_end"] = pd.to_datetime(working_features["available_month_end"], errors="coerce")
    working_features["series_key"] = (
        working_features["indicator_code"].astype("string") + "__" + working_features["geo"].astype("string")
    )

    aggregated_rows: list[dict[str, object]] = []
    for row in stage_targets.itertuples(index=False):
        base_row = {
            "quarter": row.quarter,
            "quarter_end": row.quarter_end,
            "quarter_start": row.quarter_start,
            "month_end": row.month_end,
            "month_in_quarter": int(row.month_in_quarter),
            "information_set": row.information_set,
            "target": row.target,
            "target_geo": row.target_geo,
            "target_column": row.target_column,
        }
        subset = working_features.loc[
            (working_features["month_end"] >= row.quarter_start)
            & (working_features["month_end"] <= row.month_end)
            & (working_features["available_month_end"] <= row.month_end)
        ].copy()
        if subset.empty:
            aggregated_rows.append(base_row)
            continue

        grouped = subset.groupby("series_key", sort=False)[value_columns].mean(numeric_only=True)
        for series_key, values in grouped.iterrows():
            for value_column in value_columns:
                base_row[f"{series_key}__{value_column}"] = values.get(value_column)
        aggregated_rows.append(base_row)

    design = pd.DataFrame.from_records(aggregated_rows)
    return design.sort_values("quarter_end", kind="stable").reset_index(drop=True)


def render_nowcast_summary(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "# Baseline Nowcast Summary\n\n_No backtest results were produced._\n"

    lines = [
        "# Baseline Nowcast Summary",
        "",
        "## Metrics",
        "",
        _markdown_table(
            headers=("model_name", "information_set", "target_column", "rmse", "mae", "directional_accuracy", "evaluation_count"),
            rows=[
                (
                    str(row.model_name),
                    str(row.information_set),
                    str(row.target_column),
                    f"{float(row.rmse):.4f}",
                    f"{float(row.mae):.4f}",
                    f"{float(row.directional_accuracy):.2%}",
                    str(int(row.evaluation_count)),
                )
                for row in metrics.itertuples(index=False)
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def save_actual_vs_nowcast_charts(
    predictions: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    active_logger = logger or logging.getLogger(__name__)
    if predictions.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        active_logger.warning("Skipping chart output because matplotlib is not installed.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for (model_name, information_set, target_column), group in predictions.groupby(
        ["model_name", "information_set", "target_column"],
        sort=False,
    ):
        ordered = group.sort_values("quarter_end", kind="stable")
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(ordered["quarter_end"], ordered["actual"], label="Actual", linewidth=2.0)
        axis.plot(ordered["quarter_end"], ordered["prediction"], label="Nowcast", linewidth=1.8)
        axis.set_title(f"{model_name} | {information_set} | {target_column}")
        axis.set_ylabel("GDP growth")
        axis.set_xlabel("Quarter end")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.autofmt_xdate()
        safe_name = f"{model_name}__{information_set}__{target_column}".replace("/", "_")
        chart_path = output_dir / f"{safe_name}.png"
        figure.tight_layout()
        figure.savefig(chart_path, dpi=150)
        plt.close(figure)
        saved_paths.append(chart_path)
    return saved_paths


def _rolling_design_backtest(
    design: pd.DataFrame,
    model_name: str,
    stage: str,
    min_train_quarters: int,
    model_factory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = design.sort_values("quarter_end", kind="stable").reset_index(drop=True)
    ordered = ordered.loc[ordered["target"].notna()].reset_index(drop=True)
    predictions: list[dict[str, object]] = []

    for train_slice, evaluation_index in rolling_origin_splits(ordered.shape[0], min_train_quarters):
        train = ordered.iloc[train_slice]
        test = ordered.iloc[[evaluation_index]]
        feature_columns = _active_feature_columns(train)
        if not feature_columns:
            continue

        model = model_factory()
        model.fit(train.loc[:, feature_columns], train["target"])
        prediction = float(model.predict(test.loc[:, feature_columns])[0])
        predictions.append(
            {
                "model_name": model_name,
                "information_set": stage,
                "target_column": str(test["target_column"].iloc[0]),
                "target_geo": str(test["target_geo"].iloc[0]),
                "quarter": str(test["quarter"].iloc[0]),
                "quarter_end": test["quarter_end"].iloc[0],
                "month_end": test["month_end"].iloc[0],
                "actual": float(test["target"].iloc[0]),
                "prediction": prediction,
                "error": float(prediction - float(test["target"].iloc[0])),
            }
        )

    prediction_frame = pd.DataFrame.from_records(predictions)
    importance_frame = pd.DataFrame()
    full_feature_columns = _active_feature_columns(ordered)
    if full_feature_columns:
        full_model = model_factory()
        full_model.fit(ordered.loc[:, full_feature_columns], ordered["target"])
        importance_frame = full_model.importance_summary(stage, latest_row=ordered.iloc[-1])
        importance_frame["model_name"] = model_name
        importance_frame["target_column"] = ordered["target_column"].iloc[0]
        importance_frame["target_geo"] = ordered["target_geo"].iloc[0]

    return prediction_frame, importance_frame


def _rolling_dynamic_factor_backtest(
    feature_long: pd.DataFrame,
    stage_targets: pd.DataFrame,
    min_train_quarters: int,
    n_factors: int,
    model_name: str,
    stage: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or logging.getLogger(__name__)
    ordered = stage_targets.sort_values("quarter_end", kind="stable").reset_index(drop=True)
    ordered = ordered.loc[ordered["target"].notna()].reset_index(drop=True)

    predictions: list[dict[str, object]] = []
    extraction_methods: list[str] = []
    full_loadings = pd.DataFrame()

    for train_slice, evaluation_index in rolling_origin_splits(ordered.shape[0], min_train_quarters):
        evaluation_row = ordered.iloc[evaluation_index]
        snapshot = pd.Timestamp(evaluation_row["month_end"])
        panel = _build_dynamic_factor_monthly_panel(feature_long, snapshot_month_end=snapshot)
        if panel.shape[0] < 12 or panel.shape[1] < 2:
            continue

        factors, loadings, extraction_method = _extract_monthly_factors(panel, n_factors=n_factors)
        extraction_methods.append(extraction_method)
        design = _build_factor_stage_design(ordered.iloc[: evaluation_index + 1].copy(), factors)
        train = design.iloc[train_slice]
        test = design.iloc[[evaluation_index]]
        feature_columns = _active_feature_columns(train)
        if not feature_columns:
            continue

        model = BridgeEquationModel()
        model.fit(train.loc[:, feature_columns], train["target"])
        prediction = float(model.predict(test.loc[:, feature_columns])[0])
        predictions.append(
            {
                "model_name": model_name,
                "information_set": stage,
                "target_column": str(test["target_column"].iloc[0]),
                "target_geo": str(test["target_geo"].iloc[0]),
                "quarter": str(test["quarter"].iloc[0]),
                "quarter_end": test["quarter_end"].iloc[0],
                "month_end": test["month_end"].iloc[0],
                "actual": float(test["target"].iloc[0]),
                "prediction": prediction,
                "error": float(prediction - float(test["target"].iloc[0])),
                "factor_extraction_method": extraction_method,
            }
        )
        full_loadings = loadings

    prediction_frame = pd.DataFrame.from_records(predictions)
    importance_frame = pd.DataFrame()
    if ordered.shape[0] >= min_train_quarters + 1:
        final_snapshot = pd.Timestamp(ordered.iloc[-1]["month_end"])
        panel = _build_dynamic_factor_monthly_panel(feature_long, snapshot_month_end=final_snapshot)
        if panel.shape[0] >= 12 and panel.shape[1] >= 2:
            factors, loadings, extraction_method = _extract_monthly_factors(panel, n_factors=n_factors)
            design = _build_factor_stage_design(ordered.copy(), factors)
            feature_columns = _active_feature_columns(design)
            if feature_columns:
                model = BridgeEquationModel()
                model.fit(design.loc[:, feature_columns], design["target"])
                importance_frame = model.importance_summary(stage, latest_row=design.iloc[-1])
                importance_frame["model_name"] = model_name
                importance_frame["target_column"] = ordered["target_column"].iloc[0]
                importance_frame["target_geo"] = ordered["target_geo"].iloc[0]
                importance_frame["factor_extraction_method"] = extraction_method
                if not loadings.empty:
                    loading_frame = loadings.copy()
                    loading_frame["model_name"] = model_name
                    loading_frame["information_set"] = stage
                    loading_frame["target_column"] = ordered["target_column"].iloc[0]
                    loading_frame["target_geo"] = ordered["target_geo"].iloc[0]
                    importance_frame = pd.concat([importance_frame, loading_frame], ignore_index=True, sort=False)

    if prediction_frame.empty:
        active_logger.warning("Dynamic-factor baseline produced no predictions for %s.", stage)
    elif extraction_methods:
        prediction_frame["factor_extraction_method"] = prediction_frame["factor_extraction_method"].fillna(extraction_methods[-1])

    return prediction_frame, importance_frame


class _NumericPreprocessor:
    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self.impute_values_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.scales_: pd.Series | None = None

    def fit_transform(self, design: pd.DataFrame) -> pd.DataFrame:
        self.feature_names_ = list(design.columns)
        numeric = design.apply(pd.to_numeric, errors="coerce")
        self.impute_values_ = numeric.median(axis=0, skipna=True).fillna(0.0)
        imputed = numeric.fillna(self.impute_values_)
        self.means_ = imputed.mean(axis=0).fillna(0.0)
        self.scales_ = imputed.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return (imputed - self.means_) / self.scales_

    def transform(self, design: pd.DataFrame) -> pd.DataFrame:
        numeric = design.apply(pd.to_numeric, errors="coerce").reindex(columns=self.feature_names_)
        imputed = numeric.fillna(self.impute_values_)
        return (imputed - self.means_) / self.scales_


def _build_stage_target_tables(
    bridge_targets: pd.DataFrame,
    target_geo: str,
    target_column: str,
) -> dict[str, pd.DataFrame]:
    working = bridge_targets.copy()
    working["month_end"] = pd.to_datetime(working["month_end"], errors="coerce")
    working["quarter_end"] = pd.to_datetime(working["quarter_end"], errors="coerce")
    working["quarter_start"] = pd.to_datetime(working["quarter_start"], errors="coerce")
    filtered = working.loc[working["geo"].astype("string") == target_geo].copy()
    filtered["information_set"] = "month_" + filtered["month_in_quarter"].astype(int).astype(str)
    filtered["target"] = pd.to_numeric(filtered[target_column], errors="coerce")
    filtered["target_geo"] = target_geo
    filtered["target_column"] = target_column

    return {
        stage: stage_frame.sort_values("quarter_end", kind="stable").reset_index(drop=True)
        for stage, stage_frame in filtered.groupby("information_set", sort=False)
    }


def _select_bridge_feature_panel(feature_long: pd.DataFrame, target_geo: str) -> pd.DataFrame:
    preferred_rows = []
    for indicator_code, group in feature_long.groupby("indicator_code", sort=False):
        geos = group["geo"].dropna().astype(str).unique().tolist()
        if target_geo in geos:
            preferred_geo = target_geo
        else:
            aggregate_geos = sorted([geo for geo in geos if geo.startswith("AGG_")])
            preferred_geo = aggregate_geos[0] if aggregate_geos else sorted(geos)[0]
        preferred_rows.append(group.loc[group["geo"].astype(str) == preferred_geo])
    return pd.concat(preferred_rows, ignore_index=True) if preferred_rows else feature_long.iloc[0:0].copy()


def _build_dynamic_factor_monthly_panel(feature_long: pd.DataFrame, snapshot_month_end: pd.Timestamp) -> pd.DataFrame:
    working = feature_long.copy()
    working["month_end"] = pd.to_datetime(working["month_end"], errors="coerce")
    working["available_month_end"] = pd.to_datetime(working["available_month_end"], errors="coerce")
    subset = working.loc[
        (working["available_month_end"] <= snapshot_month_end)
        & (working["month_end"] <= snapshot_month_end)
    ].copy()
    subset["series_key"] = subset["indicator_code"].astype("string") + "__" + subset["geo"].astype("string")
    panel = subset.pivot_table(index="month_end", columns="series_key", values="configured_value", aggfunc="last")
    panel = panel.sort_index()
    panel = panel.loc[:, panel.notna().sum() >= max(6, int(0.25 * max(len(panel), 1)))]
    return panel


def _extract_monthly_factors(
    monthly_panel: pd.DataFrame,
    n_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    prepared = monthly_panel.sort_index().copy()
    prepared = prepared.ffill().fillna(prepared.mean()).fillna(0.0)
    standardized = (prepared - prepared.mean()) / prepared.std(ddof=0).replace(0.0, 1.0)

    try:
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        dynamic_factor = DynamicFactor(
            standardized,
            k_factors=min(n_factors, standardized.shape[1]),
            factor_order=1,
            error_order=0,
            error_var=False,
        )
        result = dynamic_factor.fit(disp=False, maxiter=200)
        factor_values = getattr(getattr(result, "factors", None), "smoothed", None)
        if factor_values is None:
            factor_values = getattr(getattr(result, "factors", None), "filtered", None)
        if factor_values is None:
            raise RuntimeError("DynamicFactor results did not expose smoothed or filtered factors.")
        factor_array = np.asarray(factor_values)
        factor_columns = [f"factor_{index + 1}" for index in range(factor_array.shape[0])]
        factors = pd.DataFrame(factor_array.T, index=standardized.index, columns=factor_columns)

        loading_rows = []
        params = getattr(result, "params", None)
        if params is not None:
            params_series = pd.Series(params)
            for parameter_name, value in params_series.items():
                if "loading" in str(parameter_name).casefold():
                    loading_rows.append(
                        {
                            "feature_name": str(parameter_name),
                            "coefficient": float(value),
                            "abs_coefficient": abs(float(value)),
                            "importance_type": "factor_loading",
                        }
                    )
        loadings = pd.DataFrame.from_records(loading_rows)
        return factors, loadings, "statsmodels_dynamic_factor"
    except Exception:
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "Dynamic-factor baseline requires statsmodels DynamicFactor or sklearn PCA to be installed."
            ) from exc

        component_count = max(1, min(n_factors, standardized.shape[0], standardized.shape[1]))
        pca = PCA(n_components=component_count)
        transformed = pca.fit_transform(standardized.to_numpy(dtype="float64"))
        factor_columns = [f"factor_{index + 1}" for index in range(component_count)]
        factors = pd.DataFrame(transformed, index=standardized.index, columns=factor_columns)
        loadings = pd.DataFrame(pca.components_.T, index=standardized.columns, columns=factor_columns)
        loading_rows = loadings.reset_index().melt(id_vars="index", var_name="factor_name", value_name="coefficient")
        loading_rows = loading_rows.rename(columns={"index": "feature_name"})
        loading_rows["abs_coefficient"] = loading_rows["coefficient"].abs()
        loading_rows["importance_type"] = "pca_loading"
        return factors, loading_rows, "sklearn_pca_fallback"


def _build_factor_stage_design(stage_targets: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_row in stage_targets.itertuples(index=False):
        base_row = {
            "quarter": target_row.quarter,
            "quarter_end": target_row.quarter_end,
            "quarter_start": target_row.quarter_start,
            "month_end": target_row.month_end,
            "month_in_quarter": int(target_row.month_in_quarter),
            "information_set": target_row.information_set,
            "target": target_row.target,
            "target_geo": target_row.target_geo,
            "target_column": target_row.target_column,
        }
        subset = factors.loc[(factors.index >= target_row.quarter_start) & (factors.index <= target_row.month_end)]
        for factor_name in factors.columns:
            base_row[f"{factor_name}__mean"] = subset[factor_name].mean() if not subset.empty else np.nan
            base_row[f"{factor_name}__last"] = subset[factor_name].iloc[-1] if not subset.empty else np.nan
        rows.append(base_row)
    return pd.DataFrame.from_records(rows).sort_values("quarter_end", kind="stable").reset_index(drop=True)


def _fit_linear_model_with_optional_statsmodels(
    transformed_design: pd.DataFrame,
    target: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, str]:
    try:
        import statsmodels.api as sm

        result = sm.OLS(target, sm.add_constant(transformed_design.to_numpy(dtype="float64"))).fit()
        coefficients = np.asarray(result.params[1:], dtype="float64")
        intercept = float(result.params[0])
        pvalues = np.asarray(result.pvalues[1:], dtype="float64")
        return coefficients, intercept, pvalues, "statsmodels_ols"
    except Exception:
        x_matrix = transformed_design.to_numpy(dtype="float64")
        x_augmented = np.column_stack([np.ones(x_matrix.shape[0]), x_matrix])
        beta, *_ = np.linalg.lstsq(x_augmented, target, rcond=None)
        coefficients = np.asarray(beta[1:], dtype="float64")
        intercept = float(beta[0])
        pvalues = np.full_like(coefficients, np.nan, dtype="float64")
        return coefficients, intercept, pvalues, "numpy_lstsq"


def _active_feature_columns(design: pd.DataFrame) -> list[str]:
    candidate_columns = [column for column in design.columns if column not in IDENTIFIER_COLUMNS]
    active = []
    for column in candidate_columns:
        numeric = pd.to_numeric(design[column], errors="coerce")
        if numeric.notna().sum() < 3:
            continue
        if numeric.dropna().std(ddof=0) == 0:
            continue
        active.append(column)
    return active


def _write_table_with_csv_fallback(frame: pd.DataFrame, stem: Path, logger: logging.Logger) -> Path:
    csv_path = stem.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    try:
        write_dataframe_parquet(frame, stem.with_suffix(".parquet"))
    except RuntimeError as exc:
        logger.warning("Skipping parquet output for %s: %s", stem.name, exc)
    return csv_path


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "_No rows available._"
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


if __name__ == "__main__":
    raise SystemExit(main())
