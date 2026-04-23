from __future__ import annotations

from evaluation.backtests import directional_accuracy, evaluate_prediction_frame, mae, rmse, rolling_origin_splits


def test_metric_helpers_return_expected_values() -> None:
    actual = [1.0, -2.0, 3.0]
    predicted = [1.5, -1.0, 1.0]

    assert round(rmse(actual, predicted), 6) == round(((0.25 + 1.0 + 4.0) / 3.0) ** 0.5, 6)
    assert round(mae(actual, predicted), 6) == round((0.5 + 1.0 + 2.0) / 3.0, 6)
    assert directional_accuracy(actual, predicted) == 1.0


def test_rolling_origin_splits_start_after_minimum_training_window() -> None:
    splits = rolling_origin_splits(n_samples=6, min_train_size=3)
    assert splits == [(slice(0, 3), 3), (slice(0, 4), 4), (slice(0, 5), 5)]


def test_evaluate_prediction_frame_groups_metrics() -> None:
    import pandas as pd

    predictions = pd.DataFrame(
        {
            "model_name": ["bridge_ols", "bridge_ols", "elastic_net_baseline"],
            "information_set": ["month_1", "month_1", "month_2"],
            "target_column": ["qoq_real_gdp_growth"] * 3,
            "actual": [1.0, 2.0, -1.0],
            "prediction": [1.5, 1.0, -0.5],
        }
    )
    metrics = evaluate_prediction_frame(predictions)

    assert list(metrics["model_name"]) == ["bridge_ols", "elastic_net_baseline"]
    assert metrics.loc[metrics["model_name"] == "bridge_ols", "evaluation_count"].iloc[0] == 2
