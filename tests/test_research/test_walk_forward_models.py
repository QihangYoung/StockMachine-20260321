from __future__ import annotations

import numpy as np
import pandas as pd

from stockmachine.alpha import list_alpha_expert_names
from stockmachine.research.us_equities_baseline import FEATURE_COLUMNS, generate_walk_forward_predictions


def _synthetic_panel(*, end: str = "2025-03-31") -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", end, freq="B")
    symbols = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL")
    sectors = {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "NVDA": "Tech",
        "AMZN": "Consumer",
        "META": "Comm",
        "GOOGL": "Comm",
    }
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []

    for day_index, current_date in enumerate(dates):
        benchmark_future_return = 0.0005 * ((day_index % 5) - 2)
        for symbol_index, symbol in enumerate(symbols):
            base = symbol_index + 1
            feature_values = {}
            for feature_index, feature in enumerate(FEATURE_COLUMNS, start=1):
                drift = 0.002 * day_index
                cross_section = 0.05 * base
                seasonal = 0.01 * np.sin((day_index + feature_index) / 5.0)
                noise = rng.normal(0.0, 0.002)
                feature_values[feature] = cross_section + drift + seasonal + noise + feature_index / 50.0

            target = (
                0.20 * feature_values["mom_20"]
                + 0.10 * feature_values["rel_mom_20"]
                - 0.08 * feature_values["vol_20"]
                + 0.04 * feature_values["volume_ratio_20"]
                + rng.normal(0.0, 0.001)
            )
            future_return = target + benchmark_future_return

            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "sector": sectors[symbol],
                    "industry": f"{sectors[symbol]}-Industry",
                    "close": 100.0 + day_index + base,
                    "vol_20": abs(feature_values["vol_20"]) + 0.01,
                    "median_dollar_volume_20": 60_000_000.0 + base * 1_000_000.0,
                    "target": float(target),
                    "future_return": float(future_return),
                    "benchmark_future_return": float(benchmark_future_return),
                    **feature_values,
                }
            )

    return pd.DataFrame(rows)


def test_generate_walk_forward_predictions_includes_new_base_models() -> None:
    panel = _synthetic_panel()

    predictions = generate_walk_forward_predictions(
        panel,
        predict_start="2025-03-01",
        train_window_days=20,
        validation_window_days=5,
        test_window_days=10,
        purge_window_days=1,
        embargo_window_days=0,
    )

    produced_models = set(predictions["model"].unique())
    expected_models = set(list_alpha_expert_names())

    assert expected_models.issubset(produced_models)
    assert {"score", "confidence", "target", "future_return", "benchmark_future_return"}.issubset(
        predictions.columns
    )
    assert {
        "split_train_start",
        "split_train_end",
        "split_validation_start",
        "split_validation_end",
        "split_test_start",
        "split_test_end",
    }.issubset(predictions.columns)
    for model_name in expected_models:
        model_predictions = predictions[predictions["model"] == model_name]
        assert not model_predictions.empty
        assert model_predictions["date"].min() >= pd.Timestamp("2025-03-01")
    assert not predictions.duplicated(subset=["model", "date", "symbol"]).any()


def test_generate_walk_forward_predictions_can_limit_to_requested_model_scope() -> None:
    panel = _synthetic_panel()

    predictions = generate_walk_forward_predictions(
        panel,
        predict_start="2025-03-01",
        requested_models=("extra_trees",),
        train_window_days=20,
        validation_window_days=5,
        test_window_days=10,
        purge_window_days=1,
        embargo_window_days=0,
    )

    assert set(predictions["model"].unique()) == {"extra_trees"}
    assert not predictions.empty


def test_generate_walk_forward_predictions_can_limit_to_requested_ensemble_scope() -> None:
    panel = _synthetic_panel()

    predictions = generate_walk_forward_predictions(
        panel,
        predict_start="2025-03-01",
        requested_models=("ensemble_extra_trees_hist_gbm_rank",),
        train_window_days=20,
        validation_window_days=5,
        test_window_days=10,
        purge_window_days=1,
        embargo_window_days=0,
    )

    assert set(predictions["model"].unique()) == {"extra_trees", "hist_gbm", "ensemble_extra_trees_hist_gbm_rank"}
    assert not predictions.empty


def test_generate_walk_forward_predictions_can_include_partial_current_test_tail() -> None:
    panel = _synthetic_panel(end="2025-04-08")
    unlabeled_mask = panel["date"] >= pd.Timestamp("2025-04-01")
    panel.loc[unlabeled_mask, ["target", "future_return", "benchmark_future_return"]] = np.nan

    predictions = generate_walk_forward_predictions(
        panel,
        predict_start="2025-04-01",
        requested_models=("extra_trees",),
        train_window_days=20,
        validation_window_days=5,
        test_window_days=10,
        purge_window_days=1,
        embargo_window_days=0,
        include_partial_current_test=True,
    )

    assert set(predictions["model"].unique()) == {"extra_trees"}
    assert not predictions.empty
    assert predictions["date"].max() == pd.Timestamp("2025-04-08")
    assert predictions.loc[predictions["date"] == pd.Timestamp("2025-04-08"), "target"].isna().all()
