from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from stockmachine.research.us_equities_baseline import (
    FEATURE_COLUMNS,
    build_catboost_regressor,
    build_elastic_net_pipeline,
    build_extra_trees_model,
    build_hist_gbm_model,
    build_huber_pipeline,
    build_lightgbm_regressor,
    build_lightgbm_ranker,
    build_lstm_regressor,
    build_random_forest_model,
    build_ridge_pipeline,
    build_transformer_regressor,
    build_xgboost_regressor,
    get_trainable_model_builder,
)


def _toy_feature_frame() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        row = {feature: float((idx + 1) * (feature_index + 1)) / 100.0 for feature_index, feature in enumerate(FEATURE_COLUMNS)}
        rows.append(row)
    return pd.DataFrame(rows)


def _toy_target_series() -> pd.Series:
    return pd.Series([0.03, 0.02, 0.01, -0.01, 0.015, -0.005, 0.025, -0.015], dtype=float)


def test_trainable_model_builders_return_pipelines() -> None:
    builder_names = (
        "ridge",
        "huber_regression",
        "elastic_net",
        "hist_gbm",
        "extra_trees",
        "random_forest",
        "lightgbm_regressor",
        "lightgbm_ranker",
        "catboost_regressor",
        "xgboost_regressor",
        "lstm_regressor",
        "transformer_regressor",
    )

    for name in builder_names:
        model = get_trainable_model_builder(name)()
        if name in {"lightgbm_ranker", "lstm_regressor", "transformer_regressor"}:
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
        else:
            assert isinstance(model, Pipeline)


def test_trainable_model_pipelines_fit_and_predict() -> None:
    train_x = _toy_feature_frame()
    train_y = _toy_target_series()

    builders = (
        build_ridge_pipeline,
        build_huber_pipeline,
        build_elastic_net_pipeline,
        build_hist_gbm_model,
        build_extra_trees_model,
        build_random_forest_model,
        build_lightgbm_regressor,
        build_catboost_regressor,
        build_xgboost_regressor,
    )

    for builder in builders:
        model = builder()
        model.fit(train_x, train_y)
        predictions = model.predict(train_x)
        assert len(predictions) == len(train_x)


def test_lightgbm_ranker_builder_fit_and_predict() -> None:
    train_x = _toy_feature_frame()
    train_x.insert(0, "symbol", [f"S{i}" for i in range(len(train_x))])
    train_x.insert(0, "date", pd.to_datetime(["2025-01-02"] * 4 + ["2025-01-03"] * 4))
    train_y = _toy_target_series()

    model = build_lightgbm_ranker()
    model.fit(train_x, train_y, group=[4, 4])
    predictions = model.predict(train_x)

    assert len(predictions) == len(train_x)


def test_sequence_model_builders_fit_and_predict() -> None:
    dates = pd.date_range("2025-01-02", periods=12, freq="B")
    symbols = ("AAPL", "MSFT")
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols, start=1):
        for date_index, current_date in enumerate(dates, start=1):
            row = {
                "date": current_date,
                "symbol": symbol,
                "sector": "Tech",
                "industry": "Tech-Industry",
                "close": 100.0 + date_index,
                "vol_20": 0.02,
                "median_dollar_volume_20": 70_000_000.0,
                "future_return": 0.001 * date_index,
                "benchmark_future_return": 0.0003 * date_index,
            }
            for feature_index, feature in enumerate(FEATURE_COLUMNS, start=1):
                row[feature] = 0.01 * symbol_index + 0.001 * date_index + feature_index / 100.0
            row["target"] = float(0.1 * row["mom_20"] - 0.05 * row["vol_20"] + 0.02 * row["rel_mom_20"])
            rows.append(row)
    frame = pd.DataFrame(rows)

    builders = (
        lambda: build_lstm_regressor(lookback=5, epochs=1, hidden_size=8, max_train_samples=64),
        lambda: build_transformer_regressor(
            lookback=5,
            epochs=1,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            max_train_samples=64,
        ),
    )

    for builder in builders:
        model = builder()
        model.fit(frame, frame["target"])
        predictions = model.predict(frame)
        assert len(predictions) == len(frame)
