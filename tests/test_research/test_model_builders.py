from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from stockmachine.research.us_equities_baseline import (
    FEATURE_COLUMNS,
    build_elastic_net_pipeline,
    build_extra_trees_model,
    build_hist_gbm_model,
    build_huber_pipeline,
    build_random_forest_model,
    build_ridge_pipeline,
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
    )

    for name in builder_names:
        model = get_trainable_model_builder(name)()
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
    )

    for builder in builders:
        model = builder()
        model.fit(train_x, train_y)
        predictions = model.predict(train_x)
        assert len(predictions) == len(train_x)

