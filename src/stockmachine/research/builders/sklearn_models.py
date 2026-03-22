from __future__ import annotations

from typing import Callable

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline

from stockmachine.research.builders.common import build_linear_model_pipeline, build_tree_model_pipeline


def build_ridge_pipeline() -> Pipeline:
    """Construct the linear baseline model."""

    return build_linear_model_pipeline(Ridge(alpha=1.0))


def build_huber_pipeline() -> Pipeline:
    """Construct a robust linear baseline for heavy-tailed return labels."""

    return build_linear_model_pipeline(
        HuberRegressor(
            alpha=0.0001,
            epsilon=1.5,
            max_iter=300,
            tol=1e-5,
        )
    )


def build_elastic_net_pipeline() -> Pipeline:
    """Construct a sparse linear baseline over the standardized feature panel."""

    return build_linear_model_pipeline(
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.15,
            max_iter=5000,
            selection="cyclic",
            tol=1e-4,
        )
    )


def build_hist_gbm_model() -> Pipeline:
    """Construct the tree-based baseline model."""

    return build_tree_model_pipeline(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=200,
            min_samples_leaf=40,
            random_state=7,
        )
    )


def build_extra_trees_model() -> Pipeline:
    """Construct a bagging-style tree model for noisy cross-sectional ranking."""

    return build_tree_model_pipeline(
        ExtraTreesRegressor(
            bootstrap=False,
            max_depth=8,
            max_features="sqrt",
            min_samples_leaf=40,
            n_estimators=400,
            n_jobs=-1,
            random_state=7,
        )
    )


def build_random_forest_model() -> Pipeline:
    """Construct a conservative bagging tree baseline."""

    return build_tree_model_pipeline(
        RandomForestRegressor(
            bootstrap=True,
            max_depth=6,
            max_features="sqrt",
            min_samples_leaf=40,
            n_estimators=300,
            n_jobs=-1,
            random_state=7,
        )
    )


def get_sklearn_model_builders() -> dict[str, Callable[[], Pipeline]]:
    """Return the built-in scikit-learn model builders."""

    return {
        "ridge": build_ridge_pipeline,
        "huber_regression": build_huber_pipeline,
        "elastic_net": build_elastic_net_pipeline,
        "hist_gbm": build_hist_gbm_model,
        "extra_trees": build_extra_trees_model,
        "random_forest": build_random_forest_model,
    }

