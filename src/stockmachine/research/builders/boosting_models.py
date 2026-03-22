from __future__ import annotations

import importlib
from typing import Callable

from sklearn.pipeline import Pipeline

from stockmachine.research.builders.common import build_tree_model_pipeline


def build_catboost_regressor() -> Pipeline:
    """Construct a CatBoost regressor for cross-sectional return prediction."""

    CatBoostRegressor = _load_catboost_regressor()
    return build_tree_model_pipeline(
        CatBoostRegressor(
            allow_writing_files=False,
            depth=6,
            iterations=200,
            learning_rate=0.05,
            loss_function="RMSE",
            l2_leaf_reg=3.0,
            random_seed=7,
            thread_count=-1,
            verbose=False,
        )
    )


def build_xgboost_regressor() -> Pipeline:
    """Construct an XGBoost regressor for cross-sectional return prediction."""

    XGBRegressor = _load_xgboost_regressor()
    return build_tree_model_pipeline(
        XGBRegressor(
            colsample_bytree=0.8,
            eval_metric="rmse",
            learning_rate=0.05,
            max_depth=4,
            n_estimators=250,
            n_jobs=-1,
            objective="reg:squarederror",
            random_state=7,
            reg_lambda=1.0,
            subsample=0.8,
            verbosity=0,
        )
    )


def get_boosting_model_builders() -> dict[str, Callable[[], Pipeline]]:
    """Return the built-in boosting model builders."""

    return {
        "catboost_regressor": build_catboost_regressor,
        "xgboost_regressor": build_xgboost_regressor,
    }


def _load_catboost_regressor() -> type:
    try:
        module = importlib.import_module("catboost")
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "catboost is required for build_catboost_regressor(); install the optional P1 dependencies."
        ) from exc
    try:
        return getattr(module, "CatBoostRegressor")
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "catboost.CatBoostRegressor was not found; verify the installed catboost package."
        ) from exc


def _load_xgboost_regressor() -> type:
    try:
        module = importlib.import_module("xgboost")
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "xgboost is required for build_xgboost_regressor(); install the optional P1 dependencies."
        ) from exc
    try:
        return getattr(module, "XGBRegressor")
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "xgboost.XGBRegressor was not found; verify the installed xgboost package."
        ) from exc

