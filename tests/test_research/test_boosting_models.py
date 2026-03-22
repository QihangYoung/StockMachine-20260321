from __future__ import annotations

import importlib

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from stockmachine.research.builders.boosting_models import (
    build_catboost_regressor,
    build_xgboost_regressor,
    get_boosting_model_builders,
)
from stockmachine.research.builders.common import FEATURE_COLUMNS


def _toy_feature_frame() -> pd.DataFrame:
    rows = []
    for idx in range(24):
        rows.append(
            {
                feature: float((idx + 1) * (feature_index + 1)) / 100.0
                for feature_index, feature in enumerate(FEATURE_COLUMNS)
            }
        )
    return pd.DataFrame(rows)


def _toy_target_series() -> pd.Series:
    values = [0.03, 0.02, 0.01, -0.01, 0.015, -0.005, 0.025, -0.015] * 3
    return pd.Series(values[:24], dtype=float)


@pytest.mark.parametrize(
    ("module_name", "builder_name"),
    [
        ("catboost", "catboost_regressor"),
        ("xgboost", "xgboost_regressor"),
    ],
)
def test_boosting_model_builders_return_pipelines(module_name: str, builder_name: str) -> None:
    pytest.importorskip(module_name)

    builders = get_boosting_model_builders()

    assert set(builders) == {"catboost_regressor", "xgboost_regressor"}
    model = builders[builder_name]()
    assert isinstance(model, Pipeline)


@pytest.mark.parametrize(
    ("module_name", "builder_name"),
    [
        ("catboost", "catboost_regressor"),
        ("xgboost", "xgboost_regressor"),
    ],
)
def test_boosting_model_pipelines_fit_and_predict(module_name: str, builder_name: str) -> None:
    pytest.importorskip(module_name)

    train_x = _toy_feature_frame()
    train_y = _toy_target_series()

    builders = {
        "catboost_regressor": build_catboost_regressor,
        "xgboost_regressor": build_xgboost_regressor,
    }
    model = builders[builder_name]()
    model.fit(train_x, train_y)
    predictions = model.predict(train_x)
    assert len(predictions) == len(train_x)


def test_catboost_builder_raises_clear_import_error_when_dependency_missing(monkeypatch) -> None:
    def _missing_catboost(module_name: str, package=None):  # type: ignore[no-untyped-def]
        if module_name == "catboost":
            raise ImportError("missing catboost")
        return importlib.import_module(module_name, package=package)

    monkeypatch.setattr(
        "stockmachine.research.builders.boosting_models.importlib.import_module",
        _missing_catboost,
    )

    with pytest.raises(ImportError, match="catboost is required for build_catboost_regressor"):
        build_catboost_regressor()


def test_xgboost_builder_raises_clear_import_error_when_dependency_missing(monkeypatch) -> None:
    def _missing_xgboost(module_name: str, package=None):  # type: ignore[no-untyped-def]
        if module_name == "xgboost":
            raise ImportError("missing xgboost")
        return importlib.import_module(module_name, package=package)

    monkeypatch.setattr(
        "stockmachine.research.builders.boosting_models.importlib.import_module",
        _missing_xgboost,
    )

    with pytest.raises(ImportError, match="xgboost is required for build_xgboost_regressor"):
        build_xgboost_regressor()
