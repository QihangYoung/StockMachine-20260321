from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from stockmachine.research.builders import lightgbm_models


def _toy_feature_frame() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        row = {
            feature: float((idx + 1) * (feature_index + 1)) / 100.0
            for feature_index, feature in enumerate(lightgbm_models.FEATURE_COLUMNS)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _toy_rank_frame() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2025-01-02"] * 4 + ["2025-01-03"] * 4 + ["2025-01-06"] * 4)
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN"] * 3
    for idx, (date, symbol) in enumerate(zip(dates, symbols, strict=True)):
        row = {
            "date": date,
            "symbol": symbol,
            "target": [0.03, 0.02, 0.01, -0.01, 0.025, 0.015, 0.005, -0.005, 0.02, 0.01, 0.0, -0.01][idx],
        }
        for feature_index, feature in enumerate(lightgbm_models.FEATURE_COLUMNS):
            row[feature] = float((idx + 1) * (feature_index + 1)) / 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_lightgbm_regressor_fits_and_predicts() -> None:
    model = lightgbm_models.build_lightgbm_regressor()
    assert isinstance(model, Pipeline)

    frame = _toy_feature_frame()
    target = pd.Series(np.linspace(-0.02, 0.02, len(frame)))

    model.fit(frame, target)
    predictions = model.predict(frame)

    assert len(predictions) == len(frame)


def test_build_lightgbm_ranker_fits_and_predicts() -> None:
    model = lightgbm_models.build_lightgbm_ranker()

    frame = _toy_rank_frame()
    group = frame.groupby("date", sort=False).size().tolist()

    model.fit(frame, frame["target"], group=group)
    predictions = model.predict(frame)

    assert len(predictions) == len(frame)
    assert np.isfinite(predictions).all()


def test_missing_lightgbm_raises_clear_import_error(monkeypatch) -> None:
    original_import_module = lightgbm_models.importlib.import_module

    def _missing_lightgbm(name: str, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("missing lightgbm")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(lightgbm_models.importlib, "import_module", _missing_lightgbm)

    with pytest.raises(ImportError, match="LightGBM is required"):
        lightgbm_models.build_lightgbm_regressor()
