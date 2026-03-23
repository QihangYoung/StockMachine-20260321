from __future__ import annotations

import pandas as pd

from stockmachine.research.us_equities_baseline import build_ensemble_prediction_frames


def _prediction_frame(*, model: str, scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "symbol": ["AAPL", "MSFT"],
            "sector": ["Tech", "Tech"],
            "industry": ["Hardware", "Software"],
            "close": [100.0, 200.0],
            "vol_20": [0.02, 0.02],
            "median_dollar_volume_20": [1_000_000.0, 1_500_000.0],
            "target": [0.01, -0.01],
            "future_return": [0.015, -0.005],
            "benchmark_future_return": [0.005, 0.005],
            "score": scores,
            "confidence": [1.0, 0.5],
            "model": [model, model],
        }
    )


def test_build_ensemble_prediction_frames_adds_mean_and_rank_blends() -> None:
    frames = {
        "hist_gbm": _prediction_frame(model="hist_gbm", scores=[2.0, -1.0]),
        "ridge": _prediction_frame(model="ridge", scores=[0.8, 0.1]),
        "random_forest": _prediction_frame(model="random_forest", scores=[1.2, -0.2]),
        "extra_trees": _prediction_frame(model="extra_trees", scores=[1.7, -0.4]),
        "lightgbm_ranker": _prediction_frame(model="lightgbm_ranker", scores=[0.9, 0.2]),
    }

    ensemble_frames = build_ensemble_prediction_frames(frames)
    models = {frame["model"].iloc[0] for frame in ensemble_frames}

    assert models == {
        "ensemble_hist_gbm_ridge_mean",
        "ensemble_hist_gbm_ridge_rank",
        "ensemble_hist_gbm_random_forest_mean",
        "ensemble_hist_gbm_random_forest_rank",
        "ensemble_hist_gbm_ridge_random_forest_mean",
        "ensemble_hist_gbm_ridge_random_forest_rank",
        "ensemble_extra_trees_hist_gbm_rank",
        "ensemble_extra_trees_lightgbm_ranker_rank",
        "ensemble_extra_trees_hist_gbm_lightgbm_ranker_rank",
    }
    for frame in ensemble_frames:
        assert frame["symbol"].tolist() == ["AAPL", "MSFT"]
        assert frame.iloc[0]["score"] > frame.iloc[1]["score"]
        assert frame.iloc[0]["confidence"] > frame.iloc[1]["confidence"]


def test_build_ensemble_prediction_frames_skips_missing_components() -> None:
    frames = {
        "hist_gbm": _prediction_frame(model="hist_gbm", scores=[2.0, -1.0]),
    }

    ensemble_frames = build_ensemble_prediction_frames(frames)

    assert ensemble_frames == []
