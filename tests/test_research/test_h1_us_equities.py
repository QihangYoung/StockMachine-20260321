import numpy as np
import pandas as pd

from stockmachine.research.h1_us_equities import (
    H1TargetConfig,
    H1_FEATURE_COLUMNS,
    build_h1_research_frame,
    fit_predict_h1_base_model,
)


def test_build_h1_research_frame_produces_expected_columns() -> None:
    dates = pd.bdate_range("2025-01-01", periods=40)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("SPY", 2.5)]:
        for index, current_date in enumerate(dates):
            price = 100.0 + offset + index
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * 1.002,
                    "volume": 1_000_000.0 + index * 1000,
                    "adj_open": price,
                    "adj_close": price * 1.002,
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "sector": ["Tech", "Health"],
            "industry": ["Software", "Biotech"],
        }
    )

    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)

    assert not frame.empty
    assert set(H1_FEATURE_COLUMNS).issubset(frame.columns)
    assert {"future_return", "target", "target_bucket_2", "sector", "industry", "median_dollar_volume_20"}.issubset(frame.columns)
    assert frame["future_return"].notna().all()
    assert frame["target"].notna().all()
    assert frame["target_bucket_2"].isin([0, 1]).all()


def test_fit_predict_h1_base_model_emits_binary_classification_scores() -> None:
    dates = pd.bdate_range("2025-01-01", periods=55)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("CCC", -3.0), ("SPY", 2.0)]:
        for index, current_date in enumerate(dates):
            drift = 1.0 + offset / 100.0
            price = 100.0 + offset + index * drift
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1.001 + offset / 10000.0),
                    "volume": 1_200_000.0 + index * 500 + abs(offset) * 1000,
                    "adj_open": price,
                    "adj_close": price * (1.001 + offset / 10000.0),
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrials"],
            "industry": ["Software", "Biotech", "Machinery"],
        }
    )
    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)
    train_frame = frame.loc[frame["date"] < frame["date"].quantile(0.7)].copy()
    test_frame = frame.loc[frame["date"] >= frame["date"].quantile(0.7)].copy()

    predictions = fit_predict_h1_base_model(
        "hist_gbm",
        train_frame=train_frame,
        test_frame=test_frame,
        target_config=H1TargetConfig(),
    )

    assert not predictions.empty
    assert {"probability_positive", "predicted_bucket", "classification_confidence", "target_bucket_2"}.issubset(
        predictions.columns
    )
    assert np.isfinite(predictions["score"]).all()
    assert ((predictions["probability_positive"] >= 0.0) & (predictions["probability_positive"] <= 1.0)).all()
    assert predictions["predicted_bucket"].isin([0, 1]).all()


def test_fit_predict_h1_base_model_supports_point_regression() -> None:
    dates = pd.bdate_range("2025-01-01", periods=55)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("CCC", -3.0), ("SPY", 2.0)]:
        for index, current_date in enumerate(dates):
            drift = 1.0 + offset / 100.0
            price = 100.0 + offset + index * drift
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1.001 + offset / 10000.0),
                    "volume": 1_200_000.0 + index * 500 + abs(offset) * 1000,
                    "adj_open": price,
                    "adj_close": price * (1.001 + offset / 10000.0),
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrials"],
            "industry": ["Software", "Biotech", "Machinery"],
        }
    )
    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)
    train_frame = frame.loc[frame["date"] < frame["date"].quantile(0.7)].copy()
    test_frame = frame.loc[frame["date"] >= frame["date"].quantile(0.7)].copy()

    predictions = fit_predict_h1_base_model(
        "ridge",
        train_frame=train_frame,
        test_frame=test_frame,
        target_config=H1TargetConfig(task="point_regression"),
    )

    assert not predictions.empty
    assert "target_bucket_2" not in predictions.columns
    assert np.isfinite(predictions["score"]).all()
    assert predictions["probability_positive"].isna().all()
    assert predictions["predicted_bucket"].isin([0, 1]).all()
