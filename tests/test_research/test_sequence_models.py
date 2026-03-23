from __future__ import annotations

import numpy as np
import pandas as pd

from stockmachine.research.builders.sequence_models import (
    build_lstm_regressor,
    build_transformer_regressor,
)
from stockmachine.research.us_equities_baseline import FEATURE_COLUMNS


def _toy_sequence_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=30, freq="B")
    symbols = ("AAPL", "MSFT", "NVDA")
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols, start=1):
        for date_index, current_date in enumerate(dates, start=1):
            close = 100.0 + symbol_index + date_index
            open_price = close * 0.997
            high = close * 1.01
            low = close * 0.99
            volume = 1_000_000.0 + symbol_index * 25_000.0 + date_index * 1_000.0
            row = {
                "date": current_date,
                "symbol": symbol,
                "sector": "Tech",
                "industry": "Tech-Industry",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "dollar_volume": close * volume,
                "vol_20": 0.02 + symbol_index / 100.0,
                "median_dollar_volume_20": 75_000_000.0 + symbol_index * 1_000_000.0 + date_index * 100_000.0,
                "future_return": 0.001 * date_index,
                "benchmark_future_return": 0.0002 * date_index,
            }
            for feature_index, feature in enumerate(FEATURE_COLUMNS, start=1):
                row[feature] = 0.01 * symbol_index + 0.001 * date_index + feature_index / 100.0
            row["target"] = float(0.2 * row["mom_20"] - 0.1 * row["vol_20"] + 0.05 * row["rel_mom_20"])
            rows.append(row)
    return pd.DataFrame(rows)


def test_lstm_sequence_builder_fit_and_predict() -> None:
    frame = _toy_sequence_frame()
    cutoff = frame["date"].sort_values().iloc[-8]
    validation = frame.loc[frame["date"] >= cutoff].copy()
    train = frame.loc[frame["date"] < cutoff].copy()
    history = pd.concat([train, validation], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    model = build_lstm_regressor(lookback=5, epochs=2, patience=1, hidden_size=8, max_train_samples=128)

    model.fit(train, train["target"], validation_frame=validation, history_frame=history)
    predictions = model.predict(validation)

    assert len(predictions) == len(validation)
    assert np.isfinite(predictions).all()


def test_transformer_sequence_builder_fit_and_predict() -> None:
    frame = _toy_sequence_frame()
    cutoff = frame["date"].sort_values().iloc[-8]
    validation = frame.loc[frame["date"] >= cutoff].copy()
    train = frame.loc[frame["date"] < cutoff].copy()
    history = pd.concat([train, validation], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    model = build_transformer_regressor(
        lookback=5,
        epochs=2,
        patience=1,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_train_samples=128,
    )

    model.fit(train, train["target"], validation_frame=validation, history_frame=history)
    predictions = model.predict(validation)

    assert len(predictions) == len(validation)
    assert np.isfinite(predictions).all()
