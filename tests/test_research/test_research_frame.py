from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockmachine.research.us_equities_baseline import (
    FEATURE_COLUMNS,
    H5TargetConfig,
    build_research_frame,
)


def _toy_price_panel(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    rows: list[dict[str, object]] = []
    for idx, current_date in enumerate(dates):
        stock_open = 100.0 + idx * 0.7
        stock_close = stock_open * (1.0 + 0.002 * np.sin(idx / 5.0))
        benchmark_open = 400.0 + idx * 0.35
        benchmark_close = benchmark_open * (1.0 + 0.001 * np.cos(idx / 7.0))

        rows.append(
            {
                "date": current_date,
                "symbol": "AAPL",
                "open": stock_open,
                "high": stock_open * 1.01,
                "low": stock_open * 0.99,
                "close": stock_close,
                "volume": 1_000_000.0 + idx * 5_000.0,
                "price_adjust_factor": 1.0,
            }
        )
        rows.append(
            {
                "date": current_date,
                "symbol": "SPY",
                "open": benchmark_open,
                "high": benchmark_open * 1.005,
                "low": benchmark_open * 0.995,
                "close": benchmark_close,
                "volume": 10_000_000.0 + idx * 10_000.0,
                "price_adjust_factor": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _toy_price_panel_with_peer(periods: int = 90) -> pd.DataFrame:
    base = _toy_price_panel(periods=periods)
    dates = pd.Index(pd.to_datetime(base["date"].drop_duplicates())).sort_values()
    peer_rows: list[dict[str, object]] = []
    for idx, current_date in enumerate(dates):
        stock_open = 120.0 + idx * 0.45
        stock_close = stock_open * (1.0 + 0.0015 * np.cos(idx / 6.0))
        peer_rows.append(
            {
                "date": current_date,
                "symbol": "MSFT",
                "open": stock_open,
                "high": stock_open * 1.008,
                "low": stock_open * 0.992,
                "close": stock_close,
                "volume": 900_000.0 + idx * 6_000.0,
                "price_adjust_factor": 1.0,
            }
        )
    return pd.concat([base, pd.DataFrame(peer_rows)], ignore_index=True)


def _row_for(frame: pd.DataFrame, date: pd.Timestamp, symbol: str = "AAPL") -> pd.Series:
    row = frame.loc[(frame["date"] == date) & (frame["symbol"] == symbol)]
    assert not row.empty
    return row.iloc[0]


def test_build_research_frame_features_do_not_use_future_symbol_prices() -> None:
    price_data = _toy_price_panel()
    anchor_date = pd.Timestamp("2024-04-12")
    ordered_dates = pd.Index(pd.to_datetime(price_data["date"]).drop_duplicates().sort_values())
    anchor_index = int(ordered_dates.get_loc(anchor_date))
    entry_date = ordered_dates[anchor_index + 1]
    exit_date = ordered_dates[anchor_index + 6]

    base = build_research_frame(price_data, benchmark_symbol="SPY", horizon=5)
    mutated = price_data.copy()
    entry_mask = (mutated["symbol"] == "AAPL") & (mutated["date"] == entry_date)
    exit_mask = (mutated["symbol"] == "AAPL") & (mutated["date"] == exit_date)
    mutated.loc[entry_mask, "open"] *= 0.8
    mutated.loc[entry_mask, "high"] *= 0.8
    mutated.loc[entry_mask, "low"] *= 0.8
    mutated.loc[entry_mask, "close"] *= 0.8
    mutated.loc[exit_mask, "open"] *= 1.25
    mutated.loc[exit_mask, "high"] *= 1.25
    mutated.loc[exit_mask, "low"] *= 1.25
    mutated.loc[exit_mask, "close"] *= 1.25
    shifted = build_research_frame(mutated, benchmark_symbol="SPY", horizon=5)

    base_row = _row_for(base, anchor_date)
    shifted_row = _row_for(shifted, anchor_date)

    np.testing.assert_allclose(
        base_row[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        shifted_row[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    assert base_row["future_return"] != shifted_row["future_return"]
    assert base_row["target"] != shifted_row["target"]


def test_build_research_frame_target_responds_to_future_benchmark_only() -> None:
    price_data = _toy_price_panel()
    anchor_date = pd.Timestamp("2024-04-12")
    ordered_dates = pd.Index(pd.to_datetime(price_data["date"]).drop_duplicates().sort_values())
    anchor_index = int(ordered_dates.get_loc(anchor_date))
    entry_date = ordered_dates[anchor_index + 1]
    exit_date = ordered_dates[anchor_index + 6]

    base = build_research_frame(price_data, benchmark_symbol="SPY", horizon=5)
    mutated = price_data.copy()
    entry_mask = (mutated["symbol"] == "SPY") & (mutated["date"] == entry_date)
    exit_mask = (mutated["symbol"] == "SPY") & (mutated["date"] == exit_date)
    mutated.loc[entry_mask, "open"] *= 1.2
    mutated.loc[entry_mask, "high"] *= 1.2
    mutated.loc[entry_mask, "low"] *= 1.2
    mutated.loc[entry_mask, "close"] *= 1.2
    mutated.loc[exit_mask, "open"] *= 0.85
    mutated.loc[exit_mask, "high"] *= 0.85
    mutated.loc[exit_mask, "low"] *= 0.85
    mutated.loc[exit_mask, "close"] *= 0.85
    shifted = build_research_frame(mutated, benchmark_symbol="SPY", horizon=5)

    base_row = _row_for(base, anchor_date)
    shifted_row = _row_for(shifted, anchor_date)

    np.testing.assert_allclose(
        base_row[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        shifted_row[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    assert base_row["future_return"] == shifted_row["future_return"]
    assert base_row["benchmark_future_return"] != shifted_row["benchmark_future_return"]
    assert base_row["target"] != shifted_row["target"]


def test_build_research_frame_emits_h5_v2_core_features() -> None:
    price_data = _toy_price_panel()
    metadata = pd.DataFrame(
        [
            {"symbol": "AAPL", "sector": "Tech", "industry": "Hardware"},
        ]
    )

    frame = build_research_frame(
        price_data,
        benchmark_symbol="SPY",
        horizon=5,
        symbol_metadata=metadata,
    )

    expected_columns = {
        "gap_z_20",
        "intraday_return",
        "ret_2d",
        "mom_3",
        "vol_5",
        "vol_10",
        "range_5",
        "volume_ratio_5",
        "close_ma5_gap",
        "close_ma20_gap",
        "price_position_20d",
        "rel_ret_1d",
        "rel_ret_5d",
        "sector_rel_ret_1d",
    }
    assert expected_columns.issubset(frame.columns)
    sample_row = _row_for(frame, pd.Timestamp("2024-04-12"))
    assert not sample_row[list(FEATURE_COLUMNS)].isna().any()
    assert sample_row["sector"] == "Tech"
    assert sample_row["industry"] == "Hardware"
    assert sample_row["sector_rel_ret_1d"] == 0.0


def test_build_research_frame_supports_beta_residual_target() -> None:
    price_data = _toy_price_panel()

    frame = build_research_frame(
        price_data,
        benchmark_symbol="SPY",
        horizon=5,
        target_config=H5TargetConfig(kind="beta_residual", beta_lookback_days=20, beta_min_obs=10),
    )

    sample_row = _row_for(frame, pd.Timestamp("2024-04-12"))
    assert sample_row["target_kind"] == "beta_residual"
    assert np.isfinite(sample_row["target_beta_estimate"])
    assert sample_row["target"] == pytest.approx(
        sample_row["future_return"] - sample_row["target_beta_estimate"] * sample_row["benchmark_future_return"]
    )


def test_build_research_frame_supports_sector_residual_target() -> None:
    price_data = _toy_price_panel_with_peer()
    metadata = pd.DataFrame(
        [
            {"symbol": "AAPL", "sector": "Tech", "industry": "Hardware"},
            {"symbol": "MSFT", "sector": "Tech", "industry": "Software"},
        ]
    )

    frame = build_research_frame(
        price_data,
        benchmark_symbol="SPY",
        horizon=5,
        symbol_metadata=metadata,
        target_config=H5TargetConfig(kind="sector_residual"),
    )
    sample_date = pd.Timestamp("2024-04-12")
    day_slice = frame.loc[frame["date"] == sample_date].copy()

    assert set(day_slice["symbol"]) == {"AAPL", "MSFT"}
    sector_mean = float(day_slice["future_return"].mean())
    np.testing.assert_allclose(
        day_slice["target"].to_numpy(),
        (day_slice["future_return"] - sector_mean).to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )
    assert float(day_slice["target"].sum()) == pytest.approx(0.0, abs=1e-12)
