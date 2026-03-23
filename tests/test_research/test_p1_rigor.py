from __future__ import annotations

import pandas as pd
import pytest

from stockmachine.research.p1_rigor import (
    build_cost_stress_summary,
    build_period_stability_summary,
    build_strict_research_bundle,
    summarize_backtest_records,
)


def _sample_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2025-01-02",
                "entry_date": "2025-01-03",
                "exit_date": "2025-01-10",
                "gross_return": 0.02,
                "net_return": 0.019,
                "benchmark_return": 0.01,
                "turnover": 1.0,
                "cost_bps": 10.0,
                "positions": 10,
            },
            {
                "signal_date": "2025-02-03",
                "entry_date": "2025-02-04",
                "exit_date": "2025-02-11",
                "gross_return": -0.01,
                "net_return": -0.011,
                "benchmark_return": -0.005,
                "turnover": 1.0,
                "cost_bps": 10.0,
                "positions": 10,
            },
            {
                "signal_date": "2025-04-02",
                "entry_date": "2025-04-03",
                "exit_date": "2025-04-10",
                "gross_return": 0.03,
                "net_return": 0.028,
                "benchmark_return": 0.015,
                "turnover": 2.0,
                "cost_bps": 20.0,
                "positions": 10,
            },
        ]
    )


def test_summarize_backtest_records_uses_existing_net_returns_by_default() -> None:
    summary = summarize_backtest_records(_sample_records(), horizon=5)

    assert summary["sessions"] == 3
    assert summary["benchmark_total_return"] > 0
    assert summary["mean_turnover"] == 4 / 3
    assert summary["mean_cost_bps"] == 40 / 3


def test_summarize_backtest_records_can_revalue_with_new_cost() -> None:
    records = _sample_records()

    summary = summarize_backtest_records(records, horizon=5, cost_bps_per_side=40.0)

    expected_returns = records["gross_return"] - records["turnover"] * 0.004
    expected_total = float((1.0 + expected_returns).prod() - 1.0)
    assert abs(summary["total_return"] - expected_total) < 1e-12
    assert summary["mean_cost_bps"] == float((records["turnover"] * 40.0).mean())


def test_build_period_stability_summary_splits_year_and_quarter() -> None:
    records = _sample_records()

    yearly = build_period_stability_summary(records, model_name="hist_gbm", horizon=5, period="year")
    quarterly = build_period_stability_summary(records, model_name="hist_gbm", horizon=5, period="quarter")

    assert list(yearly["period_label"]) == ["2025"]
    assert set(quarterly["period_label"]) == {"2025Q1", "2025Q2"}
    assert set(yearly["model"]) == {"hist_gbm"}
    assert set(quarterly["period"]) == {"quarter"}


def test_build_cost_stress_summary_emits_one_row_per_cost_level() -> None:
    summary = build_cost_stress_summary(
        _sample_records(),
        model_name="hist_gbm",
        horizon=5,
        cost_levels_bps=(10.0, 20.0, 40.0),
    )

    assert list(summary["cost_bps_per_side"]) == [10.0, 20.0, 40.0]
    assert (summary["model"] == "hist_gbm").all()
    assert "excess_total_return" in summary.columns


def _strict_bundle_price_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    return pd.DataFrame(
        [
            {
                "date": current_date,
                "symbol": symbol,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1_000_000.0,
                "adj_open": 100.0 + index,
                "adj_close": 100.5 + index,
            }
            for index, current_date in enumerate(dates)
            for symbol in ("AAPL", "SPY")
        ]
    )


def _strict_bundle_dataset(*, include_membership: bool) -> dict[str, pd.DataFrame]:
    membership = pd.DataFrame(
        [
            {
                "session_date": "2025-01-02",
                "universe_name": "us_equities_research_v1",
                "symbol": "AAPL",
                "is_member": True,
            },
            {
                "session_date": "2025-01-03",
                "universe_name": "us_equities_research_v1",
                "symbol": "AAPL",
                "is_member": True,
            },
        ]
    )
    return {
        "daily_bar": pd.DataFrame(),
        "benchmark_index": pd.DataFrame(),
        "adj_factor": pd.DataFrame(),
        "symbol_master": pd.DataFrame(
            [
                {"as_of_date": "2025-01-02", "symbol": "AAPL", "is_active": True},
                {"as_of_date": "2025-01-03", "symbol": "AAPL", "is_active": True},
            ]
        ),
        "industry_membership": pd.DataFrame(
            [
                {"as_of_date": "2025-01-02", "symbol": "AAPL", "industry_system": "gics"},
                {"as_of_date": "2025-01-03", "symbol": "AAPL", "industry_system": "gics"},
            ]
        ),
        "universe_membership": membership if include_membership else pd.DataFrame(),
    }


def test_build_strict_research_bundle_requires_explicit_membership(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: _strict_bundle_dataset(include_membership=False),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: _strict_bundle_price_panel(),
    )

    with pytest.raises(ValueError, match="explicit universe_membership history"):
        build_strict_research_bundle(predict_start="2025-01-01", horizon=5)


def test_build_strict_research_bundle_requires_snapshot_and_explicit_coverage(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: _strict_bundle_price_panel(),
    )

    def _metadata(session_dates, **kwargs):
        captured["require_snapshot"] = kwargs.get("require_snapshot")
        captured["session_dates"] = list(pd.Index(session_dates))
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "symbol": "AAPL",
                    "company_name": "Apple",
                    "quote_type": "EQUITY",
                    "exchange": "XNAS",
                    "currency": "USD",
                    "country": "US",
                    "sector": "Technology",
                    "industry": "Hardware",
                },
                {
                    "date": pd.Timestamp("2025-01-03"),
                    "symbol": "AAPL",
                    "company_name": "Apple",
                    "quote_type": "EQUITY",
                    "exchange": "XNAS",
                    "currency": "USD",
                    "country": "US",
                    "sector": "Technology",
                    "industry": "Hardware",
                },
            ]
        )

    monkeypatch.setattr("stockmachine.research.p1_rigor.build_point_in_time_metadata_history", _metadata)
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda price_data, **kwargs: pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "symbol": "AAPL",
                    "sector": "Technology",
                    "industry": "Hardware",
                    "close": 100.0,
                    "vol_20": 0.02,
                    "median_dollar_volume_20": 100_000_000.0,
                    "target": 0.01,
                    "future_return": 0.015,
                    "benchmark_future_return": 0.005,
                    "gap_1": 0.01,
                    "ret_1d": 0.01,
                    "mom_5": 0.02,
                    "mom_10": 0.03,
                    "mom_20": 0.04,
                    "mom_60": 0.05,
                    "vol_60": 0.03,
                    "range_1d": 0.02,
                    "volume_ratio_20": 1.1,
                    "rel_mom_20": 0.01,
                    "rel_mom_60": 0.02,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda research_frame, **kwargs: pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "symbol": "AAPL",
                    "model": "hist_gbm",
                    "score": 0.9,
                    "confidence": 0.9,
                }
            ]
        ),
    )

    bundle = build_strict_research_bundle(predict_start="2025-01-01", horizon=5)

    assert captured["require_snapshot"] is True
    assert captured["session_dates"] == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert list(bundle.predictions["model"]) == ["hist_gbm"]
