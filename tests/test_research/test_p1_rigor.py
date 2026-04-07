from __future__ import annotations

import pandas as pd
import pytest

from stockmachine.research.p1_rigor import (
    RepositoryCacheState,
    StrictResearchBundle,
    build_cost_stress_summary,
    build_period_stability_summary,
    build_strict_research_bundle,
    run_model_backtest_from_bundle,
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


def test_run_model_backtest_from_bundle_forwards_turnover_control_overrides(monkeypatch) -> None:
    bundle = StrictResearchBundle(
        predict_start="2025-01-01",
        horizon=1,
        strategy_project="us_equities_h1",
        framework_id="us_equities_h1_strict_v1",
        dataset={"daily_bar": pd.DataFrame(), "benchmark_index": pd.DataFrame()},
        research_frame=pd.DataFrame(),
        predictions=pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "symbol": "AAPL",
                    "model": "ridge",
                    "score": 0.9,
                    "confidence": 0.8,
                }
            ]
        ),
    )
    captured: dict[str, object] = {}

    class _FakeEngine:
        def run(self, start_date, end_date):
            return type(
                "Result",
                (),
                {
                    "sessions": 1,
                    "total_return": 0.01,
                    "annualized_return": 0.01,
                    "annualized_volatility": 0.1,
                    "sharpe": 0.5,
                    "max_drawdown": -0.1,
                    "meta": {
                        "benchmark_total_return": 0.0,
                        "mean_turnover": 0.2,
                        "mean_cost_bps": 2.0,
                        "mean_gross_exposure": 1.0,
                        "mean_changed_symbols": 1.0,
                        "records": [],
                    },
                },
            )()

    def _build_framework_backtest_engine(**kwargs):
        captured["turnover_control_overrides"] = kwargs.get("turnover_control_overrides")
        return _FakeEngine()

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_framework_backtest_engine",
        _build_framework_backtest_engine,
    )

    run_model_backtest_from_bundle(
        bundle,
        model_name="ridge",
        turnover_control_overrides={"no_trade_band": 0.07, "max_turnover": 0.5},
    )

    assert captured["turnover_control_overrides"] == {"no_trade_band": 0.07, "max_turnover": 0.5}


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


def test_build_strict_research_bundle_uses_h1_framework_components(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: _cache_test_metadata(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("h5 research frame should not be used")),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("h5 predictions should not be used")),
    )
    def _build_h1_research_frame(price_data, **kwargs):
        captured["research_frame_called"] = True
        return pd.DataFrame(
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
                    "gap_z_20": 0.0,
                    "intraday_return": 0.01,
                    "ret_1d": 0.01,
                    "ret_2d": 0.02,
                    "mom_3": 0.03,
                    "range_1d": 0.02,
                    "range_5": 0.02,
                    "vol_5": 0.02,
                    "volume_ratio_5": 1.1,
                    "volume_ratio_20": 1.1,
                    "rel_ret_1d": 0.01,
                    "rel_mom_3": 0.01,
                    "sector_rel_ret_1d": 0.0,
                    "sector_rel_mom_3": 0.0,
                }
            ]
        )

    def _generate_h1_predictions(research_frame, **kwargs):
        captured["predictions_called"] = kwargs["predict_start"]
        captured["feature_version"] = kwargs.get("feature_version")
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "symbol": "AAPL",
                    "model": "extra_trees",
                    "score": 0.9,
                    "confidence": 0.9,
                }
            ]
        )

    monkeypatch.setattr("stockmachine.research.h1_us_equities.build_h1_research_frame", _build_h1_research_frame)
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.generate_h1_walk_forward_predictions",
        _generate_h1_predictions,
    )

    bundle = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=1,
        strategy_project="us_equities_h1",
        prediction_options={"feature_version": "v2"},
    )

    assert bundle.strategy_project == "us_equities_h1"
    assert bundle.framework_id == "us_equities_h1_strict_v1"
    assert captured["research_frame_called"] is True
    assert captured["predictions_called"] == "2025-01-01"
    assert captured["feature_version"] == "v2"


def test_build_strict_research_bundle_routes_h5_target_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: _cache_test_metadata(),
    )

    def _build_h5_research_frame(price_data, **kwargs):
        captured["target_config"] = kwargs.get("target_config")
        return _cache_test_research_frame()

    monkeypatch.setattr("stockmachine.research.p1_rigor.build_research_frame", _build_h5_research_frame)
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

    bundle = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        prediction_options={
            "target_kind": "beta_residual",
            "beta_lookback_days": 80,
            "beta_min_obs": 55,
            "beta_clip_low": 0.1,
            "beta_clip_high": 1.8,
        },
    )

    target_config = captured["target_config"]
    assert bundle.strategy_project == "us_equities_h5"
    assert target_config.kind == "beta_residual"
    assert target_config.beta_lookback_days == 80
    assert target_config.beta_min_obs == 55
    assert target_config.beta_clip_low == 0.1
    assert target_config.beta_clip_high == 1.8


def _configure_cache_safe_identity(monkeypatch, *, silver_token: str = "silver_v1") -> None:
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_repository_cache_state",
        lambda: RepositoryCacheState(
            head="deadbeef",
            clean=True,
            cache_allowed=True,
            reason="clean",
        ),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._collect_dependency_versions",
        lambda: {"python": "3.11.0", "pandas": "test"},
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_silver_input_fingerprint",
        lambda layout: {"token": silver_token},
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_alpha_registry_snapshot",
        lambda: [{"name": "hist_gbm", "family": "tree"}],
    )


def _cache_test_metadata() -> pd.DataFrame:
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


def _cache_test_research_frame() -> pd.DataFrame:
    return pd.DataFrame(
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
    )


def test_build_strict_research_bundle_reuses_valid_bundle_and_prediction_cache(tmp_path, monkeypatch) -> None:
    counts = {"load": 0, "price": 0, "metadata": 0, "research_frame": 0, "predictions": 0}
    _configure_cache_safe_identity(monkeypatch)

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: counts.__setitem__("load", counts["load"] + 1) or _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: counts.__setitem__("price", counts["price"] + 1) or _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: counts.__setitem__("metadata", counts["metadata"] + 1) or _cache_test_metadata(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda price_data, **kwargs: counts.__setitem__("research_frame", counts["research_frame"] + 1)
        or _cache_test_research_frame(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda research_frame, **kwargs: counts.__setitem__("predictions", counts["predictions"] + 1)
        or pd.DataFrame(
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

    cache_dir = tmp_path / "cache"
    first = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )
    second = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )

    assert counts == {"load": 1, "price": 1, "metadata": 1, "research_frame": 1, "predictions": 1}
    assert first.bundle_cache_hit is False
    assert first.prediction_cache_hit is False
    assert second.bundle_cache_hit is True
    assert second.prediction_cache_hit is True
    assert first.bundle_cache_key == second.bundle_cache_key
    assert first.prediction_cache_key == second.prediction_cache_key


def test_build_strict_research_bundle_reuses_bundle_but_rebuilds_predictions_for_new_predict_start(
    tmp_path,
    monkeypatch,
) -> None:
    counts = {"load": 0, "price": 0, "metadata": 0, "research_frame": 0, "predictions": 0}
    _configure_cache_safe_identity(monkeypatch)

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: counts.__setitem__("load", counts["load"] + 1) or _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: counts.__setitem__("price", counts["price"] + 1) or _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: counts.__setitem__("metadata", counts["metadata"] + 1) or _cache_test_metadata(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda price_data, **kwargs: counts.__setitem__("research_frame", counts["research_frame"] + 1)
        or _cache_test_research_frame(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda research_frame, **kwargs: counts.__setitem__("predictions", counts["predictions"] + 1)
        or pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "symbol": "AAPL",
                    "model": "hist_gbm",
                    "score": 0.5 if kwargs["predict_start"] == "2025-01-01" else 0.8,
                    "confidence": 0.9,
                }
            ]
        ),
    )

    cache_dir = tmp_path / "cache"
    first = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )
    second = build_strict_research_bundle(
        predict_start="2025-06-01",
        horizon=5,
        cache_dir=cache_dir,
    )

    assert counts["load"] == 1
    assert counts["price"] == 1
    assert counts["metadata"] == 1
    assert counts["research_frame"] == 1
    assert counts["predictions"] == 2
    assert second.bundle_cache_hit is True
    assert second.prediction_cache_hit is False
    assert first.bundle_cache_key == second.bundle_cache_key
    assert first.prediction_cache_key != second.prediction_cache_key


def test_build_strict_research_bundle_invalidates_cache_when_silver_fingerprint_changes(tmp_path, monkeypatch) -> None:
    counts = {"load": 0, "price": 0, "metadata": 0, "research_frame": 0, "predictions": 0}
    silver_state = {"token": "silver_v1"}

    _configure_cache_safe_identity(monkeypatch, silver_token=silver_state["token"])
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_silver_input_fingerprint",
        lambda layout: {"token": silver_state["token"]},
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: counts.__setitem__("load", counts["load"] + 1) or _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: counts.__setitem__("price", counts["price"] + 1) or _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: counts.__setitem__("metadata", counts["metadata"] + 1) or _cache_test_metadata(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda price_data, **kwargs: counts.__setitem__("research_frame", counts["research_frame"] + 1)
        or _cache_test_research_frame(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda research_frame, **kwargs: counts.__setitem__("predictions", counts["predictions"] + 1)
        or pd.DataFrame(
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

    cache_dir = tmp_path / "cache"
    first = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )
    silver_state["token"] = "silver_v2"
    second = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )

    assert counts["load"] == 2
    assert counts["price"] == 2
    assert counts["metadata"] == 2
    assert counts["research_frame"] == 2
    assert counts["predictions"] == 2
    assert first.bundle_cache_key != second.bundle_cache_key
    assert second.bundle_cache_hit is False
    assert second.prediction_cache_hit is False


def test_build_strict_research_bundle_invalidates_bundle_cache_when_h5_target_changes(tmp_path, monkeypatch) -> None:
    counts = {"load": 0, "price": 0, "metadata": 0, "research_frame": 0, "predictions": 0}
    _configure_cache_safe_identity(monkeypatch)

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: counts.__setitem__("load", counts["load"] + 1) or _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: counts.__setitem__("price", counts["price"] + 1) or _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: counts.__setitem__("metadata", counts["metadata"] + 1) or _cache_test_metadata(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda price_data, **kwargs: counts.__setitem__("research_frame", counts["research_frame"] + 1)
        or _cache_test_research_frame(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda research_frame, **kwargs: counts.__setitem__("predictions", counts["predictions"] + 1)
        or pd.DataFrame(
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

    cache_dir = tmp_path / "cache"
    first = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
        prediction_options={"target_kind": "benchmark_excess"},
    )
    second = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
        prediction_options={"target_kind": "beta_residual", "beta_lookback_days": 80},
    )

    assert counts["load"] == 2
    assert counts["price"] == 2
    assert counts["metadata"] == 2
    assert counts["research_frame"] == 2
    assert counts["predictions"] == 2
    assert first.bundle_cache_key != second.bundle_cache_key
    assert second.bundle_cache_hit is False
    assert second.prediction_cache_hit is False


def test_build_strict_research_bundle_skips_cache_when_repository_is_dirty(tmp_path, monkeypatch) -> None:
    counts = {"load": 0, "price": 0, "metadata": 0, "research_frame": 0, "predictions": 0}

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_repository_cache_state",
        lambda: RepositoryCacheState(
            head="deadbeef",
            clean=False,
            cache_allowed=False,
            reason="repository_dirty",
        ),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.load_us_equities_dataset",
        lambda layout=None: counts.__setitem__("load", counts["load"] + 1) or _strict_bundle_dataset(include_membership=True),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_price_panel_from_silver",
        lambda dataset: counts.__setitem__("price", counts["price"] + 1) or _strict_bundle_price_panel(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_point_in_time_metadata_history",
        lambda session_dates, **kwargs: counts.__setitem__("metadata", counts["metadata"] + 1) or _cache_test_metadata(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.build_research_frame",
        lambda price_data, **kwargs: counts.__setitem__("research_frame", counts["research_frame"] + 1)
        or _cache_test_research_frame(),
    )
    monkeypatch.setattr(
        "stockmachine.research.p1_rigor.generate_walk_forward_predictions",
        lambda research_frame, **kwargs: counts.__setitem__("predictions", counts["predictions"] + 1)
        or pd.DataFrame(
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

    cache_dir = tmp_path / "cache"
    first = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )
    second = build_strict_research_bundle(
        predict_start="2025-01-01",
        horizon=5,
        cache_dir=cache_dir,
    )

    assert counts == {"load": 2, "price": 2, "metadata": 2, "research_frame": 2, "predictions": 2}
    assert first.bundle_cache_key is None
    assert first.prediction_cache_key is None
    assert second.bundle_cache_key is None
    assert second.prediction_cache_key is None
    assert not cache_dir.exists()
