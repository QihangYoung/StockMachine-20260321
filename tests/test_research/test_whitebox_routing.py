from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from stockmachine.research.p1_rigor import StrictResearchBundle, run_model_backtest_from_bundle
from stockmachine.research.us_equities_baseline import OverlayConfig, resolve_model_whitebox_policy


def _make_bundle(*, model_name: str) -> StrictResearchBundle:
    predictions = pd.DataFrame(
        {
            "model": [model_name],
            "date": [pd.Timestamp("2020-01-02")],
        }
    )
    return StrictResearchBundle(
        predict_start="2020-01-01",
        horizon=5,
        strategy_project="us_equities_h5",
        framework_id="us_equities_h5",
        dataset={"daily_bar": pd.DataFrame(), "benchmark_index": pd.DataFrame()},
        research_frame=pd.DataFrame(),
        predictions=predictions,
    )


def test_resolve_model_whitebox_policy_uses_known_overrides() -> None:
    base = OverlayConfig()

    top_k, config = resolve_model_whitebox_policy("extra_trees", top_k=10, overlay_config=base)
    assert top_k == 8
    assert config.min_median_dollar_volume_20 == 30_000_000.0
    assert config.max_vol_20 == 0.04

    top_k, config = resolve_model_whitebox_policy("lightgbm_ranker", top_k=8, overlay_config=base)
    assert top_k == 10
    assert config.min_median_dollar_volume_20 == 30_000_000.0
    assert config.max_vol_20 == 0.065

    top_k, config = resolve_model_whitebox_policy("ridge", top_k=9, overlay_config=base)
    assert top_k == 9
    assert config == base


def test_run_model_backtest_from_bundle_routes_whitebox_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeEngine:
        def run(self, *, start_date, end_date):
            return SimpleNamespace(
                sessions=1,
                total_return=0.01,
                annualized_return=0.02,
                annualized_volatility=0.03,
                sharpe=0.5,
                max_drawdown=-0.1,
                meta={
                    "records": [],
                    "benchmark_total_return": 0.0,
                    "mean_turnover": 0.1,
                    "mean_cost_bps": 8.0,
                    "mean_gross_exposure": 1.0,
                    "mean_changed_symbols": 1.0,
                },
            )

    def _fake_build_engine(**kwargs):
        captured.update(kwargs)
        return _FakeEngine()

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_framework_backtest_engine",
        _fake_build_engine,
    )

    result = run_model_backtest_from_bundle(
        _make_bundle(model_name="extra_trees"),
        model_name="extra_trees",
        top_k=10,
        overlay_config=OverlayConfig(),
        route_model_whitebox=True,
    )

    assert captured["top_k"] == 8
    assert captured["overlay_config"].min_median_dollar_volume_20 == 30_000_000.0
    assert captured["overlay_config"].max_vol_20 == 0.04
    assert result["requested_top_k"] == 10
    assert result["effective_top_k"] == 8


def test_run_model_backtest_from_bundle_can_disable_whitebox_routing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeEngine:
        def run(self, *, start_date, end_date):
            return SimpleNamespace(
                sessions=1,
                total_return=0.01,
                annualized_return=0.02,
                annualized_volatility=0.03,
                sharpe=0.5,
                max_drawdown=-0.1,
                meta={
                    "records": [],
                    "benchmark_total_return": 0.0,
                    "mean_turnover": 0.1,
                    "mean_cost_bps": 8.0,
                    "mean_gross_exposure": 1.0,
                    "mean_changed_symbols": 1.0,
                },
            )

    def _fake_build_engine(**kwargs):
        captured.update(kwargs)
        return _FakeEngine()

    monkeypatch.setattr(
        "stockmachine.research.p1_rigor._build_framework_backtest_engine",
        _fake_build_engine,
    )

    result = run_model_backtest_from_bundle(
        _make_bundle(model_name="extra_trees"),
        model_name="extra_trees",
        top_k=10,
        overlay_config=OverlayConfig(),
        route_model_whitebox=False,
    )

    assert captured["top_k"] == 10
    assert captured["overlay_config"].min_median_dollar_volume_20 == 50_000_000.0
    assert captured["overlay_config"].max_vol_20 == 0.04
    assert result["effective_top_k"] == 10
