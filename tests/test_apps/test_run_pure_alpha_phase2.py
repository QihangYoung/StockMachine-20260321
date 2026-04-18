from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase2 import build_phase2_beta_artifacts


def test_phase2_beta_builder_uses_prior_returns_only(tmp_path) -> None:
    paths = _write_price_fixture(
        tmp_path,
        stock_returns={"AAA": [0.02, -0.04, 0.03, -0.02, 0.50]},
        benchmark_returns=[0.01, -0.02, 0.015, -0.01, 0.02],
    )
    membership_path = tmp_path / "membership.csv.gz"
    pd.DataFrame(
        {
            "session_date": ["2020-01-06"],
            "symbol": ["AAA"],
            "variant": ["top500_clean_core_beta_full"],
        }
    ).to_csv(membership_path, index=False, compression="gzip")

    result = build_phase2_beta_artifacts(
        membership_path=membership_path,
        daily_globs=(paths["stock_daily"],),
        adj_factor_globs=(paths["stock_adj"],),
        benchmark_daily_globs=(paths["benchmark_daily"],),
        benchmark_adj_factor_globs=(paths["benchmark_adj"],),
        output_root=tmp_path / "phase2",
        lookback_sessions=3,
        min_observations=3,
        shrinkage_strength=0.0,
        asof_lag_sessions=1,
    )
    panel = pd.read_csv(tmp_path / "phase2" / "phase2_beta_panel_validation.csv.gz")

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert panel.loc[0, "beta_available"] == True
    assert panel.loc[0, "beta_obs"] == 3
    assert abs(panel.loc[0, "beta_raw"] - 2.0) < 1e-10
    assert abs(panel.loc[0, "beta"] - 2.0) < 1e-10


def test_phase2_beta_builder_applies_shrinkage_and_clip(tmp_path) -> None:
    paths = _write_price_fixture(
        tmp_path,
        stock_returns={"AAA": [0.04, -0.08, 0.06, -0.04, 0.08]},
        benchmark_returns=[0.01, -0.02, 0.015, -0.01, 0.02],
    )
    membership_path = tmp_path / "membership.csv.gz"
    pd.DataFrame(
        {
            "session_date": ["2020-01-06"],
            "symbol": ["AAA"],
            "variant": ["top500_clean_core_beta_full"],
        }
    ).to_csv(membership_path, index=False, compression="gzip")

    build_phase2_beta_artifacts(
        membership_path=membership_path,
        daily_globs=(paths["stock_daily"],),
        adj_factor_globs=(paths["stock_adj"],),
        benchmark_daily_globs=(paths["benchmark_daily"],),
        benchmark_adj_factor_globs=(paths["benchmark_adj"],),
        output_root=tmp_path / "phase2",
        lookback_sessions=3,
        min_observations=3,
        shrinkage_target=1.0,
        shrinkage_strength=0.25,
        beta_clip_high=3.0,
        asof_lag_sessions=1,
    )
    panel = pd.read_csv(tmp_path / "phase2" / "phase2_beta_panel_validation.csv.gz")

    assert abs(panel.loc[0, "beta_raw"] - 4.0) < 1e-10
    assert abs(panel.loc[0, "beta_shrunk"] - 3.25) < 1e-10
    assert abs(panel.loc[0, "beta"] - 3.0) < 1e-10


def test_phase2_beta_builder_writes_variant_coverage_and_memo(tmp_path) -> None:
    paths = _write_price_fixture(
        tmp_path,
        stock_returns={
            "AAA": [0.02, -0.04, 0.03, -0.02, 0.04],
            "BBB": [0.01, -0.02, 0.015, -0.01, 0.02],
        },
        benchmark_returns=[0.01, -0.02, 0.015, -0.01, 0.02],
    )
    membership_path = tmp_path / "membership.csv.gz"
    pd.DataFrame(
        {
            "session_date": ["2020-01-06", "2020-01-06"],
            "symbol": ["AAA", "BBB"],
            "variant": ["top500_clean_core_beta_full", "top500_clean_core_beta_full"],
        }
    ).to_csv(membership_path, index=False, compression="gzip")

    result = build_phase2_beta_artifacts(
        membership_path=membership_path,
        daily_globs=(paths["stock_daily"],),
        adj_factor_globs=(paths["stock_adj"],),
        benchmark_daily_globs=(paths["benchmark_daily"],),
        benchmark_adj_factor_globs=(paths["benchmark_adj"],),
        output_root=tmp_path / "phase2",
        lookback_sessions=3,
        min_observations=3,
        shrinkage_strength=0.0,
        asof_lag_sessions=1,
    )
    coverage = pd.read_csv(tmp_path / "phase2" / "phase2_variant_beta_coverage_validation.csv")

    assert result["beta_available_rows"] == 2
    assert coverage.loc[0, "members"] == 2
    assert coverage.loc[0, "beta_available_members"] == 2
    assert coverage.loc[0, "beta_coverage"] == 1.0
    assert (tmp_path / "phase2" / "phase2_beta_panel_memo.md").exists()
    assert (tmp_path / "phase2" / "phase2_beta_coverage_summary_validation.csv").exists()


def _write_price_fixture(
    tmp_path,
    *,
    stock_returns: dict[str, list[float]],
    benchmark_returns: list[float],
) -> dict[str, object]:
    start = date(2020, 1, 1)
    dates = [(start + timedelta(days=index)).isoformat() for index in range(len(benchmark_returns) + 1)]
    stock_daily = tmp_path / "stock_daily.jsonl"
    stock_adj = tmp_path / "stock_adj.jsonl"
    benchmark_daily = tmp_path / "benchmark_daily.jsonl"
    benchmark_adj = tmp_path / "benchmark_adj.jsonl"

    stock_daily_rows = []
    stock_adj_rows = []
    for symbol, returns in stock_returns.items():
        close = 100.0
        stock_daily_rows.append(_bar(dates[0], symbol, close))
        stock_adj_rows.append(_adj(dates[0], symbol))
        for session_date, daily_return in zip(dates[1:], returns):
            close *= 1.0 + daily_return
            stock_daily_rows.append(_bar(session_date, symbol, close))
            stock_adj_rows.append(_adj(session_date, symbol))

    benchmark_close = 100.0
    benchmark_daily_rows = [_bar(dates[0], "SPY", benchmark_close)]
    benchmark_adj_rows = [_adj(dates[0], "SPY")]
    for session_date, daily_return in zip(dates[1:], benchmark_returns):
        benchmark_close *= 1.0 + daily_return
        benchmark_daily_rows.append(_bar(session_date, "SPY", benchmark_close))
        benchmark_adj_rows.append(_adj(session_date, "SPY"))

    _write_jsonl(stock_daily, stock_daily_rows)
    _write_jsonl(stock_adj, stock_adj_rows)
    _write_jsonl(benchmark_daily, benchmark_daily_rows)
    _write_jsonl(benchmark_adj, benchmark_adj_rows)
    return {
        "stock_daily": stock_daily,
        "stock_adj": stock_adj,
        "benchmark_daily": benchmark_daily,
        "benchmark_adj": benchmark_adj,
    }


def _bar(session_date: str, symbol: str, close: float) -> dict[str, object]:
    return {"session_date": session_date, "symbol": symbol, "close": close}


def _adj(session_date: str, symbol: str) -> dict[str, object]:
    return {"session_date": session_date, "symbol": symbol, "price_adjust_factor": 1.0}


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
