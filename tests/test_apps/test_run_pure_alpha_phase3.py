from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import build_phase3_signal_artifacts


def test_phase3_signal_builder_uses_open_to_open_forward_label(tmp_path) -> None:
    paths = _write_signal_fixture(tmp_path, days=35, symbols=("AAA", "BBB", "CCC"))
    session_dates = ("2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25")
    membership_path = _write_membership(tmp_path, session_dates=session_dates, symbols=("AAA", "BBB", "CCC"))
    beta_path = _write_beta_panel(tmp_path, session_dates=session_dates, symbols=("AAA", "BBB", "CCC"), beta=1.0)

    build_phase3_signal_artifacts(
        membership_path=membership_path,
        beta_panel_path=beta_path,
        daily_globs=(paths["stock_daily"],),
        adj_factor_globs=(paths["stock_adj"],),
        benchmark_daily_globs=(paths["benchmark_daily"],),
        benchmark_adj_factor_globs=(paths["benchmark_adj"],),
        output_root=tmp_path / "phase3",
        holding_period_sessions=2,
        min_cross_section=2,
    )
    panel = pd.read_csv(tmp_path / "phase3" / "phase3_baseline_signal_panel_validation.csv.gz")
    aaa = panel.loc[panel["symbol"] == "AAA"].iloc[0]

    expected_forward = (123.0 / 121.0) - 1.0
    expected_return_5d = (120.0 / 115.0) - 1.0
    assert abs(aaa["forward_return_5d"] - expected_forward) < 1e-12
    assert abs(aaa["reversal_5d"] + expected_return_5d) < 1e-12


def test_phase3_signal_builder_writes_leaderboard_and_diagnostics(tmp_path) -> None:
    paths = _write_signal_fixture(tmp_path, days=70, symbols=("AAA", "BBB", "CCC", "DDD"))
    session_dates = ("2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06")
    membership_path = _write_membership(
        tmp_path,
        session_dates=session_dates,
        symbols=("AAA", "BBB", "CCC", "DDD"),
    )
    beta_path = _write_beta_panel(
        tmp_path,
        session_dates=session_dates,
        symbols=("AAA", "BBB", "CCC", "DDD"),
        beta=1.0,
    )

    result = build_phase3_signal_artifacts(
        membership_path=membership_path,
        beta_panel_path=beta_path,
        daily_globs=(paths["stock_daily"],),
        adj_factor_globs=(paths["stock_adj"],),
        benchmark_daily_globs=(paths["benchmark_daily"],),
        benchmark_adj_factor_globs=(paths["benchmark_adj"],),
        output_root=tmp_path / "phase3",
        holding_period_sessions=2,
        min_cross_section=2,
        top_bottom_quantile=0.5,
    )
    leaderboard = pd.read_csv(tmp_path / "phase3" / "phase3_baseline_signal_leaderboard_validation.csv")
    diagnostics = pd.read_csv(
        tmp_path / "phase3" / "phase3_baseline_signal_daily_diagnostics_validation.csv"
    )

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert result["primary_target"] == "forward_beta_residual_return_5d"
    assert "random_control" in set(leaderboard["signal"])
    assert not diagnostics.empty
    assert diagnostics["test_window_used"].eq(False).all()
    assert (tmp_path / "phase3" / "phase3_baseline_signal_memo.md").exists()


def test_phase3_signal_builder_random_control_is_deterministic(tmp_path) -> None:
    paths = _write_signal_fixture(tmp_path, days=35, symbols=("AAA", "BBB", "CCC"))
    session_dates = ("2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25")
    membership_path = _write_membership(tmp_path, session_dates=session_dates, symbols=("AAA", "BBB", "CCC"))
    beta_path = _write_beta_panel(tmp_path, session_dates=session_dates, symbols=("AAA", "BBB", "CCC"), beta=1.0)

    kwargs = {
        "membership_path": membership_path,
        "beta_panel_path": beta_path,
        "daily_globs": (paths["stock_daily"],),
        "adj_factor_globs": (paths["stock_adj"],),
        "benchmark_daily_globs": (paths["benchmark_daily"],),
        "benchmark_adj_factor_globs": (paths["benchmark_adj"],),
        "holding_period_sessions": 2,
        "min_cross_section": 2,
    }
    build_phase3_signal_artifacts(output_root=tmp_path / "phase3_a", **kwargs)
    build_phase3_signal_artifacts(output_root=tmp_path / "phase3_b", **kwargs)
    panel_a = pd.read_csv(tmp_path / "phase3_a" / "phase3_baseline_signal_panel_validation.csv.gz")
    panel_b = pd.read_csv(tmp_path / "phase3_b" / "phase3_baseline_signal_panel_validation.csv.gz")

    assert panel_a[["symbol", "random_control"]].equals(panel_b[["symbol", "random_control"]])


def _write_signal_fixture(tmp_path, *, days: int, symbols: tuple[str, ...]) -> dict[str, object]:
    stock_daily = tmp_path / "stock_daily.jsonl"
    stock_adj = tmp_path / "stock_adj.jsonl"
    benchmark_daily = tmp_path / "benchmark_daily.jsonl"
    benchmark_adj = tmp_path / "benchmark_adj.jsonl"
    start = date(2020, 1, 1)

    stock_rows = []
    stock_adj_rows = []
    benchmark_rows = []
    benchmark_adj_rows = []
    for offset in range(days):
        session = (start + timedelta(days=offset)).isoformat()
        benchmark_open = 200.0 + offset
        benchmark_close = 200.5 + offset
        benchmark_rows.append(_bar(session, "SPY", benchmark_open, benchmark_close))
        benchmark_adj_rows.append(_adj(session, "SPY"))
        for symbol_index, symbol in enumerate(symbols):
            base = 100.0 + symbol_index * 20.0
            stock_rows.append(_bar(session, symbol, base + offset, base + offset))
            stock_adj_rows.append(_adj(session, symbol))
    _write_jsonl(stock_daily, stock_rows)
    _write_jsonl(stock_adj, stock_adj_rows)
    _write_jsonl(benchmark_daily, benchmark_rows)
    _write_jsonl(benchmark_adj, benchmark_adj_rows)
    return {
        "stock_daily": stock_daily,
        "stock_adj": stock_adj,
        "benchmark_daily": benchmark_daily,
        "benchmark_adj": benchmark_adj,
    }


def _write_membership(tmp_path, *, session_dates: tuple[str, ...], symbols: tuple[str, ...]):
    path = tmp_path / "membership.csv.gz"
    rows = []
    for session_date in session_dates:
        for index, symbol in enumerate(symbols, start=1):
            rows.append(
                {
                    "session_date": session_date,
                    "symbol": symbol,
                    "variant": "top500_clean_core_beta_full",
                    "liquidity_rank": index,
                }
            )
    pd.DataFrame(
        rows
    ).to_csv(path, index=False, compression="gzip")
    return path


def _write_beta_panel(tmp_path, *, session_dates: tuple[str, ...], symbols: tuple[str, ...], beta: float):
    path = tmp_path / "beta.csv.gz"
    rows = []
    for session_date in session_dates:
        for symbol in symbols:
            rows.append(
                {
                    "session_date": session_date,
                    "symbol": symbol,
                    "beta": beta,
                    "beta_available": True,
                }
            )
    pd.DataFrame(
        rows
    ).to_csv(path, index=False, compression="gzip")
    return path


def _bar(session_date: str, symbol: str, open_: float, close: float) -> dict[str, object]:
    return {"session_date": session_date, "symbol": symbol, "open": open_, "close": close}


def _adj(session_date: str, symbol: str) -> dict[str, object]:
    return {"session_date": session_date, "symbol": symbol, "price_adjust_factor": 1.0}


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
