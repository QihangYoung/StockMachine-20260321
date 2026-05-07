from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4z import _active_return_dates
from stockmachine.apps.run_pure_alpha_phase5b import (
    _build_h10_turnover_curve,
    _fmt_float_like,
    _fmt_pct_like,
    _text_table,
)
from stockmachine.research.robustness_analyzers import (
    build_cost_execution_stress_summary,
    build_tail_dependence_summary,
    build_time_stability_summary,
)
from stockmachine.research.robustness_frameworks import resolve_robustness_framework


DEFAULT_PHASE5F_ROOT = RESEARCH_ROOT / "phase5f_overlay_combo_on_turnover_aware_20260427"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase6_sota_robustness_20260507"
DEFAULT_PORTFOLIO = "short_overlay_only"
DEFAULT_MODEL_NAME = "current_sota"
DEFAULT_HOLDING_PERIOD_SESSIONS = 10
DEFAULT_POSITIONS_PATH = DEFAULT_PHASE5F_ROOT / "phase5f_positions_validation.csv.gz"
DEFAULT_DAILY_DIAGNOSTICS_PATH = DEFAULT_PHASE5F_ROOT / "phase5f_daily_validation.csv"
DEFAULT_STRICT_CURVE_PATH = (
    DEFAULT_PHASE5F_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_daily_curve.csv"
)
DEFAULT_ASSET_QA_PATH = (
    RESEARCH_ROOT / "phase0_asset_class_qa_20260417" / "top1000_asset_class_qa_by_symbol.csv"
)
DEFAULT_TRANSACTION_COST_LEVELS_BPS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 10.0)
DEFAULT_BORROW_COST_LEVELS_BPS_ANNUAL: tuple[float, ...] = (0.0, 50.0, 200.0, 500.0)
DEFAULT_MARGIN_COST_LEVELS_BPS_ANNUAL: tuple[float, ...] = (0.0, 300.0, 600.0)


def build_phase6_sota_robustness_artifacts(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS_PATH,
    daily_diagnostics_path: str | Path = DEFAULT_DAILY_DIAGNOSTICS_PATH,
    strict_curve_path: str | Path = DEFAULT_STRICT_CURVE_PATH,
    asset_qa_path: str | Path = DEFAULT_ASSET_QA_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    portfolio: str = DEFAULT_PORTFOLIO,
    model_name: str = DEFAULT_MODEL_NAME,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
    transaction_cost_levels_bps: Sequence[float] = DEFAULT_TRANSACTION_COST_LEVELS_BPS,
    borrow_cost_levels_bps_annual: Sequence[float] = DEFAULT_BORROW_COST_LEVELS_BPS_ANNUAL,
    margin_cost_levels_bps_annual: Sequence[float] = DEFAULT_MARGIN_COST_LEVELS_BPS_ANNUAL,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_strict_curve(strict_curve_path, portfolio=portfolio)
    diagnostics = _load_daily_diagnostics(daily_diagnostics_path, portfolio=portfolio)
    positions = _load_full_positions(positions_path, portfolio=portfolio)
    calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)

    turnover = _build_h10_turnover_curve(
        positions_path=positions_path,
        portfolio=portfolio,
        benchmark_calendar=calendar,
        holding_period_sessions=holding_period_sessions,
    )
    aggregate_positions, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=holding_period_sessions,
    )
    records = _build_backtest_records(curve=curve, turnover=turnover, daily_exposure=daily_exposure)
    summary = _build_summary(records, curve=curve)

    # The shared project framework is h5 by name, but the strict rolling h10 output is
    # a daily multi-sleeve return stream. Phase6 therefore annualizes with horizon=1.
    framework = resolve_robustness_framework(strategy_project="us_equities_pure_alpha_h5", horizon=5)
    yearly = build_time_stability_summary(
        records,
        model_name=model_name,
        horizon=1,
        framework=framework,
        period="year",
    )
    quarterly = build_time_stability_summary(
        records,
        model_name=model_name,
        horizon=1,
        framework=framework,
        period="quarter",
    )
    tail = build_tail_dependence_summary(
        records,
        model_name=model_name,
        horizon=1,
        framework=framework,
    )
    cost = build_cost_execution_stress_summary(
        records,
        model_name=model_name,
        horizon=1,
        framework=framework,
        cost_levels_bps=transaction_cost_levels_bps,
    )
    financing = _build_financing_borrow_stress(
        records,
        transaction_cost_levels_bps=(0.0, 2.0, 4.0),
        borrow_cost_levels_bps_annual=borrow_cost_levels_bps_annual,
        margin_cost_levels_bps_annual=margin_cost_levels_bps_annual,
    )
    market_state = _build_market_state_exposure(records)
    rolling_beta = _build_rolling_beta(records, windows=(60, 126, 252))
    rolling_beta_summary = _build_rolling_beta_summary(rolling_beta)
    leg_attribution = _build_leg_attribution(curve=curve, diagnostics=diagnostics)
    concentration = _build_position_concentration_summary(daily_exposure)
    sector = _build_sector_exposure_summary(aggregate_positions)
    borrow_proxy = _build_borrow_proxy_coverage(positions=positions, asset_qa_path=asset_qa_path)
    gate = _build_gate_check(
        summary=summary,
        yearly=yearly,
        tail=tail,
        cost=cost,
        market_state=market_state,
        rolling_beta_summary=rolling_beta_summary,
        leg_attribution=leg_attribution,
        concentration=concentration,
        borrow_proxy=borrow_proxy,
    )

    records_path = output_dir / "phase6_sota_backtest_records.csv"
    summary_path = output_dir / "phase6_sota_summary_metrics.csv"
    yearly_path = output_dir / "phase6_time_stability_yearly.csv"
    quarterly_path = output_dir / "phase6_time_stability_quarterly.csv"
    tail_path = output_dir / "phase6_tail_dependence.csv"
    cost_path = output_dir / "phase6_transaction_cost_stress.csv"
    financing_path = output_dir / "phase6_financing_borrow_stress.csv"
    market_path = output_dir / "phase6_market_state_exposure.csv"
    rolling_beta_path = output_dir / "phase6_rolling_realized_beta.csv"
    rolling_beta_summary_path = output_dir / "phase6_rolling_realized_beta_summary.csv"
    leg_path = output_dir / "phase6_leg_attribution.csv"
    concentration_path = output_dir / "phase6_position_concentration.csv"
    daily_exposure_path = output_dir / "phase6_daily_exposure.csv"
    sector_path = output_dir / "phase6_sector_exposure.csv"
    borrow_path = output_dir / "phase6_borrow_proxy_coverage.csv"
    gate_path = output_dir / "phase6_gate_check.csv"
    aggregate_positions_path = output_dir / "phase6_daily_aggregate_positions.csv.gz"
    plot_path = output_dir / "phase6_sota_robustness_plot.png"
    memo_path = output_dir / "phase6_sota_robustness_memo.md"
    rollup_path = output_dir / "phase6_rollup.json"

    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    quarterly.to_csv(quarterly_path, index=False)
    tail.to_csv(tail_path, index=False)
    cost.to_csv(cost_path, index=False)
    financing.to_csv(financing_path, index=False)
    market_state.to_csv(market_path, index=False)
    rolling_beta.to_csv(rolling_beta_path, index=False)
    rolling_beta_summary.to_csv(rolling_beta_summary_path, index=False)
    leg_attribution.to_csv(leg_path, index=False)
    concentration.to_csv(concentration_path, index=False)
    daily_exposure.to_csv(daily_exposure_path, index=False)
    sector.to_csv(sector_path, index=False)
    borrow_proxy.to_csv(borrow_path, index=False)
    gate.to_csv(gate_path, index=False)
    aggregate_positions.to_csv(aggregate_positions_path, index=False, compression="gzip")
    _plot_robustness(records=records, rolling_beta=rolling_beta, output_path=plot_path)
    memo_path.write_text(
        _memo(
            summary=summary,
            yearly=yearly,
            tail=tail,
            cost=cost,
            financing=financing,
            market_state=market_state,
            rolling_beta_summary=rolling_beta_summary,
            leg_attribution=leg_attribution,
            concentration=concentration,
            sector=sector,
            borrow_proxy=borrow_proxy,
            gate=gate,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "portfolio": portfolio,
        "model_name": model_name,
        "positions_path": Path(positions_path).as_posix(),
        "daily_diagnostics_path": Path(daily_diagnostics_path).as_posix(),
        "strict_curve_path": Path(strict_curve_path).as_posix(),
        "asset_qa_path": Path(asset_qa_path).as_posix(),
        "holding_period_sessions": int(holding_period_sessions),
        "annualization_horizon_for_robustness": 1,
        "records_artifact": records_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "yearly_artifact": yearly_path.as_posix(),
        "quarterly_artifact": quarterly_path.as_posix(),
        "tail_artifact": tail_path.as_posix(),
        "transaction_cost_artifact": cost_path.as_posix(),
        "financing_borrow_artifact": financing_path.as_posix(),
        "market_state_artifact": market_path.as_posix(),
        "rolling_beta_artifact": rolling_beta_path.as_posix(),
        "rolling_beta_summary_artifact": rolling_beta_summary_path.as_posix(),
        "leg_attribution_artifact": leg_path.as_posix(),
        "position_concentration_artifact": concentration_path.as_posix(),
        "daily_exposure_artifact": daily_exposure_path.as_posix(),
        "sector_exposure_artifact": sector_path.as_posix(),
        "borrow_proxy_artifact": borrow_path.as_posix(),
        "gate_check_artifact": gate_path.as_posix(),
        "aggregate_positions_artifact": aggregate_positions_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "validation_only_phase6_sota_robustness_packet",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "Strict h10 PnL is evaluated as a daily multi-sleeve return stream.",
            "Transaction-cost stress uses turnover times bps per traded notional.",
            "Borrow and margin stress are scenario haircuts, not broker-confirmed historical charges.",
            "Borrow coverage uses current-date Alpaca ETB metadata proxy, not point-in-time borrow data.",
            "The current top1000 data lake is not a final survivorship-bias-free full-market universe.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_strict_curve(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["return_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for portfolio '{portfolio}' in {path}.")
    if "test_window_used" in frame and frame["test_window_used"].map(_is_true).any():
        raise ValueError("Strict curve contains test-window rows; refusing Phase6 run.")
    return frame.sort_values("return_date").reset_index(drop=True)


def _load_daily_diagnostics(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No diagnostics found for portfolio '{portfolio}' in {path}.")
    if "test_window_used" in frame and frame["test_window_used"].map(_is_true).any():
        raise ValueError("Daily diagnostics contain test-window rows; refusing Phase6 run.")
    return frame.sort_values("session_date").reset_index(drop=True)


def _load_full_positions(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No positions found for portfolio '{portfolio}' in {path}.")
    if "test_window_used" in frame and frame["test_window_used"].map(_is_true).any():
        raise ValueError("Positions contain test-window rows; refusing Phase6 run.")
    frame["signed_weight"] = pd.to_numeric(frame["signed_weight"], errors="coerce")
    frame = frame.dropna(subset=["session_date", "symbol", "signed_weight"])
    for column in ("sic2_sector", "sic4_industry"):
        if column not in frame.columns:
            frame[column] = "UNKNOWN"
        frame[column] = frame[column].fillna("UNKNOWN").astype(str)
    return frame.sort_values(["session_date", "side", "symbol"]).reset_index(drop=True)


def _build_daily_aggregate_positions(
    positions: pd.DataFrame,
    *,
    benchmark_calendar: pd.Series,
    holding_period_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_map = _active_return_dates(
        positions,
        calendar=benchmark_calendar,
        holding_period_sessions=holding_period_sessions,
    )
    active = positions.merge(active_map, on="session_date", how="inner")
    active_sleeves = (
        active.groupby("return_date", as_index=False)["session_date"]
        .nunique()
        .rename(columns={"session_date": "active_sleeves"})
    )
    active = active.merge(active_sleeves, on="return_date", how="left")
    active["portfolio_weight"] = active["signed_weight"].astype(float) / active["active_sleeves"].astype(float)
    aggregate = (
        active.groupby(["return_date", "symbol"], as_index=False)
        .agg(
            portfolio_weight=("portfolio_weight", "sum"),
            sic2_sector=("sic2_sector", "first"),
            sic4_industry=("sic4_industry", "first"),
        )
        .sort_values(["return_date", "symbol"])
        .reset_index(drop=True)
    )
    aggregate["abs_weight"] = aggregate["portfolio_weight"].abs()
    daily = (
        aggregate.groupby("return_date", as_index=False)
        .apply(_daily_exposure_row, include_groups=False)
        .reset_index(drop=True)
    )
    return aggregate, daily.sort_values("return_date").reset_index(drop=True)


def _daily_exposure_row(group: pd.DataFrame) -> pd.Series:
    weights = group["portfolio_weight"].astype(float)
    abs_weights = weights.abs().sort_values(ascending=False)
    long_weights = weights[weights > 1e-12]
    short_weights = weights[weights < -1e-12]
    gross = float(abs_weights.sum())
    return pd.Series(
        {
            "gross_exposure": gross,
            "long_gross": float(long_weights.sum()),
            "short_gross": float(-short_weights.sum()),
            "net_exposure": float(weights.sum()),
            "positions": int((abs_weights > 1e-12).sum()),
            "long_positions": int((long_weights.abs() > 1e-12).sum()),
            "short_positions": int((short_weights.abs() > 1e-12).sum()),
            "max_abs_weight": float(abs_weights.iloc[0]) if not abs_weights.empty else 0.0,
            "top5_abs_share": _top_share(abs_weights, 5, gross),
            "top10_abs_share": _top_share(abs_weights, 10, gross),
            "hhi_abs_weight": float(((abs_weights / gross) ** 2).sum()) if gross > 0 else np.nan,
        }
    )


def _top_share(abs_weights: pd.Series, count: int, gross: float) -> float:
    if gross <= 0:
        return np.nan
    return float(abs_weights.head(count).sum() / gross)


def _build_backtest_records(
    *,
    curve: pd.DataFrame,
    turnover: pd.DataFrame,
    daily_exposure: pd.DataFrame,
) -> pd.DataFrame:
    records = curve.merge(turnover, on="return_date", how="left").merge(daily_exposure, on="return_date", how="left")
    records["entry_date"] = records["return_date"]
    records["exit_date"] = records["return_date"]
    records["gross_return"] = records["gross_return"].astype(float)
    records["net_return"] = records["gross_return"]
    records["benchmark_return"] = records["benchmark_oto_return"].astype(float)
    records["turnover"] = records["h10_turnover"].astype(float).fillna(0.0)
    records["cost_bps"] = 0.0
    records["positions"] = records["h10_positions"].fillna(records["positions"]).fillna(0).astype(int)
    records["test_window_used"] = False
    keep = [
        "entry_date",
        "exit_date",
        "gross_return",
        "net_return",
        "benchmark_return",
        "turnover",
        "cost_bps",
        "positions",
        "long_gross_return",
        "short_gross_return",
        "active_sleeves",
        "gross_exposure",
        "long_gross",
        "short_gross",
        "net_exposure",
        "long_positions",
        "short_positions",
        "max_abs_weight",
        "top5_abs_share",
        "top10_abs_share",
        "hhi_abs_weight",
        "test_window_used",
    ]
    return records[keep].sort_values("entry_date").reset_index(drop=True)


def _build_summary(records: pd.DataFrame, *, curve: pd.DataFrame) -> pd.DataFrame:
    returns = records["net_return"].astype(float)
    benchmark = records["benchmark_return"].astype(float)
    metrics = _daily_path_metrics(returns, benchmark=benchmark)
    metrics.update(
        {
            "model": DEFAULT_MODEL_NAME,
            "start": str(records["entry_date"].min().date()),
            "end": str(records["entry_date"].max().date()),
            "sessions": int(len(records)),
            "mean_daily_return_bps": float(returns.mean() * 10000.0),
            "mean_turnover": float(records["turnover"].mean()),
            "mean_gross_exposure": float(records["gross_exposure"].mean()),
            "mean_long_gross": float(records["long_gross"].mean()),
            "mean_short_gross": float(records["short_gross"].mean()),
            "mean_positions": float(records["positions"].mean()),
            "rolling_10_positive_rate": _rolling_positive_rate(returns, 10),
            "rolling_60_positive_rate": _rolling_positive_rate(returns, 60),
            "rolling_252_positive_rate": _rolling_positive_rate(returns, 252),
            "active_sleeve_contract_ok": bool(curve["active_sleeve_contract_ok"].map(_is_true).all()),
            "max_active_sleeves": int(curve["active_sleeves"].max()),
            "test_window_used": bool(curve["test_window_used"].map(_is_true).any())
            if "test_window_used" in curve
            else False,
        }
    )
    return pd.DataFrame([metrics])


def _daily_path_metrics(returns: pd.Series, *, benchmark: pd.Series | None = None) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe_no_rf": np.nan,
            "max_drawdown": np.nan,
            "final_equity": np.nan,
            "corr_to_spy": np.nan,
            "realized_beta_to_spy": np.nan,
        }
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else np.nan
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    corr = np.nan
    beta = np.nan
    if benchmark is not None:
        bench = pd.to_numeric(benchmark, errors="coerce").reindex(returns.index)
        aligned = pd.concat([returns.rename("strategy"), bench.rename("benchmark")], axis=1).dropna()
        if len(aligned) > 2 and aligned["benchmark"].var(ddof=1) > 0:
            corr = float(aligned["strategy"].corr(aligned["benchmark"]))
            beta = float(aligned["strategy"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1))
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(equity.iloc[-1] ** (252.0 / len(returns)) - 1.0),
        "annualized_vol": vol,
        "sharpe_no_rf": sharpe,
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(equity.iloc[-1]),
        "corr_to_spy": corr,
        "realized_beta_to_spy": beta,
    }


def _rolling_positive_rate(returns: pd.Series, window: int) -> float:
    rolling = (1.0 + returns.astype(float)).rolling(window).apply(np.prod, raw=True) - 1.0
    valid = rolling.dropna()
    return float((valid > 0).mean()) if not valid.empty else np.nan


def _build_financing_borrow_stress(
    records: pd.DataFrame,
    *,
    transaction_cost_levels_bps: Sequence[float],
    borrow_cost_levels_bps_annual: Sequence[float],
    margin_cost_levels_bps_annual: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gross = records["gross_return"].astype(float)
    turnover = records["turnover"].astype(float)
    short_gross = records["short_gross"].astype(float).fillna(0.0)
    excess_gross = (records["gross_exposure"].astype(float).fillna(0.0) - 1.0).clip(lower=0.0)
    benchmark = records["benchmark_return"].astype(float)
    for tx_bps in transaction_cost_levels_bps:
        for borrow_bps in borrow_cost_levels_bps_annual:
            for margin_bps in margin_cost_levels_bps_annual:
                net = (
                    gross
                    - turnover * float(tx_bps) / 10000.0
                    - short_gross * float(borrow_bps) / 10000.0 / 252.0
                    - excess_gross * float(margin_bps) / 10000.0 / 252.0
                )
                metrics = _daily_path_metrics(net, benchmark=benchmark)
                metrics.update(
                    {
                        "transaction_cost_bps_per_traded_notional": float(tx_bps),
                        "borrow_cost_bps_annual_on_short_gross": float(borrow_bps),
                        "margin_cost_bps_annual_on_gross_above_100pct_nav": float(margin_bps),
                        "mean_daily_transaction_cost_bps": float((turnover * float(tx_bps)).mean()),
                        "mean_daily_borrow_cost_bps": float((short_gross * float(borrow_bps) / 252.0).mean()),
                        "mean_daily_margin_cost_bps": float((excess_gross * float(margin_bps) / 252.0).mean()),
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows).sort_values(
        [
            "transaction_cost_bps_per_traded_notional",
            "borrow_cost_bps_annual_on_short_gross",
            "margin_cost_bps_annual_on_gross_above_100pct_nav",
        ]
    )


def _build_market_state_exposure(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame["strategy_return"] = frame["net_return"].astype(float)
    frame["spy_return"] = frame["benchmark_return"].astype(float)
    bottom_decile = float(frame["spy_return"].quantile(0.10))
    top_decile = float(frame["spy_return"].quantile(0.90))
    abs_top_decile = float(frame["spy_return"].abs().quantile(0.90))
    states = [
        ("all", pd.Series(True, index=frame.index)),
        ("spy_up", frame["spy_return"] > 0),
        ("spy_down", frame["spy_return"] < 0),
        ("spy_top_decile", frame["spy_return"] >= top_decile),
        ("spy_bottom_decile", frame["spy_return"] <= bottom_decile),
        ("spy_abs_move_top_decile", frame["spy_return"].abs() >= abs_top_decile),
    ]
    rows: list[dict[str, Any]] = []
    for state, mask in states:
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        metrics = _daily_path_metrics(sub["strategy_return"], benchmark=sub["spy_return"])
        rows.append(
            {
                "state": state,
                "days": int(len(sub)),
                "strategy_mean_bps": float(sub["strategy_return"].mean() * 10000.0),
                "strategy_hit_rate": float((sub["strategy_return"] > 0).mean()),
                "spy_mean_bps": float(sub["spy_return"].mean() * 10000.0),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _build_rolling_beta(records: pd.DataFrame, *, windows: Sequence[int]) -> pd.DataFrame:
    frame = records[["entry_date", "net_return", "benchmark_return"]].copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["net_return"] = frame["net_return"].astype(float)
    frame["benchmark_return"] = frame["benchmark_return"].astype(float)
    rows: list[pd.DataFrame] = []
    for window in windows:
        cov = frame["net_return"].rolling(window).cov(frame["benchmark_return"])
        var = frame["benchmark_return"].rolling(window).var()
        corr = frame["net_return"].rolling(window).corr(frame["benchmark_return"])
        out = pd.DataFrame(
            {
                "entry_date": frame["entry_date"],
                "window": int(window),
                "rolling_beta": cov / var,
                "rolling_corr": corr,
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _build_rolling_beta_summary(rolling_beta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, group in rolling_beta.dropna(subset=["rolling_beta"]).groupby("window", sort=True):
        rows.append(
            {
                "window": int(window),
                "observations": int(len(group)),
                "mean_abs_rolling_beta": float(group["rolling_beta"].abs().mean()),
                "median_abs_rolling_beta": float(group["rolling_beta"].abs().median()),
                "p90_abs_rolling_beta": float(group["rolling_beta"].abs().quantile(0.90)),
                "max_abs_rolling_beta": float(group["rolling_beta"].abs().max()),
                "mean_abs_rolling_corr": float(group["rolling_corr"].abs().mean()),
                "p90_abs_rolling_corr": float(group["rolling_corr"].abs().quantile(0.90)),
                "max_abs_rolling_corr": float(group["rolling_corr"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _build_leg_attribution(*, curve: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "source": "strict_daily_path",
            "leg": "long_raw_daily",
            "mean_bps": float(curve["long_gross_return"].astype(float).mean() * 10000.0),
            "hit_rate": float((curve["long_gross_return"].astype(float) > 0).mean()),
        },
        {
            "source": "strict_daily_path",
            "leg": "short_raw_daily",
            "mean_bps": float(curve["short_gross_return"].astype(float).mean() * 10000.0),
            "hit_rate": float((curve["short_gross_return"].astype(float) > 0).mean()),
        },
        {
            "source": "strict_daily_path",
            "leg": "spread_raw_daily",
            "mean_bps": float(curve["gross_return"].astype(float).mean() * 10000.0),
            "hit_rate": float((curve["gross_return"].astype(float) > 0).mean()),
        },
    ]
    diagnostic_columns = [
        ("long_beta_residual", "long_beta_residual_h10"),
        ("short_beta_residual_contribution", "short_beta_residual_h10"),
        ("spread_beta_residual", "spread_beta_residual_h10"),
        ("long_style_factor_sic2_residual", "long_style_factor_sic2_residual_h10"),
        ("short_style_factor_sic2_residual_contribution", "short_style_factor_sic2_residual_h10"),
        ("spread_style_factor_sic2_residual", "spread_style_factor_sic2_residual_h10"),
    ]
    for column, label in diagnostic_columns:
        if column not in diagnostics.columns:
            continue
        series = diagnostics[column].astype(float)
        rows.append(
            {
                "source": "decision_date_h10_diagnostic",
                "leg": label,
                "mean_bps": float(series.mean() * 10000.0),
                "hit_rate": float((series > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_position_concentration_summary(daily_exposure: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "gross_exposure",
        "long_gross",
        "short_gross",
        "net_exposure",
        "positions",
        "long_positions",
        "short_positions",
        "max_abs_weight",
        "top5_abs_share",
        "top10_abs_share",
        "hhi_abs_weight",
    ]
    rows = []
    for column in columns:
        series = daily_exposure[column].astype(float)
        rows.append(
            {
                "metric": column,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p90": float(series.quantile(0.90)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def _build_sector_exposure_summary(aggregate_positions: pd.DataFrame) -> pd.DataFrame:
    sector = (
        aggregate_positions.groupby(["return_date", "sic2_sector"], as_index=False)["portfolio_weight"]
        .sum()
        .rename(columns={"portfolio_weight": "net_sector_weight"})
    )
    rows: list[dict[str, Any]] = []
    for return_date, group in sector.groupby("return_date", sort=True):
        abs_exposure = group["net_sector_weight"].abs()
        idx = abs_exposure.idxmax()
        rows.append(
            {
                "return_date": return_date,
                "sic2_l1_exposure": float(abs_exposure.sum()),
                "sic2_max_abs_exposure": float(abs_exposure.max()),
                "sic2_worst_sector": str(group.loc[idx, "sic2_sector"]),
                "sic2_worst_sector_weight": float(group.loc[idx, "net_sector_weight"]),
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "return_date": "SUMMARY",
        "sic2_l1_exposure": float(daily["sic2_l1_exposure"].mean()),
        "sic2_max_abs_exposure": float(daily["sic2_max_abs_exposure"].mean()),
        "sic2_worst_sector": str(daily.sort_values("sic2_max_abs_exposure", ascending=False).iloc[0]["sic2_worst_sector"]),
        "sic2_worst_sector_weight": float(
            daily.sort_values("sic2_max_abs_exposure", ascending=False).iloc[0]["sic2_worst_sector_weight"]
        ),
    }
    return pd.concat([pd.DataFrame([summary]), daily], ignore_index=True)


def _build_borrow_proxy_coverage(*, positions: pd.DataFrame, asset_qa_path: str | Path) -> pd.DataFrame:
    short_positions = positions[positions["signed_weight"].astype(float) < -1e-12].copy()
    qa_path = Path(asset_qa_path)
    if not qa_path.exists():
        return pd.DataFrame(
            [
                {
                    "scope": "target_short_rows",
                    "rows": int(len(short_positions)),
                    "unique_symbols": int(short_positions["symbol"].nunique()),
                    "current_shortable_coverage": np.nan,
                    "current_easy_to_borrow_coverage": np.nan,
                    "missing_asset_qa_symbols": int(short_positions["symbol"].nunique()),
                    "data_status": "asset_qa_missing",
                }
            ]
        )
    qa = pd.read_csv(qa_path)
    qa = qa.drop_duplicates("symbol")
    merged = short_positions.merge(qa[["symbol", "shortable", "easy_to_borrow"]], on="symbol", how="left")
    symbol_coverage = merged.drop_duplicates("symbol")
    return pd.DataFrame(
        [
            {
                "scope": "target_short_rows",
                "rows": int(len(short_positions)),
                "unique_symbols": int(short_positions["symbol"].nunique()),
                "current_shortable_coverage": float(merged["shortable"].map(_is_true).mean()),
                "current_easy_to_borrow_coverage": float(merged["easy_to_borrow"].map(_is_true).mean()),
                "missing_asset_qa_symbols": int(merged["shortable"].isna().groupby(merged["symbol"]).max().sum()),
                "data_status": "current_metadata_proxy_not_point_in_time",
            },
            {
                "scope": "unique_short_symbols",
                "rows": int(len(symbol_coverage)),
                "unique_symbols": int(symbol_coverage["symbol"].nunique()),
                "current_shortable_coverage": float(symbol_coverage["shortable"].map(_is_true).mean()),
                "current_easy_to_borrow_coverage": float(symbol_coverage["easy_to_borrow"].map(_is_true).mean()),
                "missing_asset_qa_symbols": int(symbol_coverage["shortable"].isna().sum()),
                "data_status": "current_metadata_proxy_not_point_in_time",
            },
        ]
    )


def _build_gate_check(
    *,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    tail: pd.DataFrame,
    cost: pd.DataFrame,
    market_state: pd.DataFrame,
    rolling_beta_summary: pd.DataFrame,
    leg_attribution: pd.DataFrame,
    concentration: pd.DataFrame,
    borrow_proxy: pd.DataFrame,
) -> pd.DataFrame:
    s = summary.iloc[0]
    yearly_positive_rate = float((yearly["total_return"].astype(float) > 0).mean()) if not yearly.empty else np.nan
    top5 = tail[(tail["tail_side"].eq("top")) & (tail["trim_count"].eq(5))]
    top5_share = float(top5["removed_return_share_of_abs_sum"].iloc[0]) if not top5.empty else np.nan
    cost4 = cost[cost["cost_bps_per_side"].astype(float).eq(4.0)]
    cost4_ann = float(cost4["annualized_return"].iloc[0]) if not cost4.empty else np.nan
    full_state = market_state[market_state["state"].eq("all")]
    realized_beta = float(full_state["realized_beta_to_spy"].iloc[0]) if not full_state.empty else float(s["realized_beta_to_spy"])
    realized_corr = float(full_state["corr_to_spy"].iloc[0]) if not full_state.empty else float(s["corr_to_spy"])
    rb60 = rolling_beta_summary[rolling_beta_summary["window"].eq(60)]
    rb60_p90 = float(rb60["p90_abs_rolling_beta"].iloc[0]) if not rb60.empty else np.nan
    residual_legs = {
        row["leg"]: float(row["mean_bps"])
        for row in leg_attribution.to_dict(orient="records")
        if str(row["source"]) == "decision_date_h10_diagnostic"
    }
    borrow_rows = borrow_proxy[borrow_proxy["scope"].eq("target_short_rows")]
    etb = float(borrow_rows["current_easy_to_borrow_coverage"].iloc[0]) if not borrow_rows.empty else np.nan
    rows = [
        _gate("validation_annualized_return_gross", float(s["annualized_return"]), 0.08, ">=", "fail"),
        _gate("validation_sharpe_gross", float(s["sharpe_no_rf"]), 0.80, ">=", "fail"),
        _gate("validation_max_drawdown", float(s["max_drawdown"]), -0.15, ">=", "fail"),
        _gate("absolute_realized_beta_full_sample", abs(realized_beta), 0.05, "<=", "fail"),
        _gate("absolute_market_correlation_full_sample", abs(realized_corr), 0.10, "<=", "fail"),
        _gate("rolling_60_positive_rate", float(s["rolling_60_positive_rate"]), 0.50, ">=", "fail"),
        _gate("positive_year_ratio", yearly_positive_rate, 0.50, ">=", "fail"),
        _gate("top5_positive_day_abs_share", top5_share, 0.50, "<=", "fail"),
        _gate("cost_4bps_annualized_return", cost4_ann, 0.05, ">=", "fail"),
        _gate("p90_abs_rolling_60_beta", rb60_p90, 0.20, "<=", "warn"),
        _gate(
            "long_style_factor_sic2_residual_mean_bps",
            residual_legs.get("long_style_factor_sic2_residual_h10", np.nan),
            0.0,
            ">=",
            "warn",
        ),
        _gate(
            "short_style_factor_sic2_residual_mean_bps",
            residual_legs.get("short_style_factor_sic2_residual_h10", np.nan),
            0.0,
            ">=",
            "warn",
        ),
        _gate("current_etb_proxy_short_row_coverage", etb, 1.0, ">=", "warn"),
        {
            "gate": "point_in_time_borrow_history",
            "value": "missing",
            "threshold": "required_before_product_freeze",
            "operator": "exists",
            "status": "WARN",
            "severity": "blocker_for_production_not_for_validation_research",
        },
        {
            "gate": "survivorship_free_full_market_universe",
            "value": "not_final",
            "threshold": "required_before_product_freeze",
            "operator": "exists",
            "status": "WARN",
            "severity": "blocker_for_production_not_for_validation_research",
        },
    ]
    return pd.DataFrame(rows)


def _gate(name: str, value: float, threshold: float, operator: str, severity_if_fail: str) -> dict[str, Any]:
    if pd.isna(value):
        status = "WARN"
    elif operator == ">=":
        status = "PASS" if float(value) >= float(threshold) else severity_if_fail.upper()
    elif operator == "<=":
        status = "PASS" if float(value) <= float(threshold) else severity_if_fail.upper()
    else:
        raise ValueError(f"Unsupported gate operator: {operator}")
    return {
        "gate": name,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "status": status,
        "severity": severity_if_fail,
    }


def _plot_robustness(*, records: pd.DataFrame, rolling_beta: pd.DataFrame, output_path: Path) -> None:
    frame = records.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["gross_equity"] = (1.0 + frame["gross_return"].astype(float)).cumprod()
    frame["cost4_equity"] = (
        1.0 + frame["gross_return"].astype(float) - frame["turnover"].astype(float) * 4.0 / 10000.0
    ).cumprod()
    frame["spy_equity"] = (1.0 + frame["benchmark_return"].astype(float)).cumprod()
    frame["drawdown"] = frame["gross_equity"] / frame["gross_equity"].cummax() - 1.0
    beta60 = rolling_beta[rolling_beta["window"].eq(60)].copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(frame["entry_date"], frame["gross_equity"], label="SOTA gross", color="#0f766e", linewidth=2.0)
    axes[0].plot(frame["entry_date"], frame["cost4_equity"], label="SOTA less 4bps turnover cost", color="#2563eb")
    axes[0].plot(frame["entry_date"], frame["spy_equity"], label="SPY raw", color="#6b7280", alpha=0.8)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left")
    axes[1].plot(frame["entry_date"], frame["drawdown"] * 100.0, color="#b45309")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Drawdown %")
    axes[2].plot(beta60["entry_date"], beta60["rolling_beta"], color="#7c3aed", label="60d realized beta")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.05, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axhline(-0.05, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Rolling beta")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    fig.suptitle("Phase6 SOTA Robustness Packet (Validation Only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    tail: pd.DataFrame,
    cost: pd.DataFrame,
    financing: pd.DataFrame,
    market_state: pd.DataFrame,
    rolling_beta_summary: pd.DataFrame,
    leg_attribution: pd.DataFrame,
    concentration: pd.DataFrame,
    sector: pd.DataFrame,
    borrow_proxy: pd.DataFrame,
    gate: pd.DataFrame,
    plot_path: Path,
) -> str:
    s = summary.copy()
    for column in ("annualized_return", "annualized_vol", "max_drawdown", "rolling_60_positive_rate", "corr_to_spy"):
        if column in s:
            s[column] = s[column].map(_fmt_pct_like if column != "corr_to_spy" else _fmt_float_like)
    for column in ("sharpe_no_rf", "mean_daily_return_bps", "mean_turnover", "mean_gross_exposure", "mean_positions"):
        if column in s:
            s[column] = s[column].map(_fmt_float_like)

    y = yearly[["period_label", "sessions", "total_return", "annualized_return", "sharpe", "max_drawdown"]].copy()
    for column in ("total_return", "annualized_return", "max_drawdown"):
        y[column] = y[column].map(_fmt_pct_like)
    y["sharpe"] = y["sharpe"].map(_fmt_float_like)

    c = cost[["cost_bps_per_side", "annualized_return", "sharpe", "max_drawdown", "mean_cost_bps"]].copy()
    for column in ("annualized_return", "max_drawdown"):
        c[column] = c[column].map(_fmt_pct_like)
    for column in ("sharpe", "mean_cost_bps"):
        c[column] = c[column].map(_fmt_float_like)

    f = financing[
        [
            "transaction_cost_bps_per_traded_notional",
            "borrow_cost_bps_annual_on_short_gross",
            "margin_cost_bps_annual_on_gross_above_100pct_nav",
            "annualized_return",
            "sharpe_no_rf",
            "max_drawdown",
        ]
    ].copy()
    f = f[
        (f["transaction_cost_bps_per_traded_notional"].isin([0.0, 4.0]))
        & (f["borrow_cost_bps_annual_on_short_gross"].isin([0.0, 200.0]))
        & (f["margin_cost_bps_annual_on_gross_above_100pct_nav"].isin([0.0, 300.0]))
    ]
    for column in ("annualized_return", "max_drawdown"):
        f[column] = f[column].map(_fmt_pct_like)
    f["sharpe_no_rf"] = f["sharpe_no_rf"].map(_fmt_float_like)

    m = market_state[["state", "days", "strategy_mean_bps", "strategy_hit_rate", "realized_beta_to_spy", "corr_to_spy"]].copy()
    m["strategy_mean_bps"] = m["strategy_mean_bps"].map(_fmt_float_like)
    m["strategy_hit_rate"] = m["strategy_hit_rate"].map(_fmt_pct_like)
    m["realized_beta_to_spy"] = m["realized_beta_to_spy"].map(_fmt_float_like)
    m["corr_to_spy"] = m["corr_to_spy"].map(_fmt_float_like)

    rb = rolling_beta_summary.copy()
    for column in rb.columns:
        if column != "window":
            rb[column] = rb[column].map(_fmt_float_like)

    leg = leg_attribution.copy()
    leg["mean_bps"] = leg["mean_bps"].map(_fmt_float_like)
    leg["hit_rate"] = leg["hit_rate"].map(_fmt_pct_like)

    conc = concentration.copy()
    for column in ("mean", "median", "p90", "max"):
        conc[column] = conc[column].map(_fmt_float_like)

    sector_summary = sector[sector["return_date"].astype(str).eq("SUMMARY")].copy()
    for column in ("sic2_l1_exposure", "sic2_max_abs_exposure", "sic2_worst_sector_weight"):
        sector_summary[column] = sector_summary[column].map(_fmt_float_like)

    borrow = borrow_proxy.copy()
    for column in ("current_shortable_coverage", "current_easy_to_borrow_coverage"):
        borrow[column] = borrow[column].map(_fmt_pct_like)

    gates = gate.copy()
    lines = [
        "# Phase6 SOTA Robustness Packet",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "- Candidate: current SOTA / Phase5F `short_overlay_only`.",
        "- Window: validation only; no test-window performance is used.",
        "- PnL semantics: rolling h10 daily multi-sleeve path, annualized as daily returns.",
        "- Costs: gross/no-cost baseline plus scenario haircuts.",
        "",
        "## Headline",
        "",
        _text_table(s),
        "",
        "## Yearly Stability",
        "",
        _text_table(y),
        "",
        "## Transaction Cost Stress",
        "",
        _text_table(c),
        "",
        "## Financing And Borrow Stress Samples",
        "",
        _text_table(f),
        "",
        "## Market-State Exposure",
        "",
        _text_table(m),
        "",
        "## Rolling Realized Beta Summary",
        "",
        _text_table(rb),
        "",
        "## Leg Attribution",
        "",
        _text_table(leg),
        "",
        "## Position Concentration",
        "",
        _text_table(conc),
        "",
        "## SIC2 Sector Exposure Summary",
        "",
        _text_table(sector_summary),
        "",
        "## Borrow Proxy Coverage",
        "",
        _text_table(borrow),
        "",
        "## Gate Check",
        "",
        _text_table(gates),
        "",
        "## Plot",
        "",
        f"![Phase6 robustness plot]({plot_path.as_posix()})",
        "",
        "## First Reading",
        "",
        "- The first Phase6 packet is a challenge report, not a new strategy version.",
        "- The gross economics remain strong, but the full-sample realized beta gate is slightly above the 0.05 target and the rolling 60-day beta tail needs follow-up.",
        "- Production freeze remains blocked by point-in-time borrow history and a final survivorship-free universe, even before considering the beta follow-up.",
        "- If the beta and data blockers are accepted or mitigated, the next step is a formal Phase7 freeze memo rather than more validation tuning.",
    ]
    return "\n".join(lines) + "\n"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase6 robustness packet for pure-alpha SOTA.")
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS_PATH))
    parser.add_argument("--daily-diagnostics-path", default=str(DEFAULT_DAILY_DIAGNOSTICS_PATH))
    parser.add_argument("--strict-curve-path", default=str(DEFAULT_STRICT_CURVE_PATH))
    parser.add_argument("--asset-qa-path", default=str(DEFAULT_ASSET_QA_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    args = parser.parse_args(argv)

    result = build_phase6_sota_robustness_artifacts(
        positions_path=args.positions_path,
        daily_diagnostics_path=args.daily_diagnostics_path,
        strict_curve_path=args.strict_curve_path,
        asset_qa_path=args.asset_qa_path,
        output_root=args.output_root,
        portfolio=args.portfolio,
        model_name=args.model_name,
        holding_period_sessions=args.holding_period_sessions,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
