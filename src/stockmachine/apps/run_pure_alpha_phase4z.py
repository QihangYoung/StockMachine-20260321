from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import DEFAULT_OUTPUT_ROOT as PHASE4S_OUTPUT_ROOT


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4z_strict_h10_daily_paths_20260419"
DEFAULT_PHASE3_H10_ROLLUP = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_h10_probe_20260418"
    / "phase3_baseline_signal_rollup.json"
)
DEFAULT_POSITIONS = PHASE4S_OUTPUT_ROOT / "phase4s_positions_validation.csv.gz"
DEFAULT_EXISTING_CURVE = (
    PHASE4S_OUTPUT_ROOT
    / "daily_multisleeve_gross"
    / "phase4s_daily_multisleeve_gross_curve.csv"
)
DEFAULT_HOLDING_PERIOD_SESSIONS = 10
DEFAULT_VALIDATION_PRICE_END = "2019-12-31"
DEFAULT_LEAD_PORTFOLIO = "sic2_soft_neutral"


def build_phase4z_strict_horizon_daily_paths(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    existing_curve_path: str | Path | None = DEFAULT_EXISTING_CURVE,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    benchmark_symbol: str = "SPY",
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    lead_portfolio: str = DEFAULT_LEAD_PORTFOLIO,
) -> dict[str, Any]:
    """Rebuild strict horizon-aware daily paths from Phase4S position sleeves."""

    _validate_settings(holding_period_sessions=holding_period_sessions)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase3_contract = _load_phase3_contract(phase3_rollup_path)
    phase3_horizon = phase3_contract.get("holding_period_sessions")
    if phase3_horizon != holding_period_sessions:
        raise ValueError(
            f"Phase3 rollup horizon {phase3_horizon} does not match requested "
            f"horizon {holding_period_sessions}."
        )
    positions = _load_positions(positions_path)
    symbols = tuple(sorted(positions["symbol"].astype(str).unique()))

    stock_returns = _load_open_to_open_returns(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=validation_price_end,
    )
    benchmark_returns = _load_open_to_open_returns(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=validation_price_end,
    ).rename(columns={"oto_return": "benchmark_oto_return"})
    benchmark_calendar = (
        benchmark_returns[benchmark_returns["symbol"].eq(benchmark_symbol)]["session_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    active_map = _active_return_dates(
        positions,
        calendar=benchmark_calendar,
        holding_period_sessions=holding_period_sessions,
    )
    sleeve_returns, daily = _strict_daily_returns(
        positions,
        active_map=active_map,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        holding_period_sessions=holding_period_sessions,
    )
    curve = _add_path_metrics(daily, lead_portfolio=lead_portfolio)
    metrics = _portfolio_metrics(curve, holding_period_sessions=holding_period_sessions)
    comparison = _compare_existing_curve(
        curve,
        existing_curve_path=existing_curve_path,
        holding_period_sessions=holding_period_sessions,
    )
    audit = _active_sleeve_audit(curve, holding_period_sessions=holding_period_sessions)

    curve_path = output_dir / "phase4z_strict_h10_daily_curve.csv"
    sleeve_path = output_dir / "phase4z_strict_h10_sleeve_returns.csv.gz"
    metrics_path = output_dir / "phase4z_strict_h10_metrics.csv"
    comparison_path = output_dir / "phase4z_vs_existing_daily_curve_comparison.csv"
    audit_path = output_dir / "phase4z_horizon_active_sleeve_audit.csv"
    memo_path = output_dir / "phase4z_strict_h10_daily_paths_memo.md"
    plot_path = output_dir / "phase4z_strict_h10_daily_paths.png"
    rollup_path = output_dir / "phase4z_strict_h10_rollup.json"

    curve.to_csv(curve_path, index=False)
    sleeve_returns.to_csv(sleeve_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    audit.to_csv(audit_path, index=False)
    _plot_paths(curve, output_path=plot_path, lead_portfolio=lead_portfolio)
    memo_path.write_text(
        _memo(
            phase3_contract=phase3_contract,
            metrics=metrics,
            comparison=comparison,
            audit=audit,
            positions_path=Path(positions_path),
            phase3_rollup_path=Path(phase3_rollup_path),
            holding_period_sessions=holding_period_sessions,
            lead_portfolio=lead_portfolio,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "positions_path": Path(positions_path).as_posix(),
        "phase3_rollup_path": Path(phase3_rollup_path).as_posix(),
        "existing_curve_path": Path(existing_curve_path).as_posix()
        if existing_curve_path
        else None,
        "benchmark_symbol": benchmark_symbol,
        "holding_period_sessions": int(holding_period_sessions),
        "validation_price_end": validation_price_end,
        "phase3_holding_period_sessions": phase3_contract.get("holding_period_sessions"),
        "label_contract": {
            "decision_session": "T",
            "entry": "adjusted open at T+1",
            "daily_pnl_dates": "open-to-open returns from T+2 through T+holding+1",
            "exit": f"adjusted open after {holding_period_sessions} holding sessions",
        },
        "positions_rows": int(len(positions)),
        "portfolios": sorted(positions["portfolio"].astype(str).unique()),
        "stock_return_rows": int(len(stock_returns)),
        "curve_rows": int(len(curve)),
        "sleeve_return_rows": int(len(sleeve_returns)),
        "max_active_sleeves": int(curve["active_sleeves"].max()) if not curve.empty else 0,
        "active_sleeve_contract_ok": bool(
            (curve["active_sleeves"] <= holding_period_sessions).all()
        )
        if not curve.empty
        else True,
        "curve_artifact": curve_path.as_posix(),
        "sleeve_returns_artifact": sleeve_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "comparison_artifact": comparison_path.as_posix(),
        "audit_artifact": audit_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "method": "strict_horizon_open_to_open_daily_multisleeve_rebuild",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "Gross/no-cost path only; transaction costs and borrow costs are not charged.",
            "Uses Phase4S validation positions and adjusted open-to-open returns.",
            "This rebuild is for path and risk diagnostics; it does not change selector or portfolio construction.",
            "Legacy Phase3 label columns may still be named *_5d; this run enforces the h10 contract via rollup metadata and active-sleeve checks.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(*, holding_period_sessions: int) -> None:
    if holding_period_sessions <= 0:
        raise ValueError("holding_period_sessions must be positive.")


def _load_phase3_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "artifact_dir": payload.get("artifact_dir"),
        "holding_period_sessions": payload.get("holding_period_sessions"),
        "primary_target": payload.get("primary_target"),
        "validation_start": payload.get("validation_start"),
        "validation_end": payload.get("validation_end"),
        "signal_panel_artifact": payload.get("signal_panel_artifact"),
    }


def _load_positions(path: str | Path) -> pd.DataFrame:
    usecols = [
        "session_date",
        "portfolio",
        "side",
        "symbol",
        "signed_weight",
        "test_window_used",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame["side"] = frame["side"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["signed_weight"] = pd.to_numeric(frame["signed_weight"], errors="coerce")
    frame["test_window_used"] = frame["test_window_used"].map(_is_true)
    frame = frame.dropna(subset=["session_date", "symbol", "signed_weight"])
    if frame.empty:
        raise ValueError(f"No positions found in {path}.")
    if frame["test_window_used"].any():
        raise ValueError("Positions include test-window rows; refusing to build validation path.")
    return frame.sort_values(["portfolio", "session_date", "side", "symbol"]).reset_index(
        drop=True
    )


def _load_open_to_open_returns(
    *,
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    symbols: Sequence[str],
    end_date: str,
) -> pd.DataFrame:
    prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=end_date,
    )
    prices["session_date"] = pd.to_datetime(prices["session_date"])
    prices = prices.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    prices["oto_return"] = prices.groupby("symbol")["adjusted_open"].pct_change()
    return prices[["session_date", "symbol", "oto_return"]].dropna().reset_index(drop=True)


def _active_return_dates(
    positions: pd.DataFrame,
    *,
    calendar: pd.Series,
    holding_period_sessions: int,
) -> pd.DataFrame:
    calendar_values = pd.to_datetime(calendar).drop_duplicates().sort_values().reset_index(drop=True)
    calendar_index = {date: idx for idx, date in enumerate(calendar_values)}
    rows: list[dict[str, Any]] = []
    signal_dates = positions["session_date"].drop_duplicates().sort_values()
    for signal_date in signal_dates:
        if signal_date not in calendar_index:
            continue
        start = calendar_index[signal_date] + 2
        stop = start + holding_period_sessions
        if stop > len(calendar_values):
            continue
        for sleeve_age, return_date in enumerate(calendar_values.iloc[start:stop], start=1):
            rows.append(
                {
                    "session_date": signal_date,
                    "return_date": return_date,
                    "sleeve_age": int(sleeve_age),
                }
            )
    if not rows:
        raise ValueError("No active return dates could be mapped from positions.")
    return pd.DataFrame(rows)


def _strict_daily_returns(
    positions: pd.DataFrame,
    *,
    active_map: pd.DataFrame,
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    holding_period_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = positions.merge(active_map, on="session_date", how="inner")
    active = active.rename(columns={"session_date": "sleeve_session_date"})
    active = active.merge(
        stock_returns.rename(columns={"session_date": "return_date"}),
        on=["return_date", "symbol"],
        how="left",
    )
    active["missing_return"] = active["oto_return"].isna()
    active["weighted_return"] = active["signed_weight"] * active["oto_return"].fillna(0.0)

    sleeve_key = ["portfolio", "return_date", "sleeve_session_date"]
    sleeve_returns = (
        active.groupby(sleeve_key, sort=True)
        .agg(
            sleeve_gross_return=("weighted_return", "sum"),
            missing_positions=("missing_return", "sum"),
            positions=("symbol", "count"),
            sleeve_age=("sleeve_age", "first"),
            test_window_used=("test_window_used", "any"),
        )
        .reset_index()
    )
    if sleeve_returns["test_window_used"].any():
        raise ValueError("Encountered test-window sleeve rows; refusing to continue.")
    sleeve_returns["complete_sleeve"] = sleeve_returns["missing_positions"].eq(0)

    complete_keys = sleeve_returns.loc[sleeve_returns["complete_sleeve"], sleeve_key]
    active_complete = active.merge(complete_keys, on=sleeve_key, how="inner")

    sleeve_side = (
        active_complete.groupby([*sleeve_key, "side"], sort=True)["weighted_return"]
        .sum()
        .unstack("side")
        .reset_index()
    )
    for side in ("long", "short"):
        if side not in sleeve_side.columns:
            sleeve_side[side] = 0.0
    sleeve_side = sleeve_side.rename(
        columns={"long": "sleeve_long_return", "short": "sleeve_short_return"}
    )
    sleeve_returns = sleeve_returns.merge(sleeve_side, on=sleeve_key, how="left")
    for column in ("sleeve_long_return", "sleeve_short_return"):
        sleeve_returns[column] = sleeve_returns[column].fillna(0.0)

    complete = sleeve_returns[sleeve_returns["complete_sleeve"]].copy()
    daily = (
        complete.groupby(["portfolio", "return_date"], sort=True)
        .agg(
            gross_return=("sleeve_gross_return", "mean"),
            long_gross_return=("sleeve_long_return", "mean"),
            short_gross_return=("sleeve_short_return", "mean"),
            active_sleeves=("sleeve_session_date", "nunique"),
        )
        .reset_index()
    )
    requested = (
        sleeve_returns.groupby(["portfolio", "return_date"], sort=True)
        .agg(requested_active_sleeves=("sleeve_session_date", "nunique"))
        .reset_index()
    )
    daily = daily.merge(requested, on=["portfolio", "return_date"], how="left")
    daily["dropped_sleeves_missing_prices"] = (
        daily["requested_active_sleeves"] - daily["active_sleeves"]
    )
    daily = daily.merge(
        benchmark_returns[["session_date", "benchmark_oto_return"]]
        .drop_duplicates("session_date")
        .rename(columns={"session_date": "return_date"}),
        on="return_date",
        how="left",
    )
    daily["holding_period_sessions"] = int(holding_period_sessions)
    daily["active_sleeve_contract_ok"] = daily["active_sleeves"] <= holding_period_sessions
    daily["test_window_used"] = False
    return sleeve_returns, daily.sort_values(["portfolio", "return_date"]).reset_index(drop=True)


def _add_path_metrics(daily: pd.DataFrame, *, lead_portfolio: str) -> pd.DataFrame:
    frames = []
    lead_final = None
    for portfolio, group in daily.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date").copy()
        frame["equity"] = (1.0 + frame["gross_return"]).cumprod()
        frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
        frame["rolling_60_return"] = frame["equity"] / frame["equity"].shift(60) - 1.0
        frame["rolling_252_return"] = frame["equity"] / frame["equity"].shift(252) - 1.0
        frame["benchmark_equity"] = (1.0 + frame["benchmark_oto_return"].fillna(0.0)).cumprod()
        frame["benchmark_drawdown"] = (
            frame["benchmark_equity"] / frame["benchmark_equity"].cummax() - 1.0
        )
        if portfolio == lead_portfolio and not frame.empty:
            lead_final = float(frame["equity"].iloc[-1])
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else daily
    if lead_final and lead_final > 0:
        result = _add_endpoint_scaled_benchmark(result, target_final_equity=lead_final)
    else:
        result["benchmark_endpoint_scaled_equity"] = np.nan
        result["benchmark_endpoint_scaled_drawdown"] = np.nan
    return result


def _add_endpoint_scaled_benchmark(
    curve: pd.DataFrame,
    *,
    target_final_equity: float,
) -> pd.DataFrame:
    frames = []
    for _, group in curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date").copy()
        benchmark_returns = frame["benchmark_oto_return"].fillna(0.0)
        log_path = np.log1p(benchmark_returns).cumsum()
        endpoint = float(log_path.iloc[-1]) if len(log_path) else 0.0
        scale = np.log(target_final_equity) / endpoint if abs(endpoint) > 1e-12 else 0.0
        frame["benchmark_endpoint_scaled_equity"] = np.exp(log_path * scale)
        frame["benchmark_endpoint_scaled_drawdown"] = (
            frame["benchmark_endpoint_scaled_equity"]
            / frame["benchmark_endpoint_scaled_equity"].cummax()
            - 1.0
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else curve


def _portfolio_metrics(
    curve: pd.DataFrame,
    *,
    holding_period_sessions: int,
) -> pd.DataFrame:
    rows = []
    for portfolio, group in curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date")
        returns = frame["gross_return"].dropna()
        if returns.empty:
            continue
        equity = (1.0 + returns).cumprod()
        annualized_return = float(equity.iloc[-1] ** (252.0 / len(returns)) - 1.0)
        annualized_vol = float(returns.std(ddof=1) * np.sqrt(252.0))
        rows.append(
            {
                "portfolio": portfolio,
                "holding_period_sessions": int(holding_period_sessions),
                "start": str(frame["return_date"].min().date()),
                "end": str(frame["return_date"].max().date()),
                "daily_rows": int(len(returns)),
                "final_equity": float(equity.iloc[-1]),
                "annualized_return": annualized_return,
                "annualized_vol": annualized_vol,
                "sharpe_no_rf": annualized_return / annualized_vol
                if annualized_vol > 0
                else np.nan,
                "max_drawdown": float(frame["drawdown"].min()),
                "rolling_60_positive_rate": float((frame["rolling_60_return"] > 0).mean()),
                "rolling_252_positive_rate": float((frame["rolling_252_return"] > 0).mean()),
                "mean_active_sleeves": float(frame["active_sleeves"].mean()),
                "max_active_sleeves": int(frame["active_sleeves"].max()),
                "active_sleeve_contract_ok": bool(
                    (frame["active_sleeves"] <= holding_period_sessions).all()
                ),
                "mean_dropped_sleeves_missing_prices": float(
                    frame["dropped_sleeves_missing_prices"].mean()
                ),
                "test_window_used": bool(frame["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _compare_existing_curve(
    curve: pd.DataFrame,
    *,
    existing_curve_path: str | Path | None,
    holding_period_sessions: int,
) -> pd.DataFrame:
    if not existing_curve_path or not Path(existing_curve_path).exists():
        return pd.DataFrame(columns=_comparison_columns())
    existing = pd.read_csv(existing_curve_path)
    existing["session_date"] = pd.to_datetime(existing["session_date"])
    current = curve.copy()
    current["return_date"] = pd.to_datetime(current["return_date"])
    existing_columns = ["portfolio", "session_date", "gross_return", "active_sleeves", "equity"]
    merged = current.merge(
        existing[existing_columns],
        left_on=["portfolio", "return_date"],
        right_on=["portfolio", "session_date"],
        how="left",
        suffixes=("_strict", "_existing"),
    )
    if "gross_return_existing" not in merged.columns:
        return pd.DataFrame(columns=_comparison_columns())
    merged["gross_return_diff"] = merged["gross_return_strict"] - merged["gross_return_existing"]
    merged["active_sleeves_diff"] = (
        merged["active_sleeves_strict"] - merged["active_sleeves_existing"]
    )
    rows = []
    for portfolio, group in merged.groupby("portfolio", sort=True):
        diff = group["gross_return_diff"].dropna()
        active_diff = group["active_sleeves_diff"].dropna()
        strict_returns = group["gross_return_strict"].dropna()
        existing_returns = group["gross_return_existing"].dropna()
        rows.append(
            {
                "portfolio": portfolio,
                "holding_period_sessions": int(holding_period_sessions),
                "matched_days": int(len(diff)),
                "mean_gross_return_diff_bps": float(diff.mean() * 10000.0)
                if len(diff)
                else np.nan,
                "std_gross_return_diff_bps": float(diff.std(ddof=1) * 10000.0)
                if len(diff) > 1
                else np.nan,
                "max_abs_gross_return_diff_bps": float(diff.abs().max() * 10000.0)
                if len(diff)
                else np.nan,
                "active_sleeve_mismatch_days": int((active_diff != 0).sum()),
                "existing_max_active_sleeves": int(group["active_sleeves_existing"].max())
                if group["active_sleeves_existing"].notna().any()
                else 0,
                "strict_max_active_sleeves": int(group["active_sleeves_strict"].max())
                if group["active_sleeves_strict"].notna().any()
                else 0,
                "existing_final_equity": _final_equity(existing_returns),
                "strict_final_equity": _final_equity(strict_returns),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_comparison_columns())


def _comparison_columns() -> list[str]:
    return [
        "portfolio",
        "holding_period_sessions",
        "matched_days",
        "mean_gross_return_diff_bps",
        "std_gross_return_diff_bps",
        "max_abs_gross_return_diff_bps",
        "active_sleeve_mismatch_days",
        "existing_max_active_sleeves",
        "strict_max_active_sleeves",
        "existing_final_equity",
        "strict_final_equity",
        "test_window_used",
    ]


def _final_equity(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).cumprod().iloc[-1])


def _active_sleeve_audit(
    curve: pd.DataFrame,
    *,
    holding_period_sessions: int,
) -> pd.DataFrame:
    rows = []
    for portfolio, group in curve.groupby("portfolio", sort=True):
        rows.append(
            {
                "portfolio": portfolio,
                "holding_period_sessions": int(holding_period_sessions),
                "daily_rows": int(len(group)),
                "mean_active_sleeves": float(group["active_sleeves"].mean()),
                "min_active_sleeves": int(group["active_sleeves"].min()),
                "max_active_sleeves": int(group["active_sleeves"].max()),
                "days_above_horizon": int(
                    (group["active_sleeves"] > holding_period_sessions).sum()
                ),
                "max_requested_active_sleeves": int(group["requested_active_sleeves"].max()),
                "days_with_missing_price_drops": int(
                    (group["dropped_sleeves_missing_prices"] > 0).sum()
                ),
                "active_sleeve_contract_ok": bool(
                    (group["active_sleeves"] <= holding_period_sessions).all()
                ),
                "test_window_used": bool(group["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _plot_paths(curve: pd.DataFrame, *, output_path: Path, lead_portfolio: str) -> None:
    if curve.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for portfolio, group in curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date")
        linewidth = 2.2 if portfolio == lead_portfolio else 1.2
        axes[0].plot(frame["return_date"], frame["equity"], label=portfolio, linewidth=linewidth)
        axes[1].plot(frame["return_date"], frame["drawdown"] * 100.0, label=portfolio, linewidth=linewidth)
    lead = curve[curve["portfolio"].eq(lead_portfolio)].sort_values("return_date")
    if not lead.empty and "benchmark_endpoint_scaled_equity" in lead:
        axes[0].plot(
            lead["return_date"],
            lead["benchmark_endpoint_scaled_equity"],
            label="SPY endpoint-scaled",
            linewidth=1.6,
            linestyle="--",
            color="black",
        )
        axes[1].plot(
            lead["return_date"],
            lead["benchmark_endpoint_scaled_drawdown"] * 100.0,
            label="SPY endpoint-scaled",
            linewidth=1.2,
            linestyle="--",
            color="black",
        )
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Phase4Z Strict h10 Gross Daily Multi-Sleeve Paths")
    axes[0].set_ylabel("Growth of $1")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    phase3_contract: dict[str, Any],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    audit: pd.DataFrame,
    positions_path: Path,
    phase3_rollup_path: Path,
    holding_period_sessions: int,
    lead_portfolio: str,
) -> str:
    lines = [
        "# Phase4Z Strict h10 Daily Path Rebuild",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- positions: `{positions_path.as_posix()}`;",
        f"- Phase3 rollup: `{phase3_rollup_path.as_posix()}`;",
        f"- enforced holding period: `{holding_period_sessions}` sessions;",
        f"- Phase3 rollup holding period: `{phase3_contract.get('holding_period_sessions')}` sessions;",
        "- entry: adjusted open at T+1;",
        "- daily PnL: open-to-open returns from T+2 through T+holding+1;",
        "- cost: gross/no-cost;",
        "- lockbox/test window: not used.",
        "",
        "## Strict Metrics",
        "",
        "| portfolio | final equity | ann return | ann vol | sharpe | max DD | max active sleeves | contract ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.sort_values("portfolio").iterrows():
        lines.append(
            f"| {row['portfolio']} | {_fmt_float(row['final_equity'])} | "
            f"{_fmt_pct(row['annualized_return'])} | {_fmt_pct(row['annualized_vol'])} | "
            f"{_fmt_float(row['sharpe_no_rf'])} | {_fmt_pct(row['max_drawdown'])} | "
            f"{int(row['max_active_sleeves'])} | {bool(row['active_sleeve_contract_ok'])} |"
        )
    if not comparison.empty:
        lines.extend(
            [
                "",
                "## Difference Versus Existing Phase4S Daily Curve",
                "",
                "| portfolio | existing final | strict final | active mismatch days | existing max active | strict max active | max abs daily diff |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in comparison.sort_values("portfolio").iterrows():
            lines.append(
                f"| {row['portfolio']} | {_fmt_float(row['existing_final_equity'])} | "
                f"{_fmt_float(row['strict_final_equity'])} | "
                f"{int(row['active_sleeve_mismatch_days'])} | "
                f"{int(row['existing_max_active_sleeves'])} | "
                f"{int(row['strict_max_active_sleeves'])} | "
                f"{_fmt_bps(row['max_abs_gross_return_diff_bps'] / 10000.0)} |"
            )
    lines.extend(
        [
            "",
            "## Active-Sleeve Audit",
            "",
            "| portfolio | days | mean active | max active | days above horizon | missing-price drop days |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in audit.sort_values("portfolio").iterrows():
        lines.append(
            f"| {row['portfolio']} | {int(row['daily_rows'])} | "
            f"{_fmt_float(row['mean_active_sleeves'])} | {int(row['max_active_sleeves'])} | "
            f"{int(row['days_above_horizon'])} | "
            f"{int(row['days_with_missing_price_drops'])} |"
        )
    lead = metrics[metrics["portfolio"].eq(lead_portfolio)]
    if not lead.empty:
        row = lead.iloc[0]
        lines.extend(
            [
                "",
                "## Takeaway",
                "",
                f"The strict h10 rebuild keeps the lead `{lead_portfolio}` path in the same ballpark: "
                f"final equity `{_fmt_float(row['final_equity'])}`, annualized volatility "
                f"`{_fmt_pct(row['annualized_vol'])}`, Sharpe `{_fmt_float(row['sharpe_no_rf'])}`. "
                "Unlike the previous ad hoc curve, the strict rebuild enforces max active sleeves <= horizon.",
            ]
        )
    return "\n".join(lines) + "\n"


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def _fmt_bps(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 10000.0:.2f} bps"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild strict h10 daily multi-sleeve paths from Phase4S positions."
    )
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--existing-curve-path", default=str(DEFAULT_EXISTING_CURVE))
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--lead-portfolio", default=DEFAULT_LEAD_PORTFOLIO)
    args = parser.parse_args(argv)

    result = build_phase4z_strict_horizon_daily_paths(
        positions_path=args.positions_path,
        output_root=args.output_root,
        phase3_rollup_path=args.phase3_rollup_path,
        existing_curve_path=args.existing_curve_path,
        benchmark_symbol=args.benchmark_symbol,
        holding_period_sessions=args.holding_period_sessions,
        validation_price_end=args.validation_price_end,
        lead_portfolio=args.lead_portfolio,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
