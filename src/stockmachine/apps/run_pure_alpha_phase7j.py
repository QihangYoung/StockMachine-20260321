"""Phase7J strict daily path for Phase7I shared-capital portfolios.

This app evaluates the Phase7I target positions as a daily rebalance path:
signal date T enters/rebalances at T+1 open, earns open-to-open return through
T+2 open, and pays turnover-based costs. It is validation-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4z import _load_open_to_open_returns
from stockmachine.apps.run_pure_alpha_phase7b import _markdown_table
from stockmachine.apps.run_pure_alpha_phase7i import DEFAULT_OUTPUT_ROOT as PHASE7I_OUTPUT_ROOT


DEFAULT_POSITIONS_PATH = PHASE7I_OUTPUT_ROOT / "phase7i_shared_capital_positions.csv.gz"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7j_phase7i_strict_daily_cost_20260517"
DEFAULT_COST_BPS_PER_SIDE = 4.0
DEFAULT_VALIDATION_PRICE_END = "2019-12-31"
DEFAULT_BENCHMARK_SYMBOL = "SPY"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7J strict daily path.")
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument(
        "--benchmark-adj-factor-glob",
        action="append",
        dest="benchmark_adj_factor_globs",
    )
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7j_strict_daily_path(
        positions_path=args.positions_path,
        output_root=args.output_root,
        daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
        adj_factor_globs=tuple(args.adj_factor_globs or DEFAULT_ADJ_FACTOR_GLOBS),
        benchmark_daily_globs=tuple(args.benchmark_daily_globs or DEFAULT_BENCHMARK_DAILY_GLOBS),
        benchmark_adj_factor_globs=tuple(
            args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS
        ),
        benchmark_symbol=args.benchmark_symbol,
        validation_price_end=args.validation_price_end,
        cost_bps_per_side=args.cost_bps_per_side,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7j_strict_daily_path(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    cost_bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    calendar = (
        benchmark_returns[benchmark_returns["symbol"].eq(benchmark_symbol)]["session_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    daily = _strict_daily_rebalance_path(
        positions=positions,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        calendar=calendar,
        cost_bps_per_side=cost_bps_per_side,
    )
    curve = _add_path_metrics(daily)
    metrics = _portfolio_metrics(curve)
    turnover = _turnover_summary(curve)

    curve_path = output_dir / "phase7j_strict_daily_curve.csv"
    metrics_path = output_dir / "phase7j_strict_daily_metrics.csv"
    turnover_path = output_dir / "phase7j_turnover_summary.csv"
    memo_path = output_dir / "phase7j_strict_daily_cost_memo.md"
    rollup_path = output_dir / "phase7j_rollup.json"

    curve.to_csv(curve_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    turnover.to_csv(turnover_path, index=False)
    memo_path.write_text(
        _memo(
            positions_path=Path(positions_path),
            cost_bps_per_side=cost_bps_per_side,
            curve=curve,
            metrics=metrics,
            turnover=turnover,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7j_phase7i_strict_daily_cost_path",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "positions_path": Path(positions_path).as_posix(),
        "output_root": output_dir.as_posix(),
        "cost_bps_per_side": float(cost_bps_per_side),
        "benchmark_symbol": benchmark_symbol,
        "validation_price_end": validation_price_end,
        "rows": {
            "positions": int(len(positions)),
            "stock_returns": int(len(stock_returns)),
            "curve": int(len(curve)),
            "metrics": int(len(metrics)),
        },
        "outputs": {
            "curve": curve_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "turnover_summary": turnover_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "method": "strict_daily_rebalance_open_to_open_with_turnover_costs",
        "label_contract": {
            "signal_session": "T",
            "rebalance": "adjusted open at T+1",
            "pnl": "open-to-open return from T+1 to T+2, stored on T+2 return_date",
            "cost": "sum_abs_target_weight_change * cost_bps_per_side",
        },
        "limitations": [
            "No hard minimum holding period; this intentionally tests the Phase7I MVP as constructed.",
            "Costs are turnover-based only; no borrow, financing, spread, market impact, or capacity model.",
            "Days with missing active position returns are skipped for that portfolio.",
        ],
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_positions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "session_date",
        "portfolio",
        "side",
        "symbol",
        "signed_weight",
        "test_window_used",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"positions missing required columns: {missing}")
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame["side"] = frame["side"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["signed_weight"] = pd.to_numeric(frame["signed_weight"], errors="coerce")
    frame["test_window_used"] = frame["test_window_used"].map(_is_true)
    frame = frame.dropna(subset=["session_date", "portfolio", "symbol", "signed_weight"])
    if frame.empty:
        raise ValueError(f"No positions found in {path}.")
    if frame["test_window_used"].any():
        raise ValueError("Positions include test-window rows; refusing to continue.")
    return frame.sort_values(["portfolio", "session_date", "side", "symbol"]).reset_index(drop=True)


def _strict_daily_rebalance_path(
    *,
    positions: pd.DataFrame,
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    calendar: pd.Series,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    calendar_values = pd.to_datetime(calendar).drop_duplicates().sort_values().reset_index(drop=True)
    calendar_index = {date: idx for idx, date in enumerate(calendar_values)}
    return_lookup = stock_returns.copy()
    return_lookup["session_date"] = pd.to_datetime(return_lookup["session_date"])
    return_lookup = return_lookup.rename(columns={"session_date": "return_date"})
    benchmark = benchmark_returns.copy()
    benchmark["session_date"] = pd.to_datetime(benchmark["session_date"])
    benchmark = benchmark[["session_date", "benchmark_oto_return"]].drop_duplicates("session_date")

    rows: list[dict[str, Any]] = []
    for portfolio, group in positions.groupby("portfolio", sort=True):
        previous_weights: dict[str, float] = {}
        for signal_date, signal_positions in group.groupby("session_date", sort=True):
            if signal_date not in calendar_index:
                continue
            return_idx = calendar_index[signal_date] + 2
            if return_idx >= len(calendar_values):
                continue
            return_date = calendar_values.iloc[return_idx]
            weights = (
                signal_positions.groupby("symbol", sort=True)["signed_weight"]
                .sum()
                .loc[lambda series: series.abs() > 1e-12]
                .to_dict()
            )
            turnover = _signed_turnover(previous_weights, weights)
            active = pd.DataFrame(
                {"symbol": list(weights.keys()), "signed_weight": list(weights.values())}
            )
            active = active.merge(
                return_lookup[return_lookup["return_date"].eq(return_date)][
                    ["symbol", "oto_return"]
                ],
                on="symbol",
                how="left",
            )
            missing_positions = int(active["oto_return"].isna().sum())
            if missing_positions:
                previous_weights = weights
                continue
            gross_return = float(
                (active["signed_weight"].astype(float) * active["oto_return"].astype(float)).sum()
            )
            cost_return = float(turnover * cost_bps_per_side / 10000.0)
            benchmark_row = benchmark[benchmark["session_date"].eq(return_date)]
            benchmark_return = (
                float(benchmark_row["benchmark_oto_return"].iloc[0])
                if not benchmark_row.empty
                else np.nan
            )
            rows.append(
                {
                    "portfolio": portfolio,
                    "signal_date": str(signal_date.date()),
                    "return_date": return_date,
                    "gross_return": gross_return,
                    "net_return": gross_return - cost_return,
                    "cost_return": cost_return,
                    "cost_bps": cost_return * 10000.0,
                    "turnover": float(turnover),
                    "positions": int(len(weights)),
                    "gross_exposure": float(sum(abs(weight) for weight in weights.values())),
                    "net_exposure": float(sum(weights.values())),
                    "missing_positions": int(missing_positions),
                    "benchmark_oto_return": benchmark_return,
                    "test_window_used": False,
                }
            )
            previous_weights = weights
    if not rows:
        raise ValueError("No strict daily returns were built.")
    return pd.DataFrame(rows).sort_values(["portfolio", "return_date"]).reset_index(drop=True)


def _add_path_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for portfolio, group in daily.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date").copy()
        for kind in ("gross", "net"):
            returns = frame[f"{kind}_return"].fillna(0.0)
            frame[f"{kind}_equity"] = (1.0 + returns).cumprod()
            frame[f"{kind}_drawdown"] = (
                frame[f"{kind}_equity"] / frame[f"{kind}_equity"].cummax() - 1.0
            )
            frame[f"{kind}_rolling_60_return"] = (
                frame[f"{kind}_equity"] / frame[f"{kind}_equity"].shift(60) - 1.0
            )
        frame["benchmark_equity"] = (1.0 + frame["benchmark_oto_return"].fillna(0.0)).cumprod()
        frame["benchmark_drawdown"] = (
            frame["benchmark_equity"] / frame["benchmark_equity"].cummax() - 1.0
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _portfolio_metrics(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date")
        base = {
            "portfolio": portfolio,
            "start": str(frame["return_date"].min().date()),
            "end": str(frame["return_date"].max().date()),
            "daily_rows": int(len(frame)),
            "mean_turnover": float(frame["turnover"].mean()),
            "median_turnover": float(frame["turnover"].median()),
            "mean_cost_bps": float(frame["cost_bps"].mean()),
            "mean_positions": float(frame["positions"].mean()),
            "mean_gross_exposure": float(frame["gross_exposure"].mean()),
            "mean_abs_net_exposure": float(frame["net_exposure"].abs().mean()),
            "corr_to_spy_gross": _safe_corr(frame["gross_return"], frame["benchmark_oto_return"]),
            "corr_to_spy_net": _safe_corr(frame["net_return"], frame["benchmark_oto_return"]),
            "realized_beta_gross": _realized_beta(frame["gross_return"], frame["benchmark_oto_return"]),
            "realized_beta_net": _realized_beta(frame["net_return"], frame["benchmark_oto_return"]),
            "test_window_used": bool(frame["test_window_used"].any()),
        }
        for kind in ("gross", "net"):
            returns = pd.to_numeric(frame[f"{kind}_return"], errors="coerce").dropna()
            equity = frame[f"{kind}_equity"]
            annualized_return = float(equity.iloc[-1] ** (252.0 / len(returns)) - 1.0)
            annualized_vol = float(returns.std(ddof=1) * np.sqrt(252.0))
            row = dict(base)
            row.update(
                {
                    "return_kind": kind,
                    "final_equity": float(equity.iloc[-1]),
                    "annualized_return": annualized_return,
                    "annualized_vol": annualized_vol,
                    "sharpe_no_rf": annualized_return / annualized_vol
                    if annualized_vol > 0.0
                    else np.nan,
                    "max_drawdown": float(frame[f"{kind}_drawdown"].min()),
                    "mean_daily_return_bps": float(returns.mean() * 10000.0),
                    "hit_rate_daily": float((returns > 0.0).mean()),
                    "rolling_60_positive_rate": float(
                        (frame[f"{kind}_rolling_60_return"] > 0.0).mean()
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["return_kind", "sharpe_no_rf"], ascending=[True, False])


def _turnover_summary(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in curve.groupby("portfolio", sort=True):
        turnover = pd.to_numeric(group["turnover"], errors="coerce")
        rows.append(
            {
                "portfolio": portfolio,
                "mean_turnover": float(turnover.mean()),
                "median_turnover": float(turnover.median()),
                "p10_turnover": float(turnover.quantile(0.10)),
                "p90_turnover": float(turnover.quantile(0.90)),
                "max_turnover": float(turnover.max()),
                "mean_cost_bps": float(group["cost_bps"].mean()),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _signed_turnover(previous: dict[str, float], new: dict[str, float]) -> float:
    symbols = set(previous) | set(new)
    return float(sum(abs(float(new.get(symbol, 0.0)) - float(previous.get(symbol, 0.0))) for symbol in symbols))


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _realized_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    frame = pd.concat([returns, benchmark], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    x = frame.iloc[:, 1].to_numpy(dtype=float)
    y = frame.iloc[:, 0].to_numpy(dtype=float)
    variance = float(np.var(x, ddof=1))
    if variance <= 0.0:
        return np.nan
    return float(np.cov(y, x, ddof=1)[0, 1] / variance)


def _memo(
    *,
    positions_path: Path,
    cost_bps_per_side: float,
    curve: pd.DataFrame,
    metrics: pd.DataFrame,
    turnover: pd.DataFrame,
) -> str:
    lines = [
        "# Phase7J Strict Daily Cost Path Results",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- positions: `{positions_path.as_posix()}`",
        f"- cost: `{cost_bps_per_side}` bps per side",
        "- signal date T rebalances at T+1 adjusted open",
        "- PnL is T+1 to T+2 adjusted open-to-open return",
        "- no hard minimum holding period",
        f"- curve rows: `{len(curve)}`",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Turnover",
        "",
        _markdown_table(turnover),
        "",
        "## Notes",
        "",
        "- This is stricter than Phase7I forward-label diagnostics because it builds an actual daily target path and charges realized target turnover.",
        "- It is still not a production execution model: no borrow, financing, spread, market impact, ADV participation, or order-fill simulation is included.",
        "",
    ]
    return "\n".join(lines)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
