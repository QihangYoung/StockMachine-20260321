"""Phase7M reversal-regime exposure overlay.

Phase7M tests whether the Phase7C reliable-core reversal regime score is more
useful as a portfolio-level exposure/risk-budget overlay than as an internal
factor-weight switch. It starts from Phase7K locked-turnover positions, scales
signed weights by walk-forward gate bucket, and reruns the strict daily
open-to-open path with turnover costs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from stockmachine.apps.run_pure_alpha_phase7j import (
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_VALIDATION_PRICE_END,
    _add_path_metrics,
    _portfolio_metrics,
    _strict_daily_rebalance_path,
    _turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase7l import (
    DEFAULT_GATE_QUANTILE_COLUMN,
    DEFAULT_GATE_SCORE,
    DEFAULT_GATE_STATES_PATH,
    DEFAULT_GATE_UPPER_QUANTILE_COLUMN,
)
from stockmachine.apps.run_pure_alpha_phase7k import DEFAULT_COST_BPS_PER_SIDE


DEFAULT_PHASE7K_ROOT = RESEARCH_ROOT / "phase7k_locked_turnover_shared_capital_20260518_tb0p15"
DEFAULT_POSITIONS_PATH = DEFAULT_PHASE7K_ROOT / "phase7k_locked_positions.csv.gz"
DEFAULT_BASELINE_METRICS_PATH = DEFAULT_PHASE7K_ROOT / "phase7k_strict_daily_metrics.csv"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7m_reversal_regime_exposure_overlay_20260518"
DEFAULT_POLICIES: Mapping[str, Mapping[str, float]] = {
    "exposure_top1p25": {"bottom": 1.0, "middle": 1.0, "top": 1.25, "unready": 1.0, "missing": 1.0},
    "exposure_bottom0p75": {"bottom": 0.75, "middle": 1.0, "top": 1.0, "unready": 1.0, "missing": 1.0},
    "exposure_step0p75_1p25": {"bottom": 0.75, "middle": 1.0, "top": 1.25, "unready": 1.0, "missing": 1.0},
    "exposure_step0p90_1p10": {"bottom": 0.90, "middle": 1.0, "top": 1.10, "unready": 1.0, "missing": 1.0},
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7M reversal-regime exposure overlay.")
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS_PATH))
    parser.add_argument("--gate-states-path", default=str(DEFAULT_GATE_STATES_PATH))
    parser.add_argument("--baseline-metrics-path", default=str(DEFAULT_BASELINE_METRICS_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument("--benchmark-adj-factor-glob", action="append", dest="benchmark_adj_factor_globs")
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7m_reversal_regime_exposure_overlay(
        positions_path=args.positions_path,
        gate_states_path=args.gate_states_path,
        baseline_metrics_path=args.baseline_metrics_path,
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        validation_price_end=args.validation_price_end,
        daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
        adj_factor_globs=tuple(args.adj_factor_globs or DEFAULT_ADJ_FACTOR_GLOBS),
        benchmark_daily_globs=tuple(args.benchmark_daily_globs or DEFAULT_BENCHMARK_DAILY_GLOBS),
        benchmark_adj_factor_globs=tuple(args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS),
        cost_bps_per_side=args.cost_bps_per_side,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7m_reversal_regime_exposure_overlay(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS_PATH,
    gate_states_path: str | Path = DEFAULT_GATE_STATES_PATH,
    baseline_metrics_path: str | Path = DEFAULT_BASELINE_METRICS_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    cost_bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE,
    policies: Mapping[str, Mapping[str, float]] = DEFAULT_POLICIES,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = _load_positions(positions_path)
    gate = _load_gate_buckets(gate_states_path)
    scaled = _scale_positions(positions, gate=gate, policies=policies)

    symbols = tuple(sorted(scaled["symbol"].astype(str).unique()))
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
        positions=scaled,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        calendar=calendar,
        cost_bps_per_side=cost_bps_per_side,
    )
    curve = _add_path_metrics(daily)
    metrics = _portfolio_metrics(curve)
    turnover = _turnover_summary(curve)
    bucket_summary = _bucket_summary(curve, gate)
    comparison = _baseline_comparison(metrics, baseline_metrics_path)

    paths = {
        "positions": output_dir / "phase7m_scaled_positions.csv.gz",
        "curve": output_dir / "phase7m_strict_daily_curve.csv",
        "metrics": output_dir / "phase7m_strict_daily_metrics.csv",
        "turnover": output_dir / "phase7m_turnover_summary.csv",
        "bucket_summary": output_dir / "phase7m_bucket_summary.csv",
        "comparison": output_dir / "phase7m_vs_phase7k_net_comparison.csv",
        "memo": output_dir / "phase7m_exposure_overlay_memo.md",
        "rollup": output_dir / "phase7m_rollup.json",
    }
    scaled.to_csv(paths["positions"], index=False, compression="gzip")
    curve.to_csv(paths["curve"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    turnover.to_csv(paths["turnover"], index=False)
    bucket_summary.to_csv(paths["bucket_summary"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    paths["memo"].write_text(
        _memo(
            positions_path=Path(positions_path),
            gate_states_path=Path(gate_states_path),
            policies=policies,
            metrics=metrics,
            comparison=comparison,
            bucket_summary=bucket_summary,
            cost_bps_per_side=cost_bps_per_side,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7m_reversal_regime_exposure_overlay",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "positions_path": Path(positions_path).as_posix(),
        "gate_states_path": Path(gate_states_path).as_posix(),
        "baseline_metrics_path": Path(baseline_metrics_path).as_posix(),
        "validation_price_end": validation_price_end,
        "cost_bps_per_side": float(cost_bps_per_side),
        "policies": {key: dict(value) for key, value in policies.items()},
        "rows": {
            "positions": int(len(scaled)),
            "curve": int(len(curve)),
            "metrics": int(len(metrics)),
        },
        "outputs": {key: value.as_posix() for key, value in paths.items() if key != "rollup"},
        "method": "scale_phase7k_signed_weights_by_walk_forward_reversal_regime_bucket",
        "limitations": [
            "This is an exposure overlay, not a new stock-selection optimizer.",
            "Scaling above 1.0 increases gross exposure and therefore uses leverage.",
            "Costs include turnover caused by both stock changes and exposure multiplier changes.",
        ],
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    paths["rollup"].write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_positions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    required = {"session_date", "portfolio", "symbol", "signed_weight", "test_window_used"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"positions missing required columns: {missing}")
    test_used = frame["test_window_used"].astype(str).str.lower().isin({"true", "1", "yes"})
    if test_used.any():
        raise ValueError("positions include test-window rows; refusing to continue")
    frame = frame.copy()
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["signed_weight"] = pd.to_numeric(frame["signed_weight"], errors="coerce")
    frame = frame.dropna(subset=["session_date", "portfolio", "symbol", "signed_weight"])
    if frame.empty:
        raise ValueError(f"no valid positions found in {path}")
    return frame.sort_values(["portfolio", "session_date", "symbol"]).reset_index(drop=True)


def _load_gate_buckets(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    required = {
        "session_date",
        DEFAULT_GATE_SCORE,
        DEFAULT_GATE_QUANTILE_COLUMN,
        DEFAULT_GATE_UPPER_QUANTILE_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gate states missing required columns: {missing}")
    score = pd.to_numeric(frame[DEFAULT_GATE_SCORE], errors="coerce")
    q33 = pd.to_numeric(frame[DEFAULT_GATE_QUANTILE_COLUMN], errors="coerce")
    q67 = pd.to_numeric(frame[DEFAULT_GATE_UPPER_QUANTILE_COLUMN], errors="coerce")
    ready = score.notna() & q33.notna() & q67.notna()
    frame = frame.copy()
    frame["gate_bucket"] = "unready"
    frame.loc[ready & score.le(q33), "gate_bucket"] = "bottom"
    frame.loc[ready & score.ge(q67), "gate_bucket"] = "top"
    frame.loc[ready & frame["gate_bucket"].eq("unready"), "gate_bucket"] = "middle"
    return frame[
        [
            "session_date",
            "gate_bucket",
            DEFAULT_GATE_SCORE,
            DEFAULT_GATE_QUANTILE_COLUMN,
            DEFAULT_GATE_UPPER_QUANTILE_COLUMN,
        ]
    ]


def _scale_positions(
    positions: pd.DataFrame,
    *,
    gate: pd.DataFrame,
    policies: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    base = positions.merge(gate, on="session_date", how="left")
    base["gate_bucket"] = base["gate_bucket"].fillna("missing")
    frames = []
    for policy_name, multipliers in policies.items():
        frame = base.copy()
        frame["regime_multiplier"] = frame["gate_bucket"].map(multipliers).fillna(1.0).astype(float)
        frame["base_portfolio"] = frame["portfolio"].astype(str)
        frame["portfolio"] = frame["portfolio"].astype(str) + "_" + policy_name
        frame["signed_weight"] = frame["signed_weight"].astype(float) * frame["regime_multiplier"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _bucket_summary(curve: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    frame = curve.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame = frame.merge(gate[["session_date", "gate_bucket"]], left_on="signal_date", right_on="session_date", how="left")
    frame["gate_bucket"] = frame["gate_bucket"].fillna("missing")
    return (
        frame.groupby(["portfolio", "gate_bucket"], sort=True)
        .agg(
            days=("net_return", "size"),
            mean_net_bps=("net_return", lambda values: float(values.mean() * 10000.0)),
            mean_gross_bps=("gross_return", lambda values: float(values.mean() * 10000.0)),
            mean_cost_bps=("cost_bps", "mean"),
            hit_rate=("net_return", lambda values: float((values > 0.0).mean())),
            mean_gross_exposure=("gross_exposure", "mean"),
        )
        .reset_index()
    )


def _baseline_comparison(metrics: pd.DataFrame, baseline_metrics_path: str | Path) -> pd.DataFrame:
    path = Path(baseline_metrics_path)
    if not path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(path)
    baseline = baseline[baseline["return_kind"].eq("net")].copy()
    overlay = metrics[metrics["return_kind"].eq("net")].copy()
    if baseline.empty or overlay.empty:
        return pd.DataFrame()
    overlay["base_portfolio"] = overlay["portfolio"].map(_strip_policy_suffix)
    baseline = baseline.rename(columns={"portfolio": "base_portfolio"})
    columns = [
        "final_equity",
        "annualized_return",
        "annualized_vol",
        "sharpe_no_rf",
        "max_drawdown",
        "mean_daily_return_bps",
        "mean_turnover",
        "mean_cost_bps",
        "mean_gross_exposure",
    ]
    available = [column for column in columns if column in overlay.columns and column in baseline.columns]
    merged = overlay[["portfolio", "base_portfolio", *available]].merge(
        baseline[["base_portfolio", *available]],
        on="base_portfolio",
        suffixes=("_overlay", "_baseline"),
    )
    for column in available:
        merged[f"delta_{column}"] = merged[f"{column}_overlay"] - merged[f"{column}_baseline"]
    return merged


def _strip_policy_suffix(portfolio: str) -> str:
    for policy_name in DEFAULT_POLICIES:
        suffix = "_" + policy_name
        if portfolio.endswith(suffix):
            return portfolio[: -len(suffix)]
    return portfolio


def _memo(
    *,
    positions_path: Path,
    gate_states_path: Path,
    policies: Mapping[str, Mapping[str, float]],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    cost_bps_per_side: float,
) -> str:
    net = metrics[metrics["return_kind"].eq("net")].copy()
    metric_columns = [
        column
        for column in (
            "portfolio",
            "final_equity",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "max_drawdown",
            "mean_turnover",
            "mean_cost_bps",
            "mean_gross_exposure",
        )
        if column in net.columns
    ]
    compare_columns = [
        column
        for column in (
            "portfolio",
            "base_portfolio",
            "final_equity_overlay",
            "final_equity_baseline",
            "delta_final_equity",
            "annualized_return_overlay",
            "annualized_return_baseline",
            "delta_annualized_return",
            "sharpe_no_rf_overlay",
            "sharpe_no_rf_baseline",
            "delta_sharpe_no_rf",
            "max_drawdown_overlay",
            "max_drawdown_baseline",
            "delta_max_drawdown",
        )
        if column in comparison.columns
    ]
    lines = [
        "# Phase7M Reversal-Regime Exposure Overlay",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- positions: `{positions_path.as_posix()}`",
        f"- gate states: `{gate_states_path.as_posix()}`",
        f"- cost bps per side: `{cost_bps_per_side}`",
        "- policy multipliers:",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"policy": name, **dict(multipliers)}
                    for name, multipliers in policies.items()
                ]
            )
        ),
        "",
        "## Net Metrics",
        "",
        _markdown_table(net[metric_columns] if metric_columns else net),
        "",
        "## Versus Phase7K Baseline",
        "",
        _markdown_table(comparison[compare_columns] if compare_columns else comparison),
        "",
        "## Bucket Summary",
        "",
        _markdown_table(bucket_summary),
        "",
        "## Notes",
        "",
        "- This overlay changes total gross exposure, not stock selection or factor composition.",
        "- Top multipliers above 1.0 use more gross exposure and are not directly comparable on return alone.",
        "- Turnover cost is recomputed from the scaled target weights.",
        "",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
