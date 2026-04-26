from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_VALIDATION_PRICE_END,
    _load_open_to_open_returns,
)
from stockmachine.apps.run_pure_alpha_phase5a import (
    DEFAULT_ASSUMED_COST_BPS_PER_SIDE,
    DEFAULT_BASELINE_PORTFOLIO,
    DEFAULT_H10_COMPARE_CURVE_PATH,
    DEFAULT_H10_COMPARE_METRICS_PATH,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_TARGET_PORTFOLIO,
    DEFAULT_TARGET_POSITIONS_PATH,
    _add_path_columns,
    _benchmark_lookup,
    _count_changed_symbols,
    _count_new_symbols,
    _daily_metrics_and_actions,
    _load_beta_lookup,
    _load_h10_compare_curve,
    _load_net_targets,
    _net_beta,
    _normalize_to_side_grosses,
    _portfolio_return,
    _return_lookup,
    _update_ages,
)
from stockmachine.apps.run_pure_alpha_phase5b import (
    _build_h10_turnover_curve,
    _build_return_date_map,
    _fmt_float_like,
    _fmt_pct_like,
    _reference_metrics_from_comparison,
    _text_table,
    _utc_now,
    _weight_turnover,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5c_minhold5_newname_simple_20260427"
DEFAULT_MIN_HOLD_SESSIONS = 5
NEW_NAME_CAPS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40)
PLOT_COLORS = {
    "h10_baseline": "#0f766e",
    "h10_short_hybrid": "#2563eb",
    "minhold5_newname_cap": "#c2410c",
    "spy_raw": "#6b7280",
    "h10_turnover": "#111827",
    "simple_turnover": "#dc2626",
}


@dataclass(frozen=True, slots=True)
class Phase5CConfig:
    label: str
    min_hold_sessions: int
    max_new_names_per_day: int


SWEEP_CONFIGS: tuple[Phase5CConfig, ...] = tuple(
    Phase5CConfig(
        label=f"mh5_new{cap:02d}",
        min_hold_sessions=DEFAULT_MIN_HOLD_SESSIONS,
        max_new_names_per_day=cap,
    )
    for cap in NEW_NAME_CAPS
)


def build_phase5c_minhold5_newname_simple_artifacts(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    target_positions_path: str | Path = DEFAULT_TARGET_POSITIONS_PATH,
    h10_compare_curve_path: str | Path = DEFAULT_H10_COMPARE_CURVE_PATH,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    benchmark_symbol: str = "SPY",
    target_portfolio: str = DEFAULT_TARGET_PORTFOLIO,
    baseline_portfolio: str = DEFAULT_BASELINE_PORTFOLIO,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    assumed_cost_bps_per_side: float = DEFAULT_ASSUMED_COST_BPS_PER_SIDE,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_net = _load_net_targets(target_positions_path, portfolio=target_portfolio)
    symbols = tuple(sorted(target_net["symbol"].astype(str).unique()))
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
    beta_lookup = _load_beta_lookup(signal_panel_path, symbols=symbols)
    return_map = _build_return_date_map(target_net["session_date"].drop_duplicates(), benchmark_calendar)
    stock_return_lookup = _return_lookup(stock_returns)
    benchmark_lookup = _benchmark_lookup(benchmark_returns)

    h10_curve = _load_h10_compare_curve(
        h10_compare_curve_path,
        baseline_portfolio=baseline_portfolio,
        target_portfolio=target_portfolio,
    )
    h10_turnover_curve = _build_h10_turnover_curve(
        positions_path=target_positions_path,
        portfolio=target_portfolio,
        benchmark_calendar=benchmark_calendar,
        holding_period_sessions=10,
    )
    h10_mean_turnover = float(h10_turnover_curve["h10_turnover"].iloc[10:].mean())
    h10_mean_gross = float(h10_turnover_curve["h10_gross_exposure"].iloc[10:].mean())
    h10_mean_positions = float(h10_turnover_curve["h10_positions"].iloc[10:].mean())

    sweep_rows: list[dict[str, Any]] = []
    best_records_map: dict[str, pd.DataFrame] = {}
    for config in SWEEP_CONFIGS:
        records = _simulate_simple_newname_execution(
            target_net=target_net,
            return_map=return_map,
            stock_return_lookup=stock_return_lookup,
            benchmark_lookup=benchmark_lookup,
            beta_lookup=beta_lookup,
            config=config,
            assumed_cost_bps_per_side=assumed_cost_bps_per_side,
        )
        metric, _ = _daily_metrics_and_actions(
            records,
            strategy_label=config.label,
            assumed_cost_bps_per_side=assumed_cost_bps_per_side,
        )
        metric["config_min_hold_sessions"] = int(config.min_hold_sessions)
        metric["config_max_new_names_per_day"] = int(config.max_new_names_per_day)
        metric["turnover_target_h10_mean"] = float(h10_mean_turnover)
        metric["turnover_vs_h10_ratio"] = (
            float(metric["mean_turnover"] / h10_mean_turnover) if h10_mean_turnover > 0 else np.nan
        )
        metric["turnover_gap_to_h10"] = float(h10_mean_turnover - metric["mean_turnover"])
        sweep_rows.append(metric)
        best_records_map[config.label] = records

    sweep = pd.DataFrame(sweep_rows).sort_values("config_max_new_names_per_day").reset_index(drop=True)
    eligible = sweep[sweep["mean_turnover"].astype(float) <= h10_mean_turnover + 1e-12].copy()
    if eligible.empty:
        raise ValueError("No min-hold-5 new-name-cap config stayed within the rolling h10 mean turnover.")
    best = eligible.sort_values(
        ["turnover_gap_to_h10", "annualized_return"],
        ascending=[True, False],
    ).iloc[0]
    best_label = str(best["series"])
    best_records = best_records_map[best_label].copy()

    comparison = _build_comparison_curve(h10_curve, best_records, h10_turnover_curve)
    metrics = _build_metrics(
        comparison=comparison,
        best_metric_row=best.to_dict(),
        target_portfolio=target_portfolio,
        h10_mean_turnover=h10_mean_turnover,
        h10_mean_gross=h10_mean_gross,
        h10_mean_positions=h10_mean_positions,
    )

    sweep_path = output_dir / "phase5c_minhold5_newname_sweep.csv"
    metrics_path = output_dir / "phase5c_metrics.csv"
    curve_path = output_dir / "phase5c_comparison_curve.csv"
    turnover_path = output_dir / "phase5c_h10_turnover_curve.csv"
    records_path = output_dir / "phase5c_best_records.csv.gz"
    plot_path = output_dir / "phase5c_comparison_plot.png"
    memo_path = output_dir / "phase5c_minhold5_newname_simple_memo.md"
    rollup_path = output_dir / "phase5c_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    comparison.to_csv(curve_path, index=False)
    h10_turnover_curve.to_csv(turnover_path, index=False)
    best_records.to_csv(records_path, index=False, compression="gzip")
    _plot_comparison(comparison, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            sweep=sweep,
            best_label=best_label,
            plot_path=plot_path,
            h10_mean_turnover=h10_mean_turnover,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "target_positions_path": Path(target_positions_path).as_posix(),
        "h10_compare_curve_path": Path(h10_compare_curve_path).as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "target_portfolio": target_portfolio,
        "baseline_portfolio": baseline_portfolio,
        "benchmark_symbol": benchmark_symbol,
        "validation_price_end": validation_price_end,
        "assumed_cost_bps_per_side": float(assumed_cost_bps_per_side),
        "sweep_configs": [asdict(config) for config in SWEEP_CONFIGS],
        "best_config_label": best_label,
        "h10_mean_turnover": h10_mean_turnover,
        "sweep_artifact": sweep_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "curve_artifact": curve_path.as_posix(),
        "turnover_artifact": turnover_path.as_posix(),
        "best_records_artifact": records_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "min_hold_5_plus_only_simple_new_name_cap_calibrated_to_h10_mean_turnover",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _simulate_simple_newname_execution(
    *,
    target_net: pd.DataFrame,
    return_map: dict[pd.Timestamp, pd.Timestamp],
    stock_return_lookup: dict[tuple[pd.Timestamp, str], float],
    benchmark_lookup: dict[pd.Timestamp, float],
    beta_lookup: dict[tuple[pd.Timestamp, str], float],
    config: Phase5CConfig,
    assumed_cost_bps_per_side: float,
) -> pd.DataFrame:
    targets_by_date = {
        pd.Timestamp(session_date): {
            str(row.symbol): float(row.signed_weight)
            for row in group.itertuples(index=False)
            if abs(float(row.signed_weight)) > 1e-12
        }
        for session_date, group in target_net.groupby("session_date", sort=True)
    }
    current_weights: dict[str, float] = {}
    current_ages: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for signal_date in sorted(targets_by_date):
        return_date = return_map.get(pd.Timestamp(signal_date))
        if return_date is None:
            continue
        raw_target = dict(targets_by_date[signal_date])
        target_long_gross = float(sum(weight for weight in raw_target.values() if weight > 1e-12))
        target_short_gross = float(sum(-weight for weight in raw_target.values() if weight < -1e-12))
        raw_turnover = _weight_turnover(current_weights, raw_target)
        applied = _build_simple_candidate(
            previous_weights=current_weights,
            previous_ages=current_ages,
            raw_target=raw_target,
            min_hold_sessions=config.min_hold_sessions,
            max_new_names_per_day=config.max_new_names_per_day,
            target_long_gross=target_long_gross,
            target_short_gross=target_short_gross,
        )
        turnover = _weight_turnover(current_weights, applied)
        gross_return, missing_symbols = _portfolio_return(
            weights=applied,
            return_date=return_date,
            stock_return_lookup=stock_return_lookup,
        )
        assumed_cost = turnover * assumed_cost_bps_per_side / 10000.0
        net_return = gross_return - assumed_cost
        gross_exposure = float(sum(abs(weight) for weight in applied.values()))
        net_beta, beta_coverage = _net_beta(
            weights=applied,
            signal_date=signal_date,
            beta_lookup=beta_lookup,
        )
        rows.append(
            {
                "strategy": config.label,
                "signal_date": pd.Timestamp(signal_date),
                "return_date": pd.Timestamp(return_date),
                "gross_return": gross_return,
                "net_return_assumed_cost": net_return,
                "benchmark_oto_return": benchmark_lookup.get(pd.Timestamp(return_date), np.nan),
                "turnover": turnover,
                "raw_turnover": raw_turnover,
                "assumed_cost_bps": assumed_cost * 10000.0,
                "gross_exposure": gross_exposure,
                "net_beta": net_beta,
                "beta_coverage_ratio": beta_coverage,
                "positions": int(len(applied)),
                "changed_symbols": _count_changed_symbols(current_weights, applied),
                "new_symbols": _count_new_symbols(current_weights, applied),
                "missing_return_symbols": int(missing_symbols),
                "rebalance_action": "minhold5_plus_newname_cap",
                "test_window_used": False,
            }
        )
        previous_weights = dict(current_weights)
        current_weights = dict(applied)
        current_ages = _update_ages(previous_weights, current_ages, current_weights)

    frame = pd.DataFrame(rows).sort_values("return_date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No daily execution rows built for strategy {config.label}.")
    return _add_path_columns(frame)


def _build_simple_candidate(
    *,
    previous_weights: dict[str, float],
    previous_ages: dict[str, int],
    raw_target: dict[str, float],
    min_hold_sessions: int,
    max_new_names_per_day: int,
    target_long_gross: float,
    target_short_gross: float,
) -> dict[str, float]:
    candidate = {symbol: float(weight) for symbol, weight in raw_target.items() if abs(weight) > 1e-12}
    for symbol, previous_weight in previous_weights.items():
        age = int(previous_ages.get(symbol, 0))
        if age < min_hold_sessions and abs(previous_weight) > 1e-12:
            candidate[symbol] = float(previous_weight)

    new_symbols = [
        symbol
        for symbol, weight in candidate.items()
        if abs(weight) > 1e-12 and abs(previous_weights.get(symbol, 0.0)) <= 1e-12
    ]
    if max_new_names_per_day >= 0 and len(new_symbols) > max_new_names_per_day:
        keep = set(
            sorted(
                new_symbols,
                key=lambda symbol: (abs(candidate[symbol]), symbol),
                reverse=True,
            )[:max_new_names_per_day]
        )
        for symbol in new_symbols:
            if symbol not in keep:
                candidate.pop(symbol, None)

    candidate, _ = _normalize_to_side_grosses(
        candidate,
        target_long_gross=target_long_gross,
        target_short_gross=target_short_gross,
    )
    return {symbol: float(weight) for symbol, weight in candidate.items() if abs(weight) > 1e-12}


def _build_metrics(
    *,
    comparison: pd.DataFrame,
    best_metric_row: dict[str, Any],
    target_portfolio: str,
    h10_mean_turnover: float,
    h10_mean_gross: float,
    h10_mean_positions: float,
) -> pd.DataFrame:
    reference = _reference_metrics_from_comparison(
        comparison=comparison,
        target_portfolio=target_portfolio,
        h10_mean_turnover=h10_mean_turnover,
        h10_mean_gross=h10_mean_gross,
        h10_mean_positions=h10_mean_positions,
    )
    best_row = pd.DataFrame([best_metric_row])
    return pd.concat([reference, best_row], ignore_index=True, sort=False)


def _build_comparison_curve(
    h10_curve: pd.DataFrame,
    best_records: pd.DataFrame,
    h10_turnover_curve: pd.DataFrame,
) -> pd.DataFrame:
    out = h10_curve.merge(
        best_records[
            [
                "return_date",
                "gross_return",
                "equity_gross",
                "drawdown_gross",
                "rolling_60_gross",
                "turnover",
            ]
        ].rename(
            columns={
                "gross_return": "minhold5_newname_cap__return",
                "equity_gross": "minhold5_newname_cap__equity",
                "drawdown_gross": "minhold5_newname_cap__drawdown",
                "rolling_60_gross": "minhold5_newname_cap__rolling_60_return",
                "turnover": "minhold5_newname_cap__turnover",
            }
        ),
        on="return_date",
        how="inner",
    )
    out = out.merge(
        h10_turnover_curve[["return_date", "h10_turnover"]],
        on="return_date",
        how="left",
    )
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


def _plot_comparison(curve: pd.DataFrame, *, output_path: str | Path) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 15),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8, 1.8, 1.6]},
    )
    ax1, ax2, ax3, ax4 = axes
    dates = curve["return_date"]

    series_specs = [
        ("h10_baseline", "sic2_soft_neutral__equity", "sic2_soft_neutral__drawdown", "sic2_soft_neutral__rolling_60_return"),
        (
            "h10_short_hybrid",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__equity",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__drawdown",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__rolling_60_return",
        ),
        (
            "minhold5_newname_cap",
            "minhold5_newname_cap__equity",
            "minhold5_newname_cap__drawdown",
            "minhold5_newname_cap__rolling_60_return",
        ),
    ]
    for label, equity_col, drawdown_col, rolling_col in series_specs:
        color = PLOT_COLORS[label]
        ax1.plot(dates, curve[equity_col], label=label, color=color, linewidth=2.0)
        ax2.plot(dates, curve[drawdown_col] * 100.0, color=color, linewidth=1.8)
        ax3.plot(dates, curve[rolling_col] * 100.0, color=color, linewidth=1.8)

    ax1.plot(dates, curve["benchmark_equity"], label="spy_raw", color=PLOT_COLORS["spy_raw"], linestyle="--", linewidth=1.8)
    ax2.plot(dates, curve["benchmark_drawdown"] * 100.0, color=PLOT_COLORS["spy_raw"], linestyle="--", linewidth=1.6)
    ax3.plot(dates, curve["benchmark_rolling_60_return"] * 100.0, color=PLOT_COLORS["spy_raw"], linestyle="--", linewidth=1.6)
    ax4.plot(dates, curve["h10_turnover"], label="rolling_h10_turnover", color=PLOT_COLORS["h10_turnover"], linewidth=1.6)
    ax4.plot(dates, curve["minhold5_newname_cap__turnover"], label="simple_newname_turnover", color=PLOT_COLORS["simple_turnover"], linewidth=1.6)

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax4.set_ylabel("Daily Turnover")
    ax4.set_xlabel("Date")
    ax1.set_title("Phase5C Min-Hold-5 + Only Simple New-Name Cap")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    ax4.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
    best_label: str,
    plot_path: Path,
    h10_mean_turnover: float,
) -> str:
    display = metrics.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
        "corr_to_spy",
        "hit_rate_daily",
        "beta_coverage_ratio",
    ):
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct_like if column != "corr_to_spy" else _fmt_float_like)
    for column in (
        "sharpe_no_rf",
        "mean_daily_return_bps",
        "mean_turnover",
        "turnover_target_h10_mean",
        "turnover_vs_h10_ratio",
        "turnover_gap_to_h10",
        "mean_assumed_cost_bps",
        "mean_gross_exposure",
        "mean_abs_net_beta",
        "mean_positions",
    ):
        if column in display.columns:
            display[column] = display[column].map(_fmt_float_like)

    shortlist = sweep[
        [
            "series",
            "config_max_new_names_per_day",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "mean_turnover",
            "turnover_vs_h10_ratio",
            "mean_abs_net_beta",
        ]
    ].copy()
    shortlist = shortlist.sort_values("config_max_new_names_per_day")
    for column in ("annualized_return", "annualized_vol"):
        shortlist[column] = shortlist[column].map(_fmt_pct_like)
    for column in ("sharpe_no_rf", "mean_turnover", "turnover_vs_h10_ratio", "mean_abs_net_beta"):
        shortlist[column] = shortlist[column].map(_fmt_float_like)

    return "\n".join(
        [
            "# Phase5C Min-Hold-5 + Only Simple New-Name Cap",
            "",
            f"Generated: {_utc_now()}",
            "",
            f"Rolling h10 mean turnover benchmark = {h10_mean_turnover:.6f}",
            "",
            f"Best simple config: {best_label}",
            "",
            "## Metrics",
            "",
            _text_table(display),
            "",
            "## Sweep",
            "",
            _text_table(shortlist),
            "",
            "## Plot",
            "",
            f"![Phase5C comparison]({plot_path.as_posix()})",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase5C min-hold-5 with only a simple new-name cap."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target-positions-path", default=str(DEFAULT_TARGET_POSITIONS_PATH))
    parser.add_argument("--h10-compare-curve-path", default=str(DEFAULT_H10_COMPARE_CURVE_PATH))
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--target-portfolio", default=DEFAULT_TARGET_PORTFOLIO)
    parser.add_argument("--baseline-portfolio", default=DEFAULT_BASELINE_PORTFOLIO)
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--assumed-cost-bps-per-side", type=float, default=DEFAULT_ASSUMED_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    build_phase5c_minhold5_newname_simple_artifacts(
        output_root=args.output_root,
        target_positions_path=args.target_positions_path,
        h10_compare_curve_path=args.h10_compare_curve_path,
        signal_panel_path=args.signal_panel_path,
        target_portfolio=args.target_portfolio,
        baseline_portfolio=args.baseline_portfolio,
        benchmark_symbol=args.benchmark_symbol,
        validation_price_end=args.validation_price_end,
        assumed_cost_bps_per_side=args.assumed_cost_bps_per_side,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
