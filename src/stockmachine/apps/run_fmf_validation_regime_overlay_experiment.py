from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockmachine.apps.run_fmf_validation_baseline_experiment import (
    DEFAULT_VALIDATION_END_DATE,
    FMF_VALIDATION_UNIVERSE_NAME,
    _require_return_columns,
    load_symbol_returns_for_symbols,
    run_fmf_validation_baseline_suite,
    slice_return_frame,
    write_split_window_artifacts,
)
from stockmachine.apps.run_multi_asset_core_experiment import write_experiment_artifacts
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.multi_asset import (
    FMF_VALIDATION_BUCKETS,
    CovarianceConfig,
    RiskBudgetPolicyConfig,
    RollingAllocationConfig,
    StrategyRunArtifacts,
    build_bucket_return_frame,
    build_equity_duration_shifted_policy_config,
    run_rolling_risk_budget_backtest,
    run_state_conditioned_rolling_risk_budget_backtest,
)
from stockmachine.research.strict_reports import write_csv_artifact, write_json_artifact
from stockmachine.risk.regime import BenchmarkTrendDrawdownVolRegimeDetector


DEFAULT_LEAD_POLICY_CONFIG = RiskBudgetPolicyConfig(
    name="fmf_c2_e42_c10_d22_i20_t06",
    strategic_buckets=("equity_us", "equity_ex_us", "duration", "credit", "inflation_hedge", "trend"),
    reserve_buckets=("cash",),
    risk_budgets={
        "equity_us": 0.288,
        "equity_ex_us": 0.132,
        "duration": 0.22,
        "credit": 0.10,
        "inflation_hedge": 0.20,
        "trend": 0.06,
    },
    reserve_capital_weights={"cash": 0.05},
)
DEFAULT_REGIME_SHIFT = 0.04
DEFAULT_TREND_LOOKBACK = 126
DEFAULT_VOL_LOOKBACK = 63
DEFAULT_DRAWDOWN_THRESHOLD = -0.08
DEFAULT_HIGH_VOL_THRESHOLD = 0.14


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a validation-only FMF regime-aware overlay experiment."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/fmf_validation_regime_overlay_experiment",
        help="Artifact directory for validation-only regime-overlay outputs.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive return start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive return end date.")
    parser.add_argument(
        "--validation-end-date",
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive validation-window end date. Lockbox test remains closed.",
    )
    parser.add_argument("--benchmark-column", default="equity_us")
    parser.add_argument("--cost-bps-per-side", type=float, default=0.0)
    parser.add_argument("--rebalance-frequency", type=int, default=21)
    parser.add_argument("--effective-lag", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--long-lookback", type=int, default=252)
    parser.add_argument("--short-lookback", type=int, default=63)
    parser.add_argument("--ewma-lambda", type=float, default=0.97)
    parser.add_argument("--long-weight", type=float, default=0.70)
    parser.add_argument("--short-weight", type=float, default=0.30)
    parser.add_argument("--trend-lookback", type=int, default=DEFAULT_TREND_LOOKBACK)
    parser.add_argument("--vol-lookback", type=int, default=DEFAULT_VOL_LOOKBACK)
    parser.add_argument("--drawdown-threshold", type=float, default=DEFAULT_DRAWDOWN_THRESHOLD)
    parser.add_argument("--high-vol-threshold", type=float, default=DEFAULT_HIGH_VOL_THRESHOLD)
    parser.add_argument("--equity-duration-shift", type=float, default=DEFAULT_REGIME_SHIFT)
    return parser


def build_validation_regime_state_frame(
    bucket_returns: pd.DataFrame,
    *,
    benchmark_column: str,
    trend_lookback: int = DEFAULT_TREND_LOOKBACK,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    drawdown_threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
    high_vol_threshold: float = DEFAULT_HIGH_VOL_THRESHOLD,
) -> pd.DataFrame:
    detector = BenchmarkTrendDrawdownVolRegimeDetector(
        trend_lookback_windows=int(trend_lookback),
        drawdown_threshold=float(drawdown_threshold),
        vol_lookback_windows=int(vol_lookback),
        high_vol_annualized_threshold=float(high_vol_threshold),
        horizon_sessions=1,
    )
    detector_input = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(bucket_returns.index, utc=False),
            "benchmark_return": pd.Series(bucket_returns[benchmark_column], dtype=float).to_numpy(),
        }
    )
    labeled = detector.label_frame(
        detector_input,
        return_column="benchmark_return",
        date_column="entry_date",
    )
    state_map = {
        "bull": "risk_on",
        "correction": "neutral",
        "bear": "defensive",
        "rebound": "defensive",
        "warmup": "neutral",
    }
    labeled["overlay_state"] = labeled["regime_label"].map(state_map).fillna("neutral")
    return labeled


def build_regime_overlay_policy_map(
    *,
    base_policy_config: RiskBudgetPolicyConfig = DEFAULT_LEAD_POLICY_CONFIG,
    equity_duration_shift: float = DEFAULT_REGIME_SHIFT,
) -> dict[str, RiskBudgetPolicyConfig]:
    return {
        "neutral": base_policy_config,
        "risk_on": build_equity_duration_shifted_policy_config(
            base_policy_config,
            name=f"{base_policy_config.name}_risk_on",
            equity_duration_shift=float(equity_duration_shift),
        ),
        "defensive": build_equity_duration_shifted_policy_config(
            base_policy_config,
            name=f"{base_policy_config.name}_defensive",
            equity_duration_shift=-float(equity_duration_shift),
        ),
    }


def run_fmf_validation_regime_overlay_suite(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    regime_state_frame: pd.DataFrame,
    overlay_policy_map: dict[str, RiskBudgetPolicyConfig],
) -> tuple[tuple[StrategyRunArtifacts, ...], pd.DataFrame]:
    strategy_runs = list(
        run_fmf_validation_baseline_suite(
            bucket_returns,
            covariance_config=covariance_config,
            rolling_config=rolling_config,
            benchmark_column=benchmark_column,
            cost_bps_per_side=cost_bps_per_side,
        )
    )

    lead_allocations, lead_backtest = run_rolling_risk_budget_backtest(
        bucket_returns,
        covariance_config=covariance_config,
        policy_config=DEFAULT_LEAD_POLICY_CONFIG,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name=f"rolling_{DEFAULT_LEAD_POLICY_CONFIG.name}",
            records=lead_backtest.records,
            summary=lead_backtest.summary,
            weight_schedule=lead_allocations.weight_schedule,
            diagnostics=lead_allocations.diagnostics,
        )
    )

    state_by_date = regime_state_frame.set_index("entry_date")["overlay_state"]
    overlay_allocations, overlay_backtest = run_state_conditioned_rolling_risk_budget_backtest(
        bucket_returns,
        state_by_date=state_by_date,
        state_to_policy_config=overlay_policy_map,
        default_state="neutral",
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="rolling_fmf_c2_regime_overlay_v1",
            records=overlay_backtest.records,
            summary=overlay_backtest.summary,
            weight_schedule=overlay_allocations.weight_schedule,
            diagnostics=overlay_allocations.diagnostics,
        )
    )
    return tuple(strategy_runs), regime_state_frame


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_end = pd.Timestamp(args.validation_end_date)
    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    required_symbols = tuple(
        dict.fromkeys(symbol for bucket in FMF_VALIDATION_BUCKETS for symbol in bucket.symbols)
    )
    symbol_returns = load_symbol_returns_for_symbols(layout=layout, symbols=required_symbols)
    _require_return_columns(
        symbol_returns,
        required_columns=required_symbols,
        context="FMF regime-overlay validation universe",
    )
    symbol_returns = symbol_returns.loc[:, list(required_symbols)]
    bucket_returns = build_bucket_return_frame(symbol_returns, FMF_VALIDATION_BUCKETS)

    symbol_returns = slice_return_frame(
        symbol_returns,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    bucket_returns = slice_return_frame(
        bucket_returns,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    aligned_dates = bucket_returns.index.intersection(symbol_returns.index)
    if aligned_dates.empty:
        raise ValueError("No common dates remain between FMF bucket returns and symbol returns.")
    bucket_returns = bucket_returns.loc[aligned_dates]
    symbol_returns = symbol_returns.loc[aligned_dates]

    validation_bucket_returns = bucket_returns.loc[bucket_returns.index <= validation_end].copy()
    if validation_bucket_returns.empty:
        raise ValueError("No validation rows remain after applying validation_end_date.")
    validation_symbol_returns = symbol_returns.loc[validation_bucket_returns.index].copy()

    covariance_config = CovarianceConfig(
        long_lookback=int(args.long_lookback),
        short_lookback=int(args.short_lookback),
        ewma_lambda=float(args.ewma_lambda),
        long_weight=float(args.long_weight),
        short_weight=float(args.short_weight),
    )
    rolling_config = RollingAllocationConfig(
        rebalance_frequency=int(args.rebalance_frequency),
        effective_lag=int(args.effective_lag),
        min_history=int(args.min_history),
        start_date=args.start_date,
    )
    regime_state_frame = build_validation_regime_state_frame(
        validation_bucket_returns,
        benchmark_column=str(args.benchmark_column),
        trend_lookback=int(args.trend_lookback),
        vol_lookback=int(args.vol_lookback),
        drawdown_threshold=float(args.drawdown_threshold),
        high_vol_threshold=float(args.high_vol_threshold),
    )
    overlay_policy_map = build_regime_overlay_policy_map(
        equity_duration_shift=float(args.equity_duration_shift),
    )
    strategy_runs, regime_state_frame = run_fmf_validation_regime_overlay_suite(
        validation_bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        regime_state_frame=regime_state_frame,
        overlay_policy_map=overlay_policy_map,
    )

    write_experiment_artifacts(
        output_root=output_root,
        bucket_returns=validation_bucket_returns,
        symbol_returns=validation_symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        proxy_chain_result=None,
    )
    validation_summary = write_split_window_artifacts(
        output_root=output_root,
        strategy_runs=strategy_runs,
        window_name="validation_window",
        start_date=pd.Timestamp(validation_bucket_returns.index.min()),
        end_date=validation_end,
    )

    policy_map_frame = pd.DataFrame(
        [
            {
                "overlay_state": state_name,
                "policy_name": policy_config.name,
                **policy_config.normalized_risk_budgets().to_dict(),
                **{
                    f"reserve_{column}": value
                    for column, value in policy_config.reserve_capital_series().to_dict().items()
                },
            }
            for state_name, policy_config in overlay_policy_map.items()
        ]
    )
    state_counts = (
        regime_state_frame["overlay_state"].value_counts(dropna=False).rename_axis("overlay_state").reset_index(name="days")
    )
    write_csv_artifact(output_root / "regime_state_frame.csv", regime_state_frame)
    write_csv_artifact(output_root / "overlay_policy_map.csv", policy_map_frame)
    write_csv_artifact(output_root / "overlay_state_counts.csv", state_counts)

    overlay_diagnostics = next(
        run.diagnostics
        for run in strategy_runs
        if run.strategy_name == "rolling_fmf_c2_regime_overlay_v1"
    )
    if overlay_diagnostics is not None:
        write_csv_artifact(output_root / "overlay_allocation_diagnostics.csv", overlay_diagnostics)

    _write_report(
        output_root / "summary_report.md",
        validation_summary=validation_summary,
        state_counts=state_counts,
        policy_map_frame=policy_map_frame,
    )
    write_json_artifact(
        output_root / "run_meta.json",
        {
            "experiment_family": "fmf_validation_regime_overlay",
            "universe_name": FMF_VALIDATION_UNIVERSE_NAME,
            "lockbox_policy": {
                "test_window_locked": True,
                "test_window_exposed": False,
            },
            "validation_window": {
                "start_date": pd.Timestamp(validation_bucket_returns.index.min()).date().isoformat(),
                "end_date": validation_end.date().isoformat(),
                "summary_rows": validation_summary.to_dict(orient="records"),
            },
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "regime_overlay": {
                "benchmark_column": str(args.benchmark_column),
                "trend_lookback": int(args.trend_lookback),
                "vol_lookback": int(args.vol_lookback),
                "drawdown_threshold": float(args.drawdown_threshold),
                "high_vol_threshold": float(args.high_vol_threshold),
                "equity_duration_shift": float(args.equity_duration_shift),
                "policy_rows": policy_map_frame.to_dict(orient="records"),
                "state_counts": state_counts.to_dict(orient="records"),
            },
        },
    )
    return 0


def _write_report(
    output_path: Path,
    *,
    validation_summary: pd.DataFrame,
    state_counts: pd.DataFrame,
    policy_map_frame: pd.DataFrame,
) -> None:
    lines = ["# FMF Validation Regime Overlay Summary", ""]
    summary_sorted = validation_summary.sort_values(["sharpe", "annualized_return"], ascending=[False, False])
    if not summary_sorted.empty:
        best_row = summary_sorted.iloc[0]
        lines.extend(
            [
                "## Best Validation Result",
                "",
                (
                    f"- `{best_row['strategy_name']}`: Sharpe `{best_row['sharpe']:.3f}`, "
                    f"annualized return `{best_row['annualized_return']:.3%}`, "
                    f"max drawdown `{best_row['max_drawdown']:.3%}`"
                ),
                "",
            ]
        )
    lines.extend(["## Overlay State Counts", ""])
    for row in state_counts.to_dict(orient="records"):
        lines.append(f"- `{row['overlay_state']}`: `{int(row['days'])}` days")
    lines.extend(["", "## Overlay Policy Map", ""])
    for row in policy_map_frame.to_dict(orient="records"):
        lines.append(
            (
                f"- `{row['overlay_state']}` -> `{row['policy_name']}` "
                f"(equity_us `{row['equity_us']:.3f}`, equity_ex_us `{row['equity_ex_us']:.3f}`, "
                f"duration `{row['duration']:.3f}`, credit `{row['credit']:.3f}`, "
                f"inflation `{row['inflation_hedge']:.3f}`, trend `{row['trend']:.3f}`)"
            )
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
