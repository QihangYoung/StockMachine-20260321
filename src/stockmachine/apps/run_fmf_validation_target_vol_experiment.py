from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockmachine.apps.run_fmf_validation_baseline_experiment import (
    DEFAULT_VALIDATION_END_DATE,
    FMF_VALIDATION_UNIVERSE_NAME,
    _require_return_columns,
    _write_split_report,
    run_fmf_validation_baseline_suite,
)
from stockmachine.apps.run_multi_asset_core_experiment import (
    load_symbol_returns_for_symbols,
    slice_return_frame,
    write_experiment_artifacts,
)
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.multi_asset import (
    FMF_VALIDATION_BUCKETS,
    FMF_VALIDATION_C2_V0_POLICY_CONFIG,
    FMF_VALIDATION_STRATEGIC_BUCKET_NAMES,
    CovarianceConfig,
    RiskBudgetPolicyConfig,
    RollingAllocationConfig,
    SharpeTargetVolConfig,
    StrategyRunArtifacts,
    build_bucket_return_frame,
    run_rolling_risk_budget_backtest,
    run_rolling_sharpe_target_vol_backtest,
)
from stockmachine.research.strict_reports import write_json_artifact


FMF_VALIDATION_TARGET_VOL_LEAD_POLICY_CONFIG = RiskBudgetPolicyConfig(
    name="fmf_validation_lead_e42_c10_d22_i20_t06",
    strategic_buckets=FMF_VALIDATION_STRATEGIC_BUCKET_NAMES,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FMF validation-only target-vol experiment."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/fmf_validation_target_vol_experiment",
        help="Artifact directory for experiment outputs.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive return start date.")
    parser.add_argument(
        "--validation-end-date",
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive validation-window end date.",
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
    parser.add_argument(
        "--score-lookback-months",
        type=int,
        default=12,
        help="Trailing months used to estimate ETF Sharpe scores.",
    )
    parser.add_argument(
        "--target-volatility",
        type=float,
        default=0.10,
        help="Annualized portfolio volatility cap for the rolling target-vol strategy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    required_symbols = tuple(
        dict.fromkeys(symbol for bucket in FMF_VALIDATION_BUCKETS for symbol in bucket.symbols)
    )
    symbol_returns = load_symbol_returns_for_symbols(layout=layout, symbols=required_symbols)
    _require_return_columns(
        symbol_returns,
        required_columns=required_symbols,
        context="FMF validation target-vol universe",
    )
    symbol_returns = symbol_returns.loc[:, list(required_symbols)]
    bucket_returns = build_bucket_return_frame(symbol_returns, FMF_VALIDATION_BUCKETS)

    symbol_returns = slice_return_frame(
        symbol_returns,
        start_date=args.start_date,
        end_date=args.validation_end_date,
    )
    bucket_returns = slice_return_frame(
        bucket_returns,
        start_date=args.start_date,
        end_date=args.validation_end_date,
    )
    aligned_dates = bucket_returns.index.intersection(symbol_returns.index)
    if aligned_dates.empty:
        raise ValueError("No common dates remain between FMF bucket returns and symbol returns.")
    bucket_returns = bucket_returns.loc[aligned_dates]
    symbol_returns = symbol_returns.loc[aligned_dates]

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
    target_vol_config = SharpeTargetVolConfig(
        strategic_buckets=FMF_VALIDATION_STRATEGIC_BUCKET_NAMES,
        cash_bucket="cash",
        score_lookback=int(args.score_lookback_months) * 21,
        target_volatility=float(args.target_volatility),
    )

    strategy_runs = run_fmf_validation_target_vol_suite(
        bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        target_vol_config=target_vol_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
    )
    write_experiment_artifacts(
        output_root=output_root,
        bucket_returns=bucket_returns,
        symbol_returns=symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        proxy_chain_result=None,
    )
    _write_validation_only_report(
        path=output_root / "validation_report.md",
        strategy_runs=strategy_runs,
        validation_start=pd.Timestamp(bucket_returns.index.min()),
        validation_end=pd.Timestamp(bucket_returns.index.max()),
        target_vol_config=target_vol_config,
    )
    _augment_run_meta(
        path=output_root / "run_meta.json",
        bucket_returns=bucket_returns,
        symbol_returns=symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        target_vol_config=target_vol_config,
        validation_start=pd.Timestamp(bucket_returns.index.min()),
        validation_end=pd.Timestamp(bucket_returns.index.max()),
        strategy_runs=strategy_runs,
    )
    return 0


def run_fmf_validation_target_vol_suite(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    target_vol_config: SharpeTargetVolConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
) -> tuple[StrategyRunArtifacts, ...]:
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
        policy_config=FMF_VALIDATION_TARGET_VOL_LEAD_POLICY_CONFIG,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="rolling_fmf_c2_e42_c10_d22_i20_t06",
            records=lead_backtest.records,
            summary=lead_backtest.summary,
            weight_schedule=lead_allocations.weight_schedule,
            diagnostics=lead_allocations.diagnostics,
        )
    )

    target_vol_allocations, target_vol_backtest = run_rolling_sharpe_target_vol_backtest(
        bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        target_vol_config=target_vol_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name=f"rolling_sharpe_target_vol_{target_vol_config.score_lookback}d",
            records=target_vol_backtest.records,
            summary=target_vol_backtest.summary,
            weight_schedule=target_vol_allocations.weight_schedule,
            diagnostics=target_vol_allocations.diagnostics,
        )
    )
    return tuple(strategy_runs)


def _augment_run_meta(
    *,
    path: Path,
    bucket_returns: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    target_vol_config: SharpeTargetVolConfig,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
) -> None:
    existing: dict[str, object] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(
        {
            "experiment_family": "fmf_validation_target_vol",
            "universe_name": FMF_VALIDATION_UNIVERSE_NAME,
            "reference_benchmark_symbol": "SPY",
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "target_vol_config": asdict(target_vol_config),
            "bucket_names": list(bucket_returns.columns),
            "symbol_names": list(symbol_returns.columns),
            "bucket_symbol_mapping": {
                bucket.name: list(bucket.symbols) for bucket in FMF_VALIDATION_BUCKETS
            },
            "lockbox_policy": {
                "test_window_locked": True,
                "test_window_exposed": False,
            },
            "validation_window": {
                "start_date": validation_start.date().isoformat(),
                "end_date": validation_end.date().isoformat(),
            },
            "strategies": [strategy_run.strategy_name for strategy_run in strategy_runs],
            "summary_rows": [
                {"strategy_name": strategy_run.strategy_name, **strategy_run.summary}
                for strategy_run in strategy_runs
            ],
        }
    )
    write_json_artifact(path, existing)


def _write_validation_only_report(
    path: Path,
    *,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    target_vol_config: SharpeTargetVolConfig,
) -> None:
    summary_frame = pd.DataFrame(
        [{"strategy_name": strategy_run.strategy_name, **strategy_run.summary} for strategy_run in strategy_runs]
    )
    best_row = summary_frame.sort_values(
        ["sharpe", "annualized_return"],
        ascending=[False, False],
    ).iloc[0]
    lines = [
        "# FMF Validation Target-Vol Summary",
        "",
        f"- Validation window: `{validation_start.date().isoformat()}` to `{validation_end.date().isoformat()}`",
        f"- Target volatility: `{target_vol_config.target_volatility:.1%}`",
        f"- Sharpe score lookback: `{target_vol_config.score_lookback}` sessions",
        "- Test window: locked and not touched in this run.",
        "",
        "## Best Validation Sharpe",
        "",
        f"- `{best_row['strategy_name']}` "
        f"(Sharpe `{best_row['sharpe']:.3f}`, annualized return `{best_row['annualized_return']:.3%}`, "
        f"annualized volatility `{best_row['annualized_volatility']:.3%}`, "
        f"max drawdown `{best_row['max_drawdown']:.3%}`)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
