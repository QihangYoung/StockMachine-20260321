from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockmachine.apps.run_multi_asset_core_experiment import (
    load_symbol_returns_for_symbols,
    slice_return_frame,
    write_experiment_artifacts,
)
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.multi_asset import (
    DEFAULT_C2_V0_POLICY_CONFIG,
    DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS,
    DEFAULT_MULTI_ASSET_BUCKETS,
    DEFAULT_MULTI_ASSET_PROXY_CHAIN,
    DEFAULT_MULTI_ASSET_PROXY_SYMBOLS,
    CovarianceConfig,
    ProxyChainResult,
    RiskBudgetPolicyConfig,
    RollingAllocationConfig,
    StrategyRunArtifacts,
    ThresholdRebalanceConfig,
    ThresholdRebalanceResult,
    apply_affine_proxy_chain,
    backtest_weight_schedule,
    build_bucket_return_frame,
    build_calendar_year_return_summary,
    generate_rolling_configured_allocations,
    run_constant_weight_backtest,
    run_periodic_constant_weight_backtest,
    run_rolling_risk_budget_backtest,
    run_threshold_aware_weight_schedule_backtest,
    summarize_multi_asset_backtest_records,
)
from stockmachine.research.strict_reports import write_csv_artifact, write_json_artifact, write_summary_metrics_artifact

_LEAD_EQUITY_TOTAL = 0.35
_LEAD_EQUITY_US = 0.24
_LEAD_EQUITY_EX_US = 0.11
_LEAD_DURATION = 0.15
_LEAD_CREDIT = 0.05
_LEAD_INFLATION = 0.20
_LEAD_TREND = 0.25
_DEFAULT_RESERVE_CASH_GRID = (0.0, 0.05)
_DEFAULT_DRIFT_THRESHOLD_GRID = (0.02, 0.04, 0.06)
_DEFAULT_STALE_CAP_GRID = (21, 42)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run implementation-level rebalance tests around the current lead C2 candidate."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/c2_implementation_sweep",
        help="Artifact directory for implementation-sweep outputs.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive return start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive return end date.")
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
        "--reserve-cash-grid",
        default=",".join(f"{value:.2f}" for value in _DEFAULT_RESERVE_CASH_GRID),
        help="Comma-separated grid of reserve cash weights to test.",
    )
    parser.add_argument(
        "--drift-threshold-grid",
        default=",".join(f"{value:.2f}" for value in _DEFAULT_DRIFT_THRESHOLD_GRID),
        help="Comma-separated grid of rebalance drift thresholds.",
    )
    parser.add_argument(
        "--stale-cap-grid",
        default=",".join(str(value) for value in _DEFAULT_STALE_CAP_GRID),
        help="Comma-separated grid of stale-time caps in sessions.",
    )
    parser.add_argument(
        "--use-proxy-chain",
        action="store_true",
        help="Extend CTA/GLDM/SGOV histories with the default affine proxy chain.",
    )
    return parser


def build_lead_c2_policy_config(
    *,
    reserve_cash_weight: float,
) -> RiskBudgetPolicyConfig:
    return RiskBudgetPolicyConfig(
        name=f"c2_impl_lead_cash{int(round(reserve_cash_weight * 100.0)):02d}",
        strategic_buckets=DEFAULT_C2_V0_POLICY_CONFIG.strategic_buckets,
        reserve_buckets=DEFAULT_C2_V0_POLICY_CONFIG.reserve_buckets,
        risk_budgets={
            "equity_us": _LEAD_EQUITY_US,
            "equity_ex_us": _LEAD_EQUITY_EX_US,
            "duration": _LEAD_DURATION,
            "credit": _LEAD_CREDIT,
            "inflation_hedge": _LEAD_INFLATION,
            "trend": _LEAD_TREND,
        },
        reserve_capital_weights={"cash": reserve_cash_weight},
    )


def parse_float_grid(raw_value: str) -> tuple[float, ...]:
    values = [float(token.strip()) for token in raw_value.split(",") if token.strip()]
    if not values:
        raise ValueError("Float grid arguments must contain at least one numeric value.")
    return tuple(values)


def parse_int_grid(raw_value: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in raw_value.split(",") if token.strip()]
    if not values:
        raise ValueError("Integer grid arguments must contain at least one numeric value.")
    return tuple(values)


def build_live_window(
    actual_symbol_returns: pd.DataFrame,
    *,
    required_symbols: tuple[str, ...],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for symbol in required_symbols:
        series = actual_symbol_returns[symbol].dropna()
        if series.empty:
            return None
        starts.append(pd.Timestamp(series.index.min()))
        ends.append(pd.Timestamp(series.index.max()))
    live_start = max(starts)
    live_end = min(ends)
    if live_start > live_end:
        return None
    return live_start, live_end


def run_implementation_suite(
    bucket_returns: pd.DataFrame,
    *,
    symbol_returns: pd.DataFrame,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    reserve_cash_grid: tuple[float, ...],
    drift_threshold_grid: tuple[float, ...],
    stale_cap_grid: tuple[int, ...],
) -> tuple[tuple[StrategyRunArtifacts, ...], dict[str, pd.DataFrame], pd.DataFrame]:
    strategy_runs: list[StrategyRunArtifacts] = []
    implementation_diagnostics: dict[str, pd.DataFrame] = {}

    current_c_drift = run_constant_weight_backtest(
        symbol_returns,
        weights=DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS,
        benchmark_column="SPY",
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="current_c_policy_core",
            records=current_c_drift.records,
            summary=current_c_drift.summary,
            weight_schedule=current_c_drift.weight_schedule,
            diagnostics=None,
        )
    )

    current_c_periodic = run_periodic_constant_weight_backtest(
        symbol_returns,
        weights=DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS,
        rebalance_frequency=rolling_config.rebalance_frequency,
        benchmark_column="SPY",
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="current_c_policy_core_periodic_rebalance",
            records=current_c_periodic.records,
            summary=current_c_periodic.summary,
            weight_schedule=current_c_periodic.weight_schedule,
            diagnostics=None,
        )
    )

    c2_v0_allocations, c2_v0_backtest = run_rolling_risk_budget_backtest(
        bucket_returns,
        covariance_config=covariance_config,
        policy_config=DEFAULT_C2_V0_POLICY_CONFIG,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="rolling_c2_v0_core",
            records=c2_v0_backtest.records,
            summary=c2_v0_backtest.summary,
            weight_schedule=c2_v0_allocations.weight_schedule,
            diagnostics=c2_v0_allocations.diagnostics,
        )
    )

    implementation_rows: list[dict[str, object]] = []
    for reserve_cash_weight in reserve_cash_grid:
        policy_config = build_lead_c2_policy_config(reserve_cash_weight=reserve_cash_weight)
        allocations = generate_rolling_configured_allocations(
            bucket_returns,
            covariance_config=covariance_config,
            policy_config=policy_config,
            rolling_config=rolling_config,
        )
        calendar_backtest = backtest_weight_schedule(
            bucket_returns,
            allocations.weight_schedule,
            benchmark_column=benchmark_column,
            cost_bps_per_side=cost_bps_per_side,
        )
        calendar_name = f"{policy_config.name}_calendar"
        strategy_runs.append(
            StrategyRunArtifacts(
                strategy_name=calendar_name,
                records=calendar_backtest.records,
                summary=calendar_backtest.summary,
                weight_schedule=allocations.weight_schedule,
                diagnostics=None,
            )
        )
        implementation_rows.append(
            {
                "strategy_name": calendar_name,
                "policy_name": policy_config.name,
                "reserve_cash": reserve_cash_weight,
                "rebalance_mode": "calendar",
                "drift_threshold": None,
                "stale_time_cap": None,
                "review_days": int(len(allocations.weight_schedule)),
                "executed_rebalances": int(len(allocations.weight_schedule)),
                "target_update_days": int(len(allocations.weight_schedule)),
                "mean_max_abs_drift": None,
                "p95_max_abs_drift": None,
                "mean_turnover_to_target": None,
            }
        )

        for drift_threshold in drift_threshold_grid:
            for stale_time_cap in stale_cap_grid:
                threshold_config = ThresholdRebalanceConfig(
                    drift_threshold=float(drift_threshold),
                    stale_time_cap=int(stale_time_cap),
                )
                threshold_result, backtest_result = run_threshold_aware_weight_schedule_backtest(
                    bucket_returns,
                    allocations.weight_schedule,
                    config=threshold_config,
                    benchmark_column=benchmark_column,
                    cost_bps_per_side=cost_bps_per_side,
                )
                strategy_name = (
                    f"{policy_config.name}_band"
                    f"_t{int(round(drift_threshold * 100.0)):02d}"
                    f"_s{int(stale_time_cap):02d}"
                )
                strategy_runs.append(
                    StrategyRunArtifacts(
                        strategy_name=strategy_name,
                        records=backtest_result.records,
                        summary=backtest_result.summary,
                        weight_schedule=threshold_result.weight_schedule,
                        diagnostics=None,
                    )
                )
                implementation_diagnostics[strategy_name] = threshold_result.diagnostics
                implementation_rows.append(
                    {
                        "strategy_name": strategy_name,
                        "policy_name": policy_config.name,
                        "reserve_cash": reserve_cash_weight,
                        "rebalance_mode": "threshold",
                        "drift_threshold": drift_threshold,
                        "stale_time_cap": stale_time_cap,
                        "review_days": int(len(threshold_result.diagnostics)),
                        "executed_rebalances": int(threshold_result.diagnostics["rebalanced"].sum()),
                        "target_update_days": int(threshold_result.diagnostics["target_updated_today"].sum()),
                        "mean_max_abs_drift": float(threshold_result.diagnostics["max_abs_drift"].mean()),
                        "p95_max_abs_drift": float(threshold_result.diagnostics["max_abs_drift"].quantile(0.95)),
                        "mean_turnover_to_target": float(threshold_result.diagnostics["turnover_to_target"].mean()),
                    }
                )

    implementation_summary = pd.DataFrame(implementation_rows)
    return tuple(strategy_runs), implementation_diagnostics, implementation_summary


def write_implementation_artifacts(
    *,
    output_root: Path,
    bucket_returns: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    implementation_diagnostics: dict[str, pd.DataFrame],
    implementation_summary: pd.DataFrame,
    live_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    proxy_chain_result: ProxyChainResult | None = None,
) -> None:
    write_experiment_artifacts(
        output_root=output_root,
        bucket_returns=bucket_returns,
        symbol_returns=symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
        proxy_chain_result=proxy_chain_result,
    )

    for strategy_name, diagnostics in implementation_diagnostics.items():
        strategy_dir = output_root / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        write_csv_artifact(strategy_dir / "implementation_diagnostics.csv", diagnostics)

    write_csv_artifact(output_root / "implementation_summary.csv", implementation_summary)

    live_summary = pd.DataFrame()
    if live_window is not None:
        live_rows: list[dict[str, object]] = []
        live_yearly_frames: list[pd.DataFrame] = []
        for strategy_run in strategy_runs:
            live_records = _slice_records_to_window(
                strategy_run.records,
                start_date=live_window[0],
                end_date=live_window[1],
            )
            live_rows.append(
                {
                    "strategy_name": strategy_run.strategy_name,
                    **summarize_multi_asset_backtest_records(live_records, horizon=1),
                }
            )
            live_yearly_frames.append(
                build_calendar_year_return_summary(
                    live_records,
                    strategy_name=strategy_run.strategy_name,
                )
            )
        live_summary = write_summary_metrics_artifact(output_root / "summary_metrics_live_window.csv", live_rows)
        write_csv_artifact(
            output_root / "yearly_returns_live_window.csv",
            pd.concat(live_yearly_frames, ignore_index=True),
        )

    write_json_artifact(
        output_root / "implementation_meta.json",
        {
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "lead_policy": {
                "equity_total": _LEAD_EQUITY_TOTAL,
                "equity_us": _LEAD_EQUITY_US,
                "equity_ex_us": _LEAD_EQUITY_EX_US,
                "duration": _LEAD_DURATION,
                "credit": _LEAD_CREDIT,
                "inflation_hedge": _LEAD_INFLATION,
                "trend": _LEAD_TREND,
            },
            "live_window": (
                {
                    "start_date": live_window[0].date().isoformat(),
                    "end_date": live_window[1].date().isoformat(),
                    "summary_rows": live_summary.to_dict(orient="records"),
                }
                if live_window is not None and not live_summary.empty
                else None
            ),
        },
    )
    _write_implementation_report(
        output_root / "implementation_report.md",
        summary_frame=pd.read_csv(output_root / "summary_metrics_common_window.csv"),
        live_summary_frame=live_summary,
        implementation_summary=implementation_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    all_required_symbols = tuple(
        dict.fromkeys(
            list(DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS.index)
            + [symbol for bucket in DEFAULT_MULTI_ASSET_BUCKETS for symbol in bucket.symbols]
            + (list(DEFAULT_MULTI_ASSET_PROXY_SYMBOLS) if args.use_proxy_chain else [])
        )
    )
    actual_symbol_returns = load_symbol_returns_for_symbols(layout=layout, symbols=all_required_symbols)
    actual_symbol_returns = slice_return_frame(
        actual_symbol_returns,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    proxy_chain_result: ProxyChainResult | None = None
    research_symbol_returns = actual_symbol_returns
    if args.use_proxy_chain:
        proxy_chain_result = apply_affine_proxy_chain(
            actual_symbol_returns,
            DEFAULT_MULTI_ASSET_PROXY_CHAIN,
        )
        research_symbol_returns = proxy_chain_result.symbol_returns

    required_bucket_symbols = tuple(
        dict.fromkeys(symbol for bucket in DEFAULT_MULTI_ASSET_BUCKETS for symbol in bucket.symbols)
    )
    required_output_symbols = tuple(
        dict.fromkeys(list(DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS.index) + list(required_bucket_symbols))
    )
    research_symbol_returns = research_symbol_returns.loc[:, list(required_output_symbols)]
    bucket_returns = build_bucket_return_frame(
        research_symbol_returns.loc[:, list(required_bucket_symbols)],
        DEFAULT_MULTI_ASSET_BUCKETS,
    )
    bucket_returns = slice_return_frame(
        bucket_returns,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    aligned_dates = bucket_returns.index.intersection(research_symbol_returns.index)
    if aligned_dates.empty:
        raise ValueError("No common dates remain between bucket returns and current-C symbol returns.")
    bucket_returns = bucket_returns.loc[aligned_dates]
    research_symbol_returns = research_symbol_returns.loc[aligned_dates]

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

    strategy_runs, implementation_diagnostics, implementation_summary = run_implementation_suite(
        bucket_returns,
        symbol_returns=research_symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        reserve_cash_grid=parse_float_grid(str(args.reserve_cash_grid)),
        drift_threshold_grid=parse_float_grid(str(args.drift_threshold_grid)),
        stale_cap_grid=parse_int_grid(str(args.stale_cap_grid)),
    )
    live_window = build_live_window(
        actual_symbol_returns.loc[:, list(required_output_symbols)],
        required_symbols=required_output_symbols,
    )
    write_implementation_artifacts(
        output_root=output_root,
        bucket_returns=bucket_returns,
        symbol_returns=research_symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        implementation_diagnostics=implementation_diagnostics,
        implementation_summary=implementation_summary,
        live_window=live_window,
        proxy_chain_result=proxy_chain_result,
    )
    return 0


def _slice_records_to_window(
    records: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frame = records.copy()
    entry_dates = pd.to_datetime(frame["entry_date"], utc=False)
    return frame.loc[(entry_dates >= start_date) & (entry_dates <= end_date)].reset_index(drop=True)


def _write_implementation_report(
    output_path: Path,
    *,
    summary_frame: pd.DataFrame,
    live_summary_frame: pd.DataFrame,
    implementation_summary: pd.DataFrame,
) -> None:
    lines = ["# C2 Implementation Sweep Summary", ""]

    common_candidates = summary_frame.loc[summary_frame["strategy_name"].str.startswith("c2_impl_lead_")]
    if not common_candidates.empty:
        best_common = common_candidates.sort_values(["sharpe", "annualized_return"], ascending=[False, False]).iloc[0]
        lines.extend(
            [
                "## Best Candidate On Common Window",
                "",
                (
                    f"- `{best_common['strategy_name']}`: Sharpe `{best_common['sharpe']:.3f}`, "
                    f"annualized return `{best_common['annualized_return']:.3%}`, "
                    f"max drawdown `{best_common['max_drawdown']:.3%}`"
                ),
                "",
            ]
        )

    periodic_baseline = summary_frame.loc[
        summary_frame["strategy_name"] == "current_c_policy_core_periodic_rebalance"
    ]
    if not periodic_baseline.empty:
        baseline_row = periodic_baseline.iloc[0]
        lines.extend(
            [
                "## Current C Periodic Baseline",
                "",
                (
                    f"- Sharpe `{baseline_row['sharpe']:.3f}`, "
                    f"annualized return `{baseline_row['annualized_return']:.3%}`, "
                    f"max drawdown `{baseline_row['max_drawdown']:.3%}`"
                ),
                "",
            ]
        )

    threshold_summary = implementation_summary.loc[
        implementation_summary["rebalance_mode"] == "threshold"
    ].sort_values(["reserve_cash", "drift_threshold", "stale_time_cap"])
    if not threshold_summary.empty:
        lines.extend(
            [
                "## Threshold Grid",
                "",
                f"- tested threshold variants: `{len(threshold_summary)}`",
                "",
            ]
        )

    if not live_summary_frame.empty:
        live_candidates = live_summary_frame.loc[
            live_summary_frame["strategy_name"].str.startswith("c2_impl_lead_")
        ]
        if not live_candidates.empty:
            best_live = live_candidates.sort_values(["sharpe", "annualized_return"], ascending=[False, False]).iloc[0]
            lines.extend(
                [
                    "## Best Candidate On Live Window",
                    "",
                    (
                        f"- `{best_live['strategy_name']}`: Sharpe `{best_live['sharpe']:.3f}`, "
                        f"annualized return `{best_live['annualized_return']:.3%}`, "
                        f"max drawdown `{best_live['max_drawdown']:.3%}`"
                    ),
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
