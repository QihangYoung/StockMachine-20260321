from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockmachine.apps.run_multi_asset_core_experiment import (
    load_symbol_returns_for_symbols,
    run_default_experiment_suite,
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
    apply_affine_proxy_chain,
    build_bucket_return_frame,
    run_rolling_risk_budget_backtest,
    summarize_multi_asset_backtest_records,
)
from stockmachine.research.strict_reports import (
    write_csv_artifact,
    write_json_artifact,
    write_summary_metrics_artifact,
)

_EQUITY_TOTAL = 0.35
_EQUITY_US_SHARE = 24.0 / 35.0
_DEFAULT_CREDIT_RISK_BUDGET = 0.10
_DEFAULT_CASH_RESERVE = 0.05
_DEFAULT_EQUITY_TOTAL_GRID = (_EQUITY_TOTAL,)
_DEFAULT_CREDIT_GRID = (_DEFAULT_CREDIT_RISK_BUDGET,)
_DEFAULT_DURATION_GRID = (0.15, 0.20, 0.25)
_DEFAULT_INFLATION_GRID = (0.10, 0.15, 0.20)
_LIVE_WINDOW_DATE_COLUMN = "entry_date"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a constrained C2 risk-budget region sweep against the current C baselines."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/c2_risk_budget_sweep",
        help="Artifact directory for sweep outputs.",
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
        "--equity-total-grid",
        default=",".join(f"{value:.2f}" for value in _DEFAULT_EQUITY_TOTAL_GRID),
        help="Comma-separated grid of total equity risk budgets (decimal fractions).",
    )
    parser.add_argument(
        "--credit-grid",
        default=",".join(f"{value:.2f}" for value in _DEFAULT_CREDIT_GRID),
        help="Comma-separated grid of credit risk budgets (decimal fractions).",
    )
    parser.add_argument(
        "--duration-grid",
        default=",".join(f"{value:.2f}" for value in _DEFAULT_DURATION_GRID),
        help="Comma-separated grid of duration risk budgets (decimal fractions).",
    )
    parser.add_argument(
        "--inflation-grid",
        default=",".join(f"{value:.2f}" for value in _DEFAULT_INFLATION_GRID),
        help="Comma-separated grid of inflation-hedge risk budgets (decimal fractions).",
    )
    parser.add_argument(
        "--use-proxy-chain",
        action="store_true",
        help="Extend CTA/GLDM/SGOV histories with the default affine proxy chain.",
    )
    return parser


def build_c2_region_policy_configs(
    *,
    equity_total_grid: tuple[float, ...] = _DEFAULT_EQUITY_TOTAL_GRID,
    credit_grid: tuple[float, ...] = _DEFAULT_CREDIT_GRID,
    duration_grid: tuple[float, ...] = _DEFAULT_DURATION_GRID,
    inflation_grid: tuple[float, ...] = _DEFAULT_INFLATION_GRID,
    reserve_cash_weight: float = _DEFAULT_CASH_RESERVE,
) -> tuple[RiskBudgetPolicyConfig, ...]:
    """Build a disciplined region sweep around C2 v0 without collapsing to one point."""

    strategic_buckets = DEFAULT_C2_V0_POLICY_CONFIG.strategic_buckets
    reserve_buckets = DEFAULT_C2_V0_POLICY_CONFIG.reserve_buckets

    configs: list[RiskBudgetPolicyConfig] = []
    for equity_total in equity_total_grid:
        if equity_total <= 0.0:
            continue
        equity_us_budget = equity_total * _EQUITY_US_SHARE
        equity_ex_us_budget = equity_total - equity_us_budget
        for credit_risk_budget in credit_grid:
            if credit_risk_budget <= 0.0:
                continue
            for duration_budget in duration_grid:
                for inflation_budget in inflation_grid:
                    trend_budget = 1.0 - equity_total - credit_risk_budget - duration_budget - inflation_budget
                    if trend_budget <= 0.0:
                        continue
                    if (
                        abs(equity_total - _EQUITY_TOTAL) < 1e-12
                        and abs(credit_risk_budget - _DEFAULT_CREDIT_RISK_BUDGET) < 1e-12
                        and abs(duration_budget - 0.20) < 1e-12
                        and abs(inflation_budget - 0.15) < 1e-12
                    ):
                        # The default suite already runs C2 v0 explicitly.
                        continue
                    config_name = (
                        f"c2_region_e{int(round(equity_total * 100.0)):02d}"
                        f"_c{int(round(credit_risk_budget * 100.0)):02d}"
                        f"_d{int(round(duration_budget * 100.0)):02d}"
                        f"_i{int(round(inflation_budget * 100.0)):02d}"
                        f"_t{int(round(trend_budget * 100.0)):02d}"
                    )
                    configs.append(
                        RiskBudgetPolicyConfig(
                            name=config_name,
                            strategic_buckets=strategic_buckets,
                            reserve_buckets=reserve_buckets,
                            risk_budgets={
                                "equity_us": equity_us_budget,
                                "equity_ex_us": equity_ex_us_budget,
                                "duration": duration_budget,
                                "credit": credit_risk_budget,
                                "inflation_hedge": inflation_budget,
                                "trend": trend_budget,
                            },
                            reserve_capital_weights={"cash": reserve_cash_weight},
                        )
                    )
    return tuple(configs)


def run_c2_sweep_suite(
    bucket_returns: pd.DataFrame,
    *,
    symbol_returns: pd.DataFrame,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    policy_configs: tuple[RiskBudgetPolicyConfig, ...] | None = None,
) -> tuple[StrategyRunArtifacts, ...]:
    strategy_runs = list(
        run_default_experiment_suite(
            bucket_returns,
            symbol_returns=symbol_returns,
            covariance_config=covariance_config,
            rolling_config=rolling_config,
            benchmark_column=benchmark_column,
            cost_bps_per_side=cost_bps_per_side,
        )
    )

    for policy_config in policy_configs or build_c2_region_policy_configs():
        allocations, backtest = run_rolling_risk_budget_backtest(
            bucket_returns,
            covariance_config=covariance_config,
            policy_config=policy_config,
            rolling_config=rolling_config,
            benchmark_column=benchmark_column,
            cost_bps_per_side=cost_bps_per_side,
        )
        strategy_runs.append(
            StrategyRunArtifacts(
                strategy_name=f"rolling_{policy_config.name}",
                records=backtest.records,
                summary=backtest.summary,
                weight_schedule=allocations.weight_schedule,
                diagnostics=allocations.diagnostics,
            )
        )
    return tuple(strategy_runs)


def build_policy_grid_frame(
    policy_configs: tuple[RiskBudgetPolicyConfig, ...],
) -> pd.DataFrame:
    default_budgets = DEFAULT_C2_V0_POLICY_CONFIG.normalized_risk_budgets()
    rows: list[dict[str, object]] = [
        {
            "strategy_name": "rolling_c2_v0_core",
            "policy_name": DEFAULT_C2_V0_POLICY_CONFIG.name,
            "equity_total": float(default_budgets["equity_us"] + default_budgets["equity_ex_us"]),
            **default_budgets.to_dict(),
            "diversifier_total": float(default_budgets["inflation_hedge"] + default_budgets["trend"]),
            **{f"reserve_{key}": value for key, value in DEFAULT_C2_V0_POLICY_CONFIG.reserve_capital_series().to_dict().items()},
        }
    ]
    for policy_config in policy_configs:
        budgets = policy_config.normalized_risk_budgets()
        rows.append(
            {
                "strategy_name": f"rolling_{policy_config.name}",
                "policy_name": policy_config.name,
                "equity_total": float(budgets["equity_us"] + budgets["equity_ex_us"]),
                **budgets.to_dict(),
                "diversifier_total": float(budgets["inflation_hedge"] + budgets["trend"]),
                **{f"reserve_{key}": value for key, value in policy_config.reserve_capital_series().to_dict().items()},
            }
        )
    frame = pd.DataFrame(rows)
    ordered_columns = [
        "strategy_name",
        "policy_name",
        "equity_total",
        "equity_us",
        "equity_ex_us",
        "duration",
        "credit",
        "inflation_hedge",
        "trend",
        "diversifier_total",
    ] + [column for column in frame.columns if column.startswith("reserve_")]
    return frame.loc[:, ordered_columns]


def write_c2_sweep_artifacts(
    *,
    output_root: Path,
    bucket_returns: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    policy_configs: tuple[RiskBudgetPolicyConfig, ...],
    proxy_chain_result: ProxyChainResult | None = None,
    live_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
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

    policy_grid = build_policy_grid_frame(policy_configs)
    write_csv_artifact(output_root / "policy_grid.csv", policy_grid)

    summary_frame = pd.read_csv(output_root / "summary_metrics.csv")
    common_summary_frame = pd.read_csv(output_root / "summary_metrics_common_window.csv")
    candidate_names = {"rolling_c2_v0_core"} | {f"rolling_{config.name}" for config in policy_configs}
    candidate_common_summary = (
        common_summary_frame.loc[common_summary_frame["strategy_name"].isin(candidate_names)]
        .sort_values(["sharpe", "annualized_return"], ascending=[False, False])
        .reset_index(drop=True)
    )
    write_csv_artifact(output_root / "summary_metrics_c2_candidates_common_window.csv", candidate_common_summary)

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
                _build_calendar_year_summary(
                    live_records,
                    strategy_name=strategy_run.strategy_name,
                )
            )
        live_summary = write_summary_metrics_artifact(
            output_root / "summary_metrics_live_window.csv",
            live_rows,
        )
        write_csv_artifact(
            output_root / "yearly_returns_live_window.csv",
            pd.concat(live_yearly_frames, ignore_index=True),
        )

    write_json_artifact(
        output_root / "sweep_meta.json",
        {
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "candidate_count": int(len(policy_configs) + 1),
            "candidate_names": ["rolling_c2_v0_core"] + [f"rolling_{config.name}" for config in policy_configs],
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
    _write_c2_sweep_report(
        output_root / "sweep_report.md",
        common_summary_frame=common_summary_frame,
        candidate_common_summary=candidate_common_summary,
        live_summary_frame=live_summary,
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
    policy_configs = build_c2_region_policy_configs(
        equity_total_grid=parse_grid_argument(str(args.equity_total_grid)),
        credit_grid=parse_grid_argument(str(args.credit_grid)),
        duration_grid=parse_grid_argument(str(args.duration_grid)),
        inflation_grid=parse_grid_argument(str(args.inflation_grid)),
    )
    strategy_runs = run_c2_sweep_suite(
        bucket_returns,
        symbol_returns=research_symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        policy_configs=policy_configs,
    )
    live_window = resolve_live_window(
        actual_symbol_returns.loc[:, list(required_output_symbols)],
        required_symbols=required_output_symbols,
    )
    write_c2_sweep_artifacts(
        output_root=output_root,
        bucket_returns=bucket_returns,
        symbol_returns=research_symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        policy_configs=policy_configs,
        proxy_chain_result=proxy_chain_result,
        live_window=live_window,
    )
    return 0


def resolve_live_window(
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


def _slice_records_to_window(
    records: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frame = records.copy()
    entry_dates = pd.to_datetime(frame[_LIVE_WINDOW_DATE_COLUMN], utc=False)
    return frame.loc[(entry_dates >= start_date) & (entry_dates <= end_date)].reset_index(drop=True)


def _build_calendar_year_summary(
    records: pd.DataFrame,
    *,
    strategy_name: str,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["strategy_name", "calendar_year", "total_return", "sessions"])

    frame = records.copy()
    frame[_LIVE_WINDOW_DATE_COLUMN] = pd.to_datetime(frame[_LIVE_WINDOW_DATE_COLUMN], utc=False)
    frame["calendar_year"] = frame[_LIVE_WINDOW_DATE_COLUMN].dt.year.astype(int)
    rows: list[dict[str, object]] = []
    for calendar_year, group in frame.groupby("calendar_year", sort=True):
        total_return = float((1.0 + group["net_return"].astype(float)).prod() - 1.0)
        rows.append(
            {
                "strategy_name": strategy_name,
                "calendar_year": int(calendar_year),
                "total_return": total_return,
                "sessions": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _write_c2_sweep_report(
    output_path: Path,
    *,
    common_summary_frame: pd.DataFrame,
    candidate_common_summary: pd.DataFrame,
    live_summary_frame: pd.DataFrame,
) -> None:
    lines = ["# C2 Risk-Budget Sweep Summary", ""]

    if not candidate_common_summary.empty:
        best_candidate = candidate_common_summary.iloc[0]
        lines.extend(
            [
                "## Best Candidate On Common Window",
                "",
                (
                    f"- `{best_candidate['strategy_name']}`: Sharpe `{best_candidate['sharpe']:.3f}`, "
                    f"annualized return `{best_candidate['annualized_return']:.3%}`, "
                    f"max drawdown `{best_candidate['max_drawdown']:.3%}`"
                ),
                "",
            ]
        )

    periodic_baseline = common_summary_frame.loc[
        common_summary_frame["strategy_name"] == "current_c_policy_core_periodic_rebalance"
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

    if not live_summary_frame.empty:
        live_candidates = live_summary_frame.loc[
            live_summary_frame["strategy_name"].str.startswith("rolling_c2_")
        ].sort_values(["sharpe", "annualized_return"], ascending=[False, False])
        if not live_candidates.empty:
            best_live_candidate = live_candidates.iloc[0]
            lines.extend(
                [
                    "## Best Candidate On Live Window",
                    "",
                    (
                        f"- `{best_live_candidate['strategy_name']}`: Sharpe `{best_live_candidate['sharpe']:.3f}`, "
                        f"annualized return `{best_live_candidate['annualized_return']:.3%}`, "
                        f"max drawdown `{best_live_candidate['max_drawdown']:.3%}`"
                    ),
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_grid_argument(raw_value: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("Grid arguments must contain at least one numeric value.")
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
