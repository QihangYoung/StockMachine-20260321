from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockmachine.apps.run_fmf_validation_baseline_experiment import (
    DEFAULT_VALIDATION_END_DATE,
    FMF_VALIDATION_UNIVERSE_NAME,
    load_symbol_returns_for_symbols,
    run_fmf_validation_baseline_suite,
    slice_return_frame,
    write_split_window_artifacts,
)
from stockmachine.apps.run_multi_asset_core_experiment import write_experiment_artifacts
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.multi_asset import (
    FMF_VALIDATION_BUCKETS,
    FMF_VALIDATION_C2_V0_POLICY_CONFIG,
    CovarianceConfig,
    RiskBudgetPolicyConfig,
    RollingAllocationConfig,
    StrategyRunArtifacts,
    build_bucket_return_frame,
    run_rolling_risk_budget_backtest,
)
from stockmachine.research.strict_reports import (
    write_csv_artifact,
    write_json_artifact,
    write_summary_metrics_artifact,
)


_EQUITY_SEED_TOTAL = 0.35
_EQUITY_US_SHARE = 24.0 / 35.0
_DEFAULT_CREDIT = 0.10
_DEFAULT_CASH_RESERVE = 0.05
_DEFAULT_EQUITY_TOTAL_GRID = (0.30, 0.35, 0.40)
_DEFAULT_CREDIT_GRID = (0.05, 0.10)
_DEFAULT_DURATION_GRID = (0.10, 0.15, 0.20)
_DEFAULT_INFLATION_GRID = (0.10, 0.15, 0.20)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-only FMF C2 risk-budget grid search."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/fmf_validation_c2_grid_search",
        help="Artifact directory for grid-search outputs.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive return start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive return end date.")
    parser.add_argument(
        "--validation-end-date",
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive validation-window end date. Test window remains locked.",
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
    return parser


def build_fmf_c2_region_policy_configs(
    *,
    equity_total_grid: tuple[float, ...] = _DEFAULT_EQUITY_TOTAL_GRID,
    credit_grid: tuple[float, ...] = _DEFAULT_CREDIT_GRID,
    duration_grid: tuple[float, ...] = _DEFAULT_DURATION_GRID,
    inflation_grid: tuple[float, ...] = _DEFAULT_INFLATION_GRID,
    reserve_cash_weight: float = _DEFAULT_CASH_RESERVE,
) -> tuple[RiskBudgetPolicyConfig, ...]:
    strategic_buckets = FMF_VALIDATION_C2_V0_POLICY_CONFIG.strategic_buckets
    reserve_buckets = FMF_VALIDATION_C2_V0_POLICY_CONFIG.reserve_buckets

    configs: list[RiskBudgetPolicyConfig] = []
    for equity_total in equity_total_grid:
        if equity_total <= 0.0:
            continue
        equity_us_budget = equity_total * _EQUITY_US_SHARE
        equity_ex_us_budget = equity_total - equity_us_budget
        for credit_budget in credit_grid:
            if credit_budget <= 0.0:
                continue
            for duration_budget in duration_grid:
                if duration_budget <= 0.0:
                    continue
                for inflation_budget in inflation_grid:
                    if inflation_budget <= 0.0:
                        continue
                    trend_budget = 1.0 - equity_total - credit_budget - duration_budget - inflation_budget
                    if trend_budget <= 0.0:
                        continue
                    if (
                        abs(equity_total - _EQUITY_SEED_TOTAL) < 1e-12
                        and abs(credit_budget - _DEFAULT_CREDIT) < 1e-12
                        and abs(duration_budget - 0.20) < 1e-12
                        and abs(inflation_budget - 0.15) < 1e-12
                    ):
                        continue
                    config_name = (
                        f"fmf_c2_e{int(round(equity_total * 100.0)):02d}"
                        f"_c{int(round(credit_budget * 100.0)):02d}"
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
                                "credit": credit_budget,
                                "inflation_hedge": inflation_budget,
                                "trend": trend_budget,
                            },
                            reserve_capital_weights={"cash": reserve_cash_weight},
                        )
                    )
    return tuple(configs)


def run_fmf_c2_grid_search_suite(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    policy_configs: tuple[RiskBudgetPolicyConfig, ...],
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
    for policy_config in policy_configs:
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


def build_policy_grid_frame(policy_configs: tuple[RiskBudgetPolicyConfig, ...]) -> pd.DataFrame:
    seed_budgets = FMF_VALIDATION_C2_V0_POLICY_CONFIG.normalized_risk_budgets()
    rows: list[dict[str, object]] = [
        {
            "strategy_name": "rolling_c2_v0_seed_core",
            "policy_name": FMF_VALIDATION_C2_V0_POLICY_CONFIG.name,
            "equity_total": float(seed_budgets["equity_us"] + seed_budgets["equity_ex_us"]),
            **seed_budgets.to_dict(),
            "diversifier_total": float(seed_budgets["inflation_hedge"] + seed_budgets["trend"]),
            **{
                f"reserve_{key}": value
                for key, value in FMF_VALIDATION_C2_V0_POLICY_CONFIG.reserve_capital_series().to_dict().items()
            },
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
                **{
                    f"reserve_{key}": value
                    for key, value in policy_config.reserve_capital_series().to_dict().items()
                },
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


def write_fmf_c2_grid_search_artifacts(
    *,
    output_root: Path,
    bucket_returns: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    policy_configs: tuple[RiskBudgetPolicyConfig, ...],
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
        proxy_chain_result=None,
    )
    validation_summary = write_split_window_artifacts(
        output_root=output_root,
        strategy_runs=strategy_runs,
        window_name="validation_window",
        start_date=validation_start,
        end_date=validation_end,
    )
    policy_grid = build_policy_grid_frame(policy_configs)
    write_csv_artifact(output_root / "policy_grid.csv", policy_grid)

    summary_frame = pd.read_csv(output_root / "summary_metrics.csv")
    common_summary_frame = pd.read_csv(output_root / "summary_metrics_common_window.csv")
    validation_common_summary = pd.read_csv(output_root / "summary_metrics_validation_window_common_window.csv")
    candidate_names = {"rolling_c2_v0_seed_core"} | {f"rolling_{config.name}" for config in policy_configs}
    candidate_common_summary = (
        validation_common_summary.loc[validation_common_summary["strategy_name"].isin(candidate_names)]
        .sort_values(["sharpe", "annualized_return"], ascending=[False, False])
        .reset_index(drop=True)
    )
    write_csv_artifact(
        output_root / "summary_metrics_c2_candidates_validation_common_window.csv",
        candidate_common_summary,
    )

    write_json_artifact(
        output_root / "sweep_meta.json",
        {
            "experiment_family": "fmf_validation_c2_grid_search",
            "universe_name": FMF_VALIDATION_UNIVERSE_NAME,
            "validation_window": {
                "start_date": validation_start.date().isoformat(),
                "end_date": validation_end.date().isoformat(),
                "summary_rows": validation_summary.to_dict(orient="records"),
            },
            "lockbox_policy": {
                "test_window_locked": True,
                "test_window_exposed": False,
            },
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "candidate_count": int(len(policy_configs) + 1),
            "candidate_names": ["rolling_c2_v0_seed_core"] + [f"rolling_{config.name}" for config in policy_configs],
            "baseline_names": [
                "static_equal_weight_non_cash",
                "rolling_erc_core",
                "rolling_c2_v0_seed_core",
            ],
            "summary_rows": summary_frame.to_dict(orient="records"),
            "common_summary_rows": common_summary_frame.to_dict(orient="records"),
            "candidate_validation_common_window_rows": candidate_common_summary.to_dict(orient="records"),
        },
    )
    _write_sweep_report(
        output_root / "sweep_report.md",
        validation_common_summary=validation_common_summary,
        candidate_common_summary=candidate_common_summary,
    )


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


def _write_sweep_report(
    output_path: Path,
    *,
    validation_common_summary: pd.DataFrame,
    candidate_common_summary: pd.DataFrame,
) -> None:
    lines = ["# FMF Validation C2 Grid Search Summary", ""]
    if not candidate_common_summary.empty:
        best_candidate = candidate_common_summary.iloc[0]
        lines.extend(
            [
                "## Best Candidate On Validation Common Window",
                "",
                (
                    f"- `{best_candidate['strategy_name']}`: Sharpe `{best_candidate['sharpe']:.3f}`, "
                    f"annualized return `{best_candidate['annualized_return']:.3%}`, "
                    f"max drawdown `{best_candidate['max_drawdown']:.3%}`"
                ),
                "",
            ]
        )
    c2_seed = validation_common_summary.loc[
        validation_common_summary["strategy_name"] == "rolling_c2_v0_seed_core"
    ]
    if not c2_seed.empty:
        seed_row = c2_seed.iloc[0]
        lines.extend(
            [
                "## Seed Baseline",
                "",
                (
                    f"- `rolling_c2_v0_seed_core`: Sharpe `{seed_row['sharpe']:.3f}`, "
                    f"annualized return `{seed_row['annualized_return']:.3%}`, "
                    f"max drawdown `{seed_row['max_drawdown']:.3%}`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Lockbox",
            "",
            "- This sweep is validation-only.",
            "- The `2020-01-02 ~ 2026-04-08` test window remains locked and was not exposed in this run.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
    validation_start = pd.Timestamp(validation_bucket_returns.index.min())

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
    policy_configs = build_fmf_c2_region_policy_configs(
        equity_total_grid=parse_grid_argument(str(args.equity_total_grid)),
        credit_grid=parse_grid_argument(str(args.credit_grid)),
        duration_grid=parse_grid_argument(str(args.duration_grid)),
        inflation_grid=parse_grid_argument(str(args.inflation_grid)),
    )
    strategy_runs = run_fmf_c2_grid_search_suite(
        validation_bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        policy_configs=policy_configs,
    )
    write_fmf_c2_grid_search_artifacts(
        output_root=output_root,
        bucket_returns=validation_bucket_returns,
        symbol_returns=validation_symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
        validation_start=validation_start,
        validation_end=validation_end,
        policy_configs=policy_configs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
