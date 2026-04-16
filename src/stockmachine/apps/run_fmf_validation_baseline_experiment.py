from __future__ import annotations

import argparse
import json
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
    FMF_VALIDATION_BUCKETS,
    FMF_VALIDATION_C2_V0_POLICY_CONFIG,
    FMF_VALIDATION_ERC_POLICY_CONFIG,
    CovarianceConfig,
    RollingAllocationConfig,
    StrategyRunArtifacts,
    build_bucket_return_frame,
    build_calendar_year_return_summary,
    build_risk_share_summary_frame,
    run_constant_weight_backtest,
    run_rolling_risk_budget_backtest,
    summarize_multi_asset_backtest_records,
)
from stockmachine.research.strict_reports import (
    write_csv_artifact,
    write_json_artifact,
    write_summary_metrics_artifact,
)


FMF_VALIDATION_UNIVERSE_NAME = "us_multi_asset_fmf_validation_v1"
DEFAULT_VALIDATION_END_DATE = "2019-12-31"
DEFAULT_TEST_START_DATE = "2020-01-02"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rebuilt FMF-based multi-asset baseline experiments."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/fmf_validation_baseline_experiment",
        help="Artifact directory for experiment outputs.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive return start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive return end date.")
    parser.add_argument(
        "--validation-end-date",
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive validation-window end date.",
    )
    parser.add_argument(
        "--test-start-date",
        default=DEFAULT_TEST_START_DATE,
        help="Inclusive test-window start date.",
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
        "--include-test-window",
        action="store_true",
        help="Explicitly unlock and write test-window metrics. Leave off for routine validation-only runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_end = pd.Timestamp(args.validation_end_date)
    test_start = pd.Timestamp(args.test_start_date)
    if validation_end >= test_start:
        raise ValueError("validation_end_date must fall before test_start_date.")

    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    required_symbols = tuple(
        dict.fromkeys(symbol for bucket in FMF_VALIDATION_BUCKETS for symbol in bucket.symbols)
    )
    symbol_returns = load_symbol_returns_for_symbols(layout=layout, symbols=required_symbols)
    _require_return_columns(
        symbol_returns,
        required_columns=required_symbols,
        context="FMF validation universe",
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

    validation_bucket_returns = bucket_returns.loc[bucket_returns.index <= validation_end].copy()
    if validation_bucket_returns.empty:
        raise ValueError("No validation rows remain after applying validation_end_date.")
    validation_symbol_returns = symbol_returns.loc[validation_bucket_returns.index].copy()

    if args.include_test_window:
        experiment_bucket_returns = bucket_returns
        experiment_symbol_returns = symbol_returns
    else:
        experiment_bucket_returns = validation_bucket_returns
        experiment_symbol_returns = validation_symbol_returns

    strategy_runs = run_fmf_validation_baseline_suite(
        experiment_bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column=str(args.benchmark_column),
        cost_bps_per_side=float(args.cost_bps_per_side),
    )
    write_experiment_artifacts(
        output_root=output_root,
        bucket_returns=experiment_bucket_returns,
        symbol_returns=experiment_symbol_returns,
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
    test_summary: pd.DataFrame | None = None
    if args.include_test_window:
        test_summary = write_split_window_artifacts(
            output_root=output_root,
            strategy_runs=strategy_runs,
            window_name="test_window",
            start_date=test_start,
            end_date=pd.Timestamp(bucket_returns.index.max()),
        )
    _write_split_report(
        output_root / "split_report.md",
        validation_summary=validation_summary,
        validation_start=pd.Timestamp(validation_bucket_returns.index.min()),
        validation_end=validation_end,
        test_summary=test_summary,
        test_start=test_start,
        data_end=pd.Timestamp(bucket_returns.index.max()),
        include_test_window=bool(args.include_test_window),
    )
    _augment_run_meta(
        path=output_root / "run_meta.json",
        bucket_returns=experiment_bucket_returns,
        symbol_returns=experiment_symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        validation_start=pd.Timestamp(validation_bucket_returns.index.min()),
        validation_end=validation_end,
        test_start=test_start,
        validation_summary=validation_summary,
        test_summary=test_summary,
        include_test_window=bool(args.include_test_window),
        test_end=pd.Timestamp(bucket_returns.index.max()),
    )
    return 0


def run_fmf_validation_baseline_suite(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
) -> tuple[StrategyRunArtifacts, ...]:
    strategy_runs: list[StrategyRunArtifacts] = []

    static_equal = run_constant_weight_backtest(
        bucket_returns,
        weights=FMF_VALIDATION_ERC_POLICY_CONFIG.normalized_risk_budgets(),
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="static_equal_weight_non_cash",
            records=static_equal.records,
            summary=static_equal.summary,
            weight_schedule=static_equal.weight_schedule,
            diagnostics=None,
        )
    )

    rolling_erc_allocations, rolling_erc_backtest = run_rolling_risk_budget_backtest(
        bucket_returns,
        covariance_config=covariance_config,
        policy_config=FMF_VALIDATION_ERC_POLICY_CONFIG,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="rolling_erc_core",
            records=rolling_erc_backtest.records,
            summary=rolling_erc_backtest.summary,
            weight_schedule=rolling_erc_allocations.weight_schedule,
            diagnostics=rolling_erc_allocations.diagnostics,
        )
    )

    rolling_c2_allocations, rolling_c2_backtest = run_rolling_risk_budget_backtest(
        bucket_returns,
        covariance_config=covariance_config,
        policy_config=FMF_VALIDATION_C2_V0_POLICY_CONFIG,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="rolling_c2_v0_seed_core",
            records=rolling_c2_backtest.records,
            summary=rolling_c2_backtest.summary,
            weight_schedule=rolling_c2_allocations.weight_schedule,
            diagnostics=rolling_c2_allocations.diagnostics,
        )
    )
    return tuple(strategy_runs)


def write_split_window_artifacts(
    *,
    output_root: Path,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
    window_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    yearly_frames: list[pd.DataFrame] = []
    risk_share_frames: list[pd.DataFrame] = []
    split_runs: list[StrategyRunArtifacts] = []

    for strategy_run in strategy_runs:
        split_records = _slice_records_to_window(
            strategy_run.records,
            start_date=start_date,
            end_date=end_date,
        )
        split_runs.append(
            StrategyRunArtifacts(
                strategy_name=strategy_run.strategy_name,
                records=split_records,
                summary=summarize_multi_asset_backtest_records(split_records, horizon=1),
                weight_schedule=strategy_run.weight_schedule,
                diagnostics=(
                    _slice_effective_frame_to_window(
                        strategy_run.diagnostics,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if strategy_run.diagnostics is not None
                    else None
                ),
            )
        )
        summary_rows.append({"strategy_name": strategy_run.strategy_name, **split_runs[-1].summary})
        yearly_frames.append(
            build_calendar_year_return_summary(
                split_records,
                strategy_name=strategy_run.strategy_name,
            )
        )
        if split_runs[-1].diagnostics is not None:
            risk_share_frames.append(
                build_risk_share_summary_frame(
                    split_runs[-1].diagnostics,
                    strategy_name=strategy_run.strategy_name,
                )
            )

    summary_frame = write_summary_metrics_artifact(
        output_root / f"summary_metrics_{window_name}.csv",
        summary_rows,
    )
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    risk_share_summary = pd.concat(risk_share_frames, ignore_index=True) if risk_share_frames else pd.DataFrame()
    write_csv_artifact(output_root / f"yearly_returns_{window_name}.csv", yearly_summary)
    write_csv_artifact(output_root / f"risk_share_summary_{window_name}.csv", risk_share_summary)

    common_window = _resolve_common_window(tuple(split_runs))
    common_summary_rows: list[dict[str, object]] = []
    if common_window is not None:
        common_start, common_end = common_window
        for split_run in split_runs:
            common_records = _slice_records_to_window(
                split_run.records,
                start_date=common_start,
                end_date=common_end,
            )
            common_summary_rows.append(
                {
                    "strategy_name": split_run.strategy_name,
                    **summarize_multi_asset_backtest_records(common_records, horizon=1),
                }
            )
    write_summary_metrics_artifact(
        output_root / f"summary_metrics_{window_name}_common_window.csv",
        common_summary_rows,
    )
    return summary_frame


def _augment_run_meta(
    *,
    path: Path,
    bucket_returns: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    validation_summary: pd.DataFrame,
    test_summary: pd.DataFrame | None,
    include_test_window: bool,
    test_end: pd.Timestamp,
) -> None:
    existing: dict[str, object] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.pop("current_c_benchmark_symbol", None)
    existing.update(
        {
            "experiment_family": "fmf_validation_baseline",
            "universe_name": FMF_VALIDATION_UNIVERSE_NAME,
            "reference_benchmark_symbol": "SPY",
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "bucket_names": list(bucket_returns.columns),
            "symbol_names": list(symbol_returns.columns),
            "bucket_symbol_mapping": {
                bucket.name: list(bucket.symbols) for bucket in FMF_VALIDATION_BUCKETS
            },
            "lockbox_policy": {
                "include_test_window": bool(include_test_window),
                "test_window_locked": not bool(include_test_window),
            },
            "validation_window": {
                "start_date": validation_start.date().isoformat(),
                "end_date": validation_end.date().isoformat(),
                "summary_rows": validation_summary.to_dict(orient="records"),
            },
            "test_window": {
                "start_date": test_start.date().isoformat(),
                "end_date": test_end.date().isoformat(),
                "summary_rows": (
                    test_summary.to_dict(orient="records")
                    if test_summary is not None
                    else []
                ),
            },
        }
    )
    write_json_artifact(path, existing)


def _require_return_columns(
    returns: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    context: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in returns.columns]
    if missing_columns:
        available_preview = ", ".join(list(map(str, returns.columns[:10]))) or "(none)"
        raise ValueError(
            f"Missing required symbols for {context}: {missing_columns}. "
            f"Available columns start with: {available_preview}"
        )


def _resolve_common_window(
    strategy_runs: tuple[StrategyRunArtifacts, ...],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for strategy_run in strategy_runs:
        if strategy_run.records.empty:
            continue
        record_dates = pd.to_datetime(strategy_run.records["entry_date"], utc=False)
        windows.append((record_dates.min(), record_dates.max()))
    if not windows:
        return None
    common_start = max(start for start, _ in windows)
    common_end = min(end for _, end in windows)
    if common_end < common_start:
        return None
    return common_start, common_end


def _slice_records_to_window(
    records: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    frame = records.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], utc=False)
    return frame.loc[
        (frame["entry_date"] >= start_date) & (frame["entry_date"] <= end_date)
    ].reset_index(drop=True)


def _slice_effective_frame_to_window(
    frame: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index, utc=False)
    return normalized.loc[
        (normalized.index >= start_date) & (normalized.index <= end_date)
    ]


def _write_split_report(
    path: Path,
    *,
    validation_summary: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_summary: pd.DataFrame | None,
    test_start: pd.Timestamp,
    data_end: pd.Timestamp,
    include_test_window: bool,
) -> None:
    lines = [
        "# FMF Validation Baseline Split Summary",
        "",
        f"- Validation window: `{validation_start.date().isoformat()}` to `{validation_end.date().isoformat()}`",
        f"- Test window: `{test_start.date().isoformat()}` to `{data_end.date().isoformat()}`",
        f"- Test window exposed in this run: `{str(bool(include_test_window)).lower()}`",
        "",
    ]
    if not validation_summary.empty:
        best_validation = validation_summary.sort_values(
            ["sharpe", "annualized_return"],
            ascending=[False, False],
        ).iloc[0]
        lines.extend(
            [
                "## Validation",
                "",
                f"- Best Sharpe: `{best_validation['strategy_name']}` "
                f"(Sharpe `{best_validation['sharpe']:.3f}`, annualized return "
                f"`{best_validation['annualized_return']:.3%}`, max drawdown "
                f"`{best_validation['max_drawdown']:.3%}`)",
                "",
            ]
        )
    if test_summary is not None and not test_summary.empty:
        best_test = test_summary.sort_values(
            ["sharpe", "annualized_return"],
            ascending=[False, False],
        ).iloc[0]
        lines.extend(
            [
                "## Test",
                "",
                f"- Best Sharpe: `{best_test['strategy_name']}` "
                f"(Sharpe `{best_test['sharpe']:.3f}`, annualized return "
                f"`{best_test['annualized_return']:.3%}`, max drawdown "
                f"`{best_test['max_drawdown']:.3%}`)",
                "",
            ]
        )
    elif not include_test_window:
        lines.extend(
            [
                "## Test",
                "",
                "- Locked. This routine run did not expose any test-window metrics.",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- This rebuilt baseline experiment uses the FMF long-window ETF-only universe.",
            "- Validation is for robust-region selection; test is the untouched modern out-of-sample window.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
