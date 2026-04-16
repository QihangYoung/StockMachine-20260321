from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockmachine.data.loaders.silver import load_silver_table
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.multi_asset import (
    DEFAULT_C2_V0_POLICY_CONFIG,
    DEFAULT_ERC_POLICY_CONFIG,
    DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS,
    DEFAULT_MULTI_ASSET_BUCKETS,
    DEFAULT_MULTI_ASSET_PROXY_CHAIN,
    DEFAULT_MULTI_ASSET_PROXY_SYMBOLS,
    CovarianceConfig,
    ProxyChainResult,
    RollingAllocationConfig,
    StrategyRunArtifacts,
    apply_affine_proxy_chain,
    build_bucket_return_frame,
    build_calendar_year_return_summary,
    build_risk_share_summary_frame,
    build_symbol_return_frame,
    build_weight_summary_frame,
    run_constant_weight_backtest,
    run_periodic_constant_weight_backtest,
    run_rolling_risk_budget_backtest,
    summarize_multi_asset_backtest_records,
)
from stockmachine.research.strict_reports import (
    write_csv_artifact,
    write_json_artifact,
    write_summary_metrics_artifact,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the first multi-asset core experiment over bucket-level ETF returns."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/multi_asset_core_experiment",
        help="Artifact directory for experiment outputs.",
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
        "--use-proxy-chain",
        action="store_true",
        help="Extend CTA/GLDM/SGOV histories with the default affine proxy chain.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    symbol_returns = load_symbol_returns_for_symbols(
        layout=layout,
        symbols=tuple(
            dict.fromkeys(
                list(DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS.index)
                + [symbol for bucket in DEFAULT_MULTI_ASSET_BUCKETS for symbol in bucket.symbols]
                + (list(DEFAULT_MULTI_ASSET_PROXY_SYMBOLS) if args.use_proxy_chain else [])
            )
        ),
    )
    proxy_chain_result: ProxyChainResult | None = None
    if args.use_proxy_chain:
        proxy_chain_result = apply_affine_proxy_chain(
            symbol_returns,
            DEFAULT_MULTI_ASSET_PROXY_CHAIN,
        )
        symbol_returns = proxy_chain_result.symbol_returns

    required_bucket_symbols = tuple(
        dict.fromkeys(symbol for bucket in DEFAULT_MULTI_ASSET_BUCKETS for symbol in bucket.symbols)
    )
    _require_return_columns(
        symbol_returns,
        required_columns=tuple(DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS.index),
        context="current C policy baseline",
    )
    _require_return_columns(
        symbol_returns,
        required_columns=required_bucket_symbols,
        context="bucket return construction",
    )
    required_output_symbols = tuple(
        dict.fromkeys(list(DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS.index) + list(required_bucket_symbols))
    )
    symbol_returns = symbol_returns.loc[:, list(required_output_symbols)]
    bucket_returns = build_bucket_return_frame(
        symbol_returns.loc[:, list(required_bucket_symbols)],
        DEFAULT_MULTI_ASSET_BUCKETS,
    )
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
        raise ValueError("No common dates remain between bucket returns and current-C symbol returns.")
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

    strategy_runs = run_default_experiment_suite(
        bucket_returns,
        symbol_returns=symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
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
        proxy_chain_result=proxy_chain_result,
    )
    return 0


def load_symbol_returns_for_symbols(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    daily_bar = load_silver_table("daily_bar", layout=layout)
    benchmark_index = load_silver_table("benchmark_index", layout=layout)
    adj_factor = load_silver_table("adj_factor", layout=layout)

    combined_daily_bar = pd.concat([daily_bar, benchmark_index], ignore_index=True)
    if combined_daily_bar.empty:
        raise ValueError("No daily_bar or benchmark_index rows were found in silver storage.")

    required_symbols = tuple(dict.fromkeys(str(symbol) for symbol in symbols))
    combined_daily_bar = combined_daily_bar.loc[combined_daily_bar["symbol"].astype(str).isin(required_symbols)].copy()
    combined_daily_bar = combined_daily_bar.drop_duplicates(
        subset=["session_date", "symbol"],
        keep="last",
    )
    if not adj_factor.empty and "symbol" in adj_factor.columns:
        adj_factor = adj_factor.loc[adj_factor["symbol"].astype(str).isin(required_symbols)].copy()
    if combined_daily_bar.empty:
        raise ValueError(f"Silver price history is missing required symbols: {required_symbols}.")

    return build_symbol_return_frame(combined_daily_bar, adj_factor=adj_factor)


def slice_return_frame(
    returns: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = returns.copy()
    if start_date:
        frame = frame.loc[frame.index >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame.loc[frame.index <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("No return rows remain after date filtering.")
    return frame


def run_default_experiment_suite(
    bucket_returns: pd.DataFrame,
    *,
    symbol_returns: pd.DataFrame,
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
) -> tuple[StrategyRunArtifacts, ...]:
    strategy_runs: list[StrategyRunArtifacts] = []

    current_c_baseline = run_constant_weight_backtest(
        symbol_returns,
        weights=DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS,
        benchmark_column="SPY",
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="current_c_policy_core",
            records=current_c_baseline.records,
            summary=current_c_baseline.summary,
            weight_schedule=current_c_baseline.weight_schedule,
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

    static_erc = run_constant_weight_backtest(
        bucket_returns,
        weights=DEFAULT_ERC_POLICY_CONFIG.normalized_risk_budgets(),
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="static_equal_weight_non_cash",
            records=static_erc.records,
            summary=static_erc.summary,
            weight_schedule=static_erc.weight_schedule,
            diagnostics=None,
        )
    )

    rolling_erc_allocations, rolling_erc_backtest = run_rolling_risk_budget_backtest(
        bucket_returns,
        covariance_config=covariance_config,
        policy_config=DEFAULT_ERC_POLICY_CONFIG,
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
        policy_config=DEFAULT_C2_V0_POLICY_CONFIG,
        rolling_config=rolling_config,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    strategy_runs.append(
        StrategyRunArtifacts(
            strategy_name="rolling_c2_v0_core",
            records=rolling_c2_backtest.records,
            summary=rolling_c2_backtest.summary,
            weight_schedule=rolling_c2_allocations.weight_schedule,
            diagnostics=rolling_c2_allocations.diagnostics,
        )
    )
    return tuple(strategy_runs)


def write_experiment_artifacts(
    *,
    output_root: Path,
    bucket_returns: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    strategy_runs: tuple[StrategyRunArtifacts, ...],
    covariance_config: CovarianceConfig,
    rolling_config: RollingAllocationConfig,
    benchmark_column: str,
    cost_bps_per_side: float,
    proxy_chain_result: ProxyChainResult | None = None,
) -> None:
    summary_rows = []
    yearly_frames = []
    weight_summary_frames = []
    risk_share_frames = []

    for strategy_run in strategy_runs:
        strategy_dir = output_root / strategy_run.strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        summary_rows.append({"strategy_name": strategy_run.strategy_name, **strategy_run.summary})
        yearly_frames.append(
            build_calendar_year_return_summary(
                strategy_run.records,
                strategy_name=strategy_run.strategy_name,
            )
        )
        weight_summary_frames.append(
            build_weight_summary_frame(
                strategy_run.weight_schedule,
                strategy_name=strategy_run.strategy_name,
            )
        )
        if strategy_run.diagnostics is not None:
            risk_share_frames.append(
                build_risk_share_summary_frame(
                    strategy_run.diagnostics,
                    strategy_name=strategy_run.strategy_name,
                )
            )
            write_csv_artifact(strategy_dir / "risk_share_diagnostics.csv", strategy_run.diagnostics)
        write_csv_artifact(strategy_dir / "backtest_records.csv", strategy_run.records)
        write_csv_artifact(strategy_dir / "weight_schedule.csv", strategy_run.weight_schedule.reset_index())
        write_json_artifact(strategy_dir / "summary.json", strategy_run.summary)

    summary_frame = write_summary_metrics_artifact(output_root / "summary_metrics.csv", summary_rows)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    weight_summary = pd.concat(weight_summary_frames, ignore_index=True) if weight_summary_frames else pd.DataFrame()
    risk_share_summary = pd.concat(risk_share_frames, ignore_index=True) if risk_share_frames else pd.DataFrame()
    common_window = _resolve_common_window(strategy_runs)
    common_summary_rows: list[dict[str, object]] = []
    common_yearly_frames: list[pd.DataFrame] = []
    common_risk_share_frames: list[pd.DataFrame] = []

    write_csv_artifact(output_root / "bucket_returns.csv", bucket_returns.reset_index().rename(columns={"index": "date"}))
    write_csv_artifact(output_root / "symbol_returns.csv", symbol_returns.reset_index().rename(columns={"index": "date"}))
    write_csv_artifact(output_root / "yearly_returns.csv", yearly_summary)
    write_csv_artifact(output_root / "weight_summary.csv", weight_summary)
    write_csv_artifact(output_root / "risk_share_summary.csv", risk_share_summary)
    if proxy_chain_result is not None:
        write_csv_artifact(output_root / "proxy_chain_diagnostics.csv", proxy_chain_result.diagnostics)
    if common_window is not None:
        common_start, common_end = common_window
        for strategy_run in strategy_runs:
            common_records = _slice_records_to_window(
                strategy_run.records,
                start_date=common_start,
                end_date=common_end,
            )
            common_summary_rows.append(
                {
                    "strategy_name": strategy_run.strategy_name,
                    **summarize_multi_asset_backtest_records(common_records, horizon=1),
                }
            )
            common_yearly_frames.append(
                build_calendar_year_return_summary(
                    common_records,
                    strategy_name=strategy_run.strategy_name,
                )
            )
            if strategy_run.diagnostics is not None:
                common_diagnostics = _slice_effective_frame_to_window(
                    strategy_run.diagnostics,
                    start_date=common_start,
                    end_date=common_end,
                )
                common_risk_share_frames.append(
                    build_risk_share_summary_frame(
                        common_diagnostics,
                        strategy_name=strategy_run.strategy_name,
                    )
                )

    common_summary = write_summary_metrics_artifact(
        output_root / "summary_metrics_common_window.csv",
        common_summary_rows,
    )
    common_yearly_summary = (
        pd.concat(common_yearly_frames, ignore_index=True) if common_yearly_frames else pd.DataFrame()
    )
    common_risk_share_summary = (
        pd.concat(common_risk_share_frames, ignore_index=True) if common_risk_share_frames else pd.DataFrame()
    )
    write_csv_artifact(output_root / "yearly_returns_common_window.csv", common_yearly_summary)
    write_csv_artifact(output_root / "risk_share_summary_common_window.csv", common_risk_share_summary)
    _write_summary_report(
        output_root / "summary_report.md",
        summary_frame=summary_frame,
        common_summary_frame=common_summary,
        common_window=common_window,
    )
    write_json_artifact(
        output_root / "run_meta.json",
        {
            "benchmark_column": benchmark_column,
            "current_c_benchmark_symbol": "SPY",
            "cost_bps_per_side": cost_bps_per_side,
            "covariance_config": asdict(covariance_config),
            "rolling_config": asdict(rolling_config),
            "bucket_names": list(bucket_returns.columns),
            "symbol_names": list(symbol_returns.columns),
            "proxy_chain_enabled": proxy_chain_result is not None,
            "proxy_chain_segments": (
                proxy_chain_result.diagnostics.to_dict(orient="records")
                if proxy_chain_result is not None
                else []
            ),
            "strategies": [strategy_run.strategy_name for strategy_run in strategy_runs],
            "summary_rows": summary_frame.to_dict(orient="records"),
            "common_window": (
                {
                    "start_date": common_window[0].date().isoformat(),
                    "end_date": common_window[1].date().isoformat(),
                    "summary_rows": common_summary.to_dict(orient="records"),
                }
                if common_window is not None
                else None
            ),
        },
    )


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
    return frame.loc[(frame["entry_date"] >= start_date) & (frame["entry_date"] <= end_date)].reset_index(drop=True)


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
    return normalized.loc[(normalized.index >= start_date) & (normalized.index <= end_date)]


def _write_summary_report(
    path: Path,
    *,
    summary_frame: pd.DataFrame,
    common_summary_frame: pd.DataFrame,
    common_window: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> None:
    lines = ["# Multi-Asset Core Experiment Summary", ""]
    if not summary_frame.empty:
        best_full = summary_frame.sort_values(["sharpe", "annualized_return"], ascending=[False, False]).iloc[0]
        lines.extend(
            [
                "## Full Window",
                "",
                f"- Best Sharpe: `{best_full['strategy_name']}` "
                f"(Sharpe `{best_full['sharpe']:.3f}`, annualized return `{best_full['annualized_return']:.3%}`, "
                f"max drawdown `{best_full['max_drawdown']:.3%}`)",
                "",
            ]
        )
    if common_window is not None and not common_summary_frame.empty:
        best_common = common_summary_frame.sort_values(
            ["sharpe", "annualized_return"],
            ascending=[False, False],
        ).iloc[0]
        lines.extend(
            [
                "## Common Window",
                "",
                f"- Window: `{common_window[0].date().isoformat()}` to `{common_window[1].date().isoformat()}`",
                f"- Best Sharpe: `{best_common['strategy_name']}` "
                f"(Sharpe `{best_common['sharpe']:.3f}`, annualized return `{best_common['annualized_return']:.3%}`, "
                f"max drawdown `{best_common['max_drawdown']:.3%}`)",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- `summary_metrics.csv` keeps each strategy's full available sample.",
            "- `summary_metrics_common_window.csv` aligns all strategies to the latest common start date for fair comparison.",
            "- Rolling strategies naturally start later because they require warm-up history before the first live allocation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
