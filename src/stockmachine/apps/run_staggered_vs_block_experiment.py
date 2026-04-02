from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.research_paths import resolve_research_cache_dir
from stockmachine.backtest import DailyOpenHoldBacktestEngine, DailyStaggeredOpenHoldBacktestEngine, DataFrameSignalModel
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy
from stockmachine.research.p1_rigor import build_strict_research_bundle, run_model_backtest_from_bundle
from stockmachine.research.us_equities_baseline import OverlayConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare block 5-day and daily staggered hold backtests.")
    parser.add_argument("--model", default="extra_trees")
    parser.add_argument("--predict-start", default="2018-01-01")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-root", default="artifacts/staggered_vs_block_experiment")
    parser.add_argument("--strategy-project", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=30_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_research_cache_dir(
        cache_dir=getattr(args, "cache_dir", None),
        default_dirname="p1_rigor_suite",
        strategy_project=getattr(args, "strategy_project", None),
        horizon=args.horizon,
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )

    overlay_config = OverlayConfig(
        min_close=args.min_close,
        min_median_dollar_volume_20=args.min_median_dollar_volume_20,
        max_vol_20=args.max_vol_20,
        max_positions_per_sector=args.max_positions_per_sector,
        cost_bps_per_side=args.cost_bps_per_side,
        sector_neutral=not args.disable_sector_neutral,
    )

    selected_predictions: pd.DataFrame
    block_dir = output_root / "block"

    if args.predictions_path not in (None, ""):
        bundle = None
        selected_predictions = pd.read_csv(args.predictions_path, parse_dates=["date"]).copy()
        if "model" in selected_predictions.columns:
            selected_predictions = selected_predictions[selected_predictions["model"] == args.model].copy()
        if selected_predictions.empty:
            raise RuntimeError(f"No predictions found for model '{args.model}' in '{args.predictions_path}'.")
        dataset = load_us_equities_dataset(layout=StorageLayout())
        block_summary, block_records = _run_block_backtest(
            predictions=selected_predictions,
            dataset=dataset,
            horizon=args.horizon,
            top_k=args.top_k,
            overlay_config=overlay_config,
        )
        block_dir.mkdir(parents=True, exist_ok=True)
        block_records.to_csv(block_dir / "backtest_records.csv", index=False)
        pd.DataFrame([block_summary]).to_csv(block_dir / "backtest_summary.csv", index=False)
        selected_predictions.to_csv(block_dir / "predictions.csv", index=False)
        block_payload = {
            "summary": block_summary,
            "records": block_records,
        }
    else:
        bundle = build_strict_research_bundle(
            predict_start=args.predict_start,
            horizon=args.horizon,
            strategy_project=getattr(args, "strategy_project", None),
            cache_dir=None if args.disable_cache else cache_dir,
            reuse_cache=not args.disable_cache,
            rebuild_cache=bool(args.rebuild_cache),
        )

        block_payload = run_model_backtest_from_bundle(
            bundle,
            model_name=args.model,
            top_k=args.top_k,
            overlay_config=overlay_config,
            output_dir=block_dir,
        )

        selected_predictions = bundle.predictions[bundle.predictions["model"] == args.model].copy()
        if selected_predictions.empty:
            raise RuntimeError(f"No predictions produced for model '{args.model}'.")
        dataset = bundle.dataset

    staggered_engine = DailyStaggeredOpenHoldBacktestEngine(
        predictions=selected_predictions,
        daily_bar=dataset["daily_bar"],
        benchmark_index=dataset["benchmark_index"],
        signal_model=DataFrameSignalModel(selected_predictions, horizon_bars=args.horizon),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(
            top_k=args.top_k,
            min_close=overlay_config.min_close,
            min_median_dollar_volume_20=overlay_config.min_median_dollar_volume_20,
            max_vol_20=overlay_config.max_vol_20,
            max_positions_per_sector=overlay_config.max_positions_per_sector,
            sector_neutral=overlay_config.sector_neutral,
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=args.horizon,
        cost_bps_per_side=overlay_config.cost_bps_per_side,
    )
    block_records = block_payload["records"]
    start_date = selected_predictions["date"].min().date()
    end_date = (
        pd.to_datetime(block_records["exit_date"].iloc[-1]).date()
        if isinstance(block_records, pd.DataFrame) and not block_records.empty
        else selected_predictions["date"].max().date()
    )
    staggered_result = staggered_engine.run(start_date=start_date, end_date=end_date)

    staggered_dir = output_root / "daily_staggered"
    staggered_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(staggered_result.meta.get("records", [])).to_csv(
        staggered_dir / "backtest_records.csv",
        index=False,
    )
    selected_predictions.to_csv(staggered_dir / "predictions.csv", index=False)
    staggered_summary = _build_summary_row(
        strategy="daily_staggered",
        result=staggered_result,
    )
    pd.DataFrame([staggered_summary]).to_csv(
        staggered_dir / "backtest_summary.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [
            _build_summary_row(strategy="block", result_payload=block_payload),
            staggered_summary,
        ]
    )
    comparison["model"] = args.model
    comparison["predict_start"] = args.predict_start
    comparison["horizon"] = args.horizon
    comparison["top_k"] = args.top_k
    comparison["min_close"] = overlay_config.min_close
    comparison["min_median_dollar_volume_20"] = overlay_config.min_median_dollar_volume_20
    comparison["max_vol_20"] = overlay_config.max_vol_20
    comparison["max_positions_per_sector"] = overlay_config.max_positions_per_sector
    comparison["cost_bps_per_side"] = overlay_config.cost_bps_per_side
    comparison["sector_neutral"] = overlay_config.sector_neutral
    comparison["excess_total_return"] = np.where(
        np.isfinite(comparison["total_return"]) & np.isfinite(comparison["benchmark_total_return"]),
        comparison["total_return"] - comparison["benchmark_total_return"],
        np.nan,
    )
    comparison.to_csv(output_root / "comparison_summary.csv", index=False)

    payload = {
        "ok": True,
        "model": args.model,
        "predict_start": args.predict_start,
        "horizon": args.horizon,
        "top_k": args.top_k,
        "output_root": str(output_root),
        "block_summary_path": str(block_dir / "backtest_summary.csv"),
        "staggered_summary_path": str(staggered_dir / "backtest_summary.csv"),
        "comparison_summary_path": str(output_root / "comparison_summary.csv"),
        "common_end_date": end_date.isoformat(),
        "cache": {
            "enabled": bool(args.predictions_path in (None, "")) and not args.disable_cache,
            "cache_dir": None if args.disable_cache or args.predictions_path not in (None, "") else args.cache_dir,
            "resolved_cache_dir": None if args.disable_cache or args.predictions_path not in (None, "") else str(cache_dir),
            "bundle_cache_hit": bool(getattr(bundle, "bundle_cache_hit", False)) if bundle is not None else False,
            "prediction_cache_hit": bool(getattr(bundle, "prediction_cache_hit", False)) if bundle is not None else False,
            "bundle_cache_key": getattr(bundle, "bundle_cache_key", None) if bundle is not None else None,
            "prediction_cache_key": getattr(bundle, "prediction_cache_key", None) if bundle is not None else None,
            "rebuild_cache": bool(args.rebuild_cache),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _build_summary_row(
    *,
    strategy: str,
    result: object | None = None,
    result_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    if result_payload is not None:
        summary = dict(result_payload["summary"])
        records = result_payload["records"]
        summary["strategy"] = strategy
        summary["mean_gross_exposure"] = (
            1.0 if isinstance(records, pd.DataFrame) and not records.empty else np.nan
        )
        summary["mean_cash_weight"] = 0.0 if isinstance(records, pd.DataFrame) and not records.empty else np.nan
        summary["mean_positions"] = float(records["positions"].mean()) if isinstance(records, pd.DataFrame) and not records.empty else np.nan
        summary["mean_active_cohorts"] = 1.0 if isinstance(records, pd.DataFrame) and not records.empty else np.nan
        return summary

    if result is None:
        raise ValueError("Either result or result_payload must be provided.")

    summary = {
        "strategy": strategy,
        "sessions": result.sessions,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "annualized_volatility": result.annualized_volatility,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "benchmark_total_return": result.meta.get("benchmark_total_return"),
        "mean_turnover": result.meta.get("mean_turnover"),
        "mean_cost_bps": result.meta.get("mean_cost_bps"),
        "mean_gross_exposure": result.meta.get("mean_gross_exposure"),
        "mean_cash_weight": result.meta.get("mean_cash_weight"),
        "mean_positions": result.meta.get("mean_positions"),
        "mean_active_cohorts": result.meta.get("mean_active_cohorts"),
    }
    return summary


def _run_block_backtest(
    *,
    predictions: pd.DataFrame,
    dataset: dict[str, pd.DataFrame],
    horizon: int,
    top_k: int,
    overlay_config: OverlayConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    engine = DailyOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=dataset["daily_bar"],
        benchmark_index=dataset["benchmark_index"],
        signal_model=DataFrameSignalModel(predictions, horizon_bars=horizon),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(
            top_k=top_k,
            min_close=overlay_config.min_close,
            min_median_dollar_volume_20=overlay_config.min_median_dollar_volume_20,
            max_vol_20=overlay_config.max_vol_20,
            max_positions_per_sector=overlay_config.max_positions_per_sector,
            sector_neutral=overlay_config.sector_neutral,
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=horizon,
        cost_bps_per_side=overlay_config.cost_bps_per_side,
    )
    start_date = predictions["date"].min().date()
    end_date = predictions["date"].max().date()
    result = engine.run(start_date=start_date, end_date=end_date)
    summary = {
        "sessions": result.sessions,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "annualized_volatility": result.annualized_volatility,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "benchmark_total_return": result.meta.get("benchmark_total_return"),
        "mean_turnover": result.meta.get("mean_turnover"),
        "mean_cost_bps": result.meta.get("mean_cost_bps"),
    }
    records = pd.DataFrame(result.meta.get("records", []))
    return summary, records


if __name__ == "__main__":
    raise SystemExit(main())
