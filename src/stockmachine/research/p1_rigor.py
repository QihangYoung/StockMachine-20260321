from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.backtest import DailyOpenHoldBacktestEngine, DataFrameSignalModel
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy
from stockmachine.research import get_default_research_protocol
from stockmachine.research.us_equities_baseline import (
    BENCHMARK_SYMBOL,
    OverlayConfig,
    build_point_in_time_metadata_history,
    build_price_panel_from_silver,
    build_research_frame,
    generate_walk_forward_predictions,
)

STRICT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "status",
    "predict_start",
    "top_k",
    "horizon",
    "artifacts_dir",
    "sessions",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "benchmark_total_return",
    "mean_turnover",
    "mean_cost_bps",
)


@dataclass(slots=True, frozen=True)
class StrictResearchBundle:
    """Reusable strict-research inputs shared across multiple model reruns."""

    predict_start: str
    horizon: int
    dataset: Mapping[str, pd.DataFrame]
    research_frame: pd.DataFrame
    predictions: pd.DataFrame


def build_strict_research_bundle(
    *,
    predict_start: str,
    horizon: int = 5,
    layout: StorageLayout | None = None,
) -> StrictResearchBundle:
    """Build one strict point-in-time research bundle once and reuse it."""

    storage = layout or StorageLayout()
    dataset = load_us_equities_dataset(layout=storage)
    price_data = build_price_panel_from_silver(dataset)
    metadata = build_point_in_time_metadata_history(
        pd.Index(price_data.loc[price_data["symbol"] != BENCHMARK_SYMBOL, "date"].drop_duplicates().sort_values()),
        symbol_master_frame=dataset["symbol_master"],
        industry_membership_frame=dataset["industry_membership"],
    )
    research_frame = build_research_frame(
        price_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        horizon=horizon,
        symbol_metadata=metadata,
    )
    predictions = generate_walk_forward_predictions(
        research_frame,
        predict_start=predict_start,
    )
    return StrictResearchBundle(
        predict_start=predict_start,
        horizon=horizon,
        dataset=dataset,
        research_frame=research_frame,
        predictions=predictions,
    )


def run_model_backtest_from_bundle(
    bundle: StrictResearchBundle,
    *,
    model_name: str,
    top_k: int = 10,
    overlay_config: OverlayConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one backtest from a precomputed strict bundle without retraining."""

    config = overlay_config or OverlayConfig()
    selected_predictions = bundle.predictions[bundle.predictions["model"] == model_name].copy()
    if selected_predictions.empty:
        raise RuntimeError(f"No predictions produced for model '{model_name}'.")

    engine = DailyOpenHoldBacktestEngine(
        predictions=selected_predictions,
        daily_bar=bundle.dataset["daily_bar"],
        benchmark_index=bundle.dataset["benchmark_index"],
        signal_model=DataFrameSignalModel(selected_predictions, horizon_bars=bundle.horizon),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(
            top_k=top_k,
            min_close=config.min_close,
            min_median_dollar_volume_20=config.min_median_dollar_volume_20,
            max_vol_20=config.max_vol_20,
            max_positions_per_sector=config.max_positions_per_sector,
            sector_neutral=config.sector_neutral,
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=bundle.horizon,
        cost_bps_per_side=config.cost_bps_per_side,
    )

    start_date = selected_predictions["date"].min().date()
    end_date = selected_predictions["date"].max().date()
    result = engine.run(start_date=start_date, end_date=end_date)
    summary = _build_summary_mapping(result)
    records = pd.DataFrame(result.meta.get("records", []))

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if not records.empty:
            records.to_csv(output_path / "backtest_records.csv", index=False)
        else:
            pd.DataFrame(columns=["signal_date", "entry_date", "exit_date", "gross_return", "net_return", "benchmark_return", "turnover", "cost_bps", "positions"]).to_csv(
                output_path / "backtest_records.csv",
                index=False,
            )
        pd.DataFrame([{**summary, "model": model_name}]).to_csv(output_path / "backtest_summary.csv", index=False)
        selected_predictions.to_csv(output_path / "predictions.csv", index=False)

    return {
        "model": model_name,
        "summary": summary,
        "records": records,
        "artifacts_dir": str(Path(output_dir)) if output_dir is not None else None,
    }


def run_strict_model_sweep_from_bundle(
    bundle: StrictResearchBundle,
    *,
    model_names: Sequence[str],
    output_root: str | Path,
    top_k: int = 10,
    overlay_config: OverlayConfig | None = None,
) -> dict[str, Any]:
    """Run one strict sweep over many models while reusing the same predictions."""

    config = overlay_config or OverlayConfig()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = get_default_research_protocol().to_dict()

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model_dir = root / _safe_component(model_name)
        try:
            result = run_model_backtest_from_bundle(
                bundle,
                model_name=model_name,
                top_k=top_k,
                overlay_config=config,
                output_dir=model_dir,
            )
            rows.append(
                {
                    "model": model_name,
                    "status": "success",
                    "predict_start": bundle.predict_start,
                    "top_k": top_k,
                    "horizon": bundle.horizon,
                    "artifacts_dir": str(model_dir),
                    **result["summary"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "predict_start": bundle.predict_start,
                    "top_k": top_k,
                    "horizon": bundle.horizon,
                    "artifacts_dir": str(model_dir),
                    "sessions": None,
                    "total_return": None,
                    "annualized_return": None,
                    "annualized_volatility": None,
                    "sharpe": None,
                    "max_drawdown": None,
                    "benchmark_total_return": None,
                    "mean_turnover": None,
                    "mean_cost_bps": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    summary_frame = pd.DataFrame(rows)
    for column in STRICT_SUMMARY_COLUMNS:
        if column not in summary_frame.columns:
            summary_frame[column] = None
    summary_frame = summary_frame[[*STRICT_SUMMARY_COLUMNS, *[column for column in summary_frame.columns if column not in STRICT_SUMMARY_COLUMNS]]]
    summary_path = root / "summary_metrics.csv"
    summary_frame.to_csv(summary_path, index=False)
    (root / "research_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "ok": bool((summary_frame["status"] == "success").all()) if not summary_frame.empty else False,
        "summary_metrics_path": str(summary_path),
        "research_protocol_path": str(root / "research_protocol.json"),
        "results": rows,
    }


def summarize_backtest_records(
    records: pd.DataFrame,
    *,
    horizon: int,
    cost_bps_per_side: float | None = None,
) -> dict[str, float]:
    """Summarize one backtest record frame under one cost assumption."""

    if records.empty:
        return {
            "sessions": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "benchmark_total_return": np.nan,
            "mean_turnover": np.nan,
            "mean_cost_bps": np.nan,
        }

    frame = records.copy()
    if cost_bps_per_side is None:
        net_returns = frame["net_return"].astype(float)
        mean_cost_bps = float(frame["cost_bps"].astype(float).mean())
    else:
        net_returns = frame["gross_return"].astype(float) - frame["turnover"].astype(float) * (cost_bps_per_side / 10000.0)
        mean_cost_bps = float((frame["turnover"].astype(float) * cost_bps_per_side).mean())

    benchmark_returns = frame["benchmark_return"].astype(float)
    total_return = float((1.0 + net_returns).prod() - 1.0)
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0)
    annualized_return = _annualize_total_return(total_return, len(frame), horizon)
    annualized_volatility = float(net_returns.std(ddof=1) * np.sqrt(252 / horizon)) if len(frame) > 1 else np.nan
    sharpe = (
        float((net_returns.mean() / net_returns.std(ddof=1)) * np.sqrt(252 / horizon))
        if len(frame) > 1 and net_returns.std(ddof=1) > 0
        else np.nan
    )
    equity_curve = (1.0 + net_returns).cumprod()
    peaks = equity_curve.cummax()
    max_drawdown = float((equity_curve / peaks - 1.0).min()) if not equity_curve.empty else np.nan

    return {
        "sessions": int(len(frame)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_total_return": benchmark_total_return,
        "mean_turnover": float(frame["turnover"].astype(float).mean()),
        "mean_cost_bps": mean_cost_bps,
    }


def build_period_stability_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    period: str,
) -> pd.DataFrame:
    """Build yearly or quarterly stability summaries from backtest records."""

    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "period",
                "period_label",
                "sessions",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "benchmark_total_return",
                "mean_turnover",
                "mean_cost_bps",
            ]
        )

    frame = records.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    if period == "year":
        periods = frame["entry_date"].dt.to_period("Y")
    elif period == "quarter":
        periods = frame["entry_date"].dt.to_period("Q")
    else:
        raise ValueError("period must be 'year' or 'quarter'.")

    rows: list[dict[str, Any]] = []
    for period_key, group in frame.groupby(periods, sort=True):
        summary = summarize_backtest_records(group, horizon=horizon)
        rows.append(
            {
                "model": model_name,
                "period": period,
                "period_label": str(period_key),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def build_cost_stress_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    cost_levels_bps: Sequence[float],
) -> pd.DataFrame:
    """Revalue one backtest record frame under multiple cost assumptions."""

    rows: list[dict[str, Any]] = []
    for cost_bps in cost_levels_bps:
        summary = summarize_backtest_records(records, horizon=horizon, cost_bps_per_side=float(cost_bps))
        rows.append(
            {
                "model": model_name,
                "cost_bps_per_side": float(cost_bps),
                "excess_total_return": summary["total_return"] - summary["benchmark_total_return"]
                if np.isfinite(summary["total_return"]) and np.isfinite(summary["benchmark_total_return"])
                else np.nan,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def run_topk_parameter_sweep_from_bundle(
    bundle: StrictResearchBundle,
    *,
    model_names: Sequence[str],
    top_k_values: Sequence[int],
    output_root: str | Path,
    overlay_config: OverlayConfig | None = None,
) -> pd.DataFrame:
    """Run a small top-k robustness sweep without retraining models."""

    config = overlay_config or OverlayConfig()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for top_k in top_k_values:
            run_dir = root / _safe_component(model_name) / f"top_k_{top_k}"
            result = run_model_backtest_from_bundle(
                bundle,
                model_name=model_name,
                top_k=int(top_k),
                overlay_config=config,
                output_dir=run_dir,
            )
            rows.append(
                {
                    "model": model_name,
                    "top_k": int(top_k),
                    "artifacts_dir": str(run_dir),
                    **result["summary"],
                    "excess_total_return": result["summary"]["total_return"] - result["summary"]["benchmark_total_return"]
                    if np.isfinite(result["summary"]["total_return"]) and np.isfinite(result["summary"]["benchmark_total_return"])
                    else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = root / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def _build_summary_mapping(result: Any) -> dict[str, float]:
    return {
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


def _safe_component(value: str) -> str:
    cleaned = str(value).strip().replace("\\", "_").replace("/", "_")
    return cleaned or "model"


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)
