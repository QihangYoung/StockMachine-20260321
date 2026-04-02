from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from stockmachine.backtest import DataFrameSignalModel, DailyRebalanceOpenHoldBacktestEngine
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy
from stockmachine.research.builders import build_extra_trees_model, build_hist_gbm_model, prepare_model_frame
from stockmachine.research.builders.common import build_linear_model_pipeline
from stockmachine.research.p1_rigor import build_cost_stress_summary, build_period_stability_summary
from stockmachine.research.protocols import get_h1_research_protocol
from stockmachine.research.splitting import WalkForwardSplit, WalkForwardSplitConfig, build_walk_forward_splits
from stockmachine.research.universe import (
    DEFAULT_RESEARCH_UNIVERSE_NAME,
    build_point_in_time_metadata_history,
)
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, OverlayConfig, build_price_panel_from_silver

H1_FEATURE_COLUMNS: tuple[str, ...] = (
    "gap_1",
    "gap_z_20",
    "intraday_return",
    "ret_1d",
    "ret_2d",
    "mom_3",
    "range_1d",
    "range_5",
    "vol_5",
    "vol_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "rel_ret_1d",
    "rel_mom_3",
    "sector_rel_ret_1d",
    "sector_rel_mom_3",
)

H1_BASE_MODEL_NAMES: tuple[str, ...] = ("ridge", "hist_gbm", "extra_trees")
TRADING_DAYS_PER_MONTH = 21


@dataclass(slots=True, frozen=True)
class H1TurnoverControlConfig:
    no_trade_band: float = 0.05
    max_turnover: float = 1.0
    min_weight_change: float = 0.02
    hold_rank_buffer: int = 2
    entry_rank_buffer: int = 2
    max_new_names_per_rebalance: int = 2


@dataclass(slots=True, frozen=True)
class H1PromotionGateConfig:
    min_sharpe: float = 0.5
    min_annualized_return_at_20bps: float = 0.0
    max_drawdown_floor: float = -0.35
    max_mean_turnover: float = 1.25
    min_positive_year_ratio: float = 0.5


def build_h1_research_frame(
    price_data: pd.DataFrame,
    *,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
    symbol_metadata: pd.DataFrame | None = None,
    drop_unlabeled_rows: bool = True,
) -> pd.DataFrame:
    price_data = _ensure_adjusted_price_columns(price_data.copy())
    benchmark = (
        price_data.loc[price_data["symbol"] == benchmark_symbol, ["date", "open", "close", "adj_open"]]
        .rename(columns={"open": "benchmark_open", "close": "benchmark_close", "adj_open": "benchmark_adj_open"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    benchmark["benchmark_prev_close"] = benchmark["benchmark_close"].shift(1)
    benchmark["benchmark_ret_1d"] = benchmark["benchmark_close"].pct_change(1, fill_method=None)
    benchmark["benchmark_mom_3"] = benchmark["benchmark_close"].pct_change(3, fill_method=None)
    benchmark["benchmark_gap_1"] = benchmark["benchmark_open"] / benchmark["benchmark_prev_close"] - 1.0
    benchmark["benchmark_future_return"] = (
        benchmark["benchmark_adj_open"].shift(-2) / benchmark["benchmark_adj_open"].shift(-1) - 1.0
    )

    panel = price_data.loc[price_data["symbol"] != benchmark_symbol].copy()
    panel = panel.merge(
        benchmark[["date", "benchmark_ret_1d", "benchmark_mom_3", "benchmark_gap_1", "benchmark_future_return"]],
        on="date",
        how="left",
    )

    group = panel.groupby("symbol", group_keys=False)
    panel["prev_close"] = group["close"].shift(1)
    panel["ret_1d"] = group["close"].pct_change(1, fill_method=None)
    panel["ret_2d"] = group["close"].pct_change(2, fill_method=None)
    panel["mom_3"] = group["close"].pct_change(3, fill_method=None)
    panel["intraday_return"] = panel["close"] / panel["open"] - 1.0
    panel["range_1d"] = panel["high"] / panel["low"] - 1.0
    panel["range_5"] = group["range_1d"].rolling(5).mean().reset_index(level=0, drop=True)
    panel["gap_1"] = panel["open"] / panel["prev_close"] - 1.0
    gap_mean_20 = group["gap_1"].rolling(20).mean().reset_index(level=0, drop=True)
    gap_std_20 = group["gap_1"].rolling(20).std().reset_index(level=0, drop=True)
    panel["gap_z_20"] = ((panel["gap_1"] - gap_mean_20) / gap_std_20.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    panel["gap_z_20"] = panel["gap_z_20"].fillna(0.0)
    panel["vol_5"] = group["ret_1d"].rolling(5).std().reset_index(level=0, drop=True)
    panel["vol_20"] = group["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["median_dollar_volume_20"] = (
        group["dollar_volume"].rolling(20).median().reset_index(level=0, drop=True)
    )
    panel["volume_ratio_5"] = panel["volume"] / group["volume"].rolling(5).mean().reset_index(level=0, drop=True)
    panel["volume_ratio_20"] = panel["volume"] / group["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    panel["rel_ret_1d"] = panel["ret_1d"] - panel["benchmark_ret_1d"]
    panel["rel_mom_3"] = panel["mom_3"] - panel["benchmark_mom_3"]
    panel["future_return"] = group["adj_open"].shift(-2) / group["adj_open"].shift(-1) - 1.0
    panel["target"] = panel["future_return"] - panel["benchmark_future_return"]

    if symbol_metadata is not None:
        merge_keys = ["symbol"]
        if "date" in symbol_metadata.columns:
            merge_keys = ["date", "symbol"]
        panel = panel.merge(symbol_metadata, on=merge_keys, how="left", indicator="_metadata_match")
        if merge_keys == ["date", "symbol"]:
            panel = panel.loc[panel["_metadata_match"] == "both"].copy()
        panel = panel.drop(columns="_metadata_match")
        panel["sector"] = panel["sector"].fillna("Unknown")
        panel["industry"] = panel["industry"].fillna("Unknown")
    else:
        panel["sector"] = "Unknown"
        panel["industry"] = "Unknown"

    sector_group = panel.groupby(["date", "sector"], group_keys=False)
    panel["sector_rel_ret_1d"] = panel["ret_1d"] - sector_group["ret_1d"].transform("mean")
    panel["sector_rel_mom_3"] = panel["mom_3"] - sector_group["mom_3"].transform("mean")

    required_columns = list(H1_FEATURE_COLUMNS)
    if drop_unlabeled_rows:
        required_columns += ["future_return", "target"]
    panel = panel.dropna(subset=required_columns).reset_index(drop=True)
    return panel


def get_h1_model_builders() -> Mapping[str, Callable[[], object]]:
    return {
        "ridge": lambda: build_linear_model_pipeline(Ridge(alpha=1.0), feature_columns=H1_FEATURE_COLUMNS),
        "hist_gbm": build_hist_gbm_model,
        "extra_trees": build_extra_trees_model,
    }


def build_h1_walk_forward_split_config(
    *,
    train_window_days: int | None = None,
    validation_window_days: int | None = None,
    test_window_days: int | None = None,
    purge_window_days: int | None = None,
    embargo_window_days: int | None = None,
    roll_frequency: str | None = None,
) -> WalkForwardSplitConfig:
    protocol = get_h1_research_protocol()
    return WalkForwardSplitConfig(
        train_window_days=train_window_days or protocol.walk_forward.train_window_months * TRADING_DAYS_PER_MONTH,
        validation_window_days=validation_window_days or protocol.walk_forward.validation_window_months * TRADING_DAYS_PER_MONTH,
        test_window_days=test_window_days or protocol.walk_forward.test_window_months * TRADING_DAYS_PER_MONTH,
        purge_window_days=protocol.walk_forward.purge_window_sessions if purge_window_days is None else purge_window_days,
        embargo_window_days=protocol.walk_forward.embargo_window_sessions if embargo_window_days is None else embargo_window_days,
        roll_frequency=roll_frequency or protocol.walk_forward.roll_frequency,
    )


def generate_h1_walk_forward_predictions(
    panel: pd.DataFrame,
    *,
    predict_start: str,
    model_names: Sequence[str] = H1_BASE_MODEL_NAMES,
    split_config: WalkForwardSplitConfig | None = None,
) -> pd.DataFrame:
    predict_start_ts = pd.Timestamp(predict_start).normalize()
    config = split_config or build_h1_walk_forward_split_config()
    splits = build_walk_forward_splits(panel, config=config)

    prediction_frames: list[pd.DataFrame] = []
    for split in splits:
        test_frame = split.test_frame.loc[split.test_frame["date"] >= predict_start_ts].copy()
        if test_frame.empty:
            continue
        train_frame = pd.concat([split.train_frame, split.validation_frame], ignore_index=True)
        train_frame = train_frame.sort_values(["date", "symbol"]).reset_index(drop=True)

        for model_name in model_names:
            predictions = fit_predict_h1_base_model(
                model_name,
                train_frame=train_frame,
                test_frame=test_frame,
            )
            prediction_frames.append(_attach_split_metadata(predictions, split))

    if not prediction_frames:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "sector",
                "industry",
                "close",
                "vol_20",
                "median_dollar_volume_20",
                "target",
                "future_return",
                "benchmark_future_return",
                "score",
                "confidence",
                "model",
                "split_fold_index",
                "split_anchor_date",
                "split_train_start",
                "split_train_end",
                "split_validation_start",
                "split_validation_end",
                "split_test_start",
                "split_test_end",
            ]
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["split_anchor_date"] = pd.to_datetime(predictions["split_anchor_date"], utc=False)
    predictions = predictions.sort_values(
        ["model", "date", "symbol", "split_anchor_date", "score"],
        ascending=[True, True, True, True, False],
    )
    predictions = predictions.drop_duplicates(subset=["model", "date", "symbol"], keep="last")
    return predictions.sort_values(["model", "date", "score"], ascending=[True, True, False]).reset_index(drop=True)


def fit_predict_h1_base_model(
    name: str,
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    prepared_train = prepare_model_frame(train_frame)
    prepared_test = prepare_model_frame(test_frame)
    builders = get_h1_model_builders()
    try:
        model = builders[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported h1 model '{name}'.") from exc

    features_train = prepared_train[list(H1_FEATURE_COLUMNS)]
    features_test = prepared_test[list(H1_FEATURE_COLUMNS)]
    model.fit(features_train, prepared_train["target"])
    scores = model.predict(features_test)
    return _assemble_predictions(prepared_test, np.asarray(scores), name)


def run_h1_model_backtest(
    *,
    predictions: pd.DataFrame,
    dataset: Mapping[str, pd.DataFrame],
    top_k: int,
    overlay_config: OverlayConfig,
    turnover_control: H1TurnoverControlConfig,
):
    engine = DailyRebalanceOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=dataset["daily_bar"],
        benchmark_index=dataset["benchmark_index"],
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(
            top_k=top_k,
            min_close=overlay_config.min_close,
            min_median_dollar_volume_20=overlay_config.min_median_dollar_volume_20,
            max_vol_20=overlay_config.max_vol_20,
            max_positions_per_sector=overlay_config.max_positions_per_sector,
            sector_neutral=overlay_config.sector_neutral,
            hold_rank_buffer=turnover_control.hold_rank_buffer,
            entry_rank_buffer=turnover_control.entry_rank_buffer,
            max_new_names_per_rebalance=turnover_control.max_new_names_per_rebalance,
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=overlay_config.cost_bps_per_side,
        no_trade_band=turnover_control.no_trade_band,
        max_turnover=turnover_control.max_turnover,
        min_weight_change=turnover_control.min_weight_change,
    )
    start_date = predictions["date"].min().date()
    end_date = predictions["date"].max().date()
    return engine.run(start_date=start_date, end_date=end_date)


def run_h1_baseline_sweep(
    *,
    predict_start: str = "2025-01-01",
    model_names: Sequence[str] = H1_BASE_MODEL_NAMES,
    top_k: int = 10,
    output_dir: str | Path = "artifacts/us_equities_h1_baseline",
    overlay_config: OverlayConfig | None = None,
    turnover_control: H1TurnoverControlConfig | None = None,
    cost_levels_bps: Sequence[float] = (10.0, 15.0, 20.0, 30.0),
    gate_config: H1PromotionGateConfig | None = None,
    layout: StorageLayout | None = None,
    split_config: WalkForwardSplitConfig | None = None,
) -> dict[str, object]:
    config = overlay_config or OverlayConfig(cost_bps_per_side=10.0)
    turnover = turnover_control or H1TurnoverControlConfig()
    gate = gate_config or H1PromotionGateConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    storage = layout or StorageLayout()
    dataset = load_us_equities_dataset(layout=storage)
    price_data = build_price_panel_from_silver(dataset)
    metadata = build_point_in_time_metadata_history(
        pd.Index(price_data.loc[price_data["symbol"] != BENCHMARK_SYMBOL, "date"].drop_duplicates().sort_values()),
        universe_membership_frame=dataset.get("universe_membership", pd.DataFrame()),
        symbol_master_frame=dataset["symbol_master"],
        industry_membership_frame=dataset["industry_membership"],
        universe_name=DEFAULT_RESEARCH_UNIVERSE_NAME,
    )
    research_frame = build_h1_research_frame(
        price_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        symbol_metadata=metadata,
    )
    predictions = generate_h1_walk_forward_predictions(
        research_frame,
        predict_start=predict_start,
        model_names=model_names,
        split_config=split_config,
    )
    predictions.to_csv(output_path / "predictions.csv", index=False)
    research_frame.to_csv(output_path / "research_frame.csv", index=False)
    (output_path / "research_protocol.json").write_text(
        json.dumps(get_h1_research_protocol().to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    cost_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        model_dir = output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        selected_predictions = predictions[predictions["model"] == model_name].copy()
        if selected_predictions.empty:
            continue

        result = run_h1_model_backtest(
            predictions=selected_predictions,
            dataset=dataset,
            top_k=top_k,
            overlay_config=config,
            turnover_control=turnover,
        )
        records = pd.DataFrame(result.meta.get("records", []))
        records.to_csv(model_dir / "backtest_records.csv", index=False)
        summary_row = {
            "model": model_name,
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
            "mean_changed_symbols": result.meta.get("mean_changed_symbols"),
            "artifacts_dir": str(model_dir),
        }
        pd.DataFrame([summary_row]).to_csv(model_dir / "backtest_summary.csv", index=False)
        rows.append(summary_row)
        yearly_frames.append(build_period_stability_summary(records, model_name=model_name, horizon=1, period="year"))
        cost_frames.append(
            build_cost_stress_summary(records, model_name=model_name, horizon=1, cost_levels_bps=cost_levels_bps)
        )

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(output_path / "summary_metrics.csv", index=False)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_summary.to_csv(output_path / "yearly_summary.csv", index=False)
    cost_summary = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
    cost_summary.to_csv(output_path / "cost_stress_summary.csv", index=False)
    benchmark_summary = build_h1_benchmark_summary(predictions=predictions, benchmark_index=dataset["benchmark_index"])
    benchmark_summary.to_csv(output_path / "benchmark_summary.csv", index=False)

    gate_payload = build_h1_promotion_gate(
        summary_frame=summary_frame,
        yearly_summary=yearly_summary,
        cost_summary=cost_summary,
        gate_config=gate,
    )
    (output_path / "promotion_gate.json").write_text(
        json.dumps(gate_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    return {
        "ok": not summary_frame.empty,
        "artifacts_dir": str(output_path),
        "summary_metrics_path": str(output_path / "summary_metrics.csv"),
        "yearly_summary_path": str(output_path / "yearly_summary.csv"),
        "cost_stress_summary_path": str(output_path / "cost_stress_summary.csv"),
        "benchmark_summary_path": str(output_path / "benchmark_summary.csv"),
        "promotion_gate_path": str(output_path / "promotion_gate.json"),
        "research_protocol": get_h1_research_protocol().to_dict(),
        "overlay_config": asdict(config),
        "turnover_control": asdict(turnover),
        "promotion_gate": gate_payload,
    }


def build_h1_benchmark_summary(
    *,
    predictions: pd.DataFrame,
    benchmark_index: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "sessions",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "mean_turnover",
                "mean_cost_bps",
            ]
        )

    benchmark_frame = benchmark_index.rename(columns={"session_date": "date"}).copy()
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
    benchmark_frame = benchmark_frame.sort_values("date").reset_index(drop=True)
    if "adj_open" not in benchmark_frame.columns:
        benchmark_frame["adj_open"] = benchmark_frame["open"]

    prediction_dates = sorted(predictions["date"].dt.normalize().unique())
    returns: list[float] = []
    for idx in range(0, len(prediction_dates)):
        entry_index = idx + 1
        exit_index = idx + 2
        if entry_index >= len(prediction_dates) or exit_index >= len(prediction_dates):
            break
        entry_date = pd.Timestamp(prediction_dates[entry_index]).normalize()
        exit_date = pd.Timestamp(prediction_dates[exit_index]).normalize()
        entry_row = benchmark_frame.loc[benchmark_frame["date"] == entry_date]
        exit_row = benchmark_frame.loc[benchmark_frame["date"] == exit_date]
        if entry_row.empty or exit_row.empty:
            continue
        entry_open = float(entry_row["adj_open"].iloc[0])
        exit_open = float(exit_row["adj_open"].iloc[0])
        if entry_open <= 0:
            continue
        returns.append(exit_open / entry_open - 1.0)

    spy_series = pd.Series(returns, dtype=float)
    cash_series = pd.Series(np.zeros(len(returns), dtype=float))
    return pd.DataFrame(
        [
            _benchmark_summary_row("spy_next_open_hold", spy_series),
            _benchmark_summary_row("cash_zero", cash_series),
        ]
    )


def build_h1_promotion_gate(
    *,
    summary_frame: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    gate_config: H1PromotionGateConfig,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for row in summary_frame.itertuples(index=False):
        yearly = yearly_summary[yearly_summary["model"] == row.model].copy()
        cost_20 = cost_summary[
            (cost_summary["model"] == row.model)
            & (cost_summary["cost_bps_per_side"].astype(float) == 20.0)
        ].copy()
        positive_year_ratio = float((yearly["total_return"].astype(float) > 0).mean()) if not yearly.empty else np.nan
        annualized_return_20bps = (
            float(cost_20["annualized_return"].iloc[0])
            if not cost_20.empty and pd.notna(cost_20["annualized_return"].iloc[0])
            else np.nan
        )

        checks = {
            "sharpe": bool(pd.notna(row.sharpe) and float(row.sharpe) >= gate_config.min_sharpe),
            "cost_20bps_annualized_return": bool(
                pd.notna(annualized_return_20bps)
                and float(annualized_return_20bps) >= gate_config.min_annualized_return_at_20bps
            ),
            "max_drawdown": bool(
                pd.notna(row.max_drawdown) and float(row.max_drawdown) >= gate_config.max_drawdown_floor
            ),
            "mean_turnover": bool(
                pd.notna(row.mean_turnover) and float(row.mean_turnover) <= gate_config.max_mean_turnover
            ),
            "positive_year_ratio": bool(
                pd.notna(positive_year_ratio) and positive_year_ratio >= gate_config.min_positive_year_ratio
            ),
        }
        decisions.append(
            {
                "model": row.model,
                "promote_to_paper_research": all(checks.values()),
                "checks": checks,
                "metrics": {
                    "annualized_return": row.annualized_return,
                    "sharpe": row.sharpe,
                    "max_drawdown": row.max_drawdown,
                    "mean_turnover": row.mean_turnover,
                    "positive_year_ratio": positive_year_ratio,
                    "annualized_return_at_20bps": annualized_return_20bps,
                },
            }
        )

    return {
        "gate_config": asdict(gate_config),
        "models": decisions,
    }


def _assemble_predictions(frame: pd.DataFrame, scores: np.ndarray, model_name: str) -> pd.DataFrame:
    prediction_frame = frame[
        [
            "date",
            "symbol",
            "sector",
            "industry",
            "close",
            "vol_20",
            "median_dollar_volume_20",
            "target",
            "future_return",
            "benchmark_future_return",
        ]
    ].copy()
    prediction_frame["score"] = scores
    prediction_frame["confidence"] = (
        pd.Series(scores, index=prediction_frame.index)
        .groupby(prediction_frame["date"])
        .rank(pct=True)
        .to_numpy()
    )
    prediction_frame["model"] = model_name
    return prediction_frame


def _attach_split_metadata(frame: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["split_fold_index"] = split.fold_index
    tagged["split_anchor_date"] = split.anchor_date
    tagged["split_train_start"] = split.train_start
    tagged["split_train_end"] = split.train_end
    tagged["split_validation_start"] = split.validation_start
    tagged["split_validation_end"] = split.validation_end
    tagged["split_test_start"] = split.test_start
    tagged["split_test_end"] = split.test_end
    return tagged


def _benchmark_summary_row(name: str, returns: pd.Series) -> dict[str, float | str]:
    total_return = _compound_total_return(returns)
    annualized_return = _annualize_total_return(total_return, len(returns), 1)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    sharpe = _annualized_sharpe(returns, 1)
    max_drawdown = _max_drawdown_from_returns(returns)
    return {
        "benchmark": name,
        "sessions": int(len(returns)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "mean_turnover": 0.0,
        "mean_cost_bps": 0.0,
    }


def _ensure_adjusted_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "adj_close" not in normalized.columns:
        normalized["adj_close"] = normalized.get("close")
    if "price_adjust_factor" not in normalized.columns:
        normalized["price_adjust_factor"] = np.nan
    if "cash_dividend" not in normalized.columns:
        normalized["cash_dividend"] = normalized.get("dividends", 0.0)
    if "split_factor" not in normalized.columns:
        normalized["split_factor"] = normalized.get("stock_splits", 1.0)

    valid_close = normalized["close"].notna() & (normalized["close"] > 0)
    inferred_factor = np.where(valid_close, normalized["adj_close"] / normalized["close"], np.nan)
    normalized["price_adjust_factor"] = (
        normalized["price_adjust_factor"].where(normalized["price_adjust_factor"].notna(), inferred_factor)
    )
    normalized["price_adjust_factor"] = normalized["price_adjust_factor"].fillna(1.0)
    normalized["cash_dividend"] = normalized["cash_dividend"].fillna(0.0)
    normalized["split_factor"] = normalized["split_factor"].replace(0.0, 1.0).fillna(1.0)
    normalized["adj_open"] = normalized["open"] * normalized["price_adjust_factor"]
    normalized["adj_close"] = normalized["close"] * normalized["price_adjust_factor"]
    return normalized


def _annualized_sharpe(returns: pd.Series, horizon: int) -> float:
    if returns.std(ddof=1) == 0 or returns.empty:
        return np.nan
    return float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252 / horizon))


def _compound_total_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min())
