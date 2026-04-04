from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline

from stockmachine.alpha import get_alpha_expert, list_alpha_expert_names, list_alpha_experts
from stockmachine.backtest import DailyOpenHoldBacktestEngine, DataFrameSignalModel
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy
from stockmachine.research.builders import (
    FEATURE_COLUMNS,
    build_catboost_regressor,
    build_elastic_net_pipeline,
    build_extra_trees_model,
    build_hist_gbm_model,
    build_huber_pipeline,
    build_lightgbm_ranker,
    build_lightgbm_regressor,
    build_lstm_regressor,
    build_query_group_sizes,
    build_random_forest_model,
    build_ridge_pipeline,
    build_transformer_regressor,
    build_xgboost_regressor,
    get_boosting_model_builders,
    get_lightgbm_model_builders,
    get_sequence_model_builders,
    get_sklearn_model_builders,
    prepare_model_frame,
)
from stockmachine.research.builders.common import (
    BASELINE12_FEATURE_COLUMNS,
    BASELINE12_PLUS_SHORT_AND_RELATIVE_FEATURE_COLUMNS,
)
from stockmachine.research.comparison import (
    build_yearly_holdout_windows,
    comparison_summary_frame,
    slice_frame_by_windows,
    validate_aligned_frames,
)
from stockmachine.research.splitting import (
    WalkForwardSplit,
    WalkForwardSplitConfig,
    build_latest_partial_walk_forward_split,
    build_walk_forward_splits,
)
from stockmachine.research.universe import (
    DEFAULT_RESEARCH_UNIVERSE_NAME,
    build_point_in_time_metadata_history,
)


DEFAULT_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "NFLX", "AMD", "INTC", "QCOM", "TXN", "CSCO", "IBM", "AMAT",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK",
    "SCHW", "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "ABT", "TMO", "DHR",
    "AMGN", "GILD", "XOM", "CVX", "COP", "SLB", "HD", "LOW", "COST", "WMT",
    "TGT", "KO", "PEP", "MCD", "NKE", "SBUX", "CAT", "DE", "GE", "HON",
    "BA", "LMT", "DIS", "CMCSA", "VZ", "T",
)
BENCHMARK_SYMBOL = "SPY"
BASE_MODEL_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in list_alpha_experts() if not spec.components
)
MODEL_NAMES: tuple[str, ...] = list_alpha_expert_names()

MODEL_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "extra_trees": BASELINE12_FEATURE_COLUMNS,
    "hist_gbm": FEATURE_COLUMNS,
    "lightgbm_ranker": BASELINE12_PLUS_SHORT_AND_RELATIVE_FEATURE_COLUMNS,
}


@dataclass(slots=True, frozen=True)
class PeriodMetrics:
    """Summary metrics for one evaluation period."""

    days: int
    samples: int
    mean_rank_ic: float
    rank_ic_ir: float
    mean_top_bottom_spread: float
    top_k_total_return: float
    top_k_annualized_return: float
    top_k_sharpe: float
    benchmark_total_return: float
    annualized_excess_return: float
    hit_rate: float


@dataclass(slots=True, frozen=True)
class OverlayConfig:
    """Risk and execution assumptions applied after the model score."""

    min_close: float = 10.0
    min_median_dollar_volume_20: float = 50_000_000.0
    max_vol_20: float = 0.04
    max_positions_per_sector: int = 2
    cost_bps_per_side: float = 10.0
    sector_neutral: bool = True


@dataclass(slots=True, frozen=True)
class ModelWhiteboxOverride:
    """Per-model override for the risk-aware whitebox overlay."""

    top_k: int | None = None
    min_close: float | None = None
    min_median_dollar_volume_20: float | None = None
    max_vol_20: float | None = None
    max_positions_per_sector: int | None = None
    cost_bps_per_side: float | None = None
    sector_neutral: bool | None = None


@dataclass(slots=True, frozen=True)
class OverlayPeriodMetrics:
    """Summary metrics for the risk-aware and cost-aware strategy overlay."""

    days: int
    samples: int
    mean_rank_ic: float
    rank_ic_ir: float
    mean_top_bottom_spread: float
    gross_total_return: float
    gross_annualized_return: float
    gross_sharpe: float
    net_total_return: float
    net_annualized_return: float
    net_sharpe: float
    benchmark_total_return: float
    annualized_net_excess_return: float
    hit_rate: float
    mean_turnover: float
    mean_cost_bps: float
    mean_names_held: float
    mean_max_sector_weight: float


MODEL_WHITEBOX_OVERRIDES: dict[str, ModelWhiteboxOverride] = {
    "extra_trees": ModelWhiteboxOverride(
        top_k=8,
        min_median_dollar_volume_20=30_000_000.0,
        max_vol_20=0.04,
    ),
    "hist_gbm": ModelWhiteboxOverride(
        top_k=10,
        min_median_dollar_volume_20=50_000_000.0,
        max_vol_20=0.04,
    ),
    "lightgbm_ranker": ModelWhiteboxOverride(
        top_k=10,
        min_median_dollar_volume_20=30_000_000.0,
        max_vol_20=0.05,
    ),
}


def resolve_model_whitebox_policy(
    name: str,
    *,
    top_k: int,
    overlay_config: OverlayConfig,
) -> tuple[int, OverlayConfig]:
    """Resolve the whitebox overlay settings for one model."""

    override = MODEL_WHITEBOX_OVERRIDES.get(name)
    if override is None:
        return int(top_k), overlay_config

    routed_top_k = int(override.top_k if override.top_k is not None else top_k)
    routed_config = replace(
        overlay_config,
        min_close=override.min_close if override.min_close is not None else overlay_config.min_close,
        min_median_dollar_volume_20=(
            override.min_median_dollar_volume_20
            if override.min_median_dollar_volume_20 is not None
            else overlay_config.min_median_dollar_volume_20
        ),
        max_vol_20=override.max_vol_20 if override.max_vol_20 is not None else overlay_config.max_vol_20,
        max_positions_per_sector=(
            override.max_positions_per_sector
            if override.max_positions_per_sector is not None
            else overlay_config.max_positions_per_sector
        ),
        cost_bps_per_side=(
            override.cost_bps_per_side
            if override.cost_bps_per_side is not None
            else overlay_config.cost_bps_per_side
        ),
        sector_neutral=(
            override.sector_neutral
            if override.sector_neutral is not None
            else overlay_config.sector_neutral
        ),
    )
    return routed_top_k, routed_config


def run_baseline_experiment(
    *,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    predict_start: str = "2024-01-01",
    validation_year: int = 2024,
    top_k: int = 10,
    horizon: int = 5,
    output_dir: str | Path = "artifacts/us_equities_baseline",
) -> dict[str, object]:
    """Run the first end-to-end US equities baseline experiment."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tickers = list(DEFAULT_UNIVERSE) + [BENCHMARK_SYMBOL]
    price_data = download_history(tickers=tickers, start=start, end=end)
    research_frame = build_research_frame(price_data, benchmark_symbol=BENCHMARK_SYMBOL, horizon=horizon)
    predictions = generate_walk_forward_predictions(research_frame, predict_start=predict_start)
    model_frames = {
        model_name: predictions[predictions["model"] == model_name].copy()
        for model_name in MODEL_NAMES
    }
    validate_aligned_frames(
        model_frames,
        required_columns=("score", "target", "future_return", "benchmark_future_return"),
    )
    comparison_windows = build_yearly_holdout_windows(validation_year=validation_year)

    evaluation = {}
    for model_name in MODEL_NAMES:
        model_predictions = model_frames[model_name]
        period_frames = slice_frame_by_windows(model_predictions, comparison_windows)
        evaluation[model_name] = {
            window.name: evaluate_predictions(period_frames[window.name], top_k=top_k, horizon=horizon)
            for window in comparison_windows
        }

    save_artifacts(output_path, price_data, research_frame, predictions, evaluation)

    universe_stats = {
        "requested_symbols": len(DEFAULT_UNIVERSE),
        "loaded_symbols": int(price_data["symbol"].nunique() - 1),
        "rows": int(len(research_frame)),
        "date_min": str(research_frame["date"].min().date()),
        "date_max": str(research_frame["date"].max().date()),
    }
    return {
        "universe": universe_stats,
        "evaluation": _serialize_metrics(evaluation),
        "artifacts_dir": str(output_path),
    }


def run_risk_aware_experiment(
    *,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    predict_start: str = "2024-01-01",
    validation_year: int = 2024,
    top_k: int = 10,
    horizon: int = 5,
    output_dir: str | Path = "artifacts/us_equities_risk_aware",
    overlay_config: OverlayConfig | None = None,
) -> dict[str, object]:
    """Run the enhanced baseline with sector neutralization, filters, and costs."""

    config = overlay_config or OverlayConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tickers = list(DEFAULT_UNIVERSE) + [BENCHMARK_SYMBOL]
    metadata = download_symbol_metadata(
        symbols=list(DEFAULT_UNIVERSE),
        cache_path=output_path / "symbol_metadata.csv",
    )
    price_data = download_history(tickers=tickers, start=start, end=end)
    research_frame = build_research_frame(
        price_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        horizon=horizon,
        symbol_metadata=metadata,
    )
    predictions = generate_walk_forward_predictions(research_frame, predict_start=predict_start)
    model_frames = {
        model_name: predictions[predictions["model"] == model_name].copy()
        for model_name in MODEL_NAMES
    }
    validate_aligned_frames(
        model_frames,
        required_columns=("score", "target", "future_return", "benchmark_future_return"),
    )
    comparison_windows = build_yearly_holdout_windows(validation_year=validation_year)

    raw_evaluation: dict[str, dict[str, PeriodMetrics]] = {}
    overlay_evaluation: dict[str, dict[str, OverlayPeriodMetrics]] = {}
    selection_frames: list[pd.DataFrame] = []

    for model_name in MODEL_NAMES:
        model_predictions = model_frames[model_name]
        period_frames = slice_frame_by_windows(model_predictions, comparison_windows)
        routed_top_k, routed_config = resolve_model_whitebox_policy(
            model_name,
            top_k=top_k,
            overlay_config=config,
        )
        raw_evaluation[model_name] = {
            window.name: evaluate_predictions(period_frames[window.name], top_k=routed_top_k, horizon=horizon)
            for window in comparison_windows
        }
        validation_metrics, validation_selection = evaluate_overlay_predictions(
            period_frames["validation"],
            top_k=routed_top_k,
            horizon=horizon,
            overlay_config=routed_config,
        )
        test_metrics, test_selection = evaluate_overlay_predictions(
            period_frames["test"],
            top_k=routed_top_k,
            horizon=horizon,
            overlay_config=routed_config,
        )
        overlay_evaluation[model_name] = {
            "validation": validation_metrics,
            "test": test_metrics,
        }
        selection_frames.append(
            validation_selection.assign(
                model=model_name,
                period="validation",
                requested_top_k=int(top_k),
                effective_top_k=routed_top_k,
                effective_min_median_dollar_volume_20=routed_config.min_median_dollar_volume_20,
                effective_max_vol_20=routed_config.max_vol_20,
            )
        )
        selection_frames.append(
            test_selection.assign(
                model=model_name,
                period="test",
                requested_top_k=int(top_k),
                effective_top_k=routed_top_k,
                effective_min_median_dollar_volume_20=routed_config.min_median_dollar_volume_20,
                effective_max_vol_20=routed_config.max_vol_20,
            )
        )

    save_risk_aware_artifacts(
        output_dir=output_path,
        price_data=price_data,
        metadata=metadata,
        research_frame=research_frame,
        predictions=predictions,
        raw_evaluation=raw_evaluation,
        overlay_evaluation=overlay_evaluation,
        selections=pd.concat(selection_frames, ignore_index=True),
        overlay_config=config,
    )

    universe_stats = {
        "requested_symbols": len(DEFAULT_UNIVERSE),
        "loaded_symbols": int(price_data["symbol"].nunique() - 1),
        "rows": int(len(research_frame)),
        "date_min": str(research_frame["date"].min().date()),
        "date_max": str(research_frame["date"].max().date()),
    }
    return {
        "universe": universe_stats,
        "overlay_config": asdict(config),
        "raw_evaluation": _serialize_metrics(raw_evaluation),
        "overlay_evaluation": _serialize_overlay_metrics(overlay_evaluation),
        "artifacts_dir": str(output_path),
    }


def run_silver_chain_backtest(
    *,
    predict_start: str = "2024-01-01",
    model_name: str = "hist_gbm",
    top_k: int = 10,
    horizon: int = 5,
    output_dir: str | Path = "artifacts/us_equities_silver_chain",
    overlay_config: OverlayConfig | None = None,
    layout: StorageLayout | None = None,
) -> dict[str, object]:
    """Run the model and backtest using canonical silver tables as input."""

    config = overlay_config or OverlayConfig()
    storage = layout or StorageLayout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_us_equities_dataset(layout=storage)
    price_data = build_price_panel_from_silver(dataset)
    metadata = build_point_in_time_metadata_history(
        pd.Index(price_data.loc[price_data["symbol"] != BENCHMARK_SYMBOL, "date"].drop_duplicates().sort_values()),
        universe_membership_frame=dataset.get("universe_membership", pd.DataFrame()),
        symbol_master_frame=dataset["symbol_master"],
        industry_membership_frame=dataset["industry_membership"],
        universe_name=DEFAULT_RESEARCH_UNIVERSE_NAME,
    )
    research_frame = build_research_frame(
        price_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        horizon=horizon,
        symbol_metadata=metadata,
    )
    predictions = generate_walk_forward_predictions(research_frame, predict_start=predict_start)
    selected_predictions = predictions[predictions["model"] == model_name].copy()
    if selected_predictions.empty:
        raise RuntimeError(f"No predictions produced for model '{model_name}'.")

    effective_top_k, effective_config = resolve_model_whitebox_policy(
        model_name,
        top_k=top_k,
        overlay_config=config,
    )

    engine = DailyOpenHoldBacktestEngine(
        predictions=selected_predictions,
        daily_bar=dataset["daily_bar"],
        benchmark_index=dataset["benchmark_index"],
        signal_model=DataFrameSignalModel(selected_predictions, horizon_bars=horizon),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(
            top_k=effective_top_k,
            min_close=effective_config.min_close,
            min_median_dollar_volume_20=effective_config.min_median_dollar_volume_20,
            max_vol_20=effective_config.max_vol_20,
            max_positions_per_sector=effective_config.max_positions_per_sector,
            sector_neutral=effective_config.sector_neutral,
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=horizon,
        cost_bps_per_side=effective_config.cost_bps_per_side,
    )

    start_date = selected_predictions["date"].min().date()
    end_date = selected_predictions["date"].max().date()
    result = engine.run(start_date=start_date, end_date=end_date)

    selected_predictions.to_csv(output_path / "predictions.csv", index=False)
    research_frame.to_csv(output_path / "research_frame.csv", index=False)
    pd.DataFrame(result.meta.get("records", [])).to_csv(output_path / "backtest_records.csv", index=False)
    pd.DataFrame([{
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
    }]).to_csv(output_path / "backtest_summary.csv", index=False)

    return {
        "model": model_name,
        "requested_top_k": int(top_k),
        "effective_top_k": effective_top_k,
        "overlay_config": asdict(effective_config),
        "summary": {
            "sessions": result.sessions,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "annualized_volatility": result.annualized_volatility,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "benchmark_total_return": result.meta.get("benchmark_total_return"),
            "mean_turnover": result.meta.get("mean_turnover"),
            "mean_cost_bps": result.meta.get("mean_cost_bps"),
        },
        "artifacts_dir": str(output_path),
    }


def download_history(*, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV history from Yahoo Finance and return a long table."""

    raw = _download_history_batch(tickers=tickers, start=start, end=end)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for symbol in tickers:
        ticker_frame = _extract_download_frame(raw, symbol=symbol)
        if ticker_frame.empty:
            missing.append(symbol)
            continue
        frames.append(ticker_frame)

    for symbol in missing.copy():
        try:
            retry_raw = _download_history_batch(tickers=[symbol], start=start, end=end)
            ticker_frame = _extract_download_frame(retry_raw, symbol=symbol)
        except RuntimeError:
            ticker_frame = pd.DataFrame()
        if ticker_frame.empty:
            ticker_frame = _download_single_history_frame(symbol=symbol, start=start, end=end)
        if ticker_frame.empty:
            continue
        frames.append(ticker_frame)
        missing.remove(symbol)

    if missing:
        raise RuntimeError(f"Yahoo Finance download missing symbols after retry: {', '.join(missing)}")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=False)
    panel = _ensure_adjusted_price_columns(panel)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    return panel


def download_symbol_metadata(*, symbols: list[str], cache_path: str | Path | None = None) -> pd.DataFrame:
    """Download or reuse cached sector metadata for the selected universe."""

    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cached = pd.read_csv(cache_file)
        if set(symbols).issubset(set(cached["symbol"].tolist())):
            return cached.sort_values("symbol").reset_index(drop=True)

    rows = []
    for symbol in symbols:
        info = {}
        try:
            info = yf.Ticker(symbol).info
        except Exception:
            info = {}
        rows.append(
            {
                "symbol": symbol,
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "Unknown",
                "quote_type": info.get("quoteType") or "Unknown",
            }
        )

    metadata = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(cache_file, index=False)
    return metadata


def build_price_panel_from_silver(dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert silver daily_bar and benchmark_index tables into a price panel."""

    daily_bar = dataset["daily_bar"].copy()
    adj_factor = dataset.get("adj_factor", pd.DataFrame()).copy()
    benchmark_index = dataset["benchmark_index"].copy()

    daily_bar = daily_bar.rename(columns={"session_date": "date"})
    benchmark_index = benchmark_index.rename(columns={"session_date": "date"})
    daily_bar = daily_bar[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    benchmark_index = benchmark_index[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    combined = pd.concat([daily_bar, benchmark_index], ignore_index=True)
    if not adj_factor.empty:
        adj_factor = adj_factor.rename(columns={"session_date": "date"})
        combined = combined.merge(
            adj_factor[["date", "symbol", "split_factor", "cash_dividend", "price_adjust_factor"]],
            on=["date", "symbol"],
            how="left",
        )
    combined["date"] = pd.to_datetime(combined["date"])
    combined = _ensure_adjusted_price_columns(combined)
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_metadata_from_silver(dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build symbol metadata from silver symbol master and industry tables."""

    symbol_master = dataset["symbol_master"].copy()
    industry = dataset["industry_membership"].copy()

    if symbol_master.empty and industry.empty:
        return pd.DataFrame(columns=["symbol", "company_name", "sector", "industry", "quote_type", "exchange", "currency", "country"])

    if not industry.empty:
        industry = industry.sort_values(["symbol", "as_of_date"]).drop_duplicates(
            subset=["symbol"], keep="last"
        )
        industry = industry.rename(
            columns={
                "sector_name": "sector",
                "industry_name": "industry",
            }
        )

    if not symbol_master.empty:
        symbol_master = symbol_master.rename(
            columns={
                "country_of_listing": "country",
                "asset_class": "quote_type",
                "exchange_mic": "exchange",
            }
        )

    if symbol_master.empty:
        metadata = industry[["symbol", "sector", "industry"]].copy()
        metadata["company_name"] = metadata["symbol"]
        metadata["quote_type"] = "Unknown"
        metadata["exchange"] = "Unknown"
        metadata["currency"] = "USD"
        metadata["country"] = "US"
        return metadata

    metadata = symbol_master[["symbol", "company_name", "quote_type", "exchange", "currency", "country"]].copy()
    if not industry.empty:
        metadata = metadata.merge(industry[["symbol", "sector", "industry"]], on="symbol", how="left")
    else:
        metadata["sector"] = symbol_master.get("sector", pd.Series(index=symbol_master.index, dtype="object")).fillna("Unknown")
        metadata["industry"] = symbol_master.get("industry", pd.Series(index=symbol_master.index, dtype="object")).fillna("Unknown")
    metadata["sector"] = metadata["sector"].fillna("Unknown")
    metadata["industry"] = metadata["industry"].fillna("Unknown")
    return metadata.sort_values("symbol").reset_index(drop=True)


def build_research_frame(
    price_data: pd.DataFrame,
    *,
    benchmark_symbol: str,
    horizon: int,
    symbol_metadata: pd.DataFrame | None = None,
    drop_unlabeled_rows: bool = True,
) -> pd.DataFrame:
    """Build panel features and next-open horizon labels.

    By default the returned frame only includes rows with complete future labels,
    which is what strict historical evaluation needs. Live or paper inference
    can set ``drop_unlabeled_rows=False`` to keep the latest feature rows even
    when their forward-return labels are not known yet.
    """

    price_data = _ensure_adjusted_price_columns(price_data.copy())
    benchmark = (
        price_data.loc[price_data["symbol"] == benchmark_symbol, ["date", "open", "close", "adj_open"]]
        .rename(columns={"open": "benchmark_open", "close": "benchmark_close", "adj_open": "benchmark_adj_open"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    benchmark["benchmark_ret_1d"] = benchmark["benchmark_close"].pct_change(1, fill_method=None)
    benchmark["benchmark_ret_5d"] = benchmark["benchmark_close"].pct_change(5, fill_method=None)
    benchmark["benchmark_mom_20"] = benchmark["benchmark_close"].pct_change(20, fill_method=None)
    benchmark["benchmark_mom_60"] = benchmark["benchmark_close"].pct_change(60, fill_method=None)
    benchmark["benchmark_future_return"] = (
        benchmark["benchmark_adj_open"].shift(-(horizon + 1)) / benchmark["benchmark_adj_open"].shift(-1) - 1.0
    )

    panel = price_data.loc[price_data["symbol"] != benchmark_symbol].copy()
    panel = panel.merge(
        benchmark[
            [
                "date",
                "benchmark_ret_1d",
                "benchmark_ret_5d",
                "benchmark_mom_20",
                "benchmark_mom_60",
                "benchmark_future_return",
            ]
        ],
        on="date",
        how="left",
    )

    group = panel.groupby("symbol", group_keys=False)
    panel["prev_close"] = group["close"].shift(1)
    panel["intraday_return"] = panel["close"] / panel["open"] - 1.0
    panel["ret_1d"] = group["close"].pct_change(1, fill_method=None)
    panel["ret_2d"] = group["close"].pct_change(2, fill_method=None)
    panel["mom_3"] = group["close"].pct_change(3, fill_method=None)
    panel["mom_5"] = group["close"].pct_change(5, fill_method=None)
    panel["mom_10"] = group["close"].pct_change(10, fill_method=None)
    panel["mom_20"] = group["close"].pct_change(20, fill_method=None)
    panel["mom_60"] = group["close"].pct_change(60, fill_method=None)
    panel["vol_5"] = group["ret_1d"].rolling(5).std().reset_index(level=0, drop=True)
    panel["vol_10"] = group["ret_1d"].rolling(10).std().reset_index(level=0, drop=True)
    panel["vol_20"] = group["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    panel["vol_60"] = group["ret_1d"].rolling(60).std().reset_index(level=0, drop=True)
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["median_dollar_volume_20"] = (
        group["dollar_volume"].rolling(20).median().reset_index(level=0, drop=True)
    )
    panel["volume_ratio_5"] = panel["volume"] / group["volume"].rolling(5).mean().reset_index(level=0, drop=True)
    panel["volume_ratio_20"] = panel["volume"] / group["volume"].rolling(20).mean().reset_index(level=0, drop=True)
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
    close_ma5 = group["close"].rolling(5).mean().reset_index(level=0, drop=True)
    close_ma20 = group["close"].rolling(20).mean().reset_index(level=0, drop=True)
    panel["close_ma5_gap"] = panel["close"] / close_ma5 - 1.0
    panel["close_ma20_gap"] = panel["close"] / close_ma20 - 1.0
    rolling_high_20 = group["high"].rolling(20).max().reset_index(level=0, drop=True)
    rolling_low_20 = group["low"].rolling(20).min().reset_index(level=0, drop=True)
    panel["price_position_20d"] = (panel["close"] - rolling_low_20) / (
        rolling_high_20 - rolling_low_20 + 1e-12
    )
    panel["rel_ret_1d"] = panel["ret_1d"] - panel["benchmark_ret_1d"]
    panel["rel_ret_5d"] = panel["mom_5"] - panel["benchmark_ret_5d"]
    panel["rel_mom_20"] = panel["mom_20"] - panel["benchmark_mom_20"]
    panel["rel_mom_60"] = panel["mom_60"] - panel["benchmark_mom_60"]
    panel["future_return"] = group["adj_open"].shift(-(horizon + 1)) / group["adj_open"].shift(-1) - 1.0
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

    required_columns = list(FEATURE_COLUMNS)
    if drop_unlabeled_rows:
        required_columns += ["future_return", "target"]
    panel = panel.dropna(subset=required_columns).reset_index(drop=True)
    return panel


def _ensure_adjusted_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach adjustment-factor-derived columns used by labels and backtests."""

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


def _download_history_batch(*, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("No market data returned from Yahoo Finance.")
    return raw


def _extract_download_frame(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        ticker_frame = raw[symbol].copy()
    else:
        ticker_frame = raw.copy()

    if ticker_frame.empty:
        return pd.DataFrame()

    ticker_frame = ticker_frame.reset_index()
    ticker_frame.columns = [str(column).lower().replace(" ", "_") for column in ticker_frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in ticker_frame.columns:
            return pd.DataFrame()
    ticker_frame = ticker_frame.dropna(subset=required).copy()
    if ticker_frame.empty:
        return pd.DataFrame()
    ticker_frame["symbol"] = symbol
    return ticker_frame


def _download_single_history_frame(*, symbol: str, start: str, end: str) -> pd.DataFrame:
    ticker_frame = yf.Ticker(symbol).history(
        start=start,
        end=end,
        period="max",
        auto_adjust=False,
        actions=True,
    )
    if ticker_frame.empty:
        return pd.DataFrame()
    ticker_frame = ticker_frame.reset_index()
    ticker_frame.columns = [str(column).lower().replace(" ", "_") for column in ticker_frame.columns]
    if "date" not in ticker_frame.columns:
        ticker_frame = ticker_frame.rename(columns={"datetime": "date"})
    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in ticker_frame.columns:
            return pd.DataFrame()
    ticker_frame["date"] = pd.to_datetime(ticker_frame["date"], utc=False).dt.tz_localize(None)
    ticker_frame = ticker_frame[
        (ticker_frame["date"] >= pd.Timestamp(start))
        & (ticker_frame["date"] < pd.Timestamp(end))
    ].dropna(subset=required).copy()
    if ticker_frame.empty:
        return pd.DataFrame()
    ticker_frame["symbol"] = symbol
    return ticker_frame


def generate_walk_forward_predictions(
    panel: pd.DataFrame,
    *,
    predict_start: str,
    requested_models: tuple[str, ...] | None = None,
    train_window_days: int | None = None,
    validation_window_days: int | None = None,
    test_window_days: int | None = None,
    purge_window_days: int | None = None,
    embargo_window_days: int | None = None,
    roll_frequency: str | None = None,
    include_validation_in_training: bool = True,
    include_partial_current_test: bool = False,
) -> pd.DataFrame:
    """Generate walk-forward predictions using the shared P0 research protocol."""

    panel = panel.copy()
    predict_start_ts = pd.Timestamp(predict_start).normalize()
    requested_base_model_names, requested_output_model_names = _resolve_prediction_model_scope(requested_models)
    split_config = _build_walk_forward_split_config(
        train_window_days=train_window_days,
        validation_window_days=validation_window_days,
        test_window_days=test_window_days,
        purge_window_days=purge_window_days,
        embargo_window_days=embargo_window_days,
        roll_frequency=roll_frequency,
    )
    splits = build_walk_forward_splits(panel, config=split_config)
    splits_to_process = list(splits)
    if include_partial_current_test:
        partial_split = build_latest_partial_walk_forward_split(
            panel,
            config=split_config,
            fold_index=len(splits_to_process),
        )
        if partial_split is not None and all(split.anchor_date != partial_split.anchor_date for split in splits_to_process):
            splits_to_process.append(partial_split)

    prediction_frames: list[pd.DataFrame] = []
    for split in splits_to_process:
        test_frame = split.test_frame.loc[split.test_frame["date"] >= predict_start_ts].copy()
        if test_frame.empty:
            continue

        train_frame = _compose_walk_forward_train_frame(
            split,
            include_validation_in_training=include_validation_in_training,
        )
        if train_frame.empty:
            continue

        split_prediction_frames: dict[str, pd.DataFrame] = {}
        for model_name in requested_base_model_names:
            fit_kind = get_trainable_model_fit_kind(model_name) if model_name != "factor_baseline" else "factor"
            predictions = fit_predict_base_model(
                model_name,
                train_frame=split.train_frame.copy() if fit_kind == "sequence_regressor" else train_frame,
                test_frame=test_frame,
                validation_frame=split.validation_frame.copy() if fit_kind == "sequence_regressor" else None,
                history_frame=train_frame if fit_kind == "sequence_regressor" else None,
            )
            split_prediction_frames[model_name] = _attach_split_metadata(predictions, split)

        prediction_frames.extend(split_prediction_frames.values())
        prediction_frames.extend(
            _attach_split_metadata(frame, split)
            for frame in build_ensemble_prediction_frames(
                split_prediction_frames,
                requested_models=requested_output_model_names,
            )
        )

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


def _build_walk_forward_split_config(
    *,
    train_window_days: int | None,
    validation_window_days: int | None,
    test_window_days: int | None,
    purge_window_days: int | None,
    embargo_window_days: int | None,
    roll_frequency: str | None,
) -> WalkForwardSplitConfig:
    base_config = WalkForwardSplitConfig()
    return WalkForwardSplitConfig(
        train_window_days=train_window_days or base_config.train_window_days,
        validation_window_days=validation_window_days or base_config.validation_window_days,
        test_window_days=test_window_days or base_config.test_window_days,
        purge_window_days=base_config.purge_window_days if purge_window_days is None else purge_window_days,
        embargo_window_days=base_config.embargo_window_days if embargo_window_days is None else embargo_window_days,
        roll_frequency=roll_frequency or base_config.roll_frequency,
        date_column=base_config.date_column,
    )


def _compose_walk_forward_train_frame(
    split: WalkForwardSplit,
    *,
    include_validation_in_training: bool,
) -> pd.DataFrame:
    if not include_validation_in_training:
        return split.train_frame.copy()
    return (
        pd.concat([split.train_frame, split.validation_frame], ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


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


def build_ensemble_prediction_frames(
    model_frames: dict[str, pd.DataFrame],
    *,
    requested_models: tuple[str, ...] | None = None,
) -> list[pd.DataFrame]:
    """Build low-risk ensemble prediction frames from aligned base model outputs."""

    ensemble_frames: list[pd.DataFrame] = []
    if requested_models is None:
        target_model_names = MODEL_NAMES
    else:
        target_model_names = tuple(requested_models)

    for model_name in target_model_names:
        spec = get_alpha_expert(model_name)
        if not spec.components or not spec.combine_method:
            continue
        if any(component not in model_frames for component in spec.components):
            continue
        component_frames = {component: model_frames[component] for component in spec.components}
        validate_aligned_frames(component_frames, required_columns=("score", "confidence"))
        ensemble_frames.append(
            _combine_prediction_frames(
                component_frames=component_frames,
                model_name=spec.name,
                combine_method=spec.combine_method,
            )
        )
    return ensemble_frames


def _resolve_prediction_model_scope(
    requested_models: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Resolve the base-model training scope and final output model list."""

    if not requested_models:
        return BASE_MODEL_NAMES, None

    seen_output_models: set[str] = set()
    ordered_output_models: list[str] = []
    seen_base_models: set[str] = set()
    ordered_base_models: list[str] = []

    def visit(model_name: str) -> None:
        spec = get_alpha_expert(model_name)
        if model_name not in seen_output_models:
            seen_output_models.add(model_name)
            ordered_output_models.append(model_name)
        if spec.components:
            for component in spec.components:
                visit(component)
            return
        if model_name not in seen_base_models:
            seen_base_models.add(model_name)
            ordered_base_models.append(model_name)

    for model_name in requested_models:
        visit(model_name)

    return tuple(ordered_base_models), tuple(ordered_output_models)


def get_trainable_model_builders() -> dict[str, Callable[[], object]]:
    """Return all supported trainable model builders for the research pipeline."""

    builders: dict[str, Callable[[], object]] = {}
    builders.update(get_sklearn_model_builders())
    builders.update(get_lightgbm_model_builders())
    builders.update(get_boosting_model_builders())
    builders.update(get_sequence_model_builders())
    return builders


def get_trainable_model_builder(name: str) -> Callable[[], object]:
    """Resolve one trainable baseline model builder."""

    builders = get_trainable_model_builders()
    try:
        return builders[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported trainable base model '{name}'.") from exc


def get_trainable_model_fit_kind(name: str) -> str:
    """Return how one model should be fit inside the walk-forward loop."""

    if name == "lightgbm_ranker":
        return "ranker"
    if name in {"lstm_regressor", "transformer_regressor"}:
        return "sequence_regressor"
    if name in get_trainable_model_builders():
        return "regressor"
    raise ValueError(f"Unsupported trainable base model '{name}'.")


def resolve_model_feature_columns(name: str) -> tuple[str, ...]:
    """Resolve the h5 feature subset for one model."""

    return MODEL_FEATURE_COLUMNS.get(name, FEATURE_COLUMNS)


def score_factor_model(name: str, frame: pd.DataFrame) -> np.ndarray:
    """Score one non-trainable baseline expert."""

    if name == "factor_baseline":
        return compute_factor_scores(frame)
    raise ValueError(f"Unsupported factor-style base model '{name}'.")


def fit_predict_base_model(
    name: str,
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None = None,
    history_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit one base model on the rolling window and emit the prediction frame."""

    if name == "factor_baseline":
        scores = score_factor_model(name, test_frame)
        return _assemble_predictions(test_frame, scores, name)

    prepared_train = prepare_model_frame(train_frame)
    prepared_test = prepare_model_frame(test_frame)
    builder = get_trainable_model_builder(name)
    fit_kind = get_trainable_model_fit_kind(name)
    feature_columns = resolve_model_feature_columns(name)
    model = builder()
    if fit_kind == "ranker" and hasattr(model, "feature_columns"):
        model.feature_columns = tuple(feature_columns)
    fit_kwargs: dict[str, object] = {}
    if fit_kind == "ranker":
        fit_kwargs["group"] = build_query_group_sizes(prepared_train)
    if fit_kind == "sequence_regressor":
        fit_kwargs["validation_frame"] = prepare_model_frame(validation_frame) if validation_frame is not None else None
        fit_kwargs["history_frame"] = prepare_model_frame(history_frame) if history_frame is not None else prepared_train
        model.fit(prepared_train, prepared_train["target"], **fit_kwargs)
        scores = model.predict(prepared_test)
    else:
        model.fit(prepared_train[list(feature_columns)], prepared_train["target"], **fit_kwargs)
        scores = model.predict(prepared_test[list(feature_columns)])
    return _assemble_predictions(prepared_test, scores, name)


def compute_factor_scores(frame: pd.DataFrame) -> np.ndarray:
    """Compute a simple white-box factor blend score."""

    grouped = frame.groupby("date")
    ranked = pd.DataFrame(index=frame.index)
    ranked["mom_20_rank"] = grouped["mom_20"].rank(pct=True)
    ranked["mom_60_rank"] = grouped["mom_60"].rank(pct=True)
    ranked["rel_mom_20_rank"] = grouped["rel_mom_20"].rank(pct=True)
    ranked["rel_mom_60_rank"] = grouped["rel_mom_60"].rank(pct=True)
    ranked["vol_20_rank"] = grouped["vol_20"].rank(pct=True, ascending=False)
    ranked["volume_rank"] = grouped["volume_ratio_20"].rank(pct=True)
    return ranked.mean(axis=1).to_numpy()


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    horizon: int,
) -> PeriodMetrics:
    """Evaluate one model's predictions for one period."""

    if predictions.empty:
        return PeriodMetrics(0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    rank_ic_by_day = predictions.groupby("date")[["score", "target"]].apply(
        lambda day: day["score"].corr(day["target"], method="spearman")
    )
    spread_by_day = predictions.groupby("date")[["score", "target"]].apply(_top_bottom_spread)
    hit_rate = float((spread_by_day > 0).mean())

    rebalance_returns = []
    unique_dates = sorted(predictions["date"].unique())
    for idx in range(0, len(unique_dates), horizon):
        rebalance_date = unique_dates[idx]
        cohort = predictions[predictions["date"] == rebalance_date].nlargest(top_k, "score")
        if cohort.empty:
            continue
        rebalance_returns.append(
            {
                "date": rebalance_date,
                "portfolio_return": float(cohort["future_return"].mean()),
                "benchmark_return": float(cohort["benchmark_future_return"].iloc[0]),
            }
        )

    rebalance_frame = pd.DataFrame(rebalance_returns)
    portfolio_total = float((1.0 + rebalance_frame["portfolio_return"]).prod() - 1.0) if not rebalance_frame.empty else np.nan
    benchmark_total = float((1.0 + rebalance_frame["benchmark_return"]).prod() - 1.0) if not rebalance_frame.empty else np.nan
    excess_total = portfolio_total - benchmark_total if np.isfinite(portfolio_total) and np.isfinite(benchmark_total) else np.nan
    annualization_factor = 252 / horizon
    annualized_return = (
        float((1.0 + portfolio_total) ** (annualization_factor / len(rebalance_frame)) - 1.0)
        if len(rebalance_frame) > 0 and np.isfinite(portfolio_total)
        else np.nan
    )
    annualized_excess = (
        float((1.0 + excess_total) ** (annualization_factor / len(rebalance_frame)) - 1.0)
        if len(rebalance_frame) > 0 and np.isfinite(excess_total) and excess_total > -1.0
        else np.nan
    )
    sharpe = _annualized_sharpe(rebalance_frame["portfolio_return"], horizon) if not rebalance_frame.empty else np.nan

    return PeriodMetrics(
        days=int(predictions["date"].nunique()),
        samples=int(len(predictions)),
        mean_rank_ic=float(rank_ic_by_day.mean()),
        rank_ic_ir=float(rank_ic_by_day.mean() / rank_ic_by_day.std(ddof=1)) if rank_ic_by_day.std(ddof=1) else np.nan,
        mean_top_bottom_spread=float(spread_by_day.mean()),
        top_k_total_return=portfolio_total,
        top_k_annualized_return=annualized_return,
        top_k_sharpe=sharpe,
        benchmark_total_return=benchmark_total,
        annualized_excess_return=annualized_excess,
        hit_rate=hit_rate,
    )


def evaluate_overlay_predictions(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    horizon: int,
    overlay_config: OverlayConfig,
) -> tuple[OverlayPeriodMetrics, pd.DataFrame]:
    """Evaluate one model with risk filters, sector neutralization, and costs."""

    if predictions.empty:
        empty_metrics = OverlayPeriodMetrics(
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        return empty_metrics, pd.DataFrame()

    eligible = predictions[
        (predictions["close"] >= overlay_config.min_close)
        & (predictions["median_dollar_volume_20"] >= overlay_config.min_median_dollar_volume_20)
        & (predictions["vol_20"] <= overlay_config.max_vol_20)
    ].copy()
    if eligible.empty:
        empty_metrics = OverlayPeriodMetrics(
            int(predictions["date"].nunique()),
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        return empty_metrics, pd.DataFrame()

    if overlay_config.sector_neutral:
        eligible["final_score"] = eligible["score"] - eligible.groupby(["date", "sector"])["score"].transform("mean")
    else:
        eligible["final_score"] = eligible["score"]

    rank_ic_by_day = eligible.groupby("date")[["final_score", "target"]].apply(
        lambda day: day["final_score"].corr(day["target"], method="spearman")
    )
    spread_by_day = eligible.groupby("date")[["final_score", "target"]].apply(_top_bottom_spread_final_score)
    hit_rate = float((spread_by_day > 0).mean())

    rebalances = []
    selected_positions = []
    previous_weights: dict[str, float] = {}
    unique_dates = sorted(eligible["date"].unique())

    for idx in range(0, len(unique_dates), horizon):
        rebalance_date = unique_dates[idx]
        day = eligible[eligible["date"] == rebalance_date].copy()
        selection = _select_top_k_with_sector_cap(
            day,
            top_k=top_k,
            max_positions_per_sector=overlay_config.max_positions_per_sector,
        )
        if selection.empty:
            continue

        weight = 1.0 / len(selection)
        new_weights = {symbol: weight for symbol in selection["symbol"].tolist()}
        turnover = _weight_turnover(previous_weights, new_weights)
        cost = turnover * overlay_config.cost_bps_per_side / 10000.0
        gross_return = float(selection["future_return"].mean())
        net_return = gross_return - cost
        benchmark_return = float(selection["benchmark_future_return"].iloc[0])
        max_sector_weight = float(selection.groupby("sector").size().max() / len(selection))

        rebalances.append(
            {
                "date": rebalance_date,
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "turnover": turnover,
                "cost": cost,
                "cost_bps": cost * 10000.0,
                "names_held": len(selection),
                "sector_count": int(selection["sector"].nunique()),
                "max_sector_weight": max_sector_weight,
            }
        )

        position_frame = selection[["date", "symbol", "sector", "industry", "final_score", "future_return"]].copy()
        position_frame["weight"] = weight
        position_frame["turnover"] = turnover
        position_frame["cost"] = cost
        selected_positions.append(position_frame)
        previous_weights = new_weights

    rebalance_frame = pd.DataFrame(rebalances)
    selected_frame = pd.concat(selected_positions, ignore_index=True) if selected_positions else pd.DataFrame()

    gross_total = _compound_total_return(rebalance_frame["gross_return"])
    net_total = _compound_total_return(rebalance_frame["net_return"])
    benchmark_total = _compound_total_return(rebalance_frame["benchmark_return"])
    gross_annualized = _annualize_total_return(gross_total, len(rebalance_frame), horizon)
    net_annualized = _annualize_total_return(net_total, len(rebalance_frame), horizon)
    benchmark_annualized = _annualize_total_return(benchmark_total, len(rebalance_frame), horizon)
    annualized_net_excess = (
        net_annualized - benchmark_annualized
        if np.isfinite(net_annualized) and np.isfinite(benchmark_annualized)
        else np.nan
    )

    metrics = OverlayPeriodMetrics(
        days=int(eligible["date"].nunique()),
        samples=int(len(eligible)),
        mean_rank_ic=float(rank_ic_by_day.mean()),
        rank_ic_ir=float(rank_ic_by_day.mean() / rank_ic_by_day.std(ddof=1)) if rank_ic_by_day.std(ddof=1) else np.nan,
        mean_top_bottom_spread=float(spread_by_day.mean()),
        gross_total_return=gross_total,
        gross_annualized_return=gross_annualized,
        gross_sharpe=_annualized_sharpe(rebalance_frame["gross_return"], horizon),
        net_total_return=net_total,
        net_annualized_return=net_annualized,
        net_sharpe=_annualized_sharpe(rebalance_frame["net_return"], horizon),
        benchmark_total_return=benchmark_total,
        annualized_net_excess_return=annualized_net_excess,
        hit_rate=hit_rate,
        mean_turnover=float(rebalance_frame["turnover"].mean()) if not rebalance_frame.empty else np.nan,
        mean_cost_bps=float(rebalance_frame["cost_bps"].mean()) if not rebalance_frame.empty else np.nan,
        mean_names_held=float(rebalance_frame["names_held"].mean()) if not rebalance_frame.empty else np.nan,
        mean_max_sector_weight=float(rebalance_frame["max_sector_weight"].mean()) if not rebalance_frame.empty else np.nan,
    )
    return metrics, selected_frame


def save_artifacts(
    output_dir: Path,
    price_data: pd.DataFrame,
    research_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    evaluation: dict[str, dict[str, PeriodMetrics]],
) -> None:
    """Persist experiment artifacts for later inspection."""

    price_data.to_csv(output_dir / "price_data.csv", index=False)
    research_frame.to_csv(output_dir / "research_frame.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    comparison_summary_frame(evaluation).to_csv(output_dir / "summary_metrics.csv", index=False)


def save_risk_aware_artifacts(
    *,
    output_dir: Path,
    price_data: pd.DataFrame,
    metadata: pd.DataFrame,
    research_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    raw_evaluation: dict[str, dict[str, PeriodMetrics]],
    overlay_evaluation: dict[str, dict[str, OverlayPeriodMetrics]],
    selections: pd.DataFrame,
    overlay_config: OverlayConfig,
) -> None:
    """Persist artifacts for the risk-aware experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    price_data.to_csv(output_dir / "price_data.csv", index=False)
    metadata.to_csv(output_dir / "symbol_metadata.csv", index=False)
    research_frame.to_csv(output_dir / "research_frame.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    selections.to_csv(output_dir / "selected_positions.csv", index=False)

    comparison_summary_frame(raw_evaluation).to_csv(output_dir / "raw_summary_metrics.csv", index=False)
    comparison_summary_frame(overlay_evaluation).to_csv(output_dir / "overlay_summary_metrics.csv", index=False)
    pd.DataFrame([asdict(overlay_config)]).to_csv(output_dir / "overlay_config.csv", index=False)


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


def _combine_prediction_frames(
    *,
    component_frames: dict[str, pd.DataFrame],
    model_name: str,
    combine_method: str,
) -> pd.DataFrame:
    _, base_frame = next(iter(component_frames.items()))
    combined = base_frame.copy()
    combined = combined.drop(columns=["score", "confidence", "model"])

    score_columns: list[str] = []
    for component_name, frame in component_frames.items():
        column_name = f"score_{component_name}"
        score_columns.append(column_name)
        combined = combined.merge(
            frame[["date", "symbol", "score"]].rename(columns={"score": column_name}),
            on=["date", "symbol"],
            how="inner",
        )

    if combine_method == "mean_score":
        normalized_columns = []
        for column_name in score_columns:
            normalized_column = f"{column_name}_normalized"
            normalized_columns.append(normalized_column)
            combined[normalized_column] = combined.groupby("date")[column_name].transform(_zscore_series)
        ensemble_score = combined[normalized_columns].mean(axis=1)
    elif combine_method == "rank_average":
        ranked_columns = []
        for column_name in score_columns:
            ranked_column = f"{column_name}_rank_pct"
            ranked_columns.append(ranked_column)
            combined[ranked_column] = combined.groupby("date")[column_name].rank(pct=True)
        ensemble_score = combined[ranked_columns].mean(axis=1)
    else:  # pragma: no cover - spec validation happens in registry
        raise ValueError(f"Unsupported ensemble combine method: {combine_method}")

    return _assemble_predictions(combined, ensemble_score.to_numpy(), model_name)


def _zscore_series(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - values.mean()) / std


def _top_bottom_spread(day: pd.DataFrame) -> float:
    bucket_size = max(1, int(np.ceil(len(day) * 0.2)))
    sorted_day = day.sort_values("score", ascending=False)
    top = sorted_day.head(bucket_size)["target"].mean()
    bottom = sorted_day.tail(bucket_size)["target"].mean()
    return float(top - bottom)


def _top_bottom_spread_final_score(day: pd.DataFrame) -> float:
    bucket_size = max(1, int(np.ceil(len(day) * 0.2)))
    sorted_day = day.sort_values("final_score", ascending=False)
    top = sorted_day.head(bucket_size)["target"].mean()
    bottom = sorted_day.tail(bucket_size)["target"].mean()
    return float(top - bottom)


def _annualized_sharpe(returns: pd.Series, horizon: int) -> float:
    if returns.std(ddof=1) == 0 or returns.empty:
        return np.nan
    return float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252 / horizon))


def _select_top_k_with_sector_cap(
    frame: pd.DataFrame,
    *,
    top_k: int,
    max_positions_per_sector: int,
) -> pd.DataFrame:
    ranked = frame.sort_values("final_score", ascending=False)
    counts: dict[str, int] = {}
    selected_indices = []

    for row in ranked.itertuples():
        sector = row.sector if isinstance(row.sector, str) else "Unknown"
        if counts.get(sector, 0) >= max_positions_per_sector:
            continue
        selected_indices.append(row.Index)
        counts[sector] = counts.get(sector, 0) + 1
        if len(selected_indices) >= top_k:
            break

    return frame.loc[selected_indices].copy().sort_values("final_score", ascending=False)


def _weight_turnover(previous_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    symbols = set(previous_weights) | set(new_weights)
    return float(sum(abs(new_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in symbols))


def _compound_total_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)


def _serialize_metrics(evaluation: dict[str, dict[str, PeriodMetrics]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        model: {period: asdict(metrics) for period, metrics in periods.items()}
        for model, periods in evaluation.items()
    }


def _serialize_overlay_metrics(
    evaluation: dict[str, dict[str, OverlayPeriodMetrics]]
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        model: {period: asdict(metrics) for period, metrics in periods.items()}
        for model, periods in evaluation.items()
    }
