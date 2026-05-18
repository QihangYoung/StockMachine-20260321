"""Phase7G non-momentum factor horizon grid.

Phase7G keeps the Phase7F explicit forward-label setup, but tests non-momentum
style sleeves that are already available from the current price, size,
SEC-filing, insider, and companyfacts panels. The test lockbox is not used.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
    DEFAULT_PHASE1_MEMBERSHIP,
    DEFAULT_PHASE2_BETA_PANEL,
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase7b import (
    DEFAULT_END,
    DEFAULT_EVAL_START,
    DEFAULT_FUNDAMENTAL_PANEL,
    DEFAULT_NON_PRICE_PANEL,
    DEFAULT_QUANTILE,
    DEFAULT_SIZE_PANEL,
    DEFAULT_START,
    DEFAULT_VARIANT,
    _markdown_table,
)
from stockmachine.apps.run_pure_alpha_phase7f import (
    REVERSAL_STYLE_PREFIX,
    StyleSpec,
    _benchmark_feature_frame,
    _correlation_vs_reversal,
    _load_beta,
    _load_membership,
    _payoff_pivot,
    _rolling_corr_summary,
    _rolling_correlation_vs_reversal,
    _style_metrics,
    _style_payoffs,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7g_non_momentum_factor_grid_20260517"
DEFAULT_HOLDING_HORIZONS = (5, 10, 20, 60)
DEFAULT_VOL_WINDOWS = (20, 60, 120)
DEFAULT_ROLLING_CORR_WINDOW = 252
DEFAULT_ROLLING_CORR_MIN_PERIODS = 120

SIZE_COLUMNS = (
    "variant",
    "session_date",
    "symbol",
    "trailing_median_dollar_volume_20",
    "market_cap",
    "market_cap_log",
    "market_cap_log_z",
    "market_cap_percentile",
)
NON_PRICE_COLUMNS = (
    "session_date",
    "symbol",
    "filing_red_flag_score",
    "filing_red_flag_events_20d",
    "filing_red_flag_events_60d",
    "nt_10kq_events_252d",
    "amendment_events_60d",
    "red_8k_events_60d",
    "periodic_delay_events_252d",
    "days_since_last_filing_red_flag",
    "insider_net_buy_score",
    "insider_buy_events_20d",
    "insider_sell_events_20d",
    "insider_cluster_buy_events_60d",
    "insider_buy_intensity_60d",
    "insider_sell_intensity_60d",
    "days_since_last_insider_buy",
)
FUNDAMENTAL_COLUMNS = (
    "session_date",
    "symbol",
    "assets",
    "liabilities",
    "stockholders_equity",
    "cash",
    "cash_and_short_term_investments",
    "revenue",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "total_debt",
    "liabilities_to_assets",
    "total_debt_to_assets",
    "cash_to_assets",
    "equity_to_assets",
    "net_income_margin",
    "operating_cash_flow_margin",
    "negative_net_income_flag",
    "negative_operating_cash_flow_flag",
    "revenue_change_252d",
    "net_income_change_252d",
    "operating_cash_flow_change_252d",
    "fundamental_leverage_pressure_score",
    "fundamental_profit_stress_score",
    "fundamental_fragility_score",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7G non-momentum factor grid.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--beta-panel-path", default=str(DEFAULT_PHASE2_BETA_PANEL))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--fundamental-panel-path", default=str(DEFAULT_FUNDAMENTAL_PANEL))
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument(
        "--benchmark-adj-factor-glob",
        action="append",
        dest="benchmark_adj_factor_globs",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-bottom-quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument(
        "--holding-horizons",
        default=",".join(str(value) for value in DEFAULT_HOLDING_HORIZONS),
    )
    parser.add_argument("--rolling-corr-window", type=int, default=DEFAULT_ROLLING_CORR_WINDOW)
    parser.add_argument(
        "--rolling-corr-min-periods",
        type=int,
        default=DEFAULT_ROLLING_CORR_MIN_PERIODS,
    )
    args = parser.parse_args(argv)

    rollup = build_phase7g_non_momentum_factor_grid(
        membership_path=args.membership_path,
        beta_panel_path=args.beta_panel_path,
        size_panel_path=args.size_panel_path,
        non_price_panel_path=args.non_price_panel_path,
        fundamental_panel_path=args.fundamental_panel_path,
        daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
        adj_factor_globs=tuple(args.adj_factor_globs or DEFAULT_ADJ_FACTOR_GLOBS),
        benchmark_daily_globs=tuple(args.benchmark_daily_globs or DEFAULT_BENCHMARK_DAILY_GLOBS),
        benchmark_adj_factor_globs=tuple(
            args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS
        ),
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        top_bottom_quantile=args.top_bottom_quantile,
        holding_horizons=_parse_int_grid(args.holding_horizons),
        rolling_corr_window=args.rolling_corr_window,
        rolling_corr_min_periods=args.rolling_corr_min_periods,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7g_non_momentum_factor_grid(
    *,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP,
    beta_panel_path: str | Path = DEFAULT_PHASE2_BETA_PANEL,
    size_panel_path: str | Path = DEFAULT_SIZE_PANEL,
    non_price_panel_path: str | Path = DEFAULT_NON_PRICE_PANEL,
    fundamental_panel_path: str | Path = DEFAULT_FUNDAMENTAL_PANEL,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = "SPY",
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    top_bottom_quantile: float = DEFAULT_QUANTILE,
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    rolling_corr_window: int = DEFAULT_ROLLING_CORR_WINDOW,
    rolling_corr_min_periods: int = DEFAULT_ROLLING_CORR_MIN_PERIODS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    membership = _load_membership(membership_path, variant=variant, start=start, end=end)
    beta = _load_beta(beta_panel_path)
    size = _load_size_panel(size_panel_path, variant=variant, start=start, end=end)
    non_price = _load_optional_panel(non_price_panel_path, NON_PRICE_COLUMNS, start=start, end=end)
    fundamentals = _load_optional_panel(
        fundamental_panel_path,
        FUNDAMENTAL_COLUMNS,
        start=start,
        end=end,
    )
    symbols = tuple(sorted(membership["symbol"].astype(str).unique()))
    stock_prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=end,
    )
    benchmark_prices = _load_adjusted_prices(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=end,
    )
    panel = _build_feature_target_panel(
        membership=membership,
        beta=beta,
        size=size,
        non_price=non_price,
        fundamentals=fundamentals,
        stock_prices=stock_prices,
        benchmark_prices=benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        holding_horizons=holding_horizons,
    )
    specs = _factor_specs(holding_horizons=holding_horizons)
    payoff = _style_payoffs(panel, specs=specs, top_bottom_quantile=top_bottom_quantile)
    eval_payoff = payoff[
        payoff["session_date"].ge(eval_start) & payoff["session_date"].le(end)
    ].copy()
    metrics = _style_metrics(eval_payoff)
    coverage = _coverage_summary(eval_payoff)
    corr = _correlation_vs_reversal(eval_payoff)
    rolling_corr = _rolling_correlation_vs_reversal(
        eval_payoff,
        window=rolling_corr_window,
        min_periods=rolling_corr_min_periods,
    )
    rolling_summary = _rolling_corr_summary(rolling_corr)
    metrics = (
        metrics.merge(coverage, on=["holding", "style"], how="left")
        .merge(corr, on=["holding", "style"], how="left")
        .merge(rolling_summary, on=["holding", "style"], how="left")
    )
    family_summary = _family_summary(metrics)
    top_candidates = _top_candidates(metrics)

    panel_path = output_dir / "phase7g_feature_target_panel_sample.csv"
    payoff_path = output_dir / "phase7g_factor_payoff_panel.csv"
    metrics_path = output_dir / "phase7g_factor_metrics.csv"
    family_summary_path = output_dir / "phase7g_family_summary.csv"
    top_candidates_path = output_dir / "phase7g_top_candidates.csv"
    corr_path = output_dir / "phase7g_payoff_corr_vs_reversal.csv"
    rolling_corr_path = output_dir / "phase7g_rolling_corr_vs_reversal.csv"
    memo_path = output_dir / "phase7g_non_momentum_factor_grid_memo.md"
    rollup_path = output_dir / "phase7g_rollup.json"

    panel.head(5000).to_csv(panel_path, index=False)
    payoff.to_csv(payoff_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    family_summary.to_csv(family_summary_path, index=False)
    top_candidates.to_csv(top_candidates_path, index=False)
    corr.to_csv(corr_path, index=False)
    rolling_corr.to_csv(rolling_corr_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            holding_horizons=holding_horizons,
            panel=panel,
            payoff=payoff,
            metrics=metrics,
            family_summary=family_summary,
            top_candidates=top_candidates,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7g_non_momentum_factor_grid",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "holding_horizons": [int(value) for value in holding_horizons],
        "rows": {
            "membership": int(len(membership)),
            "panel": int(len(panel)),
            "payoff": int(len(payoff)),
            "eval_payoff": int(len(eval_payoff)),
            "styles": int(len(specs)),
        },
        "outputs": {
            "feature_target_panel_sample": panel_path.as_posix(),
            "factor_payoff_panel": payoff_path.as_posix(),
            "factor_metrics": metrics_path.as_posix(),
            "family_summary": family_summary_path.as_posix(),
            "top_candidates": top_candidates_path.as_posix(),
            "payoff_corr_vs_reversal": corr_path.as_posix(),
            "rolling_corr_vs_reversal": rolling_corr_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_size_panel(
    path: str | Path,
    *,
    variant: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column in set(SIZE_COLUMNS), low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["variant"] = frame["variant"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[
        frame["variant"].eq(variant)
        & frame["session_date"].ge(start)
        & frame["session_date"].le(end)
    ].copy()
    for column in frame.columns:
        if column not in {"variant", "session_date", "symbol"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["variant"]).sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _load_optional_panel(
    path: str | Path,
    columns: Sequence[str],
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    wanted = set(columns)
    frame = pd.read_csv(path, usecols=lambda column: column in wanted, low_memory=False)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[frame["session_date"].ge(start) & frame["session_date"].le(end)].copy()
    for column in frame.columns:
        if column not in {"session_date", "symbol"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[list(columns)].sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _build_feature_target_panel(
    *,
    membership: pd.DataFrame,
    beta: pd.DataFrame,
    size: pd.DataFrame,
    non_price: pd.DataFrame,
    fundamentals: pd.DataFrame,
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    benchmark_symbol: str,
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    price_features = _stock_feature_frame(
        stock_prices,
        holding_horizons=holding_horizons,
    )
    benchmark_features = _benchmark_feature_frame(
        benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        holding_horizons=holding_horizons,
    )
    panel = (
        membership.merge(price_features, on=["session_date", "symbol"], how="left")
        .merge(beta, on=["session_date", "symbol"], how="left")
        .merge(size, on=["session_date", "symbol"], how="left")
        .merge(non_price, on=["session_date", "symbol"], how="left")
        .merge(fundamentals, on=["session_date", "symbol"], how="left")
        .merge(benchmark_features, on="session_date", how="left")
    )
    for holding in holding_horizons:
        panel[f"forward_beta_residual_return_h{holding}"] = (
            panel[f"forward_return_h{holding}"]
            - panel["beta"] * panel[f"benchmark_forward_return_h{holding}"]
        )
    panel = _add_derived_factor_columns(panel)
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _stock_feature_frame(
    prices: pd.DataFrame,
    *,
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "session_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["return_5d"] = grouped["adjusted_close"].pct_change(5)
    frame["daily_return"] = grouped["adjusted_close"].pct_change()
    for window in DEFAULT_VOL_WINDOWS:
        min_periods = max(10, window // 2)
        frame[f"realized_vol_{window}d"] = (
            grouped["daily_return"]
            .rolling(window, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
        )
    frame["downside_return"] = frame["daily_return"].where(frame["daily_return"] < 0.0, 0.0)
    frame["downside_vol_60d"] = (
        grouped["downside_return"]
        .rolling(60, min_periods=30)
        .std()
        .reset_index(level=0, drop=True)
    )
    rolling_high_60d = (
        grouped["adjusted_close"]
        .rolling(60, min_periods=30)
        .max()
        .reset_index(level=0, drop=True)
    )
    frame["drawdown_from_60d_high"] = (frame["adjusted_close"] / rolling_high_60d - 1.0).abs()
    frame["next_adjusted_open"] = grouped["adjusted_open"].shift(-1)
    for holding in holding_horizons:
        exit_open = grouped["adjusted_open"].shift(-(holding + 1))
        frame[f"forward_return_h{holding}"] = exit_open / frame["next_adjusted_open"] - 1.0
    keep = [
        "session_date",
        "symbol",
        "return_5d",
        "realized_vol_20d",
        "realized_vol_60d",
        "realized_vol_120d",
        "downside_vol_60d",
        "drawdown_from_60d_high",
    ]
    keep.extend(f"forward_return_h{holding}" for holding in holding_horizons)
    return frame[keep]


def _add_derived_factor_columns(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    market_cap = _positive(frame.get("market_cap"))
    frame["book_to_market"] = _ratio(frame.get("stockholders_equity"), market_cap)
    frame["sales_to_price"] = _ratio(frame.get("revenue"), market_cap)
    frame["earnings_yield"] = _ratio(frame.get("net_income"), market_cap)
    frame["cash_flow_yield"] = _ratio(frame.get("operating_cash_flow"), market_cap)
    frame["total_debt_to_market"] = _ratio(frame.get("total_debt"), market_cap)
    frame["log_adv_20d"] = np.log(_positive(frame.get("trailing_median_dollar_volume_20")))

    frame = frame.sort_values(["symbol", "session_date"]).copy()
    assets = pd.to_numeric(frame.get("assets"), errors="coerce")
    lagged_assets = frame.groupby("symbol")["assets"].shift(252)
    frame["asset_growth_252d"] = _ratio(assets - lagged_assets, _positive(lagged_assets))
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _positive(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values > 0.0)


def _ratio(numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series(dtype=float)
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").where(
        pd.to_numeric(denominator, errors="coerce") > 0.0
    )
    values = num / den
    return values.replace([np.inf, -np.inf], np.nan)


def _factor_specs(*, holding_horizons: Sequence[int]) -> list[StyleSpec]:
    base_specs: list[tuple[str, str, str, float, int, str]] = [
        (
            REVERSAL_STYLE_PREFIX,
            "short_horizon_reversal",
            "return_5d",
            -1.0,
            5,
            "Baseline: long recent 5-session losers, short recent winners.",
        ),
        (
            "defensive_low_beta",
            "defensive_low_risk",
            "beta",
            -1.0,
            0,
            "Long lower-beta names, short higher-beta names.",
        ),
        (
            "defensive_low_vol_20d",
            "defensive_low_risk",
            "realized_vol_20d",
            -1.0,
            20,
            "Long lower realized-volatility names using a 20-session window.",
        ),
        (
            "defensive_low_vol_60d",
            "defensive_low_risk",
            "realized_vol_60d",
            -1.0,
            60,
            "Long lower realized-volatility names using a 60-session window.",
        ),
        (
            "defensive_low_vol_120d",
            "defensive_low_risk",
            "realized_vol_120d",
            -1.0,
            120,
            "Long lower realized-volatility names using a 120-session window.",
        ),
        (
            "defensive_low_downside_vol_60d",
            "defensive_low_risk",
            "downside_vol_60d",
            -1.0,
            60,
            "Long lower downside-volatility names using a 60-session window.",
        ),
        (
            "defensive_low_drawdown_60d",
            "defensive_low_risk",
            "drawdown_from_60d_high",
            -1.0,
            60,
            "Long names closer to their 60-session high, short larger drawdowns.",
        ),
        (
            "size_small_cap",
            "size_liquidity",
            "market_cap_log_z",
            -1.0,
            0,
            "Long smaller capitalization names within the liquid universe.",
        ),
        (
            "liquidity_high_adv",
            "size_liquidity",
            "log_adv_20d",
            1.0,
            20,
            "Long higher trailing dollar-volume names.",
        ),
        (
            "quality_low_fragility",
            "quality_profitability",
            "fundamental_fragility_score",
            -1.0,
            0,
            "Long lower composite companyfacts fragility, short higher fragility.",
        ),
        (
            "quality_low_profit_stress",
            "quality_profitability",
            "fundamental_profit_stress_score",
            -1.0,
            0,
            "Long lower operating profit stress, short higher stress.",
        ),
        (
            "quality_low_leverage_pressure",
            "quality_profitability",
            "fundamental_leverage_pressure_score",
            -1.0,
            0,
            "Long lower balance-sheet leverage pressure, short higher pressure.",
        ),
        (
            "quality_high_net_margin",
            "quality_profitability",
            "net_income_margin",
            1.0,
            0,
            "Long higher net-income-margin names.",
        ),
        (
            "quality_high_ocf_margin",
            "quality_profitability",
            "operating_cash_flow_margin",
            1.0,
            0,
            "Long higher operating-cash-flow-margin names.",
        ),
        (
            "quality_high_equity_to_assets",
            "quality_profitability",
            "equity_to_assets",
            1.0,
            0,
            "Long higher equity-to-assets names.",
        ),
        (
            "quality_high_cash_to_assets",
            "quality_profitability",
            "cash_to_assets",
            1.0,
            0,
            "Long higher cash-to-assets names.",
        ),
        (
            "quality_low_liabilities_to_assets",
            "quality_profitability",
            "liabilities_to_assets",
            -1.0,
            0,
            "Long lower liabilities-to-assets names.",
        ),
        (
            "quality_low_debt_to_assets",
            "quality_profitability",
            "total_debt_to_assets",
            -1.0,
            0,
            "Long lower total-debt-to-assets names.",
        ),
        (
            "value_book_to_market",
            "value",
            "book_to_market",
            1.0,
            0,
            "Long higher book-to-market names.",
        ),
        (
            "value_sales_to_price",
            "value",
            "sales_to_price",
            1.0,
            0,
            "Long higher sales-to-market-cap names.",
        ),
        (
            "value_earnings_yield",
            "value",
            "earnings_yield",
            1.0,
            0,
            "Long higher earnings-yield names.",
        ),
        (
            "value_cash_flow_yield",
            "value",
            "cash_flow_yield",
            1.0,
            0,
            "Long higher operating-cash-flow-yield names.",
        ),
        (
            "value_low_debt_to_market",
            "value",
            "total_debt_to_market",
            -1.0,
            0,
            "Long lower debt-to-market-cap names.",
        ),
        (
            "growth_revenue_change_252d",
            "growth_investment",
            "revenue_change_252d",
            1.0,
            252,
            "Long higher 252-session revenue-change names.",
        ),
        (
            "growth_net_income_change_252d",
            "growth_investment",
            "net_income_change_252d",
            1.0,
            252,
            "Long higher 252-session net-income-change names.",
        ),
        (
            "growth_ocf_change_252d",
            "growth_investment",
            "operating_cash_flow_change_252d",
            1.0,
            252,
            "Long higher 252-session operating-cash-flow-change names.",
        ),
        (
            "investment_low_asset_growth_252d",
            "growth_investment",
            "asset_growth_252d",
            -1.0,
            252,
            "Long lower asset-growth names, an investment-anomaly proxy.",
        ),
        (
            "filing_low_red_flag",
            "filing_insider",
            "filing_red_flag_score",
            -1.0,
            60,
            "Long lower SEC filing red-flag pressure names.",
        ),
        (
            "insider_net_buy",
            "filing_insider",
            "insider_net_buy_score",
            1.0,
            60,
            "Long higher insider net-buy-score names.",
        ),
        (
            "insider_buy_intensity_60d",
            "filing_insider",
            "insider_buy_intensity_60d",
            1.0,
            60,
            "Long higher insider buy-intensity names.",
        ),
        (
            "insider_low_sell_intensity_60d",
            "filing_insider",
            "insider_sell_intensity_60d",
            -1.0,
            60,
            "Long lower insider sell-intensity names.",
        ),
    ]
    specs: list[StyleSpec] = []
    for holding in holding_horizons:
        for style, family, signal_column, score_sign, lookback, description in base_specs:
            specs.append(
                StyleSpec(
                    style=f"{style}_h{holding}",
                    family=family,
                    signal_column=signal_column,
                    score_sign=score_sign,
                    formation_lookback=lookback,
                    skip=0,
                    holding=holding,
                    description=f"{description} Forward holding h{holding}.",
                )
            )
    return specs


def _coverage_summary(payoff: pd.DataFrame) -> pd.DataFrame:
    if payoff.empty:
        return pd.DataFrame(columns=["holding", "style"])
    rows = []
    for (holding, style), group in payoff.groupby(["holding", "style"], sort=True):
        rows.append(
            {
                "holding": int(holding),
                "style": style,
                "avg_names": float(group["names"].mean()),
                "min_names": int(group["names"].min()),
                "avg_long_names": float(group["long_names"].mean()),
                "avg_short_names": float(group["short_names"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics[~metrics["family"].eq("short_horizon_reversal")].copy()
    rows = []
    for (holding, family), group in candidates.groupby(["holding", "family"], sort=True):
        best_mean = group.sort_values("mean_payoff_bps", ascending=False).iloc[0]
        best_nw = group.sort_values("newey_west_t_stat", ascending=False).iloc[0]
        rows.append(
            {
                "holding": int(holding),
                "family": family,
                "tested_styles": int(len(group)),
                "positive_styles": int((group["mean_payoff_bps"] > 0.0).sum()),
                "median_mean_payoff_bps": float(group["mean_payoff_bps"].median()),
                "best_mean_style": best_mean["style"],
                "best_mean_payoff_bps": float(best_mean["mean_payoff_bps"]),
                "best_mean_newey_west_t_stat": float(best_mean["newey_west_t_stat"]),
                "best_newey_west_style": best_nw["style"],
                "best_newey_west_payoff_bps": float(best_nw["mean_payoff_bps"]),
                "best_newey_west_t_stat": float(best_nw["newey_west_t_stat"]),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["holding", "best_newey_west_t_stat"], ascending=[True, False])


def _top_candidates(metrics: pd.DataFrame, *, limit_per_holding: int = 12) -> pd.DataFrame:
    candidates = metrics[~metrics["family"].eq("short_horizon_reversal")].copy()
    rows = []
    for holding, group in candidates.groupby("holding", sort=True):
        rows.append(
            group.sort_values(
                ["newey_west_t_stat", "mean_payoff_bps"],
                ascending=[False, False],
            ).head(limit_per_holding)
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    top_bottom_quantile: float,
    holding_horizons: Sequence[int],
    panel: pd.DataFrame,
    payoff: pd.DataFrame,
    metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    top_candidates: pd.DataFrame,
) -> str:
    baseline = metrics[metrics["family"].eq("short_horizon_reversal")]
    lines = [
        "# Phase7G Non-Momentum Factor Grid Results",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- universe variant: `{variant}`",
        f"- data window: `{start}` through `{end}`",
        f"- evaluation window: `{eval_start}` through `{end}`",
        f"- top/bottom quantile: `{top_bottom_quantile}`",
        f"- holding horizons: `{list(holding_horizons)}` sessions",
        f"- panel rows: `{len(panel)}`",
        f"- payoff rows: `{len(payoff)}`",
        "",
        "## Family Summary",
        "",
        _markdown_table(family_summary),
        "",
        "## Top Non-Momentum Candidates",
        "",
        _markdown_table(top_candidates),
        "",
        "## Reversal Baseline",
        "",
        _markdown_table(baseline),
        "",
        "## Interpretation Notes",
        "",
        "- `newey_west_t_stat` uses lag `holding - 1` to reduce the inflation from overlapping forward labels.",
        "- A positive mean here means the factor sleeve's top-quintile basket beat its bottom-quintile basket on beta-residual forward return.",
        "- This is still a validation diagnostic, not a promoted trading policy: no transaction costs, borrow, capacity, turnover, or optimizer constraints are applied.",
        "",
    ]
    return "\n".join(lines)


def _parse_int_grid(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("grid cannot be empty")
    if any(part <= 0 for part in parsed):
        raise ValueError("holding horizons must be positive")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
