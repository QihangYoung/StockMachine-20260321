"""Validation-only mean-reversion style payoff diagnostics.

Phase6M checks whether the current pure-alpha candidate is structurally
dependent on short-horizon mean reversion. It keeps the test lockbox closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_VARIANT,
    TARGET_COLUMN,
    _load_panel,
    _load_sec_sic_map,
)
from stockmachine.apps.run_pure_alpha_phase4z import DEFAULT_VALIDATION_PRICE_END

PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
OUTDIR = RESEARCH_ROOT / "phase6m_mean_reversion_failure_diagnostics_20260508"
SIZE_PANEL_PATH = (
    RESEARCH_ROOT
    / "phase6j_true_size_mvp_20260508"
    / "phase6j_true_size_panel_validation.csv.gz"
)

PHASE6K_POSITIONS_PATH = (
    RESEARCH_ROOT
    / "phase6k_size_neutral_optimizer_20260508"
    / "phase6k_positions_validation.csv.gz"
)
PHASE6K_STRICT_CURVE_PATH = (
    RESEARCH_ROOT
    / "phase6k_size_neutral_optimizer_20260508"
    / "strict_h10_rebuild"
    / "phase4z_strict_h10_daily_curve.csv"
)
CURRENT_PORTFOLIO = "size_hard_neutral"
HOLDING_PERIOD_SESSIONS = 10
TAIL_QUANTILE = 0.2
MIN_REGRESSION_ROWS = 80
MIN_DUMMY_COUNT = 5

TRUE_SIZE_TARGET = "true_size_sic2_residual"
FACTOR_FEATURES = (
    "reversal_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d_z",
)


@dataclass(frozen=True)
class FactorSpec:
    factor: str
    feature: str
    direction: int
    description: str


FACTOR_SPECS = (
    FactorSpec(
        factor="reversal_5d_loser_minus_winner",
        feature="reversal_5d_z",
        direction=1,
        description="Long recent 5-session losers, short recent winners.",
    ),
    FactorSpec(
        factor="anti_momentum_20d",
        feature="momentum_20d_z",
        direction=-1,
        description="Long 20-session losers, short 20-session winners.",
    ),
    FactorSpec(
        factor="anti_momentum_60d",
        feature="momentum_60d_z",
        direction=-1,
        description="Long 60-session losers, short 60-session winners.",
    ),
    FactorSpec(
        factor="anti_beta_residual_momentum_20d",
        feature="beta_residual_momentum_20d_z",
        direction=-1,
        description="Long residual losers, short residual winners.",
    ),
    FactorSpec(
        factor="anti_vol_adjusted_momentum_20d",
        feature="vol_adjusted_momentum_20d_z",
        direction=-1,
        description="Long low vol-adjusted momentum, short high vol-adjusted momentum.",
    ),
)


def main() -> None:
    rollup = build_phase6m_mean_reversion_diagnostics()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6m_mean_reversion_diagnostics() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = _load_validation_panel()
    factor_payoff = _build_factor_payoffs(panel)
    payoff_metrics = _build_payoff_metrics(factor_payoff)
    strategy_link = _build_current_strategy_style_link(panel)
    strategy_link_metrics = _build_strategy_link_metrics(strategy_link)
    state_panel, state_summary = _build_state_monitor(factor_payoff)

    factor_payoff_path = OUTDIR / "phase6m_factor_payoff_validation.csv"
    payoff_metrics_path = OUTDIR / "phase6m_factor_payoff_metrics.csv"
    strategy_link_path = OUTDIR / "phase6m_current_strategy_style_link.csv"
    strategy_link_metrics_path = OUTDIR / "phase6m_current_strategy_style_link_metrics.csv"
    state_panel_path = OUTDIR / "phase6m_state_monitor_panel.csv"
    state_summary_path = OUTDIR / "phase6m_state_monitor_summary.csv"
    plot_path = OUTDIR / "phase6m_mean_reversion_payoff_plot.png"
    memo_path = OUTDIR / "phase6m_mean_reversion_failure_memo.md"
    rollup_path = OUTDIR / "phase6m_rollup.json"

    factor_payoff.to_csv(factor_payoff_path, index=False)
    payoff_metrics.to_csv(payoff_metrics_path, index=False)
    strategy_link.to_csv(strategy_link_path, index=False)
    strategy_link_metrics.to_csv(strategy_link_metrics_path, index=False)
    state_panel.to_csv(state_panel_path, index=False)
    state_summary.to_csv(state_summary_path, index=False)
    _plot_factor_payoffs(factor_payoff, plot_path)
    memo_path.write_text(
        _memo(
            payoff_metrics=payoff_metrics,
            strategy_link_metrics=strategy_link_metrics,
            state_summary=state_summary,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6m_mean_reversion_failure_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "current_portfolio": CURRENT_PORTFOLIO,
        "holding_period_sessions": HOLDING_PERIOD_SESSIONS,
        "rows": {
            "panel": int(len(panel)),
            "factor_payoff": int(len(factor_payoff)),
            "strategy_link": int(len(strategy_link)),
            "state_panel": int(len(state_panel)),
        },
        "outputs": {
            "factor_payoff": str(factor_payoff_path),
            "payoff_metrics": str(payoff_metrics_path),
            "strategy_link": str(strategy_link_path),
            "strategy_link_metrics": str(strategy_link_metrics_path),
            "state_panel": str(state_panel_path),
            "state_summary": str(state_summary_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_validation_panel() -> pd.DataFrame:
    panel = _load_panel(DEFAULT_H10_SIGNAL_PANEL, DEFAULT_LONG_VARIANT, DEFAULT_SHORT_VARIANT)
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    panel["variant"] = panel["variant"].astype(str)
    if "test_window_used" in panel.columns and panel["test_window_used"].map(_is_true).any():
        raise ValueError("Signal panel contains test-window rows; refusing to continue.")

    industry_map = _load_sec_sic_map(DEFAULT_CIK_MAPPING, DEFAULT_SEC_SUBMISSIONS_DIR)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")

    size = pd.read_csv(
        SIZE_PANEL_PATH,
        usecols=[
            "variant",
            "session_date",
            "symbol",
            "market_cap_log_z",
            "market_cap_mvp_available",
        ],
        low_memory=False,
    )
    size["session_date"] = pd.to_datetime(size["session_date"])
    size["symbol"] = size["symbol"].astype(str)
    size["variant"] = size["variant"].astype(str)
    panel = panel.merge(size, on=["variant", "session_date", "symbol"], how="left")

    for column in [
        "beta",
        "lagged_close",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
        "reversal_5d",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
        TARGET_COLUMN,
        "market_cap_log_z",
    ]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    panel["trailing_median_dollar_volume_20_log"] = np.log(
        panel["trailing_median_dollar_volume_20"].where(
            panel["trailing_median_dollar_volume_20"] > 0
        )
    )
    for column in [
        "reversal_5d",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
    ]:
        panel[f"{column}_z"] = panel.groupby(["variant", "session_date"], sort=False)[
            column
        ].transform(_zscore)

    panel[TRUE_SIZE_TARGET] = _daily_residualize(
        panel,
        target_column=TARGET_COLUMN,
        numeric_features=(
            "beta",
            "market_cap_log_z",
            "trailing_median_dollar_volume_20_log",
            "liquidity_rank",
        ),
        categorical_features=("sic2_sector",),
        min_regression_rows=MIN_REGRESSION_ROWS,
        min_dummy_count=MIN_DUMMY_COUNT,
    )
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _build_factor_payoffs(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_columns = (TARGET_COLUMN, TRUE_SIZE_TARGET)
    for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True):
        for spec in FACTOR_SPECS:
            feature = pd.to_numeric(group[spec.feature], errors="coerce")
            if feature.notna().sum() < 50:
                continue
            low = feature.quantile(TAIL_QUANTILE)
            high = feature.quantile(1.0 - TAIL_QUANTILE)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue
            if spec.direction > 0:
                long_mask = feature >= high
                short_mask = feature <= low
            else:
                long_mask = feature <= low
                short_mask = feature >= high
            long_frame = group.loc[long_mask]
            short_frame = group.loc[short_mask]
            if len(long_frame) < 5 or len(short_frame) < 5:
                continue
            base = {
                "session_date": session_date,
                "variant": variant,
                "factor": spec.factor,
                "feature": spec.feature,
                "direction": spec.direction,
                "description": spec.description,
                "long_names": int(len(long_frame)),
                "short_names": int(len(short_frame)),
                "long_feature_mean": float(long_frame[spec.feature].mean()),
                "short_feature_mean": float(short_frame[spec.feature].mean()),
            }
            for target in target_columns:
                long_target = pd.to_numeric(long_frame[target], errors="coerce")
                short_target = pd.to_numeric(short_frame[target], errors="coerce")
                base[f"long_{target}"] = float(long_target.mean())
                base[f"short_{target}"] = float(short_target.mean())
                base[f"payoff_{target}"] = float(long_target.mean() - short_target.mean())
            rows.append(base)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["variant", "factor", "session_date"]).reset_index(drop=True)
    for target in target_columns:
        payoff_col = f"payoff_{target}"
        result[f"rolling_60_{payoff_col}"] = result.groupby(
            ["variant", "factor"], sort=False
        )[payoff_col].transform(lambda s: s.rolling(60, min_periods=40).mean())
        result[f"lagged_rolling_60_{payoff_col}"] = result.groupby(
            ["variant", "factor"], sort=False
        )[f"rolling_60_{payoff_col}"].shift(1)
    return result


def _build_payoff_metrics(factor_payoff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, factor), group in factor_payoff.groupby(["variant", "factor"], sort=True):
        spec = _factor_spec(factor)
        for target in (TARGET_COLUMN, TRUE_SIZE_TARGET):
            payoff = pd.to_numeric(group[f"payoff_{target}"], errors="coerce").dropna()
            rolling = pd.to_numeric(
                group[f"rolling_60_payoff_{target}"], errors="coerce"
            ).dropna()
            if payoff.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "factor": factor,
                    "feature": spec.feature if spec else None,
                    "target": target,
                    "sessions": int(len(payoff)),
                    "mean_payoff_bps_per_h10": float(payoff.mean() * 10000.0),
                    "median_payoff_bps_per_h10": float(payoff.median() * 10000.0),
                    "std_payoff_bps_per_h10": float(payoff.std(ddof=1) * 10000.0),
                    "t_stat": float(
                        payoff.mean() / payoff.std(ddof=1) * np.sqrt(len(payoff))
                    )
                    if payoff.std(ddof=1) > 0
                    else np.nan,
                    "positive_rate": float((payoff > 0).mean()),
                    "rolling_60_positive_rate": float((rolling > 0).mean())
                    if not rolling.empty
                    else np.nan,
                    "rolling_60_min_bps": float(rolling.min() * 10000.0)
                    if not rolling.empty
                    else np.nan,
                    "rolling_60_p10_bps": float(rolling.quantile(0.10) * 10000.0)
                    if not rolling.empty
                    else np.nan,
                    "rolling_60_mean_bps": float(rolling.mean() * 10000.0)
                    if not rolling.empty
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_current_strategy_style_link(panel: pd.DataFrame) -> pd.DataFrame:
    positions = pd.read_csv(
        PHASE6K_POSITIONS_PATH,
        usecols=[
            "session_date",
            "portfolio",
            "side",
            "symbol",
            "signed_weight",
            "long_variant",
            "short_variant",
            "test_window_used",
        ],
        low_memory=False,
    )
    positions = positions[positions["portfolio"].astype(str).eq(CURRENT_PORTFOLIO)].copy()
    positions["session_date"] = pd.to_datetime(positions["session_date"])
    if positions["test_window_used"].map(_is_true).any():
        raise ValueError("Positions include test-window rows; refusing to continue.")
    positions["feature_variant"] = np.where(
        positions["side"].astype(str).eq("long"),
        positions["long_variant"].astype(str),
        positions["short_variant"].astype(str),
    )
    positions["symbol"] = positions["symbol"].astype(str)
    positions["signed_weight"] = pd.to_numeric(positions["signed_weight"], errors="coerce")
    features = panel[["session_date", "variant", "symbol", *FACTOR_FEATURES]].rename(
        columns={"variant": "feature_variant"}
    )
    positions = positions.merge(
        features,
        on=["session_date", "feature_variant", "symbol"],
        how="left",
        validate="many_to_one",
    )

    calendar = pd.Series(sorted(positions["session_date"].drop_duplicates()))
    calendar_index = {date: idx for idx, date in enumerate(calendar)}
    active_rows: list[dict[str, Any]] = []
    for signal_date in calendar:
        idx = calendar_index[signal_date]
        start = idx + 2
        stop = start + HOLDING_PERIOD_SESSIONS
        if stop > len(calendar):
            continue
        for return_date in calendar.iloc[start:stop]:
            active_rows.append({"session_date": signal_date, "return_date": return_date})
    active_map = pd.DataFrame(active_rows)
    active = positions.merge(active_map, on="session_date", how="inner")
    active["active_sleeves"] = active.groupby("return_date")["session_date"].transform(
        "nunique"
    )
    active["portfolio_weight"] = active["signed_weight"] / active["active_sleeves"]
    active["pos_weight"] = active["portfolio_weight"].clip(lower=0.0)
    active["short_abs_weight"] = (-active["portfolio_weight"]).clip(lower=0.0)

    exposure_rows: list[dict[str, Any]] = []
    for return_date, group in active.groupby("return_date", sort=True):
        row: dict[str, Any] = {"return_date": return_date}
        for feature in FACTOR_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce")
            long_w = group["pos_weight"].where(values.notna(), 0.0)
            short_w = group["short_abs_weight"].where(values.notna(), 0.0)
            long_total = float(long_w.sum())
            short_total = float(short_w.sum())
            long_mean = (
                float((long_w * values.fillna(0.0)).sum() / long_total)
                if long_total > 0
                else np.nan
            )
            short_mean = (
                float((short_w * values.fillna(0.0)).sum() / short_total)
                if short_total > 0
                else np.nan
            )
            row[f"{feature}_long"] = long_mean
            row[f"{feature}_short"] = short_mean
            row[f"{feature}_spread"] = long_mean - short_mean
        exposure_rows.append(row)
    exposure = pd.DataFrame(exposure_rows)

    curve = pd.read_csv(PHASE6K_STRICT_CURVE_PATH, parse_dates=["return_date"])
    curve = curve[curve["portfolio"].astype(str).eq(CURRENT_PORTFOLIO)].copy()
    for column in ("gross_return", "benchmark_oto_return", "long_gross_return", "short_gross_return"):
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    result = curve.merge(exposure, on="return_date", how="left")
    result = result.sort_values("return_date").reset_index(drop=True)
    result["rolling_60_gross_return"] = (
        (1.0 + result["gross_return"]).rolling(60, min_periods=40).apply(np.prod, raw=True)
        - 1.0
    )
    return result


def _build_strategy_link_metrics(strategy_link: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FACTOR_FEATURES:
        col = f"{feature}_spread"
        valid = strategy_link[[col, "gross_return"]].dropna()
        if valid.empty:
            continue
        x = valid[col].to_numpy(dtype=float)
        y = valid["gross_return"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0] if np.nanstd(x) > 0 else np.nan
        worst = strategy_link[
            strategy_link["rolling_60_gross_return"]
            <= strategy_link["rolling_60_gross_return"].quantile(0.20)
        ]
        best = strategy_link[
            strategy_link["rolling_60_gross_return"]
            >= strategy_link["rolling_60_gross_return"].quantile(0.80)
        ]
        rows.append(
            {
                "feature": feature,
                "sessions": int(len(valid)),
                "spread_mean": float(valid[col].mean()),
                "spread_mean_abs": float(valid[col].abs().mean()),
                "corr_to_daily_return": float(valid[col].corr(valid["gross_return"])),
                "slope_bps_per_zspread": float(slope * 10000.0)
                if np.isfinite(slope)
                else np.nan,
                "worst_rolling_60_spread_mean": float(worst[col].mean()),
                "best_rolling_60_spread_mean": float(best[col].mean()),
                "worst_minus_best_spread": float(worst[col].mean() - best[col].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_state_monitor(factor_payoff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = _load_spy_state()
    state_panel = factor_payoff.merge(spy, on="session_date", how="left")
    payoff_col = f"payoff_{TRUE_SIZE_TARGET}"
    lagged_col = f"lagged_rolling_60_payoff_{TRUE_SIZE_TARGET}"
    state_panel["spy_60d_state"] = _tercile_state(
        state_panel["spy_return_60d"], low_label="weak", mid_label="middle", high_label="strong"
    )
    state_panel["spy_vol20_state"] = _tercile_state(
        state_panel["spy_volatility_20d"], low_label="low_vol", mid_label="mid_vol", high_label="high_vol"
    )
    state_panel["past_mr_payoff_state"] = np.where(
        state_panel[lagged_col] < 0.0, "past_mr_negative", "past_mr_nonnegative"
    )
    state_panel.loc[state_panel[lagged_col].isna(), "past_mr_payoff_state"] = "unknown"

    rows: list[dict[str, Any]] = []
    for state_column in ("spy_60d_state", "spy_vol20_state", "past_mr_payoff_state"):
        for (variant, factor, state), group in state_panel.groupby(
            ["variant", "factor", state_column], sort=True
        ):
            payoff = pd.to_numeric(group[payoff_col], errors="coerce").dropna()
            if payoff.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "factor": factor,
                    "state_variable": state_column,
                    "state": state,
                    "sessions": int(len(payoff)),
                    "mean_payoff_bps_per_h10": float(payoff.mean() * 10000.0),
                    "median_payoff_bps_per_h10": float(payoff.median() * 10000.0),
                    "positive_rate": float((payoff > 0).mean()),
                    "spy_60d_return_mean": float(group["spy_return_60d"].mean()),
                    "spy_vol20_mean": float(group["spy_volatility_20d"].mean()),
                }
            )
    return state_panel, pd.DataFrame(rows)


def _load_spy_state() -> pd.DataFrame:
    prices = _load_adjusted_prices(
        daily_globs=DEFAULT_BENCHMARK_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        symbols=("SPY",),
        end_date=DEFAULT_VALIDATION_PRICE_END,
    )
    frame = prices[prices["symbol"].astype(str).eq("SPY")].sort_values("session_date").copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["spy_return_1d"] = frame["adjusted_close"].pct_change()
    frame["spy_return_20d"] = frame["adjusted_close"].pct_change(20)
    frame["spy_return_60d"] = frame["adjusted_close"].pct_change(60)
    frame["spy_volatility_20d"] = frame["spy_return_1d"].rolling(20, min_periods=10).std()
    return frame[
        [
            "session_date",
            "spy_return_1d",
            "spy_return_20d",
            "spy_return_60d",
            "spy_volatility_20d",
        ]
    ]


def _daily_residualize(
    frame: pd.DataFrame,
    *,
    target_column: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    min_regression_rows: int,
    min_dummy_count: int,
) -> pd.Series:
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    columns = [target_column, *numeric_features, *categorical_features]
    for _, group in frame.groupby(["variant", "session_date"], sort=False):
        valid = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < min_regression_rows:
            continue
        y = valid[target_column].to_numpy(dtype=float)
        pieces = [pd.Series(1.0, index=valid.index, name="intercept")]
        if numeric_features:
            x_num = valid[list(numeric_features)].astype(float)
            x_num = (x_num - x_num.mean(axis=0)) / x_num.std(axis=0, ddof=0).replace(
                0.0, np.nan
            )
            x_num = x_num.dropna(axis=1)
            if not x_num.empty:
                pieces.append(x_num)
        for feature in categorical_features:
            counts = valid[feature].astype(str).value_counts()
            keep = sorted(counts[counts >= min_dummy_count].index)
            if len(keep) <= 1:
                continue
            dummies = pd.get_dummies(valid[feature].astype(str), prefix=feature)
            keep_columns = [f"{feature}_{value}" for value in keep[1:]]
            keep_columns = [column for column in keep_columns if column in dummies.columns]
            if keep_columns:
                pieces.append(dummies[keep_columns].astype(float))
        x = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        try:
            coeffs, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = x.to_numpy(dtype=float) @ coeffs
        residuals.loc[valid.index] = y - fitted
    return residuals


def _plot_factor_payoffs(factor_payoff: pd.DataFrame, path: Path) -> None:
    if factor_payoff.empty:
        return
    lead_variant = DEFAULT_LONG_VARIANT
    plot_frame = factor_payoff[
        factor_payoff["variant"].astype(str).eq(lead_variant)
    ].copy()
    if plot_frame.empty:
        plot_frame = factor_payoff.copy()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for factor, group in plot_frame.groupby("factor", sort=True):
        ordered = group.sort_values("session_date").copy()
        equity = (1.0 + ordered[f"payoff_{TRUE_SIZE_TARGET}"].fillna(0.0)).cumprod()
        axes[0].plot(ordered["session_date"], equity, label=factor)
        axes[1].plot(
            ordered["session_date"],
            ordered[f"rolling_60_payoff_{TRUE_SIZE_TARGET}"] * 10000.0,
            label=factor,
        )
    axes[0].axhline(1.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[0].set_title("Phase6M Mean-Reversion Factor Payoff Diagnostics (Validation Only)")
    axes[0].set_ylabel("Diagnostic compounded payoff")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Rolling 60 mean bps / h10")
    axes[1].set_xlabel("Signal date")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    payoff_metrics: pd.DataFrame,
    strategy_link_metrics: pd.DataFrame,
    state_summary: pd.DataFrame,
    plot_path: Path,
) -> str:
    top = payoff_metrics[
        payoff_metrics["target"].astype(str).eq(TRUE_SIZE_TARGET)
        & payoff_metrics["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)
    ].copy()
    top = top.sort_values("mean_payoff_bps_per_h10", ascending=False)
    link = strategy_link_metrics.copy()
    weak_state = state_summary[
        state_summary["state_variable"].astype(str).eq("past_mr_payoff_state")
    ].copy()

    lines = [
        "# Phase6M Mean-Reversion Failure Diagnostics",
        "",
        "Scope: validation-window only. Test lockbox remains closed.",
        "",
        "## Main Read",
        "",
        "- The current candidate has large intentional mean-reversion exposure, but exposure level alone does not explain bad windows.",
        "- The more important object is factor payoff: when loser-minus-winner payoff turns negative, the selector's economic premise is inverted.",
        "- This memo uses a true-size/SIC2/liquidity/beta residual target to reduce the chance that the measured payoff is merely size, liquidity, sector, or beta.",
        "",
        "## Factor Payoff Metrics",
        "",
        "| Factor | Mean bps / h10 | Positive rate | Rolling-60 positive rate | Rolling-60 p10 bps |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.mean_payoff_bps_per_h10:.2f} | "
            f"{row.positive_rate:.2%} | {row.rolling_60_positive_rate:.2%} | "
            f"{row.rolling_60_p10_bps:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Current Strategy Exposure Link",
            "",
            "| Feature spread | Mean spread | Corr to daily return | Slope bps / z-spread | Worst60 minus best60 spread |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in link.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.spread_mean:.3f} | {row.corr_to_daily_return:.3f} | "
            f"{row.slope_bps_per_zspread:.2f} | {row.worst_minus_best_spread:.3f} |"
        )
    lines.extend(
        [
            "",
            "## State Monitor",
            "",
            "The most actionable white-box state is the factor's own lagged rolling payoff. If the mean-reversion payoff has already turned negative, continuing to run the same selector is mechanically fragile.",
            "",
            "| Factor | State | Sessions | Mean bps / h10 | Positive rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    weak_state = weak_state[
        weak_state["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)
    ].sort_values(["factor", "state"])
    for row in weak_state.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.state} | {int(row.sessions)} | "
            f"{row.mean_payoff_bps_per_h10:.2f} | {row.positive_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If a factor has positive average payoff but meaningful negative rolling-60 episodes, it is not a broken factor; it is a state-dependent factor.",
            "- If the current strategy's bad rolling windows do not coincide with more extreme exposure, then capping exposure alone may not solve the issue.",
            "- A cleaner next experiment is a white-box activation rule: reduce or disable the reversal/momentum sleeve when its own lagged rolling payoff is negative.",
            "",
            f"Plot: `{plot_path}`",
        ]
    )
    return "\n".join(lines)


def _factor_spec(factor: str) -> FactorSpec | None:
    for spec in FACTOR_SPECS:
        if spec.factor == factor:
            return spec
    return None


def _zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    std = clean.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (clean - clean.mean()) / std


def _tercile_state(
    values: pd.Series,
    *,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    low = clean.quantile(1.0 / 3.0)
    high = clean.quantile(2.0 / 3.0)
    result = pd.Series(mid_label, index=values.index, dtype=object)
    result[clean <= low] = low_label
    result[clean >= high] = high_label
    result[clean.isna()] = "unknown"
    return result


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
