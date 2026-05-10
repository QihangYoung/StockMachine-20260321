"""Validation-only mean-reversion horizon surface diagnostics.

Phase6N separates two failure modes:

1. Direction flip: recent losers keep losing and winners keep winning across
   longer horizons.
2. Horizon delay: h10 looks bad, but h20/h40 recovers, suggesting the mean
   reversion clock has lengthened rather than inverted.

The test lockbox remains closed. Forward returns are only computed when the
full exit open is available inside the validation price window.
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
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
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
OUTDIR = RESEARCH_ROOT / "phase6n_mean_reversion_horizon_surface_20260508"
SIZE_PANEL_PATH = (
    RESEARCH_ROOT
    / "phase6j_true_size_mvp_20260508"
    / "phase6j_true_size_panel_validation.csv.gz"
)

HORIZONS = (1, 3, 5, 10, 20, 40)
TAIL_QUANTILE = 0.2
MIN_REGRESSION_ROWS = 80
MIN_DUMMY_COUNT = 5

TRUE_SIZE_TARGET_PREFIX = "true_size_sic2_residual_h"
BETA_RESIDUAL_PREFIX = "beta_residual_h"
FORWARD_RETURN_PREFIX = "forward_return_h"
BENCHMARK_FORWARD_PREFIX = "benchmark_forward_return_h"


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
    rollup = build_phase6n_mean_reversion_horizon_surface()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6n_mean_reversion_horizon_surface() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = _load_horizon_panel()
    payoff = _build_factor_payoff_surface(panel)
    metrics = _build_surface_metrics(payoff)
    profile = _build_horizon_profiles(metrics)
    delay_flip = _build_delay_flip_summary(payoff)

    payoff_path = OUTDIR / "phase6n_horizon_surface_payoff_validation.csv"
    metrics_path = OUTDIR / "phase6n_horizon_surface_metrics.csv"
    profile_path = OUTDIR / "phase6n_horizon_profile.csv"
    delay_flip_path = OUTDIR / "phase6n_delay_vs_flip_summary.csv"
    plot_path = OUTDIR / "phase6n_horizon_surface_plot.png"
    memo_path = OUTDIR / "phase6n_horizon_surface_memo.md"
    rollup_path = OUTDIR / "phase6n_rollup.json"

    payoff.to_csv(payoff_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    profile.to_csv(profile_path, index=False)
    delay_flip.to_csv(delay_flip_path, index=False)
    _plot_horizon_surface(metrics, plot_path)
    memo_path.write_text(
        _memo(metrics=metrics, profile=profile, delay_flip=delay_flip, plot_path=plot_path),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6n_mean_reversion_horizon_surface",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "validation_price_end": DEFAULT_VALIDATION_PRICE_END,
        "horizons": list(HORIZONS),
        "rows": {
            "panel": int(len(panel)),
            "payoff": int(len(payoff)),
            "metrics": int(len(metrics)),
            "profile": int(len(profile)),
            "delay_flip": int(len(delay_flip)),
        },
        "outputs": {
            "payoff": str(payoff_path),
            "metrics": str(metrics_path),
            "profile": str(profile_path),
            "delay_flip": str(delay_flip_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_horizon_panel() -> pd.DataFrame:
    panel = _load_panel(DEFAULT_H10_SIGNAL_PANEL, DEFAULT_LONG_VARIANT, DEFAULT_SHORT_VARIANT)
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    panel["variant"] = panel["variant"].astype(str)

    industry_map = _load_sec_sic_map(DEFAULT_CIK_MAPPING, DEFAULT_SEC_SUBMISSIONS_DIR)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")

    size = pd.read_csv(
        SIZE_PANEL_PATH,
        usecols=["variant", "session_date", "symbol", "market_cap_log_z"],
        low_memory=False,
    )
    size["session_date"] = pd.to_datetime(size["session_date"])
    size["symbol"] = size["symbol"].astype(str)
    size["variant"] = size["variant"].astype(str)
    size["market_cap_log_z"] = pd.to_numeric(size["market_cap_log_z"], errors="coerce")
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

    symbols = sorted(panel["symbol"].dropna().unique())
    stock_forward = _load_forward_returns(
        daily_globs=DEFAULT_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_ADJ_FACTOR_GLOBS,
        symbols=symbols,
        prefix=FORWARD_RETURN_PREFIX,
    )
    benchmark_forward = _load_forward_returns(
        daily_globs=DEFAULT_BENCHMARK_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        symbols=("SPY",),
        prefix=BENCHMARK_FORWARD_PREFIX,
    )
    benchmark_forward = benchmark_forward.drop(columns=["symbol"])
    panel = panel.merge(stock_forward, on=["session_date", "symbol"], how="left")
    panel = panel.merge(benchmark_forward, on="session_date", how="left")

    beta_residual_targets: list[str] = []
    for horizon in HORIZONS:
        stock_col = f"{FORWARD_RETURN_PREFIX}{horizon}"
        benchmark_col = f"{BENCHMARK_FORWARD_PREFIX}{horizon}"
        residual_col = f"{BETA_RESIDUAL_PREFIX}{horizon}"
        panel[residual_col] = panel[stock_col] - panel["beta"] * panel[benchmark_col]
        beta_residual_targets.append(residual_col)

    residuals = _daily_residualize_many(
        panel,
        target_columns=beta_residual_targets,
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
    for horizon in HORIZONS:
        panel[f"{TRUE_SIZE_TARGET_PREFIX}{horizon}"] = residuals[f"{BETA_RESIDUAL_PREFIX}{horizon}"]

    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _load_forward_returns(
    *,
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    symbols: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=DEFAULT_VALIDATION_PRICE_END,
    )
    prices["session_date"] = pd.to_datetime(prices["session_date"])
    prices = prices.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    grouped = prices.groupby("symbol", group_keys=False)
    next_open = grouped["adjusted_open"].shift(-1)
    for horizon in HORIZONS:
        exit_open = grouped["adjusted_open"].shift(-(horizon + 1))
        prices[f"{prefix}{horizon}"] = exit_open / next_open - 1.0
    return prices[["session_date", "symbol", *[f"{prefix}{h}" for h in HORIZONS]]]


def _build_factor_payoff_surface(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
            for horizon in HORIZONS:
                for target_prefix, target_label in (
                    (BETA_RESIDUAL_PREFIX, "beta_residual"),
                    (TRUE_SIZE_TARGET_PREFIX, "true_size_sic2_residual"),
                ):
                    target_col = f"{target_prefix}{horizon}"
                    valid = group[[spec.feature, target_col]].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    long_values = valid.loc[long_mask, target_col].dropna()
                    short_values = valid.loc[short_mask, target_col].dropna()
                    if len(long_values) < 5 or len(short_values) < 5:
                        continue
                    payoff = float(long_values.mean() - short_values.mean())
                    rows.append(
                        {
                            "session_date": session_date,
                            "variant": variant,
                            "factor": spec.factor,
                            "feature": spec.feature,
                            "target": target_label,
                            "horizon": int(horizon),
                            "long_names": int(len(long_values)),
                            "short_names": int(len(short_values)),
                            "long_target_mean": float(long_values.mean()),
                            "short_target_mean": float(short_values.mean()),
                            "payoff": payoff,
                            "payoff_bps": payoff * 10000.0,
                            "payoff_bps_per_session": payoff * 10000.0 / horizon,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["variant", "factor", "target", "session_date", "horizon"]
    )


def _build_surface_metrics(payoff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, factor, target, horizon), group in payoff.groupby(
        ["variant", "factor", "target", "horizon"], sort=True
    ):
        values = pd.to_numeric(group["payoff"], errors="coerce").dropna()
        if values.empty:
            continue
        ordered = group.sort_values("session_date").copy()
        ordered["rolling_60_payoff"] = ordered["payoff"].rolling(60, min_periods=40).mean()
        rolling = ordered["rolling_60_payoff"].dropna()
        rows.append(
            {
                "variant": variant,
                "factor": factor,
                "target": target,
                "horizon": int(horizon),
                "sessions": int(len(values)),
                "mean_payoff_bps": float(values.mean() * 10000.0),
                "mean_payoff_bps_per_session": float(values.mean() * 10000.0 / horizon),
                "median_payoff_bps": float(values.median() * 10000.0),
                "positive_rate": float((values > 0).mean()),
                "t_stat": float(values.mean() / values.std(ddof=1) * np.sqrt(len(values)))
                if values.std(ddof=1) > 0
                else np.nan,
                "rolling_60_positive_rate": float((rolling > 0).mean())
                if not rolling.empty
                else np.nan,
                "rolling_60_min_bps": float(rolling.min() * 10000.0)
                if not rolling.empty
                else np.nan,
                "rolling_60_p10_bps": float(rolling.quantile(0.10) * 10000.0)
                if not rolling.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_horizon_profiles(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, factor, target), group in metrics.groupby(
        ["variant", "factor", "target"], sort=True
    ):
        by_h = group.set_index("horizon").sort_index()
        if by_h.empty:
            continue
        best_total_h = int(by_h["mean_payoff_bps"].idxmax())
        best_per_session_h = int(by_h["mean_payoff_bps_per_session"].idxmax())
        h10 = by_h["mean_payoff_bps"].get(10, np.nan)
        h20 = by_h["mean_payoff_bps"].get(20, np.nan)
        h40 = by_h["mean_payoff_bps"].get(40, np.nan)
        rows.append(
            {
                "variant": variant,
                "factor": factor,
                "target": target,
                "best_total_horizon": best_total_h,
                "best_per_session_horizon": best_per_session_h,
                "h10_mean_bps": h10,
                "h20_mean_bps": h20,
                "h40_mean_bps": h40,
                "h20_minus_h10_bps": h20 - h10 if np.isfinite(h20) and np.isfinite(h10) else np.nan,
                "h40_minus_h10_bps": h40 - h10 if np.isfinite(h40) and np.isfinite(h10) else np.nan,
                "interpretation": _profile_interpretation(h10, h20, h40),
            }
        )
    return pd.DataFrame(rows)


def _build_delay_flip_summary(payoff: pd.DataFrame) -> pd.DataFrame:
    target = payoff[payoff["target"].astype(str).eq("true_size_sic2_residual")].copy()
    pivot = target.pivot_table(
        index=["variant", "factor", "session_date"],
        columns="horizon",
        values="payoff",
        aggfunc="mean",
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for (variant, factor), group in pivot.groupby(["variant", "factor"], sort=True):
        valid = group.dropna(subset=[10])
        neg = valid[valid[10] < 0.0].copy()
        if neg.empty:
            rows.append(
                {
                    "variant": variant,
                    "factor": factor,
                    "sessions_with_h10": int(len(valid)),
                    "h10_negative_sessions": 0,
                    "h10_negative_rate": 0.0,
                    "h20_recovery_rate_given_h10_negative": np.nan,
                    "h40_recovery_rate_given_h10_negative": np.nan,
                    "persistent_negative_h20_h40_rate": np.nan,
                    "mean_h10_when_negative_bps": np.nan,
                    "mean_h20_after_h10_negative_bps": np.nan,
                    "mean_h40_after_h10_negative_bps": np.nan,
                }
            )
            continue
        has20 = neg[20].notna() if 20 in neg.columns else pd.Series(False, index=neg.index)
        has40 = neg[40].notna() if 40 in neg.columns else pd.Series(False, index=neg.index)
        persistent = (
            (neg[20] <= 0.0).fillna(False) & (neg[40] <= 0.0).fillna(False)
            if 20 in neg.columns and 40 in neg.columns
            else pd.Series(False, index=neg.index)
        )
        rows.append(
            {
                "variant": variant,
                "factor": factor,
                "sessions_with_h10": int(len(valid)),
                "h10_negative_sessions": int(len(neg)),
                "h10_negative_rate": float(len(neg) / len(valid)) if len(valid) else np.nan,
                "h20_recovery_rate_given_h10_negative": float((neg.loc[has20, 20] > 0).mean())
                if has20.any()
                else np.nan,
                "h40_recovery_rate_given_h10_negative": float((neg.loc[has40, 40] > 0).mean())
                if has40.any()
                else np.nan,
                "persistent_negative_h20_h40_rate": float(persistent.mean())
                if len(persistent)
                else np.nan,
                "mean_h10_when_negative_bps": float(neg[10].mean() * 10000.0),
                "mean_h20_after_h10_negative_bps": float(neg[20].mean() * 10000.0)
                if 20 in neg.columns
                else np.nan,
                "mean_h40_after_h10_negative_bps": float(neg[40].mean() * 10000.0)
                if 40 in neg.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _daily_residualize_many(
    frame: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    min_regression_rows: int,
    min_dummy_count: int,
) -> pd.DataFrame:
    residuals = pd.DataFrame(np.nan, index=frame.index, columns=list(target_columns), dtype=float)
    x_columns = [*numeric_features, *categorical_features]
    for _, group in frame.groupby(["variant", "session_date"], sort=False):
        base = group[x_columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(base) < min_regression_rows:
            continue
        pieces = [pd.Series(1.0, index=base.index, name="intercept")]
        if numeric_features:
            x_num = base[list(numeric_features)].astype(float)
            x_num = (x_num - x_num.mean(axis=0)) / x_num.std(axis=0, ddof=0).replace(
                0.0, np.nan
            )
            x_num = x_num.dropna(axis=1)
            if not x_num.empty:
                pieces.append(x_num)
        for feature in categorical_features:
            counts = base[feature].astype(str).value_counts()
            keep = sorted(counts[counts >= min_dummy_count].index)
            if len(keep) <= 1:
                continue
            dummies = pd.get_dummies(base[feature].astype(str), prefix=feature)
            keep_columns = [f"{feature}_{value}" for value in keep[1:]]
            keep_columns = [column for column in keep_columns if column in dummies.columns]
            if keep_columns:
                pieces.append(dummies[keep_columns].astype(float))
        x_base = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for target_column in target_columns:
            y = pd.to_numeric(frame.loc[x_base.index, target_column], errors="coerce")
            valid_idx = y.dropna().index
            if len(valid_idx) < min_regression_rows:
                continue
            x = x_base.loc[valid_idx]
            try:
                coeffs, *_ = np.linalg.lstsq(
                    x.to_numpy(dtype=float),
                    y.loc[valid_idx].to_numpy(dtype=float),
                    rcond=None,
                )
            except np.linalg.LinAlgError:
                continue
            fitted = x.to_numpy(dtype=float) @ coeffs
            residuals.loc[valid_idx, target_column] = y.loc[valid_idx].to_numpy(dtype=float) - fitted
    return residuals


def _plot_horizon_surface(metrics: pd.DataFrame, path: Path) -> None:
    plot_frame = metrics[
        metrics["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)
        & metrics["target"].astype(str).eq("true_size_sic2_residual")
    ].copy()
    if plot_frame.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for factor, group in plot_frame.groupby("factor", sort=True):
        ordered = group.sort_values("horizon")
        axes[0].plot(
            ordered["horizon"],
            ordered["mean_payoff_bps"],
            marker="o",
            label=factor,
        )
        axes[1].plot(
            ordered["horizon"],
            ordered["mean_payoff_bps_per_session"],
            marker="o",
            label=factor,
        )
    axes[0].axhline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[0].set_title("Phase6N Mean-Reversion Horizon Surface (Validation Only)")
    axes[0].set_ylabel("Mean payoff bps / horizon")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Mean payoff bps / session")
    axes[1].set_xlabel("Forward horizon sessions")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    profile: pd.DataFrame,
    delay_flip: pd.DataFrame,
    plot_path: Path,
) -> str:
    key_metrics = metrics[
        metrics["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)
        & metrics["target"].astype(str).eq("true_size_sic2_residual")
    ].copy()
    key_profile = profile[
        profile["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)
        & profile["target"].astype(str).eq("true_size_sic2_residual")
    ].copy()
    key_delay = delay_flip[delay_flip["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)].copy()

    lines = [
        "# Phase6N Mean-Reversion Horizon Surface",
        "",
        "Scope: validation-window only. Test lockbox remains closed.",
        "",
        "## Question",
        "",
        "Does mean reversion fail because the direction flips, or because the payoff horizon lengthens beyond h10?",
        "",
        "## Horizon Surface",
        "",
        "| Factor | h1 | h3 | h5 | h10 | h20 | h40 | Best total horizon | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in key_profile.sort_values("factor").itertuples(index=False):
        subset = key_metrics[key_metrics["factor"].eq(row.factor)].set_index("horizon")
        values = []
        for horizon in HORIZONS:
            value = subset["mean_payoff_bps"].get(horizon, np.nan)
            values.append("NA" if pd.isna(value) else f"{value:.2f}")
        lines.append(
            f"| {row.factor} | {' | '.join(values)} | {int(row.best_total_horizon)} | "
            f"{row.interpretation} |"
        )
    lines.extend(
        [
            "",
            "## h10 Negative Episodes",
            "",
            "| Factor | h10 negative rate | h20 recovery | h40 recovery | Persistent negative h20/h40 | h10 mean when negative | h40 mean after h10 negative |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in key_delay.sort_values("factor").itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.h10_negative_rate:.2%} | "
            f"{_fmt_pct(row.h20_recovery_rate_given_h10_negative)} | "
            f"{_fmt_pct(row.h40_recovery_rate_given_h10_negative)} | "
            f"{_fmt_pct(row.persistent_negative_h20_h40_rate)} | "
            f"{row.mean_h10_when_negative_bps:.2f} | {row.mean_h40_after_h10_negative_bps:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- If h20/h40 meaningfully recovers after h10 is negative, the failure mode is partly horizon delay.",
            "- If h20 and h40 stay negative after h10 is negative, the failure mode is closer to direction flip / momentum continuation.",
            "- If average h40 is worse than h10, extending holding period is unlikely to be a clean fix.",
            "",
            f"Plot: `{plot_path}`",
        ]
    )
    return "\n".join(lines)


def _profile_interpretation(h10: float, h20: float, h40: float) -> str:
    if not np.isfinite(h10):
        return "insufficient h10 data"
    if h10 < 0 and ((np.isfinite(h20) and h20 > 0) or (np.isfinite(h40) and h40 > 0)):
        return "horizon delay candidate"
    if h10 < 0 and (not np.isfinite(h40) or h40 <= 0):
        return "direction flip candidate"
    if np.isfinite(h40) and h40 > h10:
        return "payoff lengthens beyond h10"
    if np.isfinite(h40) and h40 < h10:
        return "payoff decays after h10"
    return "h10 roughly adequate"


def _zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    std = clean.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (clean - clean.mean()) / std


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


if __name__ == "__main__":
    main()
