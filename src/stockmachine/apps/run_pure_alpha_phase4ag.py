from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import (
    _add_former_winner_scores,
    _fmt_bps,
    _fmt_pct,
    _table,
)
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4af import _add_falling_knife_overlay_scores
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_SELECTOR,
    DEFAULT_SHORT_VARIANT,
    _add_residual_targets,
    _construct_one_session,
    _load_panel,
    _load_sec_sic_map,
    _skip_row,
    _summary,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4ag_dual_overlay_combo_20260425"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_HOLDING_PERIOD_SESSIONS = 10
PORTFOLIO_SPECS = (
    {"hard_group": None, "soft_group": "sic2_sector"},
)
COMBO_SPECS = (
    {
        "portfolio": "sic2_soft_neutral",
        "label": "baseline",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_SELECTOR,
    },
    {
        "portfolio": "sic2_soft_neutral__long_soft_structural_weak_overlay",
        "label": "long_overlay_only",
        "long_score": "long_reversal_soft_structural_weak_overlay",
        "short_selector": DEFAULT_SHORT_SELECTOR,
    },
    {
        "portfolio": "sic2_soft_neutral__short_hybrid_soft_fw_overlay",
        "label": "short_overlay_only",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": "short_hybrid_soft_fw_overlay",
    },
    {
        "portfolio": "sic2_soft_neutral__dual_soft_overlay",
        "label": "dual_overlay",
        "long_score": "long_reversal_soft_structural_weak_overlay",
        "short_selector": "short_hybrid_soft_fw_overlay",
    },
)
WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("full", "2014-08-05", "2019-12-31"),
    ("2015_peak_to_trough_signal", "2015-05-19", "2015-08-19"),
    ("2015_recovery_signal", "2015-08-07", "2015-11-05"),
    ("2018Q1_stress", "2018-01-26", "2018-04-02"),
    ("2019_may_aug_signal", "2019-04-15", "2019-08-28"),
    ("2019_may_signal", "2019-04-15", "2019-06-05"),
    ("2019_aug_signal", "2019-07-18", "2019-08-28"),
)
PLOT_LABELS = {
    "sic2_soft_neutral": "baseline",
    "sic2_soft_neutral__long_soft_structural_weak_overlay": "long soft overlay",
    "sic2_soft_neutral__short_hybrid_soft_fw_overlay": "short hybrid overlay",
    "sic2_soft_neutral__dual_soft_overlay": "dual overlay",
}
PLOT_COLORS = {
    "sic2_soft_neutral": "#0f766e",
    "sic2_soft_neutral__long_soft_structural_weak_overlay": "#c2410c",
    "sic2_soft_neutral__short_hybrid_soft_fw_overlay": "#2563eb",
    "sic2_soft_neutral__dual_soft_overlay": "#7c3aed",
    "spy_scaled": "#6b7280",
}


def build_phase4ag_dual_overlay_combo(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
    """Compare baseline, single-side overlays, and dual overlay in one validation-only pass."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = _add_falling_knife_overlay_scores(panel)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )

    positions, beta_daily, skipped = _construct_combo_positions(
        panel,
        combos=COMBO_SPECS,
        long_variant=long_variant,
        short_variant=short_variant,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        soft_group_penalty=soft_group_penalty,
    )
    beta_summary = _summary(beta_daily, skipped)
    beta_window = _beta_matched_window_summary(beta_daily)

    positions_path = output_dir / "phase4ag_beta_matched_positions_validation.csv.gz"
    daily_path = output_dir / "phase4ag_beta_matched_daily_validation.csv"
    summary_path = output_dir / "phase4ag_beta_matched_summary_validation.csv"
    window_path = output_dir / "phase4ag_beta_matched_window_summary_validation.csv"
    skipped_path = output_dir / "phase4ag_beta_matched_skipped_validation.csv"

    positions.to_csv(positions_path, index=False, compression="gzip")
    beta_daily.to_csv(daily_path, index=False)
    beta_summary.to_csv(summary_path, index=False)
    beta_window.to_csv(window_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=positions_path,
        output_root=strict_root,
        phase3_rollup_path=phase3_rollup_path,
        existing_curve_path=None,
        holding_period_sessions=holding_period_sessions,
        lead_portfolio="sic2_soft_neutral",
    )

    strict_curve_path = Path(strict_root) / "phase4z_strict_h10_daily_curve.csv"
    comparison = _comparison_curve(strict_curve_path)
    metrics = _metrics_table(comparison)
    annual = _annual_table(comparison)
    window_returns = _window_return_table(comparison)

    compare_curve_path = output_dir / "phase4ag_path_compare_curve.csv"
    compare_metrics_path = output_dir / "phase4ag_path_compare_metrics.csv"
    compare_annual_path = output_dir / "phase4ag_path_compare_annual_returns.csv"
    compare_window_path = output_dir / "phase4ag_path_compare_window_returns.csv"
    compare_plot_path = output_dir / "phase4ag_path_compare.png"
    memo_path = output_dir / "phase4ag_dual_overlay_combo_memo.md"
    rollup_path = output_dir / "phase4ag_rollup.json"

    comparison.to_csv(compare_curve_path, index=False)
    metrics.to_csv(compare_metrics_path, index=False)
    annual.to_csv(compare_annual_path, index=False)
    window_returns.to_csv(compare_window_path, index=False)
    _plot_comparison(comparison, output_path=compare_plot_path)
    memo_path.write_text(_memo(metrics, window_returns, compare_plot_path), encoding="utf-8")

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "phase3_rollup_path": Path(phase3_rollup_path).as_posix(),
        "cik_mapping_path": Path(cik_mapping_path).as_posix(),
        "sec_submissions_dir": Path(sec_submissions_dir).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "portfolio_specs": list(PORTFOLIO_SPECS),
        "combo_specs": list(COMBO_SPECS),
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "min_nonzero_names": int(min_nonzero_names),
        "score_weight": float(score_weight),
        "soft_group_penalty": float(soft_group_penalty),
        "min_regression_rows": int(min_regression_rows),
        "min_dummy_count": int(min_dummy_count),
        "holding_period_sessions": int(holding_period_sessions),
        "positions_rows": int(len(positions)),
        "beta_daily_rows": int(len(beta_daily)),
        "skipped_rows": int(len(skipped)),
        "strict_rollup": strict_rollup,
        "beta_matched_positions_artifact": positions_path.as_posix(),
        "beta_matched_daily_artifact": daily_path.as_posix(),
        "beta_matched_summary_artifact": summary_path.as_posix(),
        "beta_matched_window_artifact": window_path.as_posix(),
        "beta_matched_skipped_artifact": skipped_path.as_posix(),
        "comparison_curve_artifact": compare_curve_path.as_posix(),
        "comparison_metrics_artifact": compare_metrics_path.as_posix(),
        "comparison_annual_artifact": compare_annual_path.as_posix(),
        "comparison_window_artifact": compare_window_path.as_posix(),
        "comparison_plot_artifact": compare_plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "baseline_vs_single_side_overlays_vs_dual_overlay",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _construct_combo_positions(
    panel: pd.DataFrame,
    *,
    combos: Sequence[dict[str, str]],
    long_variant: str,
    short_variant: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    for combo in combos:
        for portfolio_spec in PORTFOLIO_SPECS:
            for session_date in sessions:
                long_group = groups.get((long_variant, session_date))
                short_group = groups.get((short_variant, session_date))
                if long_group is None or short_group is None:
                    skipped.append(
                        _skip_row(session_date, combo["portfolio"], 0, 0, "missing_group")
                    )
                    continue
                book, diagnostic, skip = _construct_one_session(
                    long_group,
                    short_group,
                    session_date=session_date,
                    portfolio=combo["portfolio"],
                    long_variant=long_variant,
                    short_variant=short_variant,
                    long_score=combo["long_score"],
                    short_selector=combo["short_selector"],
                    hard_group=portfolio_spec["hard_group"],
                    soft_group=portfolio_spec["soft_group"],
                    candidate_pool_per_side=candidate_pool_per_side,
                    max_single_name_side_weight=max_single_name_side_weight,
                    min_nonzero_names=min_nonzero_names,
                    score_weight=score_weight,
                    soft_group_penalty=soft_group_penalty,
                )
                positions.extend(book)
                if diagnostic is not None:
                    daily_rows.append(diagnostic)
                if skip is not None:
                    skipped.append(skip)
    return pd.DataFrame(positions), pd.DataFrame(daily_rows), pd.DataFrame(skipped)


def _beta_matched_window_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["session_date_dt"] = pd.to_datetime(frame["session_date"])
    rows: list[dict[str, Any]] = []
    for window, start, end in WINDOWS:
        subset = frame[
            (frame["session_date_dt"] >= pd.Timestamp(start))
            & (frame["session_date_dt"] <= pd.Timestamp(end))
        ]
        for portfolio, group in subset.groupby("portfolio", sort=True):
            rows.append(
                {
                    "window": window,
                    "start": start,
                    "end": end,
                    "portfolio": portfolio,
                    "long_score": str(group["long_score"].iloc[0]),
                    "short_selector": str(group["short_selector"].iloc[0]),
                    "sessions": int(len(group)),
                    "mean_spread_raw": float(group["spread_raw"].mean()),
                    "hit_rate_raw": float((group["spread_raw"] > 0).mean()),
                    "mean_long_raw": float(group["long_raw"].mean()),
                    "mean_short_raw_contribution": float(
                        group["short_raw_contribution"].mean()
                    ),
                    "mean_spread_beta_residual": float(group["spread_beta_residual"].mean()),
                    "hit_rate_beta_residual": float(
                        (group["spread_beta_residual"] > 0).mean()
                    ),
                    "mean_long_beta_residual": float(group["long_beta_residual"].mean()),
                    "mean_short_beta_residual_contribution": float(
                        group["short_beta_residual_contribution"].mean()
                    ),
                    "mean_spread_risk_factor_sic2_residual": float(
                        group["spread_risk_factor_sic2_residual"].mean()
                    ),
                    "mean_spread_style_factor_sic2_residual": float(
                        group["spread_style_factor_sic2_residual"].mean()
                    ),
                    "mean_sic2_l1_exposure": float(group["sic2_l1_exposure"].mean()),
                    "mean_abs_net_beta": float(group["net_beta"].abs().mean()),
                    "test_window_used": bool(group["test_window_used"].any()),
                }
            )
    return pd.DataFrame(rows)


def _comparison_curve(strict_curve_path: str | Path) -> pd.DataFrame:
    curve = pd.read_csv(strict_curve_path)
    curve["return_date"] = pd.to_datetime(curve["return_date"])
    if curve["test_window_used"].astype(str).str.lower().eq("true").any():
        raise ValueError("Strict path curve includes test-window rows; refusing to compare.")

    merged: pd.DataFrame | None = None
    for portfolio in PLOT_LABELS:
        subset = curve[curve["portfolio"].eq(portfolio)].copy()
        if subset.empty:
            raise ValueError(f"Portfolio {portfolio!r} missing from {strict_curve_path}.")
        keep = subset[
            [
                "return_date",
                "gross_return",
                "long_gross_return",
                "short_gross_return",
                "equity",
                "drawdown",
                "rolling_60_return",
                "benchmark_oto_return",
                "benchmark_equity",
                "benchmark_drawdown",
            ]
        ].rename(
            columns={
                "gross_return": f"{portfolio}__return",
                "long_gross_return": f"{portfolio}__long_return",
                "short_gross_return": f"{portfolio}__short_return",
                "equity": f"{portfolio}__equity",
                "drawdown": f"{portfolio}__drawdown",
                "rolling_60_return": f"{portfolio}__rolling_60_return",
            }
        )
        if merged is None:
            merged = keep
        else:
            merged = merged.merge(
                keep.drop(
                    columns=[
                        "benchmark_oto_return",
                        "benchmark_equity",
                        "benchmark_drawdown",
                    ],
                    errors="ignore",
                ),
                on="return_date",
                how="inner",
            )
    if merged is None or merged.empty:
        raise ValueError("Unable to build comparison curve.")
    merged["benchmark_rolling_60_return"] = (
        merged["benchmark_equity"] / merged["benchmark_equity"].shift(60)
        - 1.0
    )
    merged["test_window_used"] = False
    return merged.sort_values("return_date").reset_index(drop=True)


def _metrics_table(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spy_return = curve["benchmark_oto_return"].astype(float)
    for portfolio, label in PLOT_LABELS.items():
        returns = curve[f"{portfolio}__return"].astype(float)
        equity = curve[f"{portfolio}__equity"].astype(float)
        rolling_60 = curve[f"{portfolio}__rolling_60_return"].dropna().astype(float)
        rows.append(
            {
                "series": label,
                "portfolio": portfolio,
                "start": str(curve["return_date"].min().date()),
                "end": str(curve["return_date"].max().date()),
                "final_equity": float(equity.iloc[-1]),
                "annualized_return": float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0),
                "annualized_vol": float(returns.std(ddof=1) * np.sqrt(252.0)),
                "sharpe_no_rf": float(
                    returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
                    if returns.std(ddof=1) > 0
                    else np.nan
                ),
                "max_drawdown": float(curve[f"{portfolio}__drawdown"].min()),
                "rolling_60_positive_rate": float((rolling_60 > 0).mean()),
                "corr_to_spy": float(returns.corr(spy_return)),
                "mean_daily_return_bps": float(returns.mean() * 10000.0),
                "test_window_used": False,
            }
        )
    spy_equity = curve["benchmark_equity"].astype(float)
    spy_returns = spy_equity.pct_change().dropna()
    spy_rolling_60 = curve["benchmark_rolling_60_return"].dropna().astype(float)
    rows.append(
        {
            "series": "spy_raw",
            "portfolio": "SPY",
            "start": str(curve["return_date"].min().date()),
            "end": str(curve["return_date"].max().date()),
            "final_equity": float(spy_equity.iloc[-1]),
            "annualized_return": float(spy_equity.iloc[-1] ** (252.0 / len(spy_equity)) - 1.0),
            "annualized_vol": float(spy_returns.std(ddof=1) * np.sqrt(252.0)),
            "sharpe_no_rf": float(
                spy_returns.mean() / spy_returns.std(ddof=1) * np.sqrt(252.0)
                if spy_returns.std(ddof=1) > 0
                else np.nan
            ),
            "max_drawdown": float(curve["benchmark_drawdown"].min()),
            "rolling_60_positive_rate": float((spy_rolling_60 > 0).mean()),
            "corr_to_spy": 1.0,
            "mean_daily_return_bps": float(spy_return.mean() * 10000.0),
            "test_window_used": False,
        }
    )
    return pd.DataFrame(rows)


def _annual_table(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = curve.copy()
    frame["year"] = frame["return_date"].dt.year
    for year, group in frame.groupby("year", sort=True):
        row: dict[str, Any] = {"year": int(year)}
        for portfolio, label in PLOT_LABELS.items():
            returns = group[f"{portfolio}__return"].astype(float)
            row[label] = float((1.0 + returns).prod() - 1.0)
        row["spy_raw"] = float(
            (1.0 + group["benchmark_oto_return"].astype(float)).prod() - 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _window_return_table(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, start, end in WINDOWS:
        subset = curve[
            (curve["return_date"] >= pd.Timestamp(start)) & (curve["return_date"] <= pd.Timestamp(end))
        ].copy()
        if subset.empty:
            continue
        row: dict[str, Any] = {
            "window": window,
            "start": start,
            "end": end,
        }
        for portfolio, label in PLOT_LABELS.items():
            returns = subset[f"{portfolio}__return"].astype(float)
            row[f"{label}_compound_return"] = float((1.0 + returns).prod() - 1.0)
            row[f"{label}_mean_daily_bps"] = float(returns.mean() * 10000.0)
        row["spy_raw_compound_return"] = float(
            (1.0 + subset["benchmark_oto_return"].astype(float)).prod() - 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_comparison(curve: pd.DataFrame, *, output_path: str | Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8, 1.8]},
    )
    ax1, ax2, ax3 = axes
    dates = curve["return_date"]

    for portfolio, label in PLOT_LABELS.items():
        color = PLOT_COLORS[portfolio]
        ax1.plot(dates, curve[f"{portfolio}__equity"], label=label, color=color, linewidth=2.0)
        ax2.plot(dates, curve[f"{portfolio}__drawdown"] * 100.0, color=color, linewidth=1.8)
        ax3.plot(
            dates,
            curve[f"{portfolio}__rolling_60_return"] * 100.0,
            color=color,
            linewidth=1.8,
        )

    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["spy_scaled"],
        linestyle="--",
        linewidth=1.8,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["spy_scaled"],
        linestyle="--",
        linewidth=1.6,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["spy_scaled"],
        linestyle="--",
        linewidth=1.6,
    )

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax3.set_xlabel("Date")
    ax1.set_title("Strict H10 Gross Paths: Baseline vs Long/Short/Dual Overlay vs Raw SPY")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(metrics: pd.DataFrame, window_returns: pd.DataFrame, plot_path: Path) -> str:
    metrics_display = metrics.copy()
    metrics_display["annualized_return"] = metrics_display["annualized_return"].map(_fmt_pct)
    metrics_display["annualized_vol"] = metrics_display["annualized_vol"].map(_fmt_pct)
    metrics_display["sharpe_no_rf"] = metrics_display["sharpe_no_rf"].map(lambda x: f"{x:.2f}")
    metrics_display["max_drawdown"] = metrics_display["max_drawdown"].map(_fmt_pct)
    metrics_display["rolling_60_positive_rate"] = metrics_display["rolling_60_positive_rate"].map(
        _fmt_pct
    )
    metrics_display["corr_to_spy"] = metrics_display["corr_to_spy"].map(lambda x: f"{x:.3f}")
    metrics_display["mean_daily_return_bps"] = metrics_display["mean_daily_return_bps"].map(
        lambda x: f"{x:.2f}"
    )

    windows = window_returns[
        window_returns["window"].isin(
            ["2015_peak_to_trough_signal", "2015_recovery_signal", "2019_may_signal", "2019_aug_signal"]
        )
    ].copy()
    for col in windows.columns:
        if col.endswith("_compound_return"):
            windows[col] = windows[col].map(_fmt_pct)
        if col.endswith("_mean_daily_bps"):
            windows[col] = windows[col].map(lambda x: f"{x:.2f}")

    lines = [
        "# Phase4AG Dual Overlay Combo Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Question",
        "",
        "If long-side falling-knife overlay and short-side weakening-confirmation overlay are both sensible on their own, does combining them create a better whole, or does it over-smooth the portfolio and give up too much alpha?",
        "",
        "## Portfolio Definitions",
        "",
        "- `baseline`: reversal_5d long + short_core_plus_overextension short.",
        "- `long soft overlay`: long_reversal_soft_structural_weak_overlay + old short selector.",
        "- `short hybrid overlay`: old long selector + short_hybrid_soft_fw_overlay.",
        "- `dual overlay`: long_reversal_soft_structural_weak_overlay + short_hybrid_soft_fw_overlay.",
        "",
        "## Full-Window Strict H10 Metrics",
        "",
        _table(
            metrics_display[
                [
                    "series",
                    "annualized_return",
                    "annualized_vol",
                    "sharpe_no_rf",
                    "max_drawdown",
                    "rolling_60_positive_rate",
                    "corr_to_spy",
                    "mean_daily_return_bps",
                ]
            ]
        ),
        "",
        "## Key Window Compound Returns",
        "",
        _table(windows),
        "",
        "## Plot",
        "",
        f"![Phase4AG comparison]({plot_path.as_posix()})",
        "",
        "## Readout",
        "",
        "Dual overlay is useful only if the path quality gain is worth the extra alpha drag versus the stronger single-side overlay variants.",
        "",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build validation-only comparison for baseline, single-side overlays, and dual overlay."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--long-variant", default=DEFAULT_LONG_VARIANT)
    parser.add_argument("--short-variant", default=DEFAULT_SHORT_VARIANT)
    parser.add_argument("--candidate-pool-per-side", type=int, default=80)
    parser.add_argument("--max-single-name-side-weight", type=float, default=1.0 / 30.0)
    parser.add_argument("--min-nonzero-names", type=int, default=20)
    parser.add_argument("--score-weight", type=float, default=0.01)
    parser.add_argument("--soft-group-penalty", type=float, default=25.0)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    parser.add_argument("--min-dummy-count", type=int, default=5)
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    args = parser.parse_args(argv)

    build_phase4ag_dual_overlay_combo(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        long_variant=args.long_variant,
        short_variant=args.short_variant,
        candidate_pool_per_side=args.candidate_pool_per_side,
        max_single_name_side_weight=args.max_single_name_side_weight,
        min_nonzero_names=args.min_nonzero_names,
        score_weight=args.score_weight,
        soft_group_penalty=args.soft_group_penalty,
        min_regression_rows=args.min_regression_rows,
        min_dummy_count=args.min_dummy_count,
        holding_period_sessions=args.holding_period_sessions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
