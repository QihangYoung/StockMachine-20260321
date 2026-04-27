from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import _add_former_winner_scores
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4af import _add_falling_knife_overlay_scores
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_SELECTOR as BASELINE_SHORT_SELECTOR,
    DEFAULT_SHORT_VARIANT,
    _add_residual_targets,
    _load_panel,
    _load_sec_sic_map,
    _skip_row,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _construct_one_session_turnover_aware,
    _diagnostic_summary,
    _strict_metric_summary,
    _target_turnover_summary,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5f_overlay_combo_on_turnover_aware_20260427"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_TURNOVER_PENALTY = 0.005
DEFAULT_LONG_OVERLAY = "long_reversal_soft_structural_weak_overlay"
DEFAULT_SHORT_OVERLAY = "short_hybrid_soft_fw_overlay"
TURNOVER_AWARE_ALIAS_SERIES = "turnover-aware daily optimizer"

COMBO_SPECS: tuple[dict[str, str], ...] = (
    {
        "portfolio": "baseline_old_selectors",
        "series": "baseline old-old",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": BASELINE_SHORT_SELECTOR,
    },
    {
        "portfolio": "short_overlay_only",
        "series": "short overlay only",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "long_overlay_only",
        "series": "long overlay only",
        "long_score": DEFAULT_LONG_OVERLAY,
        "short_selector": BASELINE_SHORT_SELECTOR,
    },
    {
        "portfolio": "dual_overlay",
        "series": "dual overlay",
        "long_score": DEFAULT_LONG_OVERLAY,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
)

PLOT_COLORS = {
    "baseline old-old": "#0f766e",
    "short overlay only": "#2563eb",
    "long overlay only": "#c2410c",
    "dual overlay": "#7c3aed",
    TURNOVER_AWARE_ALIAS_SERIES: "#2563eb",
    "SPY raw": "#6b7280",
}

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough_path", "2015-06-04", "2015-08-21"),
    ("2019_may_aug_path", "2019-05-01", "2019-08-31"),
    ("2019_may_path", "2019-05-01", "2019-06-05"),
    ("2019_aug_path", "2019-08-01", "2019-08-31"),
)


def build_phase5f_overlay_combo_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
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

    positions, diagnostics, skipped = _construct_combo_positions(
        panel,
        long_variant=long_variant,
        short_variant=short_variant,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        soft_group_penalty=soft_group_penalty,
        turnover_penalty=turnover_penalty,
    )

    positions_path = output_dir / "phase5f_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5f_daily_validation.csv"
    skipped_path = output_dir / "phase5f_skipped_validation.csv"
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=positions_path,
        output_root=strict_root,
        phase3_rollup_path=phase3_rollup_path,
        existing_curve_path=None,
        holding_period_sessions=holding_period_sessions,
        lead_portfolio="short_overlay_only",
    )

    strict_curve_path = Path(strict_root) / "phase4z_strict_h10_daily_curve.csv"
    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])

    benchmark_calendar = (
        strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    target_turnover = _target_turnover_summary(positions)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=benchmark_calendar.to_frame(name="session_date"),
        holding_period_sessions=holding_period_sessions,
    )
    strict_metrics = _strict_metric_summary(strict_curve)
    diagnostic_summary = _diagnostic_summary(diagnostics)

    sweep = (
        strict_metrics.merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(diagnostic_summary, on="portfolio", how="left")
    )
    sweep["series"] = sweep["portfolio"].map(_portfolio_to_series)
    sweep["long_score"] = sweep["portfolio"].map(_portfolio_to_long_score)
    sweep["short_selector"] = sweep["portfolio"].map(_portfolio_to_short_selector)
    sweep["turnover_penalty"] = float(turnover_penalty)

    metrics = _build_metrics(sweep=sweep, strict_curve=strict_curve, turnover_penalty=turnover_penalty)
    windows = _window_return_summary(strict_curve)
    comparison_curve = _build_comparison_curve(strict_curve)
    comparison_curve_long = _build_comparison_curve_long(comparison_curve)

    sweep_path = output_dir / "phase5f_sweep.csv"
    metrics_path = output_dir / "phase5f_metrics.csv"
    windows_path = output_dir / "phase5f_window_returns.csv"
    comparison_curve_path = output_dir / "phase5f_comparison_curve.csv"
    comparison_curve_long_path = output_dir / "phase5f_comparison_curve_long.csv"
    plot_path = output_dir / "phase5f_comparison_plot.png"
    memo_path = output_dir / "phase5f_overlay_combo_memo.md"
    rollup_path = output_dir / "phase5f_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    windows.to_csv(windows_path, index=False)
    comparison_curve.to_csv(comparison_curve_path, index=False)
    comparison_curve_long.to_csv(comparison_curve_long_path, index=False)
    _plot_comparison(comparison_curve, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            sweep=sweep,
            windows=windows,
            plot_path=plot_path,
            turnover_penalty=turnover_penalty,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "phase3_rollup_path": Path(phase3_rollup_path).as_posix(),
        "positions_artifact": positions_path.as_posix(),
        "daily_artifact": diagnostics_path.as_posix(),
        "skipped_artifact": skipped_path.as_posix(),
        "strict_rollup": strict_rollup,
        "sweep_artifact": sweep_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "window_returns_artifact": windows_path.as_posix(),
        "comparison_curve_artifact": comparison_curve_path.as_posix(),
        "comparison_curve_long_artifact": comparison_curve_long_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "combo_specs": list(COMBO_SPECS),
        "turnover_penalty": float(turnover_penalty),
        "method": "turnover_aware_daily_optimizer_selector_overlay_comparison",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _construct_combo_positions(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    short_variant: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    previous_by_portfolio: dict[str, dict[str, dict[str, float]]] = {
        str(spec["portfolio"]): {"long": {}, "short": {}} for spec in COMBO_SPECS
    }

    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        previous_side = previous_by_portfolio[portfolio]
        for session_date in sessions:
            long_group = groups.get((long_variant, session_date))
            short_group = groups.get((short_variant, session_date))
            if long_group is None or short_group is None:
                skipped.append(_skip_row(session_date, portfolio, 0, 0, "missing_group"))
                continue
            book, diagnostic, skip, updated = _construct_one_session_turnover_aware(
                long_group,
                short_group,
                session_date=session_date,
                portfolio=portfolio,
                long_variant=long_variant,
                short_variant=short_variant,
                long_score=str(spec["long_score"]),
                short_selector=str(spec["short_selector"]),
                soft_group="sic2_sector",
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
                turnover_penalty=turnover_penalty,
                previous_weights=previous_side,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostic["series"] = str(spec["series"])
                diagnostics.append(diagnostic)
            if skip is not None:
                skip["series"] = str(spec["series"])
                skipped.append(skip)
            previous_side = updated
        previous_by_portfolio[portfolio] = previous_side

    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _portfolio_to_series(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["series"])
    return str(portfolio)


def _series_key(series: str) -> str:
    return (
        str(series)
        .replace(" ", "_")
        .replace("-", "_")
    )


def _portfolio_to_long_score(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["long_score"])
    return ""


def _portfolio_to_short_selector(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["short_selector"])
    return ""


def _build_metrics(
    *,
    sweep: pd.DataFrame,
    strict_curve: pd.DataFrame,
    turnover_penalty: float,
) -> pd.DataFrame:
    ordered = [_portfolio_to_series(str(spec["portfolio"])) for spec in COMBO_SPECS]
    rows: list[dict[str, Any]] = []
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        row = sweep[sweep["portfolio"].eq(portfolio)]
        if row.empty:
            continue
        item = row.iloc[0]
        rows.append(
            {
                "series": str(spec["series"]),
                "portfolio": portfolio,
                "alias_of": "",
                "long_score": str(spec["long_score"]),
                "short_selector": str(spec["short_selector"]),
                "annualized_return": float(item["strict_annualized_return"]),
                "annualized_vol": float(item["strict_annualized_vol"]),
                "sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                "max_drawdown": float(item["strict_max_drawdown"]),
                "rolling_60_positive_rate": float(item["strict_rolling_60_positive_rate"]),
                "corr_to_spy": float(item["strict_corr_to_spy"]),
                "final_equity": float(item["strict_final_equity"]),
                "mean_daily_return_bps": float(
                    strict_curve[strict_curve["portfolio"].eq(portfolio)]["gross_return"].mean() * 10000.0
                ),
                "target_turnover_mean": float(item["target_turnover_mean"]),
                "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                "aggregate_positions_mean": float(item["aggregate_positions_mean"]),
                "aggregate_gross_mean": float(item["aggregate_gross_mean"]),
                "mean_abs_net_beta": float(item["mean_abs_net_beta"]),
                "mean_long_count": float(item["mean_long_count"]),
                "mean_short_count": float(item["mean_short_count"]),
                "turnover_penalty": float(turnover_penalty),
                "test_window_used": False,
            }
        )
        if portfolio == "short_overlay_only":
            rows.append(
                {
                    "series": TURNOVER_AWARE_ALIAS_SERIES,
                    "portfolio": portfolio,
                    "alias_of": "short overlay only",
                    "long_score": str(spec["long_score"]),
                    "short_selector": str(spec["short_selector"]),
                    "annualized_return": float(item["strict_annualized_return"]),
                    "annualized_vol": float(item["strict_annualized_vol"]),
                    "sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                    "max_drawdown": float(item["strict_max_drawdown"]),
                    "rolling_60_positive_rate": float(item["strict_rolling_60_positive_rate"]),
                    "corr_to_spy": float(item["strict_corr_to_spy"]),
                    "final_equity": float(item["strict_final_equity"]),
                    "mean_daily_return_bps": float(
                        strict_curve[strict_curve["portfolio"].eq(portfolio)]["gross_return"].mean()
                        * 10000.0
                    ),
                    "target_turnover_mean": float(item["target_turnover_mean"]),
                    "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                    "aggregate_positions_mean": float(item["aggregate_positions_mean"]),
                    "aggregate_gross_mean": float(item["aggregate_gross_mean"]),
                    "mean_abs_net_beta": float(item["mean_abs_net_beta"]),
                    "mean_long_count": float(item["mean_long_count"]),
                    "mean_short_count": float(item["mean_short_count"]),
                    "turnover_penalty": float(turnover_penalty),
                    "test_window_used": False,
                }
            )

    benchmark = strict_curve.sort_values(["return_date", "portfolio"]).drop_duplicates("return_date")
    spy_returns = benchmark["benchmark_oto_return"].astype(float)
    spy_equity = (1.0 + spy_returns).cumprod()
    spy_drawdown = spy_equity / spy_equity.cummax() - 1.0
    spy_rolling_60 = spy_equity / spy_equity.shift(60) - 1.0
    rows.append(
        {
            "series": "SPY raw",
            "portfolio": "SPY",
            "alias_of": "",
            "long_score": "",
            "short_selector": "",
            "annualized_return": float(spy_equity.iloc[-1] ** (252.0 / len(spy_equity)) - 1.0),
            "annualized_vol": float(spy_returns.std(ddof=1) * np.sqrt(252.0)),
            "sharpe_no_rf": float(
                spy_returns.mean() / spy_returns.std(ddof=1) * np.sqrt(252.0)
                if spy_returns.std(ddof=1) > 0
                else np.nan
            ),
            "max_drawdown": float(spy_drawdown.min()),
            "rolling_60_positive_rate": float((spy_rolling_60 > 0).mean()),
            "corr_to_spy": 1.0,
            "final_equity": float(spy_equity.iloc[-1]),
            "mean_daily_return_bps": float(spy_returns.mean() * 10000.0),
            "target_turnover_mean": np.nan,
            "aggregate_turnover_mean": np.nan,
            "aggregate_positions_mean": np.nan,
            "aggregate_gross_mean": np.nan,
            "mean_abs_net_beta": np.nan,
            "mean_long_count": np.nan,
            "mean_short_count": np.nan,
            "turnover_penalty": np.nan,
            "test_window_used": False,
        }
    )
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["series"].map(
        {
            name: i
            for i, name in enumerate([*ordered, TURNOVER_AWARE_ALIAS_SERIES, "SPY raw"])
        }
    )
    return frame.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def _window_return_summary(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for portfolio, group in strict_curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date").copy()
        for window, start, end in WINDOWS:
            subset = frame[
                (frame["return_date"] >= pd.Timestamp(start))
                & (frame["return_date"] <= pd.Timestamp(end))
            ]
            if subset.empty:
                continue
            compound = float((1.0 + subset["gross_return"].astype(float)).prod() - 1.0)
            rows.append(
                {
                    "portfolio": portfolio,
                    "series": _portfolio_to_series(portfolio),
                    "window": window,
                    "start": start,
                    "end": end,
                    "compound_return": compound,
                    "hit_rate_daily": float((subset["gross_return"].astype(float) > 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "portfolio"]).reset_index(drop=True)


def _build_comparison_curve(strict_curve: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        strict_curve.sort_values(["return_date", "portfolio"])
        .drop_duplicates("return_date")[
            [
                "return_date",
                "benchmark_oto_return",
                "benchmark_equity",
                "benchmark_drawdown",
            ]
        ]
        .copy()
    )
    benchmark["benchmark_rolling_60_return"] = (
        benchmark["benchmark_equity"].astype(float)
        / benchmark["benchmark_equity"].astype(float).shift(60)
        - 1.0
    )
    out = benchmark
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        series = _series_key(str(spec["series"]))
        subset = strict_curve[strict_curve["portfolio"].eq(portfolio)].copy()
        subset = subset.rename(
            columns={
                "gross_return": f"{series}__return",
                "equity": f"{series}__equity",
                "drawdown": f"{series}__drawdown",
                "rolling_60_return": f"{series}__rolling_60_return",
            }
        )
        out = out.merge(
            subset[
                [
                    "return_date",
                    f"{series}__return",
                    f"{series}__equity",
                    f"{series}__drawdown",
                    f"{series}__rolling_60_return",
                ]
            ],
            on="return_date",
            how="left",
        )
    out["turnover_aware_daily_optimizer__return"] = out["short_overlay_only__return"]
    out["turnover_aware_daily_optimizer__equity"] = out["short_overlay_only__equity"]
    out["turnover_aware_daily_optimizer__drawdown"] = out["short_overlay_only__drawdown"]
    out["turnover_aware_daily_optimizer__rolling_60_return"] = out[
        "short_overlay_only__rolling_60_return"
    ]
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


def _build_comparison_curve_long(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in COMBO_SPECS:
        label = str(spec["series"])
        key = _series_key(label)
        for row in curve.itertuples(index=False):
            rows.append(
                {
                    "return_date": row.return_date,
                    "series": label,
                    "equity": getattr(row, f"{key}__equity"),
                    "drawdown": getattr(row, f"{key}__drawdown"),
                    "rolling_60_return": getattr(row, f"{key}__rolling_60_return"),
                    "daily_return": getattr(row, f"{key}__return"),
                    "alias_of": "",
                    "test_window_used": False,
                }
            )
    for row in curve.itertuples(index=False):
        rows.append(
            {
                "return_date": row.return_date,
                "series": TURNOVER_AWARE_ALIAS_SERIES,
                "equity": row.turnover_aware_daily_optimizer__equity,
                "drawdown": row.turnover_aware_daily_optimizer__drawdown,
                "rolling_60_return": row.turnover_aware_daily_optimizer__rolling_60_return,
                "daily_return": row.turnover_aware_daily_optimizer__return,
                "alias_of": "short overlay only",
                "test_window_used": False,
            }
        )
        rows.append(
            {
                "return_date": row.return_date,
                "series": "SPY raw",
                "equity": row.benchmark_equity,
                "drawdown": row.benchmark_drawdown,
                "rolling_60_return": row.benchmark_rolling_60_return,
                "daily_return": row.benchmark_oto_return,
                "alias_of": "",
                "test_window_used": False,
            }
        )
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

    for spec in COMBO_SPECS:
        label = str(spec["series"])
        key = _series_key(label)
        color = PLOT_COLORS[label]
        plot_label = (
            "turnover-aware daily optimizer (= short overlay only)"
            if label == "short overlay only"
            else label
        )
        ax1.plot(dates, curve[f"{key}__equity"], label=plot_label, color=color, linewidth=2.0)
        ax2.plot(dates, curve[f"{key}__drawdown"] * 100.0, color=color, linewidth=1.8)
        ax3.plot(dates, curve[f"{key}__rolling_60_return"] * 100.0, color=color, linewidth=1.8)

    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.8,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.6,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
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
    ax1.set_title("Phase5F Selector Overlays on Turnover-Aware Optimizer (Validation Only)")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
    windows: pd.DataFrame,
    plot_path: Path,
    turnover_penalty: float,
) -> str:
    display_metrics = metrics.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
    ):
        if column in display_metrics.columns:
            display_metrics[column] = display_metrics[column].map(_fmt_pct_like)
    for column in (
        "sharpe_no_rf",
        "corr_to_spy",
        "mean_daily_return_bps",
        "target_turnover_mean",
        "aggregate_turnover_mean",
        "aggregate_positions_mean",
        "aggregate_gross_mean",
        "mean_abs_net_beta",
        "mean_long_count",
        "mean_short_count",
        "turnover_penalty",
    ):
        if column in display_metrics.columns:
            display_metrics[column] = display_metrics[column].map(_fmt_float_like)
    if "alias_of" in display_metrics.columns:
        display_metrics["alias_of"] = display_metrics["alias_of"].fillna("")

    display_windows = windows.copy()
    if not display_windows.empty:
        display_windows["compound_return"] = display_windows["compound_return"].map(_fmt_pct_like)
        display_windows["hit_rate_daily"] = display_windows["hit_rate_daily"].map(_fmt_pct_like)

    shortlist = sweep[
        [
            "series",
            "long_score",
            "short_selector",
            "strict_annualized_return",
            "strict_sharpe_no_rf",
            "strict_max_drawdown",
            "aggregate_turnover_mean",
            "mean_abs_net_beta",
        ]
    ].copy()
    shortlist["strict_annualized_return"] = shortlist["strict_annualized_return"].map(_fmt_pct_like)
    shortlist["strict_max_drawdown"] = shortlist["strict_max_drawdown"].map(_fmt_pct_like)
    for column in ("strict_sharpe_no_rf", "aggregate_turnover_mean", "mean_abs_net_beta"):
        shortlist[column] = shortlist[column].map(_fmt_float_like)

    lines = [
        "# Phase5F Overlay Combo on Turnover-Aware Optimizer",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Flexible schedule: turnover-aware daily optimizer with fixed `lambda = {turnover_penalty:.4f}`.",
        "- Only selector overlays change; turnover penalty, candidate pool, beta match, and SIC2 soft-neutral construction stay fixed.",
        f"- `{TURNOVER_AWARE_ALIAS_SERIES}` is exactly the same series as `short overlay only` under this fixed-`lambda` setup; it is shown explicitly for easier comparison.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## Metrics",
        "",
        _text_table(display_metrics),
        "",
        "## Selector Summary",
        "",
        _text_table(shortlist),
        "",
        "## Stress Windows",
        "",
        _text_table(display_windows),
        "",
        "## Plot",
        "",
        f"![Phase5F comparison]({plot_path.as_posix()})",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run selector overlay combos on the turnover-aware daily optimizer."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    args = parser.parse_args(argv)

    build_phase5f_overlay_combo_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        turnover_penalty=args.turnover_penalty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
