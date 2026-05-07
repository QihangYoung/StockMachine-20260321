from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

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
    DEFAULT_SHORT_VARIANT,
    _load_panel,
    _load_sec_sic_map,
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5o import (
    DEFAULT_FUNDAMENTAL_TRIGGER_THRESHOLD,
    DEFAULT_LONG_FUNDAMENTAL_PANEL,
    DEFAULT_LONG_NON_PRICE_PANEL,
    DEFAULT_RECENT_FILING_DAYS,
    DEFAULT_SOFT_PENALTY_WEIGHT,
    DEFAULT_TRIGGER_THRESHOLD,
    _add_non_price_long_scores,
)
from stockmachine.apps.run_pure_alpha_phase5p import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PHASE5P_ROOT,
    _add_price_falling_knife_scores,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5q_price_fk_diagnostics_20260506"
DEFAULT_PHASE5P_POSITIONS = DEFAULT_PHASE5P_ROOT / "phase5p_positions_validation.csv.gz"
DEFAULT_PHASE5P_SLEEVE_RETURNS = (
    DEFAULT_PHASE5P_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_sleeve_returns.csv.gz"
)
DEFAULT_PHASE5P_ACTIVATION = DEFAULT_PHASE5P_ROOT / "phase5p_price_state_activation_summary.csv"
DEFAULT_PHASE5P_METRICS = DEFAULT_PHASE5P_ROOT / "phase5p_metrics.csv"
DEFAULT_PHASE5P_WINDOWS = DEFAULT_PHASE5P_ROOT / "phase5p_window_returns.csv"

CURRENT_PORTFOLIO = "current_sota"
COMPARISON_PORTFOLIOS = (
    "np_fof_low_mom60_veto_t075",
    "price_fk_soft_w100",
    "price_fk_x_np_soft_w075",
)
SERIES_BY_PORTFOLIO = {
    "current_sota": "current sota",
    "np_fof_low_mom60_veto_t075": "np fof low mom60 veto t075",
    "price_fk_soft_w100": "price fk soft w100",
    "price_fk_x_np_soft_w075": "price fk x np soft w075",
}

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)

FACTOR_COLUMNS = (
    "price_falling_knife_score",
    "price_falling_knife_x_np_score",
    "long_np_plus_fund_score",
    "price_fk_sector_mom20_down",
    "price_fk_sector_mom60_down",
    "price_fk_sector_relative_mom20_down",
    "price_fk_sector_relative_mom60_down",
    "price_fk_residual_mom20_down",
    "price_fk_vol_adjusted_down",
    "price_fk_no_stabilization",
    "falling_knife_structural_weakness",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "beta_z",
    "forward_return_5d",
    "forward_beta_residual_return_5d",
)


def build_phase5q_price_fk_diagnostics_artifacts(
    *,
    positions_path: str | Path = DEFAULT_PHASE5P_POSITIONS,
    sleeve_returns_path: str | Path = DEFAULT_PHASE5P_SLEEVE_RETURNS,
    activation_path: str | Path = DEFAULT_PHASE5P_ACTIVATION,
    metrics_path: str | Path = DEFAULT_PHASE5P_METRICS,
    windows_path: str | Path = DEFAULT_PHASE5P_WINDOWS,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    long_non_price_panel_path: str | Path = DEFAULT_LONG_NON_PRICE_PANEL,
    long_fundamental_panel_path: str | Path = DEFAULT_LONG_FUNDAMENTAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = pd.read_csv(positions_path, low_memory=False)
    sleeve_returns = pd.read_csv(sleeve_returns_path, parse_dates=["return_date"], low_memory=False)
    activation = pd.read_csv(activation_path)
    metrics = pd.read_csv(metrics_path)
    windows = pd.read_csv(windows_path)
    factors = _build_long_factor_panel(
        signal_panel_path=signal_panel_path,
        long_non_price_panel_path=long_non_price_panel_path,
        long_fundamental_panel_path=long_fundamental_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        long_variant=long_variant,
        short_variant=short_variant,
    )

    side_attr = _side_attribution(sleeve_returns)
    exposure = _long_exposure_by_window(positions=positions, factors=factors)
    deltas = _long_reweight_delta_table(positions=positions, factors=factors)
    delta_rollup = _long_reweight_delta_rollup(deltas)
    score_breadth = _score_breadth_summary(activation)

    side_attr_path = output_dir / "phase5q_side_attribution_by_window.csv"
    exposure_path = output_dir / "phase5q_long_exposure_by_window.csv"
    deltas_path = output_dir / "phase5q_long_reweight_deltas.csv"
    delta_rollup_path = output_dir / "phase5q_long_reweight_delta_rollup.csv"
    score_breadth_path = output_dir / "phase5q_score_breadth_summary.csv"
    memo_path = output_dir / "phase5q_price_fk_diagnostics_memo.md"
    rollup_path = output_dir / "phase5q_rollup.json"

    side_attr.to_csv(side_attr_path, index=False)
    exposure.to_csv(exposure_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    delta_rollup.to_csv(delta_rollup_path, index=False)
    score_breadth.to_csv(score_breadth_path, index=False)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            windows=windows,
            side_attr=side_attr,
            exposure=exposure,
            delta_rollup=delta_rollup,
            score_breadth=score_breadth,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "positions_path": Path(positions_path).as_posix(),
        "sleeve_returns_path": Path(sleeve_returns_path).as_posix(),
        "metrics_path": Path(metrics_path).as_posix(),
        "windows_path": Path(windows_path).as_posix(),
        "artifacts": {
            "side_attribution": side_attr_path.as_posix(),
            "long_exposure_by_window": exposure_path.as_posix(),
            "long_reweight_deltas": deltas_path.as_posix(),
            "long_reweight_delta_rollup": delta_rollup_path.as_posix(),
            "score_breadth": score_breadth_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "method": "diagnose_why_phase5p_price_falling_knife_overlay_repair_is_modest",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_long_factor_panel(
    *,
    signal_panel_path: str | Path,
    long_non_price_panel_path: str | Path,
    long_fundamental_panel_path: str | Path,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    long_variant: str,
    short_variant: str,
) -> pd.DataFrame:
    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = _add_falling_knife_overlay_scores(panel)
    panel = _add_non_price_long_scores(
        panel,
        long_non_price_panel_path=long_non_price_panel_path,
        long_fundamental_panel_path=long_fundamental_panel_path,
        long_variant=long_variant,
        trigger_threshold=DEFAULT_TRIGGER_THRESHOLD,
        fundamental_trigger_threshold=DEFAULT_FUNDAMENTAL_TRIGGER_THRESHOLD,
        recent_filing_days=DEFAULT_RECENT_FILING_DAYS,
        soft_penalty_weight=DEFAULT_SOFT_PENALTY_WEIGHT,
    )
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel = _add_price_falling_knife_scores(panel, long_variant=long_variant)
    long_panel = panel[panel["variant"].astype(str).eq(long_variant)].copy()
    columns = ["session_date", "symbol", "sic2_sector", "sic4_industry", DEFAULT_LONG_SCORE, *FACTOR_COLUMNS]
    return long_panel[columns].drop_duplicates(["session_date", "symbol"]).reset_index(drop=True)


def _side_attribution(sleeve_returns: pd.DataFrame) -> pd.DataFrame:
    daily = (
        sleeve_returns.groupby(["portfolio", "return_date"], as_index=False)
        .agg(
            gross_return=("sleeve_gross_return", "mean"),
            long_return=("sleeve_long_return", "mean"),
            short_return=("sleeve_short_return", "mean"),
            active_sleeves=("sleeve_session_date", "nunique"),
        )
    )
    rows: list[dict[str, Any]] = []
    for portfolio, group in daily.groupby("portfolio", sort=True):
        for window, start, end in WINDOWS:
            subset = group[
                (group["return_date"] >= pd.Timestamp(start))
                & (group["return_date"] <= pd.Timestamp(end))
            ]
            if subset.empty:
                continue
            rows.append(
                {
                    "portfolio": portfolio,
                    "series": SERIES_BY_PORTFOLIO.get(portfolio, portfolio),
                    "window": window,
                    "start": start,
                    "end": end,
                    "compound_return": float((1.0 + subset["gross_return"]).prod() - 1.0),
                    "long_sum_bps": float(subset["long_return"].sum() * 10000.0),
                    "short_sum_bps": float(subset["short_return"].sum() * 10000.0),
                    "gross_sum_bps": float(subset["gross_return"].sum() * 10000.0),
                    "hit_rate_daily": float((subset["gross_return"] > 0).mean()),
                    "mean_active_sleeves": float(subset["active_sleeves"].mean()),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "portfolio"]).reset_index(drop=True)


def _long_exposure_by_window(*, positions: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    longs = positions[positions["side"].eq("long")].copy()
    factor_features = factors.drop(
        columns=["forward_return_5d", "forward_beta_residual_return_5d"],
        errors="ignore",
    )
    merged = longs.merge(factor_features, on=["session_date", "symbol"], how="left")
    rows: list[dict[str, Any]] = []
    exposure_cols = [
        "price_falling_knife_score",
        "price_falling_knife_x_np_score",
        "long_np_plus_fund_score",
        "falling_knife_structural_weakness",
        "momentum_60d_z",
        "beta_residual_momentum_20d_z",
        "beta_z",
        "forward_return_5d",
        "forward_beta_residual_return_5d",
    ]
    for portfolio, group in merged.groupby("portfolio", sort=True):
        for window, start, end in WINDOWS:
            subset = group[(group["session_date"] >= start) & (group["session_date"] <= end)].copy()
            if subset.empty:
                continue
            weights = pd.to_numeric(subset["side_weight"], errors="coerce").fillna(0.0)
            denom = float(weights.sum())
            row: dict[str, Any] = {
                "portfolio": portfolio,
                "series": SERIES_BY_PORTFOLIO.get(portfolio, portfolio),
                "window": window,
                "start": start,
                "end": end,
                "position_rows": int(len(subset)),
                "sessions": int(subset["session_date"].nunique()),
                "test_window_used": False,
            }
            for column in exposure_cols:
                values = pd.to_numeric(subset[column], errors="coerce").fillna(0.0)
                row[f"weighted_{column}"] = float(np.dot(weights, values) / denom) if denom > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["window", "portfolio"]).reset_index(drop=True)


def _long_reweight_delta_table(*, positions: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    longs = positions[positions["side"].eq("long")].copy()
    weight_frame = longs[["session_date", "portfolio", "symbol", "side_weight"]].copy()
    rows: list[pd.DataFrame] = []
    for compare_portfolio in COMPARISON_PORTFOLIOS:
        subset = weight_frame[weight_frame["portfolio"].isin([CURRENT_PORTFOLIO, compare_portfolio])]
        pivot = (
            subset.pivot_table(
                index=["session_date", "symbol"],
                columns="portfolio",
                values="side_weight",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        if CURRENT_PORTFOLIO not in pivot.columns:
            pivot[CURRENT_PORTFOLIO] = 0.0
        if compare_portfolio not in pivot.columns:
            pivot[compare_portfolio] = 0.0
        pivot["compare_portfolio"] = compare_portfolio
        pivot["compare_series"] = SERIES_BY_PORTFOLIO.get(compare_portfolio, compare_portfolio)
        pivot["current_weight"] = pivot[CURRENT_PORTFOLIO].astype(float)
        pivot["compare_weight"] = pivot[compare_portfolio].astype(float)
        pivot["delta_weight"] = pivot["compare_weight"] - pivot["current_weight"]
        rows.append(pivot)
    delta = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if delta.empty:
        return delta
    delta = delta.merge(factors, on=["session_date", "symbol"], how="left")
    delta["delta_forward_return_contribution"] = (
        delta["delta_weight"].astype(float) * pd.to_numeric(delta["forward_return_5d"], errors="coerce").fillna(0.0)
    )
    delta["delta_beta_residual_contribution"] = (
        delta["delta_weight"].astype(float)
        * pd.to_numeric(delta["forward_beta_residual_return_5d"], errors="coerce").fillna(0.0)
    )
    delta["direction"] = np.where(delta["delta_weight"] >= 0, "added_or_increased", "removed_or_reduced")
    delta["future_outcome"] = np.where(
        pd.to_numeric(delta["forward_beta_residual_return_5d"], errors="coerce").fillna(0.0) >= 0,
        "future_winner",
        "future_loser",
    )
    windowed: list[pd.DataFrame] = []
    for window, start, end in WINDOWS:
        subset = delta[(delta["session_date"] >= start) & (delta["session_date"] <= end)].copy()
        if subset.empty:
            continue
        subset["window"] = window
        subset["start"] = start
        subset["end"] = end
        windowed.append(subset)
    out = pd.concat(windowed, ignore_index=True) if windowed else pd.DataFrame()
    keep_cols = [
        "compare_portfolio",
        "compare_series",
        "window",
        "session_date",
        "symbol",
        "current_weight",
        "compare_weight",
        "delta_weight",
        "direction",
        "future_outcome",
        "forward_return_5d",
        "forward_beta_residual_return_5d",
        "delta_forward_return_contribution",
        "delta_beta_residual_contribution",
        "price_falling_knife_score",
        "price_falling_knife_x_np_score",
        "long_np_plus_fund_score",
        "falling_knife_structural_weakness",
        "momentum_60d_z",
        "beta_residual_momentum_20d_z",
        "beta_z",
        "sic2_sector",
    ]
    return out[keep_cols].sort_values(
        ["window", "compare_portfolio", "delta_beta_residual_contribution"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _long_reweight_delta_rollup(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (compare_portfolio, compare_series, window), group in deltas.groupby(
        ["compare_portfolio", "compare_series", "window"],
        sort=True,
    ):
        meaningful = group[group["delta_weight"].abs() > 1e-8].copy()
        harmful = meaningful[meaningful["delta_beta_residual_contribution"] < 0]
        helpful = meaningful[meaningful["delta_beta_residual_contribution"] > 0]
        removed_future_winners = meaningful[
            (meaningful["direction"].eq("removed_or_reduced"))
            & (meaningful["future_outcome"].eq("future_winner"))
        ]
        removed_future_losers = meaningful[
            (meaningful["direction"].eq("removed_or_reduced"))
            & (meaningful["future_outcome"].eq("future_loser"))
        ]
        added_future_losers = meaningful[
            (meaningful["direction"].eq("added_or_increased"))
            & (meaningful["future_outcome"].eq("future_loser"))
        ]
        rows.append(
            {
                "compare_portfolio": compare_portfolio,
                "compare_series": compare_series,
                "window": window,
                "rows_changed": int(len(meaningful)),
                "sum_abs_delta_weight": float(meaningful["delta_weight"].abs().sum()),
                "delta_beta_residual_contribution_bps": float(
                    meaningful["delta_beta_residual_contribution"].sum() * 10000.0
                ),
                "helpful_delta_bps": float(helpful["delta_beta_residual_contribution"].sum() * 10000.0),
                "harmful_delta_bps": float(harmful["delta_beta_residual_contribution"].sum() * 10000.0),
                "removed_future_winner_delta_bps": float(
                    removed_future_winners["delta_beta_residual_contribution"].sum() * 10000.0
                ),
                "removed_future_loser_delta_bps": float(
                    removed_future_losers["delta_beta_residual_contribution"].sum() * 10000.0
                ),
                "added_future_loser_delta_bps": float(
                    added_future_losers["delta_beta_residual_contribution"].sum() * 10000.0
                ),
                "mean_removed_future_winner_price_fk": float(
                    removed_future_winners["price_falling_knife_score"].mean()
                )
                if not removed_future_winners.empty
                else np.nan,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["window", "compare_portfolio"]).reset_index(drop=True)


def _score_breadth_summary(activation: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scope",
        "window",
        "mean_price_falling_knife_score",
        "share_price_falling_knife_gt075",
        "mean_price_fk_x_np_score",
        "mean_np_plus_fund_score",
    ]
    return activation[cols].copy()


def _memo(
    *,
    metrics: pd.DataFrame,
    windows: pd.DataFrame,
    side_attr: pd.DataFrame,
    exposure: pd.DataFrame,
    delta_rollup: pd.DataFrame,
    score_breadth: pd.DataFrame,
) -> str:
    selected_metrics = metrics[
        metrics["series"].isin(
            [
                "current sota",
                "np fof low mom60 veto t075",
                "price fk soft w100",
                "price fk x np soft w075",
            ]
        )
    ].copy()
    metric_cols = [
        "series",
        "annualized_return",
        "annualized_vol",
        "sharpe_no_rf",
        "max_drawdown",
        "rolling_60_positive_rate",
        "aggregate_turnover_mean",
        "mean_selected_long_price_falling_knife_score",
    ]
    window_focus = windows[
        windows["series"].isin(selected_metrics["series"])
        & windows["window"].isin(["2015_peak_to_trough", "2016_jan_feb", "2019_may", "2019_aug"])
    ].copy()
    side_focus = side_attr[
        side_attr["series"].isin(selected_metrics["series"])
        & side_attr["window"].isin(["2015_peak_to_trough", "2019_may", "2019_aug"])
    ].copy()
    exposure_focus = exposure[
        exposure["series"].isin(selected_metrics["series"])
        & exposure["window"].isin(["2015_peak_to_trough", "2019_may", "2019_aug"])
    ].copy()
    breadth_focus = score_breadth[
        score_breadth["scope"].eq("baseline_top80")
        & score_breadth["window"].isin(["full", "2015_peak_to_trough", "2019_may", "2019_aug"])
    ].copy()
    delta_focus = delta_rollup[
        delta_rollup["compare_portfolio"].isin(["price_fk_soft_w100", "price_fk_x_np_soft_w075"])
        & delta_rollup["window"].isin(["2015_peak_to_trough", "2019_may", "2019_aug"])
    ].copy()

    lines = [
        "# Phase5Q Price Falling-Knife Diagnostics",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Question",
        "",
        "Phase5P repaired the long-leg falling-knife windows, but the repair was not large enough to promote a new lead. This memo diagnoses why.",
        "",
        "## Selected Metrics",
        "",
        _text_table(_display_frame(selected_metrics[metric_cols])),
        "",
        "## Key Window Returns",
        "",
        _text_table(_display_frame(window_focus[["series", "window", "compound_return", "hit_rate_daily"]])),
        "",
        "## Actual Side Attribution From Strict h10 Sleeves",
        "",
        _text_table(
            _display_frame(
                side_focus[
                    [
                        "series",
                        "window",
                        "compound_return",
                        "long_sum_bps",
                        "short_sum_bps",
                        "hit_rate_daily",
                    ]
                ]
            )
        ),
        "",
        "## Decision-Window Long Exposure",
        "",
        _text_table(
            _display_frame(
                exposure_focus[
                    [
                        "series",
                        "window",
                        "weighted_price_falling_knife_score",
                        "weighted_price_falling_knife_x_np_score",
                        "weighted_long_np_plus_fund_score",
                        "weighted_forward_beta_residual_return_5d",
                    ]
                ]
            )
        ),
        "",
        "## Score Breadth",
        "",
        _text_table(_display_frame(breadth_focus)),
        "",
        "## Reweighting Rollup Versus Current SOTA",
        "",
        _text_table(
            _display_frame(
                delta_focus[
                    [
                        "compare_series",
                        "window",
                        "delta_beta_residual_contribution_bps",
                        "helpful_delta_bps",
                        "harmful_delta_bps",
                        "removed_future_winner_delta_bps",
                        "removed_future_loser_delta_bps",
                        "added_future_loser_delta_bps",
                        "mean_removed_future_winner_price_fk",
                    ]
                ]
            )
        ),
        "",
        "## Read",
        "",
        "- The price falling-knife score is broad inside the baseline top80 long basket. That makes it a useful risk temperature, but a blunt selector by itself.",
        "- Stronger price penalties reduce selected long falling-knife exposure and improve 2015 / 2016, but they also tax the ordinary rebound engine.",
        "- The hybrid `price fk x np soft w075` has the best full-sample balance because non-price moderates the penalty, but that same moderation leaves some bad long exposure in place.",
        "- 2019 Aug remains tricky because some high price-falling-knife names are future rebound winners; penalty-only logic can reduce good rebound exposure as well as bad falling-knife exposure.",
        "- The next candidate should likely be conditional: penalize when price-state is bad and there is no short-term stabilization, but relax when recent stabilization or sector-relative rebound appears.",
        "",
        "## Suggested Next Test",
        "",
        "Use a two-stage score: keep the price falling-knife penalty, but multiply it by a stronger `no stabilization` term and reduce it for names with 3-5 session stabilization / sector-relative uptick.",
    ]
    return "\n".join(lines) + "\n"


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pct_columns = {
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
        "compound_return",
        "hit_rate_daily",
        "share_price_falling_knife_gt075",
    }
    for column in result.columns:
        if column in pct_columns:
            result[column] = result[column].map(_fmt_pct_like)
        elif pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].map(_fmt_float_like)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase5Q price falling-knife diagnostics.")
    parser.add_argument("--positions-path", default=str(DEFAULT_PHASE5P_POSITIONS))
    parser.add_argument("--sleeve-returns-path", default=str(DEFAULT_PHASE5P_SLEEVE_RETURNS))
    parser.add_argument("--activation-path", default=str(DEFAULT_PHASE5P_ACTIVATION))
    parser.add_argument("--metrics-path", default=str(DEFAULT_PHASE5P_METRICS))
    parser.add_argument("--windows-path", default=str(DEFAULT_PHASE5P_WINDOWS))
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--long-non-price-panel-path", default=str(DEFAULT_LONG_NON_PRICE_PANEL))
    parser.add_argument("--long-fundamental-panel-path", default=str(DEFAULT_LONG_FUNDAMENTAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    rollup = build_phase5q_price_fk_diagnostics_artifacts(
        positions_path=args.positions_path,
        sleeve_returns_path=args.sleeve_returns_path,
        activation_path=args.activation_path,
        metrics_path=args.metrics_path,
        windows_path=args.windows_path,
        signal_panel_path=args.signal_panel_path,
        long_non_price_panel_path=args.long_non_price_panel_path,
        long_fundamental_panel_path=args.long_fundamental_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
