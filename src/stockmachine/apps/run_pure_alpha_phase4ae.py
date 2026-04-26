from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
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
    _zscore,
)
from stockmachine.apps.run_pure_alpha_phase4aa import (
    _fmt_bps,
    _fmt_float,
    _fmt_pct,
    _gated_zscore,
    _table,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4ae_stabilized_loser_long_selector_20260419"
LONG_SELECTORS = (
    DEFAULT_LONG_SCORE,
    "long_former_loser_20_5_soft",
    "long_former_loser_20_5_gate",
    "long_former_loser_residual_20_5_gate",
    "long_former_loser_mixed_20_60_5_gate",
)
PORTFOLIO_SPECS = (
    {"portfolio": "sic2_soft_neutral", "hard_group": None, "soft_group": "sic2_sector"},
)
WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("full", "2014-08-05", "2019-12-31"),
    ("2015_peak_to_trough_signal", "2015-05-19", "2015-08-19"),
    ("2015_peak_to_trough_same_dates", "2015-06-04", "2015-08-21"),
    ("2015_recovery_signal", "2015-08-07", "2015-11-05"),
    ("2019_may_aug_signal", "2019-04-15", "2019-08-28"),
    ("2019_may_signal", "2019-04-15", "2019-06-05"),
    ("2019_aug_signal", "2019-07-18", "2019-08-28"),
)


def build_phase4ae_stabilized_loser_long_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    short_selector: str = DEFAULT_SHORT_SELECTOR,
    standalone_candidate_count: int = 30,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
) -> dict[str, Any]:
    """Test former-loser long selectors requiring recent stabilization."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_loser_scores(panel)

    standalone_daily = _standalone_long_daily(
        panel,
        selectors=LONG_SELECTORS,
        candidate_count=standalone_candidate_count,
    )
    standalone_window = _long_window_summary(standalone_daily)
    standalone_bins = _long_score_bin_summary(panel, selectors=LONG_SELECTORS)

    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )

    positions, beta_daily, skipped = _beta_matched_recheck(
        panel,
        long_selectors=LONG_SELECTORS,
        long_variant=long_variant,
        short_variant=short_variant,
        short_selector=short_selector,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        soft_group_penalty=soft_group_penalty,
    )
    beta_summary = _summary(beta_daily, skipped)
    beta_window = _beta_matched_window_summary(beta_daily)

    memo = _memo(
        standalone_window,
        standalone_bins,
        beta_summary,
        beta_window,
        long_variant=long_variant,
        short_variant=short_variant,
        short_selector=short_selector,
        standalone_candidate_count=standalone_candidate_count,
    )

    memo_path = output_dir / "phase4ae_stabilized_loser_long_memo.md"
    standalone_daily_path = output_dir / "phase4ae_standalone_long_daily_validation.csv"
    standalone_window_path = output_dir / "phase4ae_standalone_long_window_summary.csv"
    standalone_bins_path = output_dir / "phase4ae_standalone_long_score_bin_summary.csv"
    positions_path = output_dir / "phase4ae_beta_matched_positions_validation.csv.gz"
    beta_daily_path = output_dir / "phase4ae_beta_matched_daily_validation.csv"
    beta_summary_path = output_dir / "phase4ae_beta_matched_summary_validation.csv"
    beta_window_path = output_dir / "phase4ae_beta_matched_window_summary_validation.csv"
    skipped_path = output_dir / "phase4ae_beta_matched_skipped_validation.csv"
    rollup_path = output_dir / "phase4ae_rollup.json"

    memo_path.write_text(memo, encoding="utf-8")
    standalone_daily.to_csv(standalone_daily_path, index=False)
    standalone_window.to_csv(standalone_window_path, index=False)
    standalone_bins.to_csv(standalone_bins_path, index=False)
    positions.to_csv(positions_path, index=False, compression="gzip")
    beta_daily.to_csv(beta_daily_path, index=False)
    beta_summary.to_csv(beta_summary_path, index=False)
    beta_window.to_csv(beta_window_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "cik_mapping_path": Path(cik_mapping_path).as_posix(),
        "sec_submissions_dir": Path(sec_submissions_dir).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "short_selector": short_selector,
        "baseline_long_score": DEFAULT_LONG_SCORE,
        "long_selectors": list(LONG_SELECTORS),
        "portfolio_specs": list(PORTFOLIO_SPECS),
        "standalone_candidate_count": int(standalone_candidate_count),
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "min_nonzero_names": int(min_nonzero_names),
        "score_weight": float(score_weight),
        "soft_group_penalty": float(soft_group_penalty),
        "min_regression_rows": int(min_regression_rows),
        "min_dummy_count": int(min_dummy_count),
        "panel_rows": int(len(panel)),
        "standalone_daily_rows": int(len(standalone_daily)),
        "standalone_window_rows": int(len(standalone_window)),
        "beta_matched_positions_rows": int(len(positions)),
        "beta_matched_daily_rows": int(len(beta_daily)),
        "beta_matched_skipped_rows": int(len(skipped)),
        "memo_artifact": memo_path.as_posix(),
        "standalone_daily_artifact": standalone_daily_path.as_posix(),
        "standalone_window_artifact": standalone_window_path.as_posix(),
        "standalone_score_bin_artifact": standalone_bins_path.as_posix(),
        "beta_matched_positions_artifact": positions_path.as_posix(),
        "beta_matched_daily_artifact": beta_daily_path.as_posix(),
        "beta_matched_summary_artifact": beta_summary_path.as_posix(),
        "beta_matched_window_artifact": beta_window_path.as_posix(),
        "beta_matched_skipped_artifact": skipped_path.as_posix(),
        "method": "former_loser_long_selector_requiring_recent_stabilization_confirmation",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "The h10 signal panel keeps legacy *_5d column names; values are h10 labels.",
            "The current panel has 5-session but not 1-session or 3-session confirmation features.",
            "Former-loser stabilization uses medium-term weakness plus positive recent 5-session return.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_former_loser_scores(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frame = group.copy()
        momentum_20_down = (-frame["momentum_20d_z"]).clip(lower=0.0)
        momentum_60_down = (-frame["momentum_60d_z"]).clip(lower=0.0)
        residual_20_down = (-frame["beta_residual_momentum_20d_z"]).clip(lower=0.0)
        vol_adjusted_down = (-frame["vol_adjusted_momentum_20d_z"]).clip(lower=0.0)
        recent_strength = frame["return_5d_z"].clip(lower=0.0)
        recent_is_up = frame["return_5d"] > 0.0

        weakness_20 = (momentum_20_down + residual_20_down + vol_adjusted_down) / 3.0
        weakness_mixed = (
            momentum_20_down + momentum_60_down + residual_20_down + vol_adjusted_down
        ) / 4.0
        raw_20 = weakness_20 * recent_strength
        raw_residual = residual_20_down * recent_strength
        raw_mixed = weakness_mixed * recent_strength

        frame["former_loser_weakness_20"] = weakness_20
        frame["former_loser_weakness_mixed_20_60"] = weakness_mixed
        frame["former_loser_recent_stabilization"] = recent_strength
        frame["former_loser_recent_is_up"] = recent_is_up
        frame["long_former_loser_20_5_soft"] = _zscore(raw_20.to_numpy(dtype=float))
        frame["long_former_loser_20_5_gate"] = _gated_zscore(
            raw_20,
            recent_is_up & (weakness_20 > 0.0) & (recent_strength > 0.0),
        )
        frame["long_former_loser_residual_20_5_gate"] = _gated_zscore(
            raw_residual,
            recent_is_up & (residual_20_down > 0.0) & (recent_strength > 0.0),
        )
        frame["long_former_loser_mixed_20_60_5_gate"] = _gated_zscore(
            raw_mixed,
            recent_is_up & (weakness_mixed > 0.0) & (recent_strength > 0.0),
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else panel


def _standalone_long_daily(
    panel: pd.DataFrame,
    *,
    selectors: Sequence[str],
    candidate_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    long_panel = panel[panel["variant"].eq(DEFAULT_LONG_VARIANT)].copy()
    for session_date, group in long_panel.groupby("session_date", sort=True):
        target_frame = group.dropna(subset=[TARGET_COLUMN, "forward_return_5d"])
        if len(target_frame) < candidate_count * 2:
            continue
        universe_mean = float(target_frame[TARGET_COLUMN].mean())
        oracle = target_frame.nlargest(candidate_count, TARGET_COLUMN)
        oracle_symbols = set(oracle["symbol"])
        winner_cutoff = float(target_frame[TARGET_COLUMN].quantile(0.80))
        winner_symbols = set(
            target_frame.loc[target_frame[TARGET_COLUMN] >= winner_cutoff, "symbol"]
        )
        for selector in selectors:
            subset = target_frame.dropna(subset=[selector])
            if len(subset) < candidate_count:
                continue
            selected = subset.sort_values([selector, "symbol"], ascending=[False, True]).head(
                candidate_count
            )
            target = selected[TARGET_COLUMN]
            raw = selected["forward_return_5d"]
            rank_ic = subset[selector].corr(subset[TARGET_COLUMN], method="spearman")
            selected_mean = float(target.mean())
            rows.append(
                {
                    "session_date": session_date,
                    "variant": DEFAULT_LONG_VARIANT,
                    "selector": selector,
                    "eligible_names": int(len(subset)),
                    "selected_names": int(len(selected)),
                    "selected_mean_target_return": selected_mean,
                    "long_contribution_target": selected_mean,
                    "selected_positive_target_share": float((target > 0).mean()),
                    "selected_negative_target_share": float((target < 0).mean()),
                    "selected_mean_forward_return_h10": float(raw.mean()),
                    "long_contribution_return": float(raw.mean()),
                    "long_edge_vs_universe": float(selected_mean - universe_mean),
                    "oracle_long_contribution_target": float(oracle[TARGET_COLUMN].mean()),
                    "oracle_overlap_rate": float(
                        len(set(selected["symbol"]).intersection(oracle_symbols)) / candidate_count
                    ),
                    "winner_quantile_capture_rate": float(
                        len(set(selected["symbol"]).intersection(winner_symbols)) / candidate_count
                    ),
                    "rank_ic_selector_vs_target": float(rank_ic),
                    "rank_ic_bad": bool(rank_ic < 0) if pd.notna(rank_ic) else False,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _long_window_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["session_date_dt"] = pd.to_datetime(frame["session_date"])
    rows = []
    for window, start, end in WINDOWS:
        subset = frame[
            (frame["session_date_dt"] >= pd.Timestamp(start))
            & (frame["session_date_dt"] <= pd.Timestamp(end))
        ]
        for selector, group in subset.groupby("selector", sort=True):
            contribution = group["long_contribution_target"]
            rows.append(
                {
                    "window": window,
                    "start": start,
                    "end": end,
                    "selector": selector,
                    "sessions": int(len(group)),
                    "mean_long_contribution_target": float(contribution.mean()),
                    "long_target_hit_rate": float((contribution > 0).mean()),
                    "mean_selected_positive_target_share": float(
                        group["selected_positive_target_share"].mean()
                    ),
                    "mean_long_edge_vs_universe": float(group["long_edge_vs_universe"].mean()),
                    "mean_oracle_long_contribution_target": float(
                        group["oracle_long_contribution_target"].mean()
                    ),
                    "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                    "mean_winner_quantile_capture_rate": float(
                        group["winner_quantile_capture_rate"].mean()
                    ),
                    "mean_rank_ic_selector_vs_target": float(
                        group["rank_ic_selector_vs_target"].mean()
                    ),
                    "rank_ic_bad_share": float(group["rank_ic_bad"].mean()),
                    "test_window_used": bool(group["test_window_used"].any()),
                }
            )
    return pd.DataFrame(rows)


def _long_score_bin_summary(
    panel: pd.DataFrame,
    *,
    selectors: Sequence[str],
    score_bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    long_panel = panel[panel["variant"].eq(DEFAULT_LONG_VARIANT)].copy()
    long_panel["session_date_dt"] = pd.to_datetime(long_panel["session_date"])
    for window, start, end in WINDOWS:
        window_panel = long_panel[
            (long_panel["session_date_dt"] >= pd.Timestamp(start))
            & (long_panel["session_date_dt"] <= pd.Timestamp(end))
        ]
        for session_date, group in window_panel.groupby("session_date", sort=True):
            for selector in selectors:
                subset = group.dropna(subset=[selector, TARGET_COLUMN]).copy()
                if len(subset) < score_bins * 2:
                    continue
                subset["score_bin"] = _score_bins(subset[selector], score_bins)
                for score_bin, bin_group in subset.groupby("score_bin", sort=True):
                    target = bin_group[TARGET_COLUMN]
                    rows.append(
                        {
                            "window": window,
                            "start": start,
                            "end": end,
                            "session_date": session_date,
                            "selector": selector,
                            "score_bin": int(score_bin),
                            "names": int(len(bin_group)),
                            "mean_target_return": float(target.mean()),
                            "positive_target_share": float((target > 0).mean()),
                            "test_window_used": False,
                        }
                    )
    daily_bins = pd.DataFrame(rows)
    if daily_bins.empty:
        return daily_bins
    return (
        daily_bins.groupby(["window", "start", "end", "selector", "score_bin"], sort=True)
        .agg(
            sessions=("session_date", "nunique"),
            mean_names=("names", "mean"),
            mean_target_return=("mean_target_return", "mean"),
            mean_positive_target_share=("positive_target_share", "mean"),
            test_window_used=("test_window_used", "any"),
        )
        .reset_index()
    )


def _score_bins(values: pd.Series, score_bins: int) -> pd.Series:
    ranks = values.rank(method="first", ascending=True)
    bins = pd.qcut(ranks, q=score_bins, labels=False, duplicates="drop")
    return bins.astype(float) + 1.0


def _beta_matched_recheck(
    panel: pd.DataFrame,
    *,
    long_selectors: Sequence[str],
    long_variant: str,
    short_variant: str,
    short_selector: str,
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
    for long_selector in long_selectors:
        for spec in PORTFOLIO_SPECS:
            portfolio = f"{spec['portfolio']}__long_{long_selector}"
            for session_date in sessions:
                long_group = groups.get((long_variant, session_date))
                short_group = groups.get((short_variant, session_date))
                if long_group is None or short_group is None:
                    skipped.append(_skip_row(session_date, portfolio, 0, 0, "missing_group"))
                    continue
                book, diagnostic, skip = _construct_one_session(
                    long_group,
                    short_group,
                    session_date=session_date,
                    portfolio=portfolio,
                    long_variant=long_variant,
                    short_variant=short_variant,
                    long_score=long_selector,
                    short_selector=short_selector,
                    hard_group=spec["hard_group"],
                    soft_group=spec["soft_group"],
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
    rows = []
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


def _memo(
    standalone_window: pd.DataFrame,
    standalone_bins: pd.DataFrame,
    beta_summary: pd.DataFrame,
    beta_window: pd.DataFrame,
    *,
    long_variant: str,
    short_variant: str,
    short_selector: str,
    standalone_candidate_count: int,
) -> str:
    lines = [
        "# Phase4AE Stabilized-Loser Long Selector Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Question",
        "",
        "Can we make the long side symmetric with the short-side former-winner rule: do not buy a stock merely because it fell; require a weak medium-term background plus recent 5-session stabilization first?",
        "",
        "## Selector Definitions",
        "",
        "- `reversal_5d`: old lead long score; high score means the stock recently fell.",
        "- `long_former_loser_20_5_soft`: weak 20-session composite times positive recent 5-session strength, without hard exclusion.",
        "- `long_former_loser_20_5_gate`: same idea, but only names with positive recent 5-session return are eligible.",
        "- `long_former_loser_residual_20_5_gate`: residual-momentum loser plus positive recent 5-session return.",
        "- `long_former_loser_mixed_20_60_5_gate`: 20d, 60d, residual, and vol-adjusted weakness plus positive recent 5-session return.",
        "",
        f"Standalone diagnostics use `{long_variant}` and select top `{standalone_candidate_count}` names per session. Beta-matched diagnostics keep short side fixed at `{short_variant}` / `{short_selector}`.",
        "",
        "## Standalone Long Summary",
        "",
        _table(
            _display_standalone(
                standalone_window[
                    standalone_window["window"].isin(
                        [
                            "full",
                            "2015_peak_to_trough_signal",
                            "2015_recovery_signal",
                            "2019_may_aug_signal",
                            "2019_may_signal",
                        ]
                    )
                ]
            )
        ),
        "",
        "## Beta-Matched Portfolio Summary",
        "",
        _table(_display_beta_summary(beta_summary)),
        "",
        "## Beta-Matched Stress Windows",
        "",
        _table(
            _display_beta_window(
                beta_window[
                    beta_window["window"].isin(
                        [
                            "2015_peak_to_trough_signal",
                            "2015_recovery_signal",
                            "2019_may_aug_signal",
                            "2019_may_signal",
                        ]
                    )
                ]
            )
        ),
        "",
        "## Score-Bin Check",
        "",
        "Bins are ranked low-to-high by selector. For a useful long selector, higher bins should have higher future residual return.",
        "",
        _table(
            _display_bins(
                standalone_bins[
                    (standalone_bins["window"].isin(["full", "2015_peak_to_trough_signal"]))
                    & (standalone_bins["selector"].isin(
                        [
                            DEFAULT_LONG_SCORE,
                            "long_former_loser_mixed_20_60_5_gate",
                        ]
                    ))
                ]
            )
        ),
        "",
        "## Guardrails",
        "",
        "- Validation only; test window remains closed.",
        "- h10 labels are used even though legacy columns retain `*_5d` names.",
        "- This first pass uses 5-session stabilization because 1-session and 3-session confirmation features are not yet in the signal panel.",
    ]
    return "\n".join(lines) + "\n"


def _display_standalone(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[
        [
            "window",
            "selector",
            "sessions",
            "mean_long_contribution_target",
            "long_target_hit_rate",
            "mean_selected_positive_target_share",
            "mean_long_edge_vs_universe",
            "mean_rank_ic_selector_vs_target",
        ]
    ].copy()
    for column in ("mean_long_contribution_target", "mean_long_edge_vs_universe"):
        out[column] = out[column].map(_fmt_bps)
    out["long_target_hit_rate"] = out["long_target_hit_rate"].map(_fmt_pct)
    out["mean_selected_positive_target_share"] = out[
        "mean_selected_positive_target_share"
    ].map(_fmt_pct)
    out["mean_rank_ic_selector_vs_target"] = out["mean_rank_ic_selector_vs_target"].map(
        _fmt_float
    )
    return out.sort_values(["window", "selector"])


def _display_beta_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = [
        "portfolio",
        "requested_sessions",
        "constructed_sessions",
        "construction_rate",
        "mean_spread_raw",
        "hit_rate_raw",
        "mean_long_raw",
        "mean_short_raw_contribution",
        "mean_spread_beta_residual",
        "mean_long_beta_residual",
        "mean_short_beta_residual_contribution",
        "mean_spread_risk_factor_sic2_residual",
        "mean_spread_style_factor_sic2_residual",
        "mean_sic2_l1_exposure",
    ]
    out = frame[[c for c in keep if c in frame.columns]].copy()
    for column in [
        c
        for c in out.columns
        if c.startswith("mean_spread")
        or c.startswith("mean_long")
        or c.startswith("mean_short")
    ]:
        out[column] = out[column].map(_fmt_bps)
    for column in ("construction_rate", "hit_rate_raw"):
        if column in out:
            out[column] = out[column].map(_fmt_pct)
    return out


def _display_beta_window(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = [
        "window",
        "portfolio",
        "long_score",
        "sessions",
        "mean_spread_raw",
        "mean_long_raw",
        "mean_short_raw_contribution",
        "mean_spread_beta_residual",
    ]
    out = frame[keep].copy()
    for column in (
        "mean_spread_raw",
        "mean_long_raw",
        "mean_short_raw_contribution",
        "mean_spread_beta_residual",
    ):
        out[column] = out[column].map(_fmt_bps)
    return out.sort_values(["window", "portfolio"])


def _display_bins(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[["window", "selector", "score_bin", "sessions", "mean_target_return"]].copy()
    out["mean_target_return"] = out["mean_target_return"].map(_fmt_bps)
    return out.sort_values(["window", "selector", "score_bin"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase4AE stabilized-loser long selector diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--standalone-candidate-count", type=int, default=30)
    parser.add_argument("--candidate-pool-per-side", type=int, default=80)
    parser.add_argument("--soft-group-penalty", type=float, default=25.0)
    args = parser.parse_args(argv)

    result = build_phase4ae_stabilized_loser_long_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        standalone_candidate_count=args.standalone_candidate_count,
        candidate_pool_per_side=args.candidate_pool_per_side,
        soft_group_penalty=args.soft_group_penalty,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
