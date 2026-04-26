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


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4aa_former_winner_short_selector_20260419"
WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("full", "2014-08-05", "2019-12-31"),
    ("2018H1", "2018-01-01", "2018-06-30"),
    ("2018Q1_stress", "2018-01-26", "2018-04-02"),
    ("2018Q2_rebound", "2018-04-01", "2018-06-30"),
    ("2019_mid", "2019-05-01", "2019-08-31"),
    ("2019_may_drawdown", "2019-05-01", "2019-06-07"),
    ("2019_jun_jul_rebound", "2019-06-07", "2019-07-31"),
    ("2019_aug_drawdown", "2019-08-01", "2019-08-31"),
)
FORMER_WINNER_SELECTORS = (
    DEFAULT_SHORT_SELECTOR,
    "short_former_winner_20_5_soft",
    "short_former_winner_20_5_gate",
    "short_former_winner_residual_20_5_gate",
    "short_former_winner_mixed_20_60_5_gate",
)
BETA_MATCHED_SELECTORS = (
    DEFAULT_SHORT_SELECTOR,
    "short_former_winner_20_5_gate",
    "short_former_winner_residual_20_5_gate",
    "short_former_winner_mixed_20_60_5_gate",
)
PORTFOLIO_SPECS = (
    {"portfolio": "beta_only_top120", "hard_group": None, "soft_group": None},
    {"portfolio": "sic2_soft_neutral", "hard_group": None, "soft_group": "sic2_sector"},
)


def build_phase4aa_former_winner_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    long_score: str = DEFAULT_LONG_SCORE,
    standalone_candidate_count: int = 30,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
) -> dict[str, Any]:
    """Study mean-reversion failure and test former-winner short selectors."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    standalone_daily = _standalone_daily(panel, candidate_count=standalone_candidate_count)
    standalone_window = _window_summary(standalone_daily)
    standalone_bins = _score_bin_summary(panel, selectors=FORMER_WINNER_SELECTORS)

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
        long_variant=long_variant,
        short_variant=short_variant,
        long_score=long_score,
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
        short_variant=short_variant,
        standalone_candidate_count=standalone_candidate_count,
    )

    research_memo_path = output_dir / "phase4aa_mean_reversion_failure_research_memo.md"
    standalone_daily_path = output_dir / "phase4aa_standalone_daily_validation.csv"
    standalone_window_path = output_dir / "phase4aa_standalone_window_summary_validation.csv"
    standalone_bins_path = output_dir / "phase4aa_standalone_score_bin_summary_validation.csv"
    positions_path = output_dir / "phase4aa_beta_matched_positions_validation.csv.gz"
    beta_daily_path = output_dir / "phase4aa_beta_matched_daily_validation.csv"
    beta_summary_path = output_dir / "phase4aa_beta_matched_summary_validation.csv"
    beta_window_path = output_dir / "phase4aa_beta_matched_window_summary_validation.csv"
    skipped_path = output_dir / "phase4aa_beta_matched_skipped_validation.csv"
    rollup_path = output_dir / "phase4aa_rollup.json"

    research_memo_path.write_text(memo, encoding="utf-8")
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
        "long_score": long_score,
        "baseline_short_selector": DEFAULT_SHORT_SELECTOR,
        "former_winner_selectors": list(FORMER_WINNER_SELECTORS),
        "beta_matched_selectors": list(BETA_MATCHED_SELECTORS),
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
        "memo_artifact": research_memo_path.as_posix(),
        "standalone_daily_artifact": standalone_daily_path.as_posix(),
        "standalone_window_artifact": standalone_window_path.as_posix(),
        "standalone_score_bin_artifact": standalone_bins_path.as_posix(),
        "beta_matched_positions_artifact": positions_path.as_posix(),
        "beta_matched_daily_artifact": beta_daily_path.as_posix(),
        "beta_matched_summary_artifact": beta_summary_path.as_posix(),
        "beta_matched_window_artifact": beta_window_path.as_posix(),
        "beta_matched_skipped_artifact": skipped_path.as_posix(),
        "method": "former_winner_short_selector_requiring_recent_weakening_confirmation",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "The h10 signal panel keeps legacy *_5d column names; values are h10 labels.",
            "Former-winner selectors use only price-derived features available at the decision session.",
            "This first pass is gross/no-cost and does not model borrow costs or squeeze constraints.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_former_winner_scores(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frame = group.copy()
        momentum_20 = frame["momentum_20d_z"].clip(lower=0.0)
        momentum_60 = frame["momentum_60d_z"].clip(lower=0.0)
        residual_20 = frame["beta_residual_momentum_20d_z"].clip(lower=0.0)
        vol_adjusted = frame["vol_adjusted_momentum_20d_z"].clip(lower=0.0)
        recent_weak_relative = (-frame["return_5d_z"]).clip(lower=0.0)
        recent_is_down = frame["return_5d"] < 0.0

        strength_20 = (momentum_20 + residual_20 + vol_adjusted) / 3.0
        strength_mixed = (momentum_20 + momentum_60 + residual_20 + vol_adjusted) / 4.0
        raw_20 = strength_20 * recent_weak_relative
        raw_residual = residual_20 * recent_weak_relative
        raw_mixed = strength_mixed * recent_weak_relative

        frame["former_winner_strength_20"] = strength_20
        frame["former_winner_strength_mixed_20_60"] = strength_mixed
        frame["former_winner_recent_weakness"] = recent_weak_relative
        frame["former_winner_recent_is_down"] = recent_is_down
        frame["short_former_winner_20_5_soft"] = _zscore(raw_20.to_numpy(dtype=float))
        frame["short_former_winner_20_5_gate"] = _gated_zscore(
            raw_20,
            recent_is_down & (strength_20 > 0.0) & (recent_weak_relative > 0.0),
        )
        frame["short_former_winner_residual_20_5_gate"] = _gated_zscore(
            raw_residual,
            recent_is_down & (residual_20 > 0.0) & (recent_weak_relative > 0.0),
        )
        frame["short_former_winner_mixed_20_60_5_gate"] = _gated_zscore(
            raw_mixed,
            recent_is_down & (strength_mixed > 0.0) & (recent_weak_relative > 0.0),
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else panel


def _gated_zscore(raw_score: pd.Series, gate: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=raw_score.index, dtype=float)
    if gate.sum() == 0:
        return out
    out.loc[gate] = _zscore(raw_score.loc[gate].to_numpy(dtype=float))
    return out


def _standalone_daily(panel: pd.DataFrame, *, candidate_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    short_panel = panel[panel["variant"].eq(DEFAULT_SHORT_VARIANT)].copy()
    for session_date, group in short_panel.groupby("session_date", sort=True):
        target_frame = group.dropna(subset=[TARGET_COLUMN, "forward_return_5d"])
        if len(target_frame) < candidate_count * 2:
            continue
        universe_mean = float(target_frame[TARGET_COLUMN].mean())
        oracle = target_frame.nsmallest(candidate_count, TARGET_COLUMN)
        oracle_symbols = set(oracle["symbol"])
        loser_cutoff = float(target_frame[TARGET_COLUMN].quantile(0.20))
        loser_symbols = set(target_frame.loc[target_frame[TARGET_COLUMN] <= loser_cutoff, "symbol"])
        for selector in FORMER_WINNER_SELECTORS:
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
                    "variant": DEFAULT_SHORT_VARIANT,
                    "selector": selector,
                    "eligible_names": int(len(subset)),
                    "selected_names": int(len(selected)),
                    "selected_mean_target_return": selected_mean,
                    "short_contribution_target": float(-selected_mean),
                    "selected_negative_target_share": float((target < 0).mean()),
                    "selected_positive_target_share": float((target > 0).mean()),
                    "selected_mean_forward_return_h10": float(raw.mean()),
                    "short_contribution_return": float(-raw.mean()),
                    "short_edge_vs_universe": float(universe_mean - selected_mean),
                    "oracle_short_contribution_target": float(-oracle[TARGET_COLUMN].mean()),
                    "oracle_overlap_rate": float(
                        len(set(selected["symbol"]).intersection(oracle_symbols)) / candidate_count
                    ),
                    "loser_quantile_capture_rate": float(
                        len(set(selected["symbol"]).intersection(loser_symbols)) / candidate_count
                    ),
                    "rank_ic_selector_vs_target": float(rank_ic),
                    "rank_ic_bad": bool(rank_ic > 0) if pd.notna(rank_ic) else False,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _window_summary(daily: pd.DataFrame) -> pd.DataFrame:
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
            contribution = group["short_contribution_target"]
            rows.append(
                {
                    "window": window,
                    "start": start,
                    "end": end,
                    "selector": selector,
                    "sessions": int(len(group)),
                    "mean_short_contribution_target": float(contribution.mean()),
                    "short_target_hit_rate": float((contribution > 0).mean()),
                    "mean_selected_negative_target_share": float(
                        group["selected_negative_target_share"].mean()
                    ),
                    "mean_selected_positive_target_share": float(
                        group["selected_positive_target_share"].mean()
                    ),
                    "mean_short_edge_vs_universe": float(group["short_edge_vs_universe"].mean()),
                    "mean_oracle_short_contribution_target": float(
                        group["oracle_short_contribution_target"].mean()
                    ),
                    "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                    "mean_loser_quantile_capture_rate": float(
                        group["loser_quantile_capture_rate"].mean()
                    ),
                    "mean_rank_ic_selector_vs_target": float(
                        group["rank_ic_selector_vs_target"].mean()
                    ),
                    "rank_ic_bad_share": float(group["rank_ic_bad"].mean()),
                    "test_window_used": bool(group["test_window_used"].any()),
                }
            )
    return pd.DataFrame(rows)


def _score_bin_summary(
    panel: pd.DataFrame,
    *,
    selectors: Sequence[str],
    score_bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    short_panel = panel[panel["variant"].eq(DEFAULT_SHORT_VARIANT)].copy()
    short_panel["session_date_dt"] = pd.to_datetime(short_panel["session_date"])
    for window, start, end in WINDOWS:
        window_panel = short_panel[
            (short_panel["session_date_dt"] >= pd.Timestamp(start))
            & (short_panel["session_date_dt"] <= pd.Timestamp(end))
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
                            "mean_short_contribution_target": float(-target.mean()),
                            "negative_target_share": float((target < 0).mean()),
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
            mean_short_contribution_target=("mean_short_contribution_target", "mean"),
            mean_negative_target_share=("negative_target_share", "mean"),
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
    long_variant: str,
    short_variant: str,
    long_score: str,
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
    for selector in BETA_MATCHED_SELECTORS:
        for spec in PORTFOLIO_SPECS:
            portfolio = f"{spec['portfolio']}__{selector}"
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
                    long_score=long_score,
                    short_selector=selector,
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
            selector = str(group["short_selector"].iloc[0])
            rows.append(
                {
                    "window": window,
                    "start": start,
                    "end": end,
                    "portfolio": portfolio,
                    "short_selector": selector,
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
    short_variant: str,
    standalone_candidate_count: int,
) -> str:
    lines = [
        "# Phase4AA Mean-Reversion Failure and Former-Winner Short Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Research Thesis",
        "",
        "Our previous short selector implicitly assumed short-horizon mean reversion: strong recent winners and overextended residual winners should underperform over the next h10 window. Phase4Y showed this is not stable. In 2019 May and 2018Q1, the highest short-preference buckets continued to outperform, so the old selector was effectively shorting momentum continuation.",
        "",
        "The industry framing is not `reversal is right` versus `momentum is right`; it is a horizon/state problem. Short-term reversal, intermediate momentum, sector/theme flow, and squeeze risk can dominate each other in different windows.",
        "",
        "## Proposed Direction",
        "",
        "This experiment tests a more conservative short rule: do not short a stock merely because it is strong. Require a strong 20d/60d or residual-momentum background, plus recent 5d weakening confirmation. In plain words: avoid shorting a train while it is still accelerating; only test shorts on former winners that have started to stall.",
        "",
        "## Selector Definitions",
        "",
        "- `short_core_plus_overextension`: old lead; high momentum/overextension, including recent continuation.",
        "- `short_former_winner_20_5_soft`: strong 20d composite times relative 5d weakness, without hard exclusion.",
        "- `short_former_winner_20_5_gate`: same idea, but only names with negative recent 5d return are eligible.",
        "- `short_former_winner_residual_20_5_gate`: residual-momentum winner plus negative recent 5d return.",
        "- `short_former_winner_mixed_20_60_5_gate`: 20d, 60d, residual, and vol-adjusted strength plus negative recent 5d return.",
        "",
        f"Standalone diagnostics use `{short_variant}` and select top `{standalone_candidate_count}` names per session. Higher short contribution is better.",
        "",
        "## Standalone Window Summary",
        "",
        _table(
            _display_standalone(
                standalone_window[
                    standalone_window["window"].isin(
                        ["full", "2018Q1_stress", "2019_may_drawdown", "2019_mid", "2019_aug_drawdown"]
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
                        ["2018Q1_stress", "2019_may_drawdown", "2019_mid", "2019_aug_drawdown"]
                    )
                ]
            )
        ),
        "",
        "## Score-Bin Sanity Check",
        "",
        "For a useful short selector, higher score bins should generally have higher short contribution. The old selector failed this check in 2019 May. Former-winner gates should be judged by whether they flatten or reverse that failure without destroying full-window performance.",
        "",
        _table(
            _display_bins(
                standalone_bins[
                    (standalone_bins["window"].isin(["full", "2019_may_drawdown"]))
                    & (standalone_bins["selector"].isin(
                        ["short_core_plus_overextension", "short_former_winner_mixed_20_60_5_gate"]
                    ))
                ]
            )
        ),
        "",
        "## Guardrails",
        "",
        "- Validation only; test window remains closed.",
        "- h10 labels are used even though legacy columns retain `*_5d` names.",
        "- This is not yet a production short model. Borrow availability, event risk, and squeeze risk remain outside this pass.",
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
            "mean_short_contribution_target",
            "short_target_hit_rate",
            "mean_selected_positive_target_share",
            "mean_rank_ic_selector_vs_target",
        ]
    ].copy()
    out["mean_short_contribution_target"] = out["mean_short_contribution_target"].map(_fmt_bps)
    out["short_target_hit_rate"] = out["short_target_hit_rate"].map(_fmt_pct)
    out["mean_selected_positive_target_share"] = out["mean_selected_positive_target_share"].map(
        _fmt_pct
    )
    out["mean_rank_ic_selector_vs_target"] = out["mean_rank_ic_selector_vs_target"].map(_fmt_float)
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
        "mean_spread_risk_factor_sic2_residual",
        "mean_spread_style_factor_sic2_residual",
        "mean_sic2_l1_exposure",
    ]
    out = frame[[c for c in keep if c in frame.columns]].copy()
    for column in [c for c in out.columns if c.startswith("mean_spread") or c.startswith("mean_long") or c.startswith("mean_short")]:
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
        "short_selector",
        "sessions",
        "mean_spread_raw",
        "mean_long_raw",
        "mean_short_raw_contribution",
        "mean_spread_beta_residual",
    ]
    out = frame[keep].copy()
    for column in ("mean_spread_raw", "mean_long_raw", "mean_short_raw_contribution", "mean_spread_beta_residual"):
        out[column] = out[column].map(_fmt_bps)
    return out.sort_values(["window", "portfolio"])


def _display_bins(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[
        ["window", "selector", "score_bin", "sessions", "mean_short_contribution_target"]
    ].copy()
    out["mean_short_contribution_target"] = out["mean_short_contribution_target"].map(_fmt_bps)
    return out.sort_values(["window", "selector", "score_bin"])


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def _fmt_bps(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 10000.0:.2f} bps"


def _fmt_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.1f}%"


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase4AA former-winner short selector diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--standalone-candidate-count", type=int, default=30)
    parser.add_argument("--candidate-pool-per-side", type=int, default=80)
    parser.add_argument("--soft-group-penalty", type=float, default=25.0)
    args = parser.parse_args(argv)

    result = build_phase4aa_former_winner_artifacts(
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
