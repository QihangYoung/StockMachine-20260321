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
    WINDOWS,
    _add_former_winner_scores,
    _beta_matched_window_summary,
    _fmt_bps,
    _fmt_float,
    _fmt_pct,
    _score_bin_summary,
    _table,
    _window_summary,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4ab_hybrid_short_veto_20260419"
HYBRID_SELECTORS = (
    DEFAULT_SHORT_SELECTOR,
    "short_hybrid_veto_nonweak_fw_t050",
    "short_hybrid_veto_nonweak_fw_t075",
    "short_hybrid_top70_veto_nonweak_fw_t050",
    "short_hybrid_soft_fw_overlay",
)
PORTFOLIO_SPECS = (
    {"portfolio": "sic2_soft_neutral", "hard_group": None, "soft_group": "sic2_sector"},
)


def build_phase4ab_hybrid_short_veto_artifacts(
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
    """Test hybrid short selectors that veto strong winners without weakening."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)

    standalone_daily = _standalone_daily(
        panel,
        selectors=HYBRID_SELECTORS,
        candidate_count=standalone_candidate_count,
    )
    standalone_window = _window_summary(standalone_daily)
    standalone_bins = _score_bin_summary(panel, selectors=HYBRID_SELECTORS)

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
        selectors=HYBRID_SELECTORS,
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

    memo_path = output_dir / "phase4ab_hybrid_short_veto_memo.md"
    standalone_daily_path = output_dir / "phase4ab_standalone_daily_validation.csv"
    standalone_window_path = output_dir / "phase4ab_standalone_window_summary_validation.csv"
    standalone_bins_path = output_dir / "phase4ab_standalone_score_bin_summary_validation.csv"
    positions_path = output_dir / "phase4ab_beta_matched_positions_validation.csv.gz"
    beta_daily_path = output_dir / "phase4ab_beta_matched_daily_validation.csv"
    beta_summary_path = output_dir / "phase4ab_beta_matched_summary_validation.csv"
    beta_window_path = output_dir / "phase4ab_beta_matched_window_summary_validation.csv"
    skipped_path = output_dir / "phase4ab_beta_matched_skipped_validation.csv"
    rollup_path = output_dir / "phase4ab_rollup.json"

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
        "long_score": long_score,
        "baseline_short_selector": DEFAULT_SHORT_SELECTOR,
        "hybrid_selectors": list(HYBRID_SELECTORS),
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
        "method": "hybrid_baseline_short_selector_with_former_winner_weakening_veto",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "The h10 signal panel keeps legacy *_5d column names; values are h10 labels.",
            "Hybrid selectors are ex-ante and use only same-session price-derived features.",
            "This pass focuses on the current lead SIC2 soft-neutral construction.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_hybrid_scores(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frame = group.copy()
        baseline = frame[DEFAULT_SHORT_SELECTOR].astype(float)
        strength = frame["former_winner_strength_mixed_20_60"].clip(lower=0.0).astype(float)
        recent_down = frame["former_winner_recent_is_down"].fillna(False).astype(bool)
        recent_weakness = frame["former_winner_recent_weakness"].clip(lower=0.0).astype(float)

        strong_050 = strength > 0.50
        strong_075 = strength > 0.75
        top70_score = baseline >= baseline.quantile(0.70)
        nonweak_050 = strong_050 & ~recent_down
        nonweak_075 = strong_075 & ~recent_down
        confirmed_weak_050 = strong_050 & recent_down

        frame["short_hybrid_veto_nonweak_fw_t050"] = baseline.where(~nonweak_050)
        frame["short_hybrid_veto_nonweak_fw_t075"] = baseline.where(~nonweak_075)
        frame["short_hybrid_top70_veto_nonweak_fw_t050"] = baseline.where(
            ~(top70_score & nonweak_050)
        )
        soft_raw = (
            baseline
            - 0.75 * strength.where(nonweak_050, 0.0)
            + 0.35 * recent_weakness.where(confirmed_weak_050, 0.0)
        )
        frame["short_hybrid_soft_fw_overlay"] = _zscore(soft_raw.to_numpy(dtype=float))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else panel


def _standalone_daily(
    panel: pd.DataFrame,
    *,
    selectors: Sequence[str],
    candidate_count: int,
) -> pd.DataFrame:
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


def _beta_matched_recheck(
    panel: pd.DataFrame,
    *,
    selectors: Sequence[str],
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
    for selector in selectors:
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
        "# Phase4AB Hybrid Short Veto Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Question",
        "",
        "Phase4AA showed that a pure former-winner weakening gate can repair some mean-reversion failure windows, but it sacrifices too much full-window performance. Phase4AB asks a narrower question: can weakening confirmation work better as a risk overlay on the old selector rather than as a full replacement?",
        "",
        "## Selector Definitions",
        "",
        "- `short_core_plus_overextension`: old lead short selector.",
        "- `short_hybrid_veto_nonweak_fw_t050`: keep the old score, but exclude strong former winners with composite strength > 0.50 unless their latest 5-session return is negative.",
        "- `short_hybrid_veto_nonweak_fw_t075`: same veto, but only for stronger former winners with composite strength > 0.75.",
        "- `short_hybrid_top70_veto_nonweak_fw_t050`: apply the 0.50 veto only when the old selector already ranks the name in the top 30% of same-day short candidates.",
        "- `short_hybrid_soft_fw_overlay`: keep all names but penalize strong non-weakening winners and add a small boost to strong winners that have started weakening.",
        "",
        f"Standalone diagnostics use `{short_variant}` and select top `{standalone_candidate_count}` names per session. Higher short contribution is better. Beta-matched diagnostics focus on the current lead `sic2_soft_neutral` construction.",
        "",
        "## Standalone Window Summary",
        "",
        _table(
            _display_standalone(
                standalone_window[
                    standalone_window["window"].isin(
                        [
                            "full",
                            "2018Q1_stress",
                            "2019_may_drawdown",
                            "2019_mid",
                            "2019_jun_jul_rebound",
                            "2019_aug_drawdown",
                        ]
                    )
                ]
            )
        ),
        "",
        "## Beta-Matched Summary",
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
                            "2018Q1_stress",
                            "2019_may_drawdown",
                            "2019_mid",
                            "2019_jun_jul_rebound",
                            "2019_aug_drawdown",
                        ]
                    )
                ]
            )
        ),
        "",
        "## Score-Bin Check",
        "",
        "The bin table below checks whether higher same-day selector scores correspond to better short outcomes. It is included mainly to catch inverted regimes like 2019 May.",
        "",
        _table(
            _display_bins(
                standalone_bins[
                    (standalone_bins["window"].isin(["full", "2019_may_drawdown"]))
                    & (standalone_bins["selector"].isin(
                        [
                            DEFAULT_SHORT_SELECTOR,
                            "short_hybrid_top70_veto_nonweak_fw_t050",
                            "short_hybrid_soft_fw_overlay",
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
        "- This pass does not use borrow, squeeze, options, news, or event-risk data.",
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
        "short_selector",
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
    out = frame[
        ["window", "selector", "score_bin", "sessions", "mean_short_contribution_target"]
    ].copy()
    out["mean_short_contribution_target"] = out["mean_short_contribution_target"].map(_fmt_bps)
    return out.sort_values(["window", "selector", "score_bin"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase4AB hybrid short veto diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--standalone-candidate-count", type=int, default=30)
    parser.add_argument("--candidate-pool-per-side", type=int, default=80)
    parser.add_argument("--soft-group-penalty", type=float, default=25.0)
    args = parser.parse_args(argv)

    result = build_phase4ab_hybrid_short_veto_artifacts(
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
