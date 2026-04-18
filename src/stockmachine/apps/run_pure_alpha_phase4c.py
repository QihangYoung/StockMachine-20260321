from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_PHASE3_SIGNAL_PANEL = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_20260418"
    / "phase3_baseline_signal_panel_validation.csv.gz"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4c_residual_loser_lab_20260418"
BASE_COLUMNS = (
    "session_date",
    "variant",
    "symbol",
    "lagged_close",
    "trailing_median_dollar_volume_20",
    "liquidity_rank",
    "beta",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "forward_return_5d",
    "forward_beta_residual_return_5d",
    "random_control",
)
TARGET_COLUMN = "forward_beta_residual_return_5d"


def build_phase4c_residual_loser_lab(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] | None = None,
    candidate_count: int = 30,
    loser_quantile: float = 0.20,
) -> dict[str, Any]:
    """Study better same-universe selectors for future beta-residual losers."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be in (0, 1).")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_signal_panel(signal_panel_path, variant_names=variant_names)
    daily = _selector_daily_diagnostics(
        panel,
        candidate_count=candidate_count,
        loser_quantile=loser_quantile,
    )
    summary = _selector_summary(daily)
    improvement = _improvement_vs_reversal(summary)
    opportunity = _universe_opportunity_summary(panel, candidate_count=candidate_count)
    roadmap = _feature_roadmap()

    daily_path = output_dir / "phase4c_residual_loser_selector_daily_validation.csv"
    summary_path = output_dir / "phase4c_residual_loser_selector_summary_validation.csv"
    improvement_path = output_dir / "phase4c_residual_loser_improvement_vs_reversal_validation.csv"
    opportunity_path = output_dir / "phase4c_residual_loser_universe_opportunity_validation.csv"
    roadmap_path = output_dir / "phase4c_residual_loser_feature_roadmap.csv"
    memo_path = output_dir / "phase4c_residual_loser_lab_memo.md"
    rollup_path = output_dir / "phase4c_residual_loser_lab_rollup.json"

    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    improvement.to_csv(improvement_path, index=False)
    opportunity.to_csv(opportunity_path, index=False)
    roadmap.to_csv(roadmap_path, index=False)
    memo_path.write_text(
        _lab_memo(
            summary,
            improvement,
            opportunity,
            roadmap,
            candidate_count=candidate_count,
            loser_quantile=loser_quantile,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "variant_names": sorted(panel["variant"].astype(str).unique()) if not panel.empty else [],
        "candidate_count": int(candidate_count),
        "loser_quantile": float(loser_quantile),
        "validation_start": _first_or_none(daily["session_date"]) if not daily.empty else None,
        "validation_end": _last_or_none(daily["session_date"]) if not daily.empty else None,
        "signal_panel_rows_loaded": int(len(panel)),
        "daily_selector_rows": int(len(daily)),
        "selector_summary_rows": int(len(summary)),
        "improvement_rows": int(len(improvement)),
        "opportunity_rows": int(len(opportunity)),
        "roadmap_rows": int(len(roadmap)),
        "daily_selector_artifact": daily_path.as_posix(),
        "selector_summary_artifact": summary_path.as_posix(),
        "improvement_artifact": improvement_path.as_posix(),
        "opportunity_artifact": opportunity_path.as_posix(),
        "feature_roadmap_artifact": roadmap_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is a same-universe residual-loser research lab, not a portfolio constructor.",
            "The oracle columns use future residual returns only to measure headroom.",
            "No universe expansion, transaction costs, borrow costs, model selection, or test-window performance are computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_signal_panel(
    signal_panel_path: str | Path,
    *,
    variant_names: Sequence[str] | None,
) -> pd.DataFrame:
    panel = pd.read_csv(signal_panel_path, usecols=list(BASE_COLUMNS), low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    if variant_names is not None:
        panel = panel[panel["variant"].isin(set(variant_names))].copy()
    for column in BASE_COLUMNS[3:]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["beta", TARGET_COLUMN, "forward_return_5d"])
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _selector_daily_diagnostics(
    panel: pd.DataFrame,
    *,
    candidate_count: int,
    loser_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = panel.groupby(["variant", "session_date"], sort=True)
    for (variant, session_date), group in grouped:
        frame = group.drop_duplicates("symbol", keep="last").copy()
        if len(frame) < candidate_count * 2:
            continue
        frame = _add_selector_scores(frame)
        oracle = frame.nsmallest(candidate_count, TARGET_COLUMN)
        oracle_symbols = set(oracle["symbol"])
        loser_cutoff = frame[TARGET_COLUMN].quantile(loser_quantile)
        loser_symbols = set(frame.loc[frame[TARGET_COLUMN] <= loser_cutoff, "symbol"])
        for selector_name, selector_label in _selector_catalog():
            subset = frame.dropna(subset=[selector_name, TARGET_COLUMN])
            if len(subset) < candidate_count * 2:
                continue
            selected = subset.sort_values(
                [selector_name, "symbol"],
                ascending=[False, True],
            ).head(candidate_count)
            residual = selected[TARGET_COLUMN]
            raw_return = selected["forward_return_5d"]
            oracle_overlap = len(set(selected["symbol"]).intersection(oracle_symbols)) / candidate_count
            loser_capture = len(set(selected["symbol"]).intersection(loser_symbols)) / candidate_count
            rank_ic = subset[selector_name].corr(subset[TARGET_COLUMN], method="spearman")
            rows.append(
                {
                    "session_date": session_date,
                    "variant": variant,
                    "selector": selector_name,
                    "selector_label": selector_label,
                    "eligible_names": int(len(subset)),
                    "selected_names": int(len(selected)),
                    "selected_mean_forward_return_5d": float(raw_return.mean()),
                    "selected_mean_forward_beta_residual_return_5d": float(residual.mean()),
                    "short_contribution_return": float(-raw_return.mean()),
                    "short_contribution_beta_residual_return": float(-residual.mean()),
                    "selected_negative_residual_share": float((residual < 0).mean()),
                    "selected_negative_raw_share": float((raw_return < 0).mean()),
                    "oracle_mean_forward_beta_residual_return_5d": float(oracle[TARGET_COLUMN].mean()),
                    "oracle_short_contribution_beta_residual_return": float(
                        -oracle[TARGET_COLUMN].mean()
                    ),
                    "oracle_overlap_rate": float(oracle_overlap),
                    "loser_quantile_capture_rate": float(loser_capture),
                    "rank_ic_loser_score_vs_forward_residual": float(rank_ic),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_daily_columns())


def _add_selector_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    safe_adv = result["trailing_median_dollar_volume_20"].where(
        result["trailing_median_dollar_volume_20"] > 0
    )
    safe_price = result["lagged_close"].where(result["lagged_close"] > 0)
    result["short_reversal_winner"] = -result["reversal_5d"]
    result["short_weak_momentum_20"] = -result["momentum_20d"]
    result["short_weak_momentum_60"] = -result["momentum_60d"]
    result["short_weak_residual_momentum_20"] = -result["beta_residual_momentum_20d"]
    result["short_vol_adjusted_weakness_20"] = -result["vol_adjusted_momentum_20d"]
    result["short_illiquidity_proxy"] = result["liquidity_rank"]
    result["short_high_beta_proxy"] = result["beta"]
    result["short_low_price_proxy"] = -np.log(safe_price)
    result["short_low_adv_proxy"] = -np.log(safe_adv)
    result["short_random_control"] = result["random_control"]
    result["short_exhausted_winner_20_5"] = _zscore(result["momentum_20d"]) + _zscore(
        -result["reversal_5d"]
    )
    result["short_residual_overextension_20_5"] = _zscore(
        result["beta_residual_momentum_20d"]
    ) + _zscore(-result["reversal_5d"])
    result["short_breakdown_after_60d_strength"] = _zscore(result["momentum_60d"]) + _zscore(
        result["reversal_5d"]
    )
    result["short_fragile_winner_proxy"] = (
        _zscore(-result["reversal_5d"])
        + _zscore(result["beta"])
        + _zscore(-np.log(safe_adv))
    )
    return result


def _selector_catalog() -> list[tuple[str, str]]:
    return [
        ("short_reversal_winner", "Current Phase4 short rule: recent 5-session winner"),
        ("short_weak_momentum_20", "Weak 20-session momentum continuation"),
        ("short_weak_momentum_60", "Weak 60-session momentum continuation"),
        ("short_weak_residual_momentum_20", "Weak 20-session residual momentum"),
        ("short_vol_adjusted_weakness_20", "Weak volatility-adjusted 20-session momentum"),
        ("short_illiquidity_proxy", "Lower-liquidity risk proxy inside current universe"),
        ("short_high_beta_proxy", "High-beta fragility proxy"),
        ("short_low_price_proxy", "Low-price fragility proxy inside current universe"),
        ("short_low_adv_proxy", "Lower-ADV fragility proxy inside current universe"),
        ("short_exhausted_winner_20_5", "20-day winner plus 5-day extension"),
        ("short_residual_overextension_20_5", "Residual momentum plus 5-day extension"),
        ("short_breakdown_after_60d_strength", "60-day strength with recent 5-day weakness"),
        ("short_fragile_winner_proxy", "5-day winner plus high beta and lower ADV"),
        ("short_random_control", "Random control"),
    ]


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / std


def _selector_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_summary_columns())
    rows = []
    for (variant, selector), group in daily.groupby(["variant", "selector"], sort=True):
        short_residual = group["short_contribution_beta_residual_return"]
        oracle_gap = group["oracle_short_contribution_beta_residual_return"] - short_residual
        rows.append(
            {
                "variant": variant,
                "selector": selector,
                "selector_label": group["selector_label"].iloc[0],
                "sessions": int(len(group)),
                "mean_short_contribution_return": float(group["short_contribution_return"].mean()),
                "mean_short_contribution_beta_residual_return": float(short_residual.mean()),
                "short_residual_hit_rate": float((short_residual > 0).mean()),
                "mean_selected_negative_residual_share": float(
                    group["selected_negative_residual_share"].mean()
                ),
                "mean_selected_negative_raw_share": float(
                    group["selected_negative_raw_share"].mean()
                ),
                "mean_oracle_short_contribution_beta_residual_return": float(
                    group["oracle_short_contribution_beta_residual_return"].mean()
                ),
                "mean_oracle_gap_beta_residual_return": float(oracle_gap.mean()),
                "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                "mean_loser_quantile_capture_rate": float(
                    group["loser_quantile_capture_rate"].mean()
                ),
                "mean_rank_ic_loser_score_vs_forward_residual": float(
                    group["rank_ic_loser_score_vs_forward_residual"].mean()
                ),
                "rank_ic_negative_share": float(
                    (group["rank_ic_loser_score_vs_forward_residual"] < 0).mean()
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_summary_columns()).sort_values(
        ["variant", "mean_short_contribution_beta_residual_return", "mean_oracle_overlap_rate"],
        ascending=[True, False, False],
    )


def _improvement_vs_reversal(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=_improvement_columns())
    rows = []
    for variant, group in summary.groupby("variant", sort=True):
        baseline = group[group["selector"] == "short_reversal_winner"]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in group.iterrows():
            rows.append(
                {
                    "variant": variant,
                    "selector": row["selector"],
                    "selector_label": row["selector_label"],
                    "baseline_selector": "short_reversal_winner",
                    "delta_short_contribution_beta_residual_return": float(
                        row["mean_short_contribution_beta_residual_return"]
                        - base["mean_short_contribution_beta_residual_return"]
                    ),
                    "delta_oracle_overlap_rate": float(
                        row["mean_oracle_overlap_rate"] - base["mean_oracle_overlap_rate"]
                    ),
                    "delta_loser_quantile_capture_rate": float(
                        row["mean_loser_quantile_capture_rate"]
                        - base["mean_loser_quantile_capture_rate"]
                    ),
                    "beats_reversal_on_residual": bool(
                        row["mean_short_contribution_beta_residual_return"]
                        > base["mean_short_contribution_beta_residual_return"]
                    ),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_improvement_columns()).sort_values(
        ["variant", "delta_short_contribution_beta_residual_return"],
        ascending=[True, False],
    )


def _universe_opportunity_summary(
    panel: pd.DataFrame,
    *,
    candidate_count: int,
) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        daily = []
        for _, frame in group.groupby("session_date", sort=False):
            clean = frame.dropna(subset=[TARGET_COLUMN, "forward_return_5d"])
            if len(clean) < candidate_count * 2:
                continue
            oracle = clean.nsmallest(candidate_count, TARGET_COLUMN)
            daily.append(
                {
                    "eligible_names": len(clean),
                    "negative_raw_share": float((clean["forward_return_5d"] < 0).mean()),
                    "negative_residual_share": float((clean[TARGET_COLUMN] < 0).mean()),
                    "oracle_short_contribution_beta_residual_return": float(
                        -oracle[TARGET_COLUMN].mean()
                    ),
                    "oracle_short_contribution_return": float(
                        -oracle["forward_return_5d"].mean()
                    ),
                }
            )
        daily_frame = pd.DataFrame(daily)
        rows.append(
            {
                "variant": variant,
                "sessions": int(len(daily_frame)),
                "mean_eligible_names": float(daily_frame["eligible_names"].mean()),
                "mean_negative_raw_share": float(daily_frame["negative_raw_share"].mean()),
                "mean_negative_residual_share": float(
                    daily_frame["negative_residual_share"].mean()
                ),
                "mean_oracle_short_contribution_beta_residual_return": float(
                    daily_frame["oracle_short_contribution_beta_residual_return"].mean()
                ),
                "mean_oracle_short_contribution_return": float(
                    daily_frame["oracle_short_contribution_return"].mean()
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_opportunity_columns())


def _feature_roadmap() -> pd.DataFrame:
    rows = [
        {
            "feature_family": "price_exhaustion",
            "availability": "derivable_now",
            "examples": "5d winner, 20d winner plus 5d extension, residual winner extension",
            "why_it_may_help_short": "Tests whether crowded recent winners mean-revert on residual returns.",
            "next_action": "Promote only if Phase4C selector beats reversal across core variants.",
        },
        {
            "feature_family": "weakness_continuation",
            "availability": "derivable_now",
            "examples": "weak 20d momentum, weak 60d momentum, weak residual momentum",
            "why_it_may_help_short": "Tests whether losers keep underperforming instead of rebounding.",
            "next_action": "Compare as asymmetric short-only signal; do not reuse long-side direction blindly.",
        },
        {
            "feature_family": "fragility_proxies",
            "availability": "derivable_now",
            "examples": "high beta, low price, lower ADV inside current universe",
            "why_it_may_help_short": "Captures names that may sell off harder when attention or liquidity fades.",
            "next_action": "Use only as tie-breaker unless standalone residual loser diagnostics are strong.",
        },
        {
            "feature_family": "fundamental_quality",
            "availability": "needs_new_data",
            "examples": "profitability, accruals, leverage, dilution, gross margin deterioration",
            "why_it_may_help_short": "Short books often need business-quality deterioration, not just price action.",
            "next_action": "Evaluate Sharadar, Norgate add-ons, or another fundamentals source before Phase5 freeze.",
        },
        {
            "feature_family": "earnings_revision_events",
            "availability": "needs_new_data",
            "examples": "estimate revisions, earnings surprises, guidance cuts, post-earnings drift",
            "why_it_may_help_short": "Future residual losers are often tied to negative information flow.",
            "next_action": "Add only with strict announcement-time and filing-time lag controls.",
        },
        {
            "feature_family": "valuation_and_growth_mismatch",
            "availability": "needs_new_data",
            "examples": "expensive valuation with slowing sales or margin pressure",
            "why_it_may_help_short": "Helps separate justified winners from fragile overvaluation.",
            "next_action": "Require sector-relative transforms to avoid pure style bets.",
        },
        {
            "feature_family": "borrow_and_short_interest",
            "availability": "needs_new_data",
            "examples": "borrow fee, utilization, shares available, short interest, days to cover",
            "why_it_may_help_short": "Filters untradable shorts and identifies squeeze-prone names.",
            "next_action": "Use as risk filter before using it as alpha input.",
        },
        {
            "feature_family": "event_and_news_risk",
            "availability": "needs_new_data",
            "examples": "fraud/legal flags, regulatory events, downgrades, product failures",
            "why_it_may_help_short": "Can identify idiosyncratic losers not visible in price-only features.",
            "next_action": "Keep out of core model until timestamp quality is auditable.",
        },
    ]
    return pd.DataFrame(rows)


def _lab_memo(
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    opportunity: pd.DataFrame,
    roadmap: pd.DataFrame,
    *,
    candidate_count: int,
    loser_quantile: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 4C Residual Loser Lab Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only same-universe research lab for identifying future "
        "beta-residual losers. It does not expand the universe, construct a new "
        "portfolio, select a final model, or inspect the test lockbox.",
        "",
        "## Method",
        "",
        f"- candidates selected per session: `{candidate_count}`",
        f"- loser quantile for capture diagnostics: `{loser_quantile}`",
        "- selector scores are oriented so higher score means more likely residual loser;",
        "- oracle diagnostics use future residual returns only to measure headroom;",
        "- feature-roadmap rows separate currently derivable proxies from data that must be sourced later.",
        "",
        "## Top Selector Snapshot",
        "",
        "| Variant | Selector | Mean Short Residual Contribution | Oracle Overlap | RankIC |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.head(30).iterrows():
        lines.append(
            f"| {row['variant']} | {row['selector']} | "
            f"{row['mean_short_contribution_beta_residual_return']} | "
            f"{row['mean_oracle_overlap_rate']} | "
            f"{row['mean_rank_ic_loser_score_vs_forward_residual']} |"
        )
    lines.extend(
        [
            "",
            "## Best Improvements Versus Current Reversal Short Rule",
            "",
            "| Variant | Selector | Delta Residual Contribution | Delta Oracle Overlap |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in improvement.head(24).iterrows():
        lines.append(
            f"| {row['variant']} | {row['selector']} | "
            f"{row['delta_short_contribution_beta_residual_return']} | "
            f"{row['delta_oracle_overlap_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Universe Opportunity",
            "",
            "| Variant | Negative Residual Share | Oracle Short Residual Contribution |",
            "|---|---:|---:|",
        ]
    )
    for _, row in opportunity.iterrows():
        lines.append(
            f"| {row['variant']} | {row['mean_negative_residual_share']} | "
            f"{row['mean_oracle_short_contribution_beta_residual_return']} |"
        )
    lines.extend(
        [
            "",
            "## Feature Roadmap",
            "",
            "| Family | Availability | Next Action |",
            "|---|---|---|",
        ]
    )
    for _, row in roadmap.iterrows():
        lines.append(
            f"| {row['feature_family']} | {row['availability']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not treat the oracle columns as implementable signals.",
            "- Do not expand the universe in this phase.",
            "- Any selected short feature must go through a new Phase 4 construction pass before Phase 5.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "selector",
        "selector_label",
        "eligible_names",
        "selected_names",
        "selected_mean_forward_return_5d",
        "selected_mean_forward_beta_residual_return_5d",
        "short_contribution_return",
        "short_contribution_beta_residual_return",
        "selected_negative_residual_share",
        "selected_negative_raw_share",
        "oracle_mean_forward_beta_residual_return_5d",
        "oracle_short_contribution_beta_residual_return",
        "oracle_overlap_rate",
        "loser_quantile_capture_rate",
        "rank_ic_loser_score_vs_forward_residual",
        "test_window_used",
    ]


def _summary_columns() -> list[str]:
    return [
        "variant",
        "selector",
        "selector_label",
        "sessions",
        "mean_short_contribution_return",
        "mean_short_contribution_beta_residual_return",
        "short_residual_hit_rate",
        "mean_selected_negative_residual_share",
        "mean_selected_negative_raw_share",
        "mean_oracle_short_contribution_beta_residual_return",
        "mean_oracle_gap_beta_residual_return",
        "mean_oracle_overlap_rate",
        "mean_loser_quantile_capture_rate",
        "mean_rank_ic_loser_score_vs_forward_residual",
        "rank_ic_negative_share",
        "test_window_used",
    ]


def _improvement_columns() -> list[str]:
    return [
        "variant",
        "selector",
        "selector_label",
        "baseline_selector",
        "delta_short_contribution_beta_residual_return",
        "delta_oracle_overlap_rate",
        "delta_loser_quantile_capture_rate",
        "beats_reversal_on_residual",
        "test_window_used",
    ]


def _opportunity_columns() -> list[str]:
    return [
        "variant",
        "sessions",
        "mean_eligible_names",
        "mean_negative_raw_share",
        "mean_negative_residual_share",
        "mean_oracle_short_contribution_beta_residual_return",
        "mean_oracle_short_contribution_return",
        "test_window_used",
    ]


def _first_or_none(series: pd.Series) -> str | None:
    if len(series) == 0:
        return None
    return str(series.min())


def _last_or_none(series: pd.Series) -> str | None:
    if len(series) == 0:
        return None
    return str(series.max())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 4C residual-loser lab.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    args = parser.parse_args(argv)

    result = build_phase4c_residual_loser_lab(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants) if args.variants else None,
        candidate_count=args.candidate_count,
        loser_quantile=args.loser_quantile,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
