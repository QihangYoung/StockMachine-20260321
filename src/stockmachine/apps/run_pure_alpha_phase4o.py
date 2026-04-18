from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4f import DEFAULT_PHASE3_SIGNAL_PANEL
from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
)
from stockmachine.apps.run_pure_alpha_phase4k import (
    DEFAULT_SHORT_SELECTOR,
    _construct_books,
    _load_panel,
    _summary,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4o_long_selector_refresh_20260418"
DEFAULT_LONG_VARIANTS = ("top500_clean_core_beta_full", "top1000_clean_core_beta_full")
DEFAULT_SHORT_VARIANTS = ("adv30m_clean_core_beta_full",)
DEFAULT_LONG_SELECTORS = (
    "reversal_5d",
    "transparent_composite",
    "momentum_20d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "long_dip_in_20d_winner",
    "long_dip_in_residual_winner",
    "long_dip_in_vol_adjusted_winner",
    "long_reversal_beta_penalty",
)
LONG_SELECTOR_LABELS = {
    "reversal_5d": "5-session reversal",
    "transparent_composite": "Phase3 transparent composite",
    "momentum_20d": "20-session momentum",
    "beta_residual_momentum_20d": "20-session beta-residual momentum",
    "vol_adjusted_momentum_20d": "20-session vol-adjusted momentum",
    "long_dip_in_20d_winner": "5d dip inside 20d winner",
    "long_dip_in_residual_winner": "5d dip inside beta-residual winner",
    "long_dip_in_vol_adjusted_winner": "5d dip inside vol-adjusted winner",
    "long_reversal_beta_penalty": "5d reversal with high-beta penalty",
}


def build_phase4o_long_selector_refresh_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variants: Sequence[str] = DEFAULT_LONG_VARIANTS,
    short_variants: Sequence[str] = DEFAULT_SHORT_VARIANTS,
    long_selectors: Sequence[str] = DEFAULT_LONG_SELECTORS,
    short_selector: str = DEFAULT_SHORT_SELECTOR,
    residual_target: str = TARGET_CS_DEMEANED,
    candidate_count: int = 30,
    min_names_per_side: int = 10,
    target_names_per_side: int = 20,
    max_names_per_side: int = 30,
    max_single_name_side_weight: float = 0.05,
    beta_match_tolerance: float = 0.05,
    min_regression_rows: int = 80,
) -> dict[str, Any]:
    """Refresh transparent long selectors on standalone and beta-matched validation metrics."""

    _validate_settings(
        long_variants=long_variants,
        short_variants=short_variants,
        long_selectors=long_selectors,
        candidate_count=candidate_count,
        min_names_per_side=min_names_per_side,
        target_names_per_side=target_names_per_side,
        max_names_per_side=max_names_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        beta_match_tolerance=beta_match_tolerance,
        min_regression_rows=min_regression_rows,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel(
        signal_panel_path,
        variant_names=tuple(dict.fromkeys([*long_variants, *short_variants])),
        min_regression_rows=min_regression_rows,
    )
    panel = _add_long_selector_scores(panel)
    selector_daily = _standalone_long_daily(
        panel,
        long_variants=long_variants,
        long_selectors=long_selectors,
        residual_target=residual_target,
        candidate_count=candidate_count,
    )
    selector_summary = _standalone_long_summary(selector_daily)
    positions, portfolio_daily, portfolio_skipped, portfolio_summary = _portfolio_refresh(
        panel,
        long_variants=long_variants,
        short_variants=short_variants,
        long_selectors=long_selectors,
        short_selector=short_selector,
        min_names_per_side=min_names_per_side,
        target_names_per_side=target_names_per_side,
        max_names_per_side=max_names_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        beta_match_tolerance=beta_match_tolerance,
    )
    paths = _write_outputs(
        output_dir,
        selector_daily=selector_daily,
        selector_summary=selector_summary,
        positions=positions,
        portfolio_daily=portfolio_daily,
        portfolio_skipped=portfolio_skipped,
        portfolio_summary=portfolio_summary,
    )
    memo_path = output_dir / "phase4o_long_selector_refresh_memo.md"
    memo_path.write_text(
        _memo(
            selector_summary,
            portfolio_summary,
            long_variants=long_variants,
            short_variants=short_variants,
            long_selectors=long_selectors,
            short_selector=short_selector,
        ),
        encoding="utf-8",
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "long_variants": list(long_variants),
        "short_variants": list(short_variants),
        "long_selectors": list(long_selectors),
        "short_selector": short_selector,
        "residual_target": residual_target,
        "candidate_count": int(candidate_count),
        "panel_rows_loaded": int(len(panel)),
        "selector_daily_rows": int(len(selector_daily)),
        "selector_summary_rows": int(len(selector_summary)),
        "portfolio_daily_rows": int(len(portfolio_daily)),
        "portfolio_summary_rows": int(len(portfolio_summary)),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "transparent_long_selector_refresh_with_beta_matched_recheck",
        "primary_evaluation": "standalone_long_selector_performance",
        **paths,
    }
    (output_dir / "phase4o_long_selector_refresh_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return rollup


def _validate_settings(
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
    long_selectors: Sequence[str],
    candidate_count: int,
    min_names_per_side: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
    min_regression_rows: int,
) -> None:
    if not long_variants:
        raise ValueError("At least one long variant is required.")
    if not short_variants:
        raise ValueError("At least one short variant is required.")
    if not long_selectors:
        raise ValueError("At least one long selector is required.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if not min_names_per_side <= target_names_per_side <= max_names_per_side:
        raise ValueError("Name counts must satisfy min <= target <= max.")
    if max_names_per_side * max_single_name_side_weight < 1.0:
        raise ValueError("Max names and single-name cap cannot sum to one full side.")
    if beta_match_tolerance < 0:
        raise ValueError("beta_match_tolerance must be non-negative.")
    if min_regression_rows <= 0:
        raise ValueError("min_regression_rows must be positive.")


def _add_long_selector_scores(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frame = group.copy()
        reversal_z = _zscore(frame["reversal_5d"])
        momentum_z = _zscore(frame["momentum_20d"])
        residual_momentum_z = _zscore(frame["beta_residual_momentum_20d"])
        vol_adjusted_z = _zscore(frame["vol_adjusted_momentum_20d"])
        beta_z = _zscore(frame["beta"])
        frame["long_dip_in_20d_winner"] = pd.concat(
            [reversal_z, momentum_z], axis=1
        ).mean(axis=1)
        frame["long_dip_in_residual_winner"] = pd.concat(
            [reversal_z, residual_momentum_z], axis=1
        ).mean(axis=1)
        frame["long_dip_in_vol_adjusted_winner"] = pd.concat(
            [reversal_z, vol_adjusted_z], axis=1
        ).mean(axis=1)
        frame["long_reversal_beta_penalty"] = reversal_z - 0.25 * beta_z.clip(lower=0.0)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else panel


def _standalone_long_daily(
    panel: pd.DataFrame,
    *,
    long_variants: Sequence[str],
    long_selectors: Sequence[str],
    residual_target: str,
    candidate_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    long_panel = panel.loc[panel["variant"].isin(set(long_variants))].copy()
    for (variant, session_date), group in long_panel.groupby(["variant", "session_date"], sort=True):
        target_frame = group.dropna(subset=[residual_target, "forward_return_5d"]).copy()
        if len(target_frame) < candidate_count * 2:
            continue
        oracle = target_frame.nlargest(candidate_count, residual_target)
        oracle_symbols = set(oracle["symbol"])
        for selector in long_selectors:
            subset = target_frame.dropna(subset=[selector]).copy()
            if len(subset) < candidate_count * 2:
                continue
            selected = subset.sort_values([selector, "symbol"], ascending=[False, True]).head(
                candidate_count
            )
            target = selected[residual_target]
            beta_residual = selected[TARGET_BETA_RESIDUAL]
            cs_residual = selected[TARGET_CS_DEMEANED]
            risk_residual = selected[TARGET_RISK_NEUTRAL]
            rank_ic = subset[selector].corr(subset[residual_target], method="spearman")
            rows.append(
                {
                    "session_date": session_date,
                    "variant": variant,
                    "residual_target": residual_target,
                    "selector": selector,
                    "selector_label": LONG_SELECTOR_LABELS.get(selector, selector),
                    "eligible_names": int(len(subset)),
                    "selected_names": int(len(selected)),
                    "selected_mean_target_return": float(target.mean()),
                    "long_contribution_target": float(target.mean()),
                    "selected_positive_target_share": float((target > 0).mean()),
                    "long_contribution_beta_residual": float(beta_residual.mean()),
                    "long_contribution_cs_demeaned_residual": float(cs_residual.mean()),
                    "long_contribution_risk_neutral_residual": float(risk_residual.mean()),
                    "long_contribution_return": float(selected["forward_return_5d"].mean()),
                    "oracle_long_contribution_target": float(oracle[residual_target].mean()),
                    "oracle_overlap_rate": float(
                        len(set(selected["symbol"]).intersection(oracle_symbols))
                        / candidate_count
                    ),
                    "rank_ic_selector_vs_target": float(rank_ic),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_selector_daily_columns())


def _standalone_long_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_selector_summary_columns())
    rows = []
    for (variant, target, selector), group in daily.groupby(
        ["variant", "residual_target", "selector"], sort=True
    ):
        contribution = group["long_contribution_target"]
        rows.append(
            {
                "variant": variant,
                "residual_target": target,
                "selector": selector,
                "selector_label": group["selector_label"].iloc[0],
                "sessions": int(len(group)),
                "mean_long_contribution_target": float(contribution.mean()),
                "target_hit_rate": float((contribution > 0).mean()),
                "mean_selected_positive_target_share": float(
                    group["selected_positive_target_share"].mean()
                ),
                "mean_long_contribution_beta_residual": float(
                    group["long_contribution_beta_residual"].mean()
                ),
                "mean_long_contribution_cs_demeaned_residual": float(
                    group["long_contribution_cs_demeaned_residual"].mean()
                ),
                "mean_long_contribution_risk_neutral_residual": float(
                    group["long_contribution_risk_neutral_residual"].mean()
                ),
                "mean_long_contribution_return": float(
                    group["long_contribution_return"].mean()
                ),
                "mean_oracle_long_contribution_target": float(
                    group["oracle_long_contribution_target"].mean()
                ),
                "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                "mean_rank_ic_selector_vs_target": float(
                    group["rank_ic_selector_vs_target"].mean()
                ),
                "rank_ic_positive_share": float(
                    (group["rank_ic_selector_vs_target"] > 0).mean()
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_selector_summary_columns()).sort_values(
        ["variant", "mean_long_contribution_target", "target_hit_rate"],
        ascending=[True, False, False],
    )


def _portfolio_refresh(
    panel: pd.DataFrame,
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
    long_selectors: Sequence[str],
    short_selector: str,
    min_names_per_side: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    position_frames = []
    daily_frames = []
    skipped_frames = []
    summary_frames = []
    required_nonzero_names = max(min_names_per_side, int(np.ceil(1.0 / max_single_name_side_weight)))
    for long_selector in long_selectors:
        positions, daily, skipped = _construct_books(
            panel,
            long_variants=long_variants,
            short_variants=short_variants,
            long_score=long_selector,
            short_selector=short_selector,
            required_nonzero_names=required_nonzero_names,
            target_names_per_side=target_names_per_side,
            max_names_per_side=max_names_per_side,
            max_single_name_side_weight=max_single_name_side_weight,
            beta_match_tolerance=beta_match_tolerance,
        )
        summary = _summary(daily, skipped)
        for frame in (positions, daily, skipped, summary):
            if not frame.empty:
                frame["long_selector_label"] = LONG_SELECTOR_LABELS.get(
                    long_selector, long_selector
                )
        position_frames.append(positions)
        daily_frames.append(daily)
        skipped_frames.append(skipped)
        summary_frames.append(summary)
    return (
        pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame(),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
        pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
    )


def _write_outputs(
    output_dir: Path,
    *,
    selector_daily: pd.DataFrame,
    selector_summary: pd.DataFrame,
    positions: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    portfolio_skipped: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "selector_daily_artifact": output_dir / "phase4o_long_selector_daily_validation.csv",
        "selector_summary_artifact": output_dir / "phase4o_long_selector_summary_validation.csv",
        "portfolio_positions_artifact": output_dir
        / "phase4o_long_selector_portfolio_positions_validation.csv.gz",
        "portfolio_daily_artifact": output_dir
        / "phase4o_long_selector_portfolio_daily_validation.csv",
        "portfolio_skipped_artifact": output_dir
        / "phase4o_long_selector_portfolio_skipped_validation.csv",
        "portfolio_summary_artifact": output_dir
        / "phase4o_long_selector_portfolio_summary_validation.csv",
    }
    selector_daily.to_csv(paths["selector_daily_artifact"], index=False)
    selector_summary.to_csv(paths["selector_summary_artifact"], index=False)
    positions.to_csv(paths["portfolio_positions_artifact"], index=False, compression="gzip")
    portfolio_daily.to_csv(paths["portfolio_daily_artifact"], index=False)
    portfolio_skipped.to_csv(paths["portfolio_skipped_artifact"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary_artifact"], index=False)
    return {key: path.as_posix() for key, path in paths.items()}


def _memo(
    selector_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
    long_selectors: Sequence[str],
    short_selector: str,
) -> str:
    lines = [
        "# Pure Alpha Phase 4O Long Selector Refresh Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Validation-only refresh of transparent long-side selectors, with top500 and "
        "top1000 long universes evaluated standalone before beta-matched rechecks.",
        "",
        "## Inputs",
        "",
        f"- long variants: `{', '.join(long_variants)}`",
        f"- short variants for portfolio recheck: `{', '.join(short_variants)}`",
        f"- short selector: `{short_selector}`",
        f"- long selectors: `{', '.join(long_selectors)}`",
        "",
        "## Standalone Long Selector Summary",
        "",
        "| Variant | Selector | Contribution | Hit Rate | Positive Share | RankIC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in selector_summary.head(18).iterrows():
        lines.append(
            f"| {row['variant']} | {row['selector']} | "
            f"{row['mean_long_contribution_target']} | {row['target_hit_rate']} | "
            f"{row['mean_selected_positive_target_share']} | "
            f"{row['mean_rank_ic_selector_vs_target']} |"
        )
    lines.extend(
        [
            "",
            "## Beta-Matched Portfolio Summary",
            "",
            "| Pair | Long Selector | Constructed | CS Spread | Long CS | Short CS | Abs Net Beta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    if portfolio_summary.empty:
        lines.append("| n/a | n/a | 0 | n/a | n/a | n/a | n/a |")
    else:
        top_portfolio = portfolio_summary.sort_values(
            ["mean_spread_cs_demeaned_residual", "constructed_sessions"],
            ascending=[False, False],
        ).head(18)
        for _, row in top_portfolio.iterrows():
            long_cs = (
                row["mean_spread_cs_demeaned_residual"]
                - row["mean_short_cs_demeaned_residual_contribution"]
            )
            lines.append(
                f"| {row['pair']} | {row['long_score']} | {row['constructed_sessions']} | "
                f"{row['mean_spread_cs_demeaned_residual']} | {long_cs} | "
                f"{row['mean_short_cs_demeaned_residual_contribution']} | "
                f"{row['mean_abs_net_beta']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Standalone long selector performance is the primary diagnostic.",
            "- Portfolio rechecks reuse the transparent short selector and beta-matching constructor.",
            "- No transaction costs, borrow costs, turnover model, or test-window performance is computed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / std


def _selector_daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "residual_target",
        "selector",
        "selector_label",
        "eligible_names",
        "selected_names",
        "selected_mean_target_return",
        "long_contribution_target",
        "selected_positive_target_share",
        "long_contribution_beta_residual",
        "long_contribution_cs_demeaned_residual",
        "long_contribution_risk_neutral_residual",
        "long_contribution_return",
        "oracle_long_contribution_target",
        "oracle_overlap_rate",
        "rank_ic_selector_vs_target",
        "test_window_used",
    ]


def _selector_summary_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "selector",
        "selector_label",
        "sessions",
        "mean_long_contribution_target",
        "target_hit_rate",
        "mean_selected_positive_target_share",
        "mean_long_contribution_beta_residual",
        "mean_long_contribution_cs_demeaned_residual",
        "mean_long_contribution_risk_neutral_residual",
        "mean_long_contribution_return",
        "mean_oracle_long_contribution_target",
        "mean_oracle_overlap_rate",
        "mean_rank_ic_selector_vs_target",
        "rank_ic_positive_share",
        "test_window_used",
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 4O long selector refresh.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--long-variant", action="append", dest="long_variants")
    parser.add_argument("--short-variant", action="append", dest="short_variants")
    parser.add_argument("--long-selector", action="append", dest="long_selectors")
    parser.add_argument("--short-selector", default=DEFAULT_SHORT_SELECTOR)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    args = parser.parse_args(argv)
    result = build_phase4o_long_selector_refresh_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        long_variants=tuple(args.long_variants or DEFAULT_LONG_VARIANTS),
        short_variants=tuple(args.short_variants or DEFAULT_SHORT_VARIANTS),
        long_selectors=tuple(args.long_selectors or DEFAULT_LONG_SELECTORS),
        short_selector=args.short_selector,
        candidate_count=args.candidate_count,
        min_regression_rows=args.min_regression_rows,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
