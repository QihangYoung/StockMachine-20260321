from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import (
    DEFAULT_VARIANTS,
    RESEARCH_ROOT,
    TARGET_COLUMN,
    _zscore,
)
from stockmachine.apps.run_pure_alpha_phase4f import (
    DEFAULT_PHASE3_SIGNAL_PANEL,
    _load_posterior_panel,
)
from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
    _add_residual_targets,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4j_residual_target_short_selector_20260418"
DEFAULT_TARGETS = (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
)
CORE_SELECTOR_FEATURES = (
    "momentum_20d",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d",
)
OVEREXTENSION_SELECTOR_FEATURES = (
    "exhausted_winner_20_5",
    "residual_overextension_20_5",
)


def build_phase4j_short_selector_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_VARIANTS,
    residual_targets: Sequence[str] = DEFAULT_TARGETS,
    candidate_count: int = 30,
    loser_quantile: float = 0.20,
    min_regression_rows: int = 80,
) -> dict[str, Any]:
    """Evaluate transparent short selectors across residual target definitions."""

    _validate_settings(
        variant_names=variant_names,
        residual_targets=residual_targets,
        candidate_count=candidate_count,
        loser_quantile=loser_quantile,
        min_regression_rows=min_regression_rows,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_posterior_panel(
        signal_panel_path,
        variant_names=variant_names,
        target_column=TARGET_COLUMN,
    )
    panel = _add_residual_targets(panel, min_regression_rows=min_regression_rows)
    daily = _selector_daily_diagnostics(
        panel,
        residual_targets=residual_targets,
        candidate_count=candidate_count,
        loser_quantile=loser_quantile,
    )
    summary = _selector_summary(daily)

    daily_path = output_dir / "phase4j_short_selector_daily_validation.csv"
    summary_path = output_dir / "phase4j_short_selector_summary_validation.csv"
    memo_path = output_dir / "phase4j_short_selector_memo.md"
    rollup_path = output_dir / "phase4j_short_selector_rollup.json"

    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    memo_path.write_text(
        _selector_memo(
            summary,
            candidate_count=candidate_count,
            loser_quantile=loser_quantile,
            min_regression_rows=min_regression_rows,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "variant_names": list(variant_names),
        "residual_targets": list(residual_targets),
        "candidate_count": int(candidate_count),
        "loser_quantile": float(loser_quantile),
        "min_regression_rows": int(min_regression_rows),
        "panel_rows_loaded": int(len(panel)),
        "daily_rows": int(len(daily)),
        "summary_rows": int(len(summary)),
        "daily_artifact": daily_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "transparent_short_selector_across_residual_targets",
        "limitations": [
            "This is a selector diagnostic, not a beta-matched portfolio constructor.",
            "Residual targets are validation-only cross-sectional diagnostics.",
            "No transaction costs, borrow costs, or test-window performance are computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    variant_names: Sequence[str],
    residual_targets: Sequence[str],
    candidate_count: int,
    loser_quantile: float,
    min_regression_rows: int,
) -> None:
    if not variant_names:
        raise ValueError("At least one variant is required.")
    if not residual_targets:
        raise ValueError("At least one residual target is required.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")
    if min_regression_rows <= 0:
        raise ValueError("min_regression_rows must be positive.")


def _selector_daily_diagnostics(
    panel: pd.DataFrame,
    *,
    residual_targets: Sequence[str],
    candidate_count: int,
    loser_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True):
        frame = _add_selector_scores(group.drop_duplicates("symbol", keep="last"))
        if len(frame) < candidate_count * 2:
            continue
        for target_name in residual_targets:
            target_frame = frame.dropna(subset=[target_name, "forward_return_5d"])
            if len(target_frame) < candidate_count * 2:
                continue
            oracle = target_frame.nsmallest(candidate_count, target_name)
            oracle_symbols = set(oracle["symbol"])
            loser_cutoff = float(target_frame[target_name].quantile(loser_quantile))
            loser_symbols = set(
                target_frame.loc[target_frame[target_name] <= loser_cutoff, "symbol"]
            )
            for selector_name, selector_label in _selector_catalog():
                subset = target_frame.dropna(subset=[selector_name]).copy()
                if len(subset) < candidate_count * 2:
                    continue
                selected = subset.sort_values(
                    [selector_name, "symbol"],
                    ascending=[False, True],
                ).head(candidate_count)
                target = selected[target_name]
                beta_residual = selected[TARGET_BETA_RESIDUAL]
                cs_residual = selected[TARGET_CS_DEMEANED]
                risk_residual = selected[TARGET_RISK_NEUTRAL]
                raw = selected["forward_return_5d"]
                rank_ic = subset[selector_name].corr(subset[target_name], method="spearman")
                rows.append(
                    {
                        "session_date": session_date,
                        "variant": variant,
                        "residual_target": target_name,
                        "selector": selector_name,
                        "selector_label": selector_label,
                        "eligible_names": int(len(subset)),
                        "selected_names": int(len(selected)),
                        "selected_mean_target_return": float(target.mean()),
                        "short_contribution_target": float(-target.mean()),
                        "selected_negative_target_share": float((target < 0).mean()),
                        "selected_mean_beta_residual": float(beta_residual.mean()),
                        "short_contribution_beta_residual": float(-beta_residual.mean()),
                        "selected_mean_cs_demeaned_residual": float(cs_residual.mean()),
                        "short_contribution_cs_demeaned_residual": float(
                            -cs_residual.mean()
                        ),
                        "selected_mean_risk_neutral_residual": float(risk_residual.mean()),
                        "short_contribution_risk_neutral_residual": float(
                            -risk_residual.mean()
                        ),
                        "selected_mean_forward_return_5d": float(raw.mean()),
                        "short_contribution_return": float(-raw.mean()),
                        "oracle_short_contribution_target": float(-oracle[target_name].mean()),
                        "oracle_overlap_rate": float(
                            len(set(selected["symbol"]).intersection(oracle_symbols))
                            / candidate_count
                        ),
                        "loser_quantile_capture_rate": float(
                            len(set(selected["symbol"]).intersection(loser_symbols))
                            / candidate_count
                        ),
                        "rank_ic_selector_vs_target": float(rank_ic),
                        "test_window_used": False,
                    }
                )
    return pd.DataFrame(rows, columns=_daily_columns())


def _add_selector_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["short_momentum_20d"] = result["momentum_20d"]
    result["short_beta_residual_momentum_20d_z"] = result[
        "beta_residual_momentum_20d_z"
    ]
    result["short_vol_adjusted_momentum_20d"] = result["vol_adjusted_momentum_20d"]
    result["short_exhausted_winner_20_5"] = result["exhausted_winner_20_5"]
    result["short_residual_overextension_20_5"] = result[
        "residual_overextension_20_5"
    ]
    result["short_core_equal"] = _mean_zscore(result, CORE_SELECTOR_FEATURES)
    result["short_overextension_equal"] = _mean_zscore(
        result, OVEREXTENSION_SELECTOR_FEATURES
    )
    result["short_core_plus_overextension"] = _mean_zscore(
        result, (*CORE_SELECTOR_FEATURES, *OVEREXTENSION_SELECTOR_FEATURES)
    )
    positive_beta = result["beta_z"].clip(lower=0.0)
    result["short_core_plus_overextension_beta_penalty"] = (
        result["short_core_plus_overextension"] - 0.25 * positive_beta
    )
    return result


def _mean_zscore(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    pieces = [_zscore(frame[column]) for column in columns]
    return pd.concat(pieces, axis=1).mean(axis=1)


def _selector_catalog() -> list[tuple[str, str]]:
    return [
        ("short_momentum_20d", "High 20-session momentum"),
        ("short_beta_residual_momentum_20d_z", "High z-scored residual momentum"),
        ("short_vol_adjusted_momentum_20d", "High vol-adjusted momentum"),
        ("short_exhausted_winner_20_5", "Exhausted 20d winner plus 5d extension"),
        ("short_residual_overextension_20_5", "Residual overextension plus 5d extension"),
        ("short_core_equal", "Equal-weight Phase4G first-tier core"),
        ("short_overextension_equal", "Equal-weight overextension pair"),
        ("short_core_plus_overextension", "Equal-weight core plus overextension"),
        (
            "short_core_plus_overextension_beta_penalty",
            "Core plus overextension with high-beta penalty",
        ),
    ]


def _selector_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_summary_columns())
    rows = []
    for (variant, target, selector), group in daily.groupby(
        ["variant", "residual_target", "selector"], sort=True
    ):
        target_contribution = group["short_contribution_target"]
        oracle_gap = group["oracle_short_contribution_target"] - target_contribution
        rows.append(
            {
                "variant": variant,
                "residual_target": target,
                "selector": selector,
                "selector_label": group["selector_label"].iloc[0],
                "sessions": int(len(group)),
                "mean_short_contribution_target": float(target_contribution.mean()),
                "target_hit_rate": float((target_contribution > 0).mean()),
                "mean_selected_negative_target_share": float(
                    group["selected_negative_target_share"].mean()
                ),
                "mean_short_contribution_beta_residual": float(
                    group["short_contribution_beta_residual"].mean()
                ),
                "mean_short_contribution_cs_demeaned_residual": float(
                    group["short_contribution_cs_demeaned_residual"].mean()
                ),
                "mean_short_contribution_risk_neutral_residual": float(
                    group["short_contribution_risk_neutral_residual"].mean()
                ),
                "mean_short_contribution_return": float(
                    group["short_contribution_return"].mean()
                ),
                "mean_oracle_short_contribution_target": float(
                    group["oracle_short_contribution_target"].mean()
                ),
                "mean_oracle_gap_target": float(oracle_gap.mean()),
                "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                "mean_loser_quantile_capture_rate": float(
                    group["loser_quantile_capture_rate"].mean()
                ),
                "mean_rank_ic_selector_vs_target": float(
                    group["rank_ic_selector_vs_target"].mean()
                ),
                "rank_ic_negative_share": float(
                    (group["rank_ic_selector_vs_target"] < 0).mean()
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_summary_columns()).sort_values(
        ["residual_target", "mean_short_contribution_target", "selector"],
        ascending=[True, False, True],
    )


def _selector_memo(
    summary: pd.DataFrame,
    *,
    candidate_count: int,
    loser_quantile: float,
    min_regression_rows: int,
) -> str:
    lines = [
        "# Pure Alpha Phase 4J Residual-Target-Aware Short Selector Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only selector diagnostic. It evaluates transparent "
        "Phase4G short features across beta, cross-section-demeaned, and risk-neutral residual targets.",
        "",
        "## Method",
        "",
        f"- selected names per session: `{candidate_count}`;",
        f"- residual-loser cutoff: bottom `{loser_quantile}`;",
        f"- minimum residual-neutralization rows: `{min_regression_rows}`;",
        "- no beta-matched portfolio, transaction cost, borrow cost, or test-window performance is computed.",
        "",
        "## Top Selectors By Target",
        "",
        "| Target | Selector | Contribution | Hit Rate | Capture | RankIC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if not summary.empty:
        top = summary.groupby("residual_target", sort=True).head(6)
        for _, row in top.iterrows():
            lines.append(
                f"| {row['residual_target']} | {row['selector']} | "
                f"{row['mean_short_contribution_target']} | {row['target_hit_rate']} | "
                f"{row['mean_loser_quantile_capture_rate']} | "
                f"{row['mean_rank_ic_selector_vs_target']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- A selector that works only on beta residual but not on cross-section-demeaned residual is not yet robust.",
            "- This phase does not prove portfolio tradability because it does not beta-match the books.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "residual_target",
        "selector",
        "selector_label",
        "eligible_names",
        "selected_names",
        "selected_mean_target_return",
        "short_contribution_target",
        "selected_negative_target_share",
        "selected_mean_beta_residual",
        "short_contribution_beta_residual",
        "selected_mean_cs_demeaned_residual",
        "short_contribution_cs_demeaned_residual",
        "selected_mean_risk_neutral_residual",
        "short_contribution_risk_neutral_residual",
        "selected_mean_forward_return_5d",
        "short_contribution_return",
        "oracle_short_contribution_target",
        "oracle_overlap_rate",
        "loser_quantile_capture_rate",
        "rank_ic_selector_vs_target",
        "test_window_used",
    ]


def _summary_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "selector",
        "selector_label",
        "sessions",
        "mean_short_contribution_target",
        "target_hit_rate",
        "mean_selected_negative_target_share",
        "mean_short_contribution_beta_residual",
        "mean_short_contribution_cs_demeaned_residual",
        "mean_short_contribution_risk_neutral_residual",
        "mean_short_contribution_return",
        "mean_oracle_short_contribution_target",
        "mean_oracle_gap_target",
        "mean_oracle_overlap_rate",
        "mean_loser_quantile_capture_rate",
        "mean_rank_ic_selector_vs_target",
        "rank_ic_negative_share",
        "test_window_used",
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4J residual-target-aware short selectors."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--residual-target", action="append", dest="residual_targets")
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    args = parser.parse_args(argv)

    result = build_phase4j_short_selector_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_VARIANTS),
        residual_targets=tuple(args.residual_targets or DEFAULT_TARGETS),
        candidate_count=args.candidate_count,
        loser_quantile=args.loser_quantile,
        min_regression_rows=args.min_regression_rows,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
