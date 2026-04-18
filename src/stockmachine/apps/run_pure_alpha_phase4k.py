from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4 import _shared_beta_target, _weights_to_target_beta
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN, _add_model_features
from stockmachine.apps.run_pure_alpha_phase4f import DEFAULT_PHASE3_SIGNAL_PANEL
from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
    _add_residual_targets,
)
from stockmachine.apps.run_pure_alpha_phase4j import _add_selector_scores


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4k_asymmetric_universe_constructor_20260418"
DEFAULT_LONG_VARIANTS = ("top500_clean_core_beta_full", "top1000_clean_core_beta_full")
DEFAULT_SHORT_VARIANTS = (
    "top1000_clean_core_beta_full",
    "adv50m_clean_core_beta_full",
    "adv30m_clean_core_beta_full",
    "adv20m_clean_core_beta_full",
)
DEFAULT_LONG_SCORE = "reversal_5d"
DEFAULT_SHORT_SELECTOR = "short_core_plus_overextension"
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
    "transparent_composite",
    "forward_return_5d",
    "benchmark_forward_return_5d",
    "forward_beta_residual_return_5d",
)


def build_phase4k_asymmetric_universe_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variants: Sequence[str] = DEFAULT_LONG_VARIANTS,
    short_variants: Sequence[str] = DEFAULT_SHORT_VARIANTS,
    long_score: str = DEFAULT_LONG_SCORE,
    short_selector: str = DEFAULT_SHORT_SELECTOR,
    min_names_per_side: int = 10,
    target_names_per_side: int = 20,
    max_names_per_side: int = 30,
    max_single_name_side_weight: float = 0.05,
    beta_match_tolerance: float = 0.05,
    min_regression_rows: int = 80,
) -> dict[str, Any]:
    """Construct validation-only beta-matched books with asymmetric universes."""

    _validate_settings(
        long_variants=long_variants,
        short_variants=short_variants,
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
    positions, daily, skipped = _construct_books(
        panel,
        long_variants=long_variants,
        short_variants=short_variants,
        long_score=long_score,
        short_selector=short_selector,
        required_nonzero_names=max(min_names_per_side, int(np.ceil(1.0 / max_single_name_side_weight))),
        target_names_per_side=target_names_per_side,
        max_names_per_side=max_names_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        beta_match_tolerance=beta_match_tolerance,
    )
    summary = _summary(daily, skipped)
    positions_path = output_dir / "phase4k_asymmetric_positions_validation.csv.gz"
    daily_path = output_dir / "phase4k_asymmetric_daily_validation.csv"
    skipped_path = output_dir / "phase4k_asymmetric_skipped_validation.csv"
    summary_path = output_dir / "phase4k_asymmetric_summary_validation.csv"
    memo_path = output_dir / "phase4k_asymmetric_universe_memo.md"
    rollup_path = output_dir / "phase4k_asymmetric_rollup.json"
    positions.to_csv(positions_path, index=False, compression="gzip")
    daily.to_csv(daily_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    summary.to_csv(summary_path, index=False)
    memo_path.write_text(_memo(summary, long_variants, short_variants), encoding="utf-8")
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "long_variants": list(long_variants),
        "short_variants": list(short_variants),
        "long_score": long_score,
        "short_selector": short_selector,
        "panel_rows_loaded": int(len(panel)),
        "position_rows": int(len(positions)),
        "daily_rows": int(len(daily)),
        "skipped_rows": int(len(skipped)),
        "summary_rows": int(len(summary)),
        "positions_artifact": positions_path.as_posix(),
        "daily_artifact": daily_path.as_posix(),
        "skipped_artifact": skipped_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "asymmetric_long_short_universe_beta_matched_constructor",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
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
    if not min_names_per_side <= target_names_per_side <= max_names_per_side:
        raise ValueError("Name counts must satisfy min <= target <= max.")
    if max_names_per_side * max_single_name_side_weight < 1.0:
        raise ValueError("Max names and single-name cap cannot sum to one full side.")
    if beta_match_tolerance < 0:
        raise ValueError("beta_match_tolerance must be non-negative.")
    if min_regression_rows <= 0:
        raise ValueError("min_regression_rows must be positive.")


def _load_panel(
    signal_panel_path: str | Path,
    *,
    variant_names: Sequence[str],
    min_regression_rows: int,
) -> pd.DataFrame:
    panel = pd.read_csv(signal_panel_path, usecols=list(BASE_COLUMNS), low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[panel["variant"].isin(set(variant_names))].copy()
    for column in BASE_COLUMNS:
        if column not in ("session_date", "variant", "symbol"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(
        subset=["beta", "forward_return_5d", "benchmark_forward_return_5d", TARGET_COLUMN]
    )
    panel = _add_model_features(panel)
    panel = _add_residual_targets(panel, min_regression_rows=min_regression_rows)
    scored = [_add_selector_scores(group) for _, group in panel.groupby(["variant", "session_date"])]
    if not scored:
        return panel
    return pd.concat(scored, ignore_index=True).sort_values(
        ["variant", "session_date", "symbol"]
    )


def _construct_books(
    panel: pd.DataFrame,
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
    long_score: str,
    short_selector: str,
    required_nonzero_names: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    minimum_names = max(required_nonzero_names, target_names_per_side)
    for long_variant in long_variants:
        for short_variant in short_variants:
            pair = f"{long_variant}__short_{short_variant}"
            for session_date in sessions:
                long_group = groups.get((long_variant, session_date))
                short_group = groups.get((short_variant, session_date))
                if long_group is None or short_group is None:
                    skipped.append(
                        _skip(session_date, pair, long_variant, short_variant, 0, 0, "missing_session")
                    )
                    continue
                pos, diag, skip = _one_session(
                    long_group,
                    short_group,
                    session_date=session_date,
                    pair=pair,
                    long_variant=long_variant,
                    short_variant=short_variant,
                    long_score=long_score,
                    short_selector=short_selector,
                    minimum_names=minimum_names,
                    required_nonzero_names=required_nonzero_names,
                    max_names_per_side=max_names_per_side,
                    max_single_name_side_weight=max_single_name_side_weight,
                    beta_match_tolerance=beta_match_tolerance,
                )
                positions.extend(pos)
                if diag is not None:
                    daily.append(diag)
                if skip is not None:
                    skipped.append(skip)
    return (
        pd.DataFrame(positions, columns=_position_columns()),
        pd.DataFrame(daily, columns=_daily_columns()),
        pd.DataFrame(skipped, columns=_skipped_columns()),
    )


def _one_session(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    pair: str,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    minimum_names: int,
    required_nonzero_names: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    needed = [
        "symbol",
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_BETA_RESIDUAL,
        TARGET_CS_DEMEANED,
        TARGET_RISK_NEUTRAL,
    ]
    longs = long_group.dropna(subset=[long_score, *needed]).drop_duplicates("symbol").copy()
    shorts = short_group.dropna(subset=[short_selector, *needed]).drop_duplicates("symbol").copy()
    if len(longs) < minimum_names or len(shorts) < minimum_names:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            "insufficient_candidates",
        )
    candidate_count = min(max_names_per_side, len(longs), len(shorts))
    long_candidates = longs.sort_values([long_score, "symbol"], ascending=[False, True]).head(
        candidate_count
    )
    short_pool = shorts.loc[~shorts["symbol"].isin(set(long_candidates["symbol"]))]
    if len(short_pool) < minimum_names:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            "insufficient_non_overlapping_short_candidates",
        )
    short_candidates = short_pool.sort_values(
        [short_selector, "symbol"],
        ascending=[False, True],
    ).head(min(candidate_count, len(short_pool)))
    beta_target, reason = _shared_beta_target(
        long_candidates["beta"].to_numpy(dtype=float),
        short_candidates["beta"].to_numpy(dtype=float),
        max_single_name_side_weight=max_single_name_side_weight,
    )
    if beta_target is None:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            reason or "beta_range_no_overlap",
        )
    try:
        long_weights = _weights_to_target_beta(
            long_candidates["beta"].to_numpy(dtype=float),
            target_beta=beta_target,
            max_weight=max_single_name_side_weight,
        )
        short_weights = _weights_to_target_beta(
            short_candidates["beta"].to_numpy(dtype=float),
            target_beta=beta_target,
            max_weight=max_single_name_side_weight,
        )
    except ValueError as exc:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            f"weight_solver_failed:{exc}",
        )
    if (long_weights > 1e-10).sum() < required_nonzero_names:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            "insufficient_long_nonzero_names",
        )
    if (short_weights > 1e-10).sum() < required_nonzero_names:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            "insufficient_short_nonzero_names",
        )
    long_beta = float(np.dot(long_weights, long_candidates["beta"].to_numpy(dtype=float)))
    short_beta = float(np.dot(short_weights, short_candidates["beta"].to_numpy(dtype=float)))
    net_beta = float(long_beta - short_beta)
    if abs(net_beta) > beta_match_tolerance:
        return [], None, _skip(
            session_date,
            pair,
            long_variant,
            short_variant,
            len(longs),
            len(shorts),
            "beta_match_tolerance_breach",
        )
    positions = [
        *_position_rows(
            long_candidates,
            long_weights,
            session_date=session_date,
            pair=pair,
            long_variant=long_variant,
            short_variant=short_variant,
            side="long",
            score_column=long_score,
        ),
        *_position_rows(
            short_candidates,
            short_weights,
            session_date=session_date,
            pair=pair,
            long_variant=long_variant,
            short_variant=short_variant,
            side="short",
            score_column=short_selector,
        ),
    ]
    metrics = _metrics(long_candidates, short_candidates, long_weights, short_weights)
    diagnostic = {
        "session_date": session_date,
        "pair": pair,
        "long_variant": long_variant,
        "short_variant": short_variant,
        "long_score": long_score,
        "short_selector": short_selector,
        "eligible_long_names": int(len(longs)),
        "eligible_short_names": int(len(shorts)),
        "long_count": int((long_weights > 1e-10).sum()),
        "short_count": int((short_weights > 1e-10).sum()),
        "long_beta": long_beta,
        "short_beta": short_beta,
        "net_beta": net_beta,
        "beta_match_error": net_beta,
        "beta_target": float(beta_target),
        "max_abs_position_weight": float(max(abs(row["signed_weight"]) for row in positions)),
        "benchmark_return": float(long_candidates["benchmark_forward_return_5d"].iloc[0]),
        **metrics,
        "test_window_used": False,
    }
    return positions, diagnostic, None


def _metrics(
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, column in (
        ("return", "forward_return_5d"),
        ("beta_residual", TARGET_BETA_RESIDUAL),
        ("cs_demeaned_residual", TARGET_CS_DEMEANED),
        ("risk_neutral_residual", TARGET_RISK_NEUTRAL),
    ):
        long_value = float(np.dot(long_weights, long_candidates[column].to_numpy(dtype=float)))
        short_value = float(-np.dot(short_weights, short_candidates[column].to_numpy(dtype=float)))
        out[f"long_{label}"] = long_value
        out[f"short_{label}_contribution"] = short_value
        out[f"spread_{label}"] = float(long_value + short_value)
    out["selected_short_negative_beta_residual_share"] = float(
        (short_candidates[TARGET_BETA_RESIDUAL] < 0).mean()
    )
    out["selected_short_negative_cs_demeaned_share"] = float(
        (short_candidates[TARGET_CS_DEMEANED] < 0).mean()
    )
    out["selected_short_negative_risk_neutral_share"] = float(
        (short_candidates[TARGET_RISK_NEUTRAL] < 0).mean()
    )
    return out


def _position_rows(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    *,
    session_date: str,
    pair: str,
    long_variant: str,
    short_variant: str,
    side: str,
    score_column: str,
) -> list[dict[str, Any]]:
    sign = 1.0 if side == "long" else -1.0
    rows = []
    for (_, row), weight in zip(candidates.iterrows(), weights):
        if weight <= 1e-12:
            continue
        rows.append(
            {
                "session_date": session_date,
                "pair": pair,
                "long_variant": long_variant,
                "short_variant": short_variant,
                "side": side,
                "symbol": row["symbol"],
                "score_column": score_column,
                "score": float(row[score_column]),
                "beta": float(row["beta"]),
                "side_weight": float(weight),
                "signed_weight": float(sign * weight),
                "forward_return_5d": float(row["forward_return_5d"]),
                "beta_residual": float(row[TARGET_BETA_RESIDUAL]),
                "cs_demeaned_beta_residual": float(row[TARGET_CS_DEMEANED]),
                "risk_neutral_beta_residual": float(row[TARGET_RISK_NEUTRAL]),
                "test_window_used": False,
            }
        )
    return rows


def _summary(daily: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    if daily.empty and skipped.empty:
        return pd.DataFrame(columns=_summary_columns())
    requested = _requested_counts(daily, skipped)
    if daily.empty:
        summary = requested.copy()
        summary["constructed_sessions"] = 0
        summary["construction_rate"] = 0.0
        for column in _summary_columns():
            if column not in summary.columns:
                summary[column] = np.nan
        summary["test_window_used"] = False
        return summary[_summary_columns()]
    rows = []
    for pair, group in daily.groupby("pair", sort=True):
        rows.append(
            {
                "pair": pair,
                "long_score": group["long_score"].iloc[0],
                "short_selector": group["short_selector"].iloc[0],
                "constructed_sessions": int(len(group)),
                "mean_spread_return": float(group["spread_return"].mean()),
                "spread_hit_rate": float((group["spread_return"] > 0).mean()),
                "mean_spread_beta_residual": float(group["spread_beta_residual"].mean()),
                "mean_spread_cs_demeaned_residual": float(
                    group["spread_cs_demeaned_residual"].mean()
                ),
                "mean_spread_risk_neutral_residual": float(
                    group["spread_risk_neutral_residual"].mean()
                ),
                "mean_long_return": float(group["long_return"].mean()),
                "mean_short_return_contribution": float(
                    group["short_return_contribution"].mean()
                ),
                "mean_short_cs_demeaned_residual_contribution": float(
                    group["short_cs_demeaned_residual_contribution"].mean()
                ),
                "mean_short_negative_cs_demeaned_share": float(
                    group["selected_short_negative_cs_demeaned_share"].mean()
                ),
                "mean_abs_net_beta": float(group["net_beta"].abs().mean()),
                "max_abs_net_beta": float(group["net_beta"].abs().max()),
                "median_long_count": float(group["long_count"].median()),
                "median_short_count": float(group["short_count"].median()),
                "test_window_used": False,
            }
        )
    detail = pd.DataFrame(rows)
    summary = requested.merge(detail, on="pair", how="left")
    summary["constructed_sessions"] = summary["constructed_sessions"].fillna(0).astype(int)
    summary["construction_rate"] = (
        summary["constructed_sessions"] / summary["requested_sessions"].replace(0, np.nan)
    ).fillna(0.0)
    summary["test_window_used"] = summary["test_window_used"].fillna(False)
    return summary[_summary_columns()].sort_values(
        ["mean_spread_cs_demeaned_residual", "mean_spread_return", "pair"],
        ascending=[False, False, True],
    )


def _requested_counts(daily: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    frames = []
    columns = ["pair", "long_variant", "short_variant", "session_date"]
    if not daily.empty:
        frames.append(daily[columns])
    if not skipped.empty:
        frames.append(skipped[columns])
    requested = pd.concat(frames, ignore_index=True)
    return (
        requested.groupby(["pair", "long_variant", "short_variant"], sort=True)
        .agg(requested_sessions=("session_date", "nunique"))
        .reset_index()
    )


def _skip(
    session_date: str,
    pair: str,
    long_variant: str,
    short_variant: str,
    eligible_long_names: int,
    eligible_short_names: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "pair": pair,
        "long_variant": long_variant,
        "short_variant": short_variant,
        "eligible_long_names": int(eligible_long_names),
        "eligible_short_names": int(eligible_short_names),
        "skip_reason": reason,
        "test_window_used": False,
    }


def _memo(summary: pd.DataFrame, long_variants: Sequence[str], short_variants: Sequence[str]) -> str:
    lines = [
        "# Pure Alpha Phase 4K Asymmetric Universe Constructor Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Validation-only diagnostic for pairing a high-quality long universe with broader, still-liquid short proxy universes under beta matching.",
        "",
        "## Inputs",
        "",
        f"- long variants: `{', '.join(long_variants)}`",
        f"- short variants: `{', '.join(short_variants)}`",
        f"- long score: `{DEFAULT_LONG_SCORE}`",
        f"- short selector: `{DEFAULT_SHORT_SELECTOR}`",
        "",
        "## Summary Snapshot",
        "",
        "| Pair | Constructed | Rate | Spread Ret | CS Resid Spread | Short CS Contrib | Abs Net Beta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.head(16).iterrows():
        lines.append(
            f"| {row['pair']} | {row['constructed_sessions']} | {row['construction_rate']} | "
            f"{row['mean_spread_return']} | {row['mean_spread_cs_demeaned_residual']} | "
            f"{row['mean_short_cs_demeaned_residual_contribution']} | {row['mean_abs_net_beta']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Selected long and short names are forced not to overlap.",
            "- Beta matching reuses the Phase 4 ex-ante beta solver.",
            "- This uses available ADV proxy variants, not a true top2000 membership file.",
            "- No costs, borrow, turnover, or test-window performance are computed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _position_columns() -> list[str]:
    return [
        "session_date",
        "pair",
        "long_variant",
        "short_variant",
        "side",
        "symbol",
        "score_column",
        "score",
        "beta",
        "side_weight",
        "signed_weight",
        "forward_return_5d",
        "beta_residual",
        "cs_demeaned_beta_residual",
        "risk_neutral_beta_residual",
        "test_window_used",
    ]


def _daily_columns() -> list[str]:
    return [
        "session_date",
        "pair",
        "long_variant",
        "short_variant",
        "long_score",
        "short_selector",
        "eligible_long_names",
        "eligible_short_names",
        "long_count",
        "short_count",
        "long_beta",
        "short_beta",
        "net_beta",
        "beta_match_error",
        "beta_target",
        "max_abs_position_weight",
        "benchmark_return",
        "long_return",
        "short_return_contribution",
        "spread_return",
        "long_beta_residual",
        "short_beta_residual_contribution",
        "spread_beta_residual",
        "long_cs_demeaned_residual",
        "short_cs_demeaned_residual_contribution",
        "spread_cs_demeaned_residual",
        "long_risk_neutral_residual",
        "short_risk_neutral_residual_contribution",
        "spread_risk_neutral_residual",
        "selected_short_negative_beta_residual_share",
        "selected_short_negative_cs_demeaned_share",
        "selected_short_negative_risk_neutral_share",
        "test_window_used",
    ]


def _skipped_columns() -> list[str]:
    return [
        "session_date",
        "pair",
        "long_variant",
        "short_variant",
        "eligible_long_names",
        "eligible_short_names",
        "skip_reason",
        "test_window_used",
    ]


def _summary_columns() -> list[str]:
    return [
        "pair",
        "long_variant",
        "short_variant",
        "requested_sessions",
        "constructed_sessions",
        "construction_rate",
        "long_score",
        "short_selector",
        "mean_spread_return",
        "spread_hit_rate",
        "mean_spread_beta_residual",
        "mean_spread_cs_demeaned_residual",
        "mean_spread_risk_neutral_residual",
        "mean_long_return",
        "mean_short_return_contribution",
        "mean_short_cs_demeaned_residual_contribution",
        "mean_short_negative_cs_demeaned_share",
        "mean_abs_net_beta",
        "max_abs_net_beta",
        "median_long_count",
        "median_short_count",
        "test_window_used",
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4K asymmetric universe constructor."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--long-variant", action="append", dest="long_variants")
    parser.add_argument("--short-variant", action="append", dest="short_variants")
    parser.add_argument("--max-names-per-side", type=int, default=30)
    args = parser.parse_args(argv)
    result = build_phase4k_asymmetric_universe_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        long_variants=tuple(args.long_variants or DEFAULT_LONG_VARIANTS),
        short_variants=tuple(args.short_variants or DEFAULT_SHORT_VARIANTS),
        max_names_per_side=args.max_names_per_side,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
