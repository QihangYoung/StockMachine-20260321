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
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4_beta_matched_portfolios_20260418"
DEFAULT_SIGNALS = ("reversal_5d",)
BASE_COLUMNS = (
    "session_date",
    "variant",
    "symbol",
    "beta",
    "forward_return_5d",
    "benchmark_forward_return_5d",
    "forward_beta_residual_return_5d",
)


def build_phase4_portfolio_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    signal_names: Sequence[str] = DEFAULT_SIGNALS,
    variant_names: Sequence[str] | None = None,
    long_gross: float = 1.0,
    short_gross: float = 1.0,
    min_names_per_side: int = 10,
    target_names_per_side: int = 20,
    max_names_per_side: int = 30,
    max_single_name_side_weight: float = 0.05,
    beta_match_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Build validation-only beta-matched long-short portfolio diagnostics."""

    _validate_portfolio_settings(
        signal_names=signal_names,
        long_gross=long_gross,
        short_gross=short_gross,
        min_names_per_side=min_names_per_side,
        target_names_per_side=target_names_per_side,
        max_names_per_side=max_names_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        beta_match_tolerance=beta_match_tolerance,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_signal_panel(
        signal_panel_path=signal_panel_path,
        signal_names=signal_names,
        variant_names=variant_names,
    )
    required_nonzero_names = max(
        min_names_per_side,
        int(np.ceil(1.0 / max_single_name_side_weight)),
    )

    positions, diagnostics, skipped = _construct_portfolios(
        panel,
        signal_names=tuple(signal_names),
        long_gross=long_gross,
        short_gross=short_gross,
        required_nonzero_names=required_nonzero_names,
        target_names_per_side=target_names_per_side,
        max_names_per_side=max_names_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        beta_match_tolerance=beta_match_tolerance,
    )
    summary = _portfolio_summary(diagnostics, skipped)

    positions_path = output_dir / "phase4_beta_matched_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase4_portfolio_daily_diagnostics_validation.csv"
    skipped_path = output_dir / "phase4_skipped_sessions_validation.csv"
    summary_path = output_dir / "phase4_portfolio_summary_validation.csv"
    memo_path = output_dir / "phase4_beta_matched_portfolio_memo.md"
    rollup_path = output_dir / "phase4_beta_matched_portfolio_rollup.json"

    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    summary.to_csv(summary_path, index=False)
    memo_path.write_text(
        _portfolio_memo(
            summary,
            signal_names=signal_names,
            long_gross=long_gross,
            short_gross=short_gross,
            min_names_per_side=min_names_per_side,
            target_names_per_side=target_names_per_side,
            max_names_per_side=max_names_per_side,
            max_single_name_side_weight=max_single_name_side_weight,
            beta_match_tolerance=beta_match_tolerance,
            required_nonzero_names=required_nonzero_names,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "signal_names": list(signal_names),
        "variant_names": sorted(panel["variant"].astype(str).unique()) if not panel.empty else [],
        "long_gross": float(long_gross),
        "short_gross": float(short_gross),
        "gross_exposure": float(long_gross + short_gross),
        "net_exposure": float(long_gross - short_gross),
        "min_names_per_side": int(min_names_per_side),
        "target_names_per_side": int(target_names_per_side),
        "max_names_per_side": int(max_names_per_side),
        "required_nonzero_names_per_side": int(required_nonzero_names),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "beta_match_tolerance": float(beta_match_tolerance),
        "validation_start": _first_or_none(diagnostics["session_date"]) if not diagnostics.empty else None,
        "validation_end": _last_or_none(diagnostics["session_date"]) if not diagnostics.empty else None,
        "signal_panel_rows_loaded": int(len(panel)),
        "position_rows": int(len(positions)),
        "constructed_session_rows": int(len(diagnostics)),
        "skipped_session_rows": int(len(skipped)),
        "summary_rows": int(len(summary)),
        "positions_artifact": positions_path.as_posix(),
        "daily_diagnostics_artifact": diagnostics_path.as_posix(),
        "skipped_sessions_artifact": skipped_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is a portfolio-construction diagnostic, not a full backtest artifact contract.",
            "Returns use Phase 3 forward validation labels and are overlapping h5 diagnostics.",
            "No trading costs, borrow costs, or turnover costs are charged in Phase 4.",
            "No test-window performance, production claim, or candidate freeze is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_portfolio_settings(
    *,
    signal_names: Sequence[str],
    long_gross: float,
    short_gross: float,
    min_names_per_side: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> None:
    if not signal_names:
        raise ValueError("At least one signal name is required.")
    if long_gross <= 0 or short_gross <= 0:
        raise ValueError("Long and short gross exposure must be positive.")
    if min_names_per_side <= 0 or target_names_per_side <= 0 or max_names_per_side <= 0:
        raise ValueError("Name-count settings must be positive.")
    if not min_names_per_side <= target_names_per_side <= max_names_per_side:
        raise ValueError("Name counts must satisfy min <= target <= max.")
    if not 0 < max_single_name_side_weight <= 1:
        raise ValueError("Max single-name side weight must be in (0, 1].")
    if max_names_per_side * max_single_name_side_weight < 1.0:
        raise ValueError("Max names and single-name cap cannot sum to one full side.")
    if beta_match_tolerance < 0:
        raise ValueError("Beta match tolerance must be non-negative.")


def _load_signal_panel(
    *,
    signal_panel_path: str | Path,
    signal_names: Sequence[str],
    variant_names: Sequence[str] | None,
) -> pd.DataFrame:
    usecols = list(dict.fromkeys([*BASE_COLUMNS, *signal_names]))
    panel = pd.read_csv(signal_panel_path, usecols=usecols, low_memory=False)
    missing = sorted(set(usecols).difference(panel.columns))
    if missing:
        raise ValueError(f"Phase 3 signal panel is missing required columns: {missing}")
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    if variant_names is not None:
        panel = panel[panel["variant"].isin(set(variant_names))].copy()
    numeric_columns = [
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        "forward_beta_residual_return_5d",
        *signal_names,
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(
        subset=[
            "beta",
            "forward_return_5d",
            "benchmark_forward_return_5d",
            "forward_beta_residual_return_5d",
        ]
    )
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _construct_portfolios(
    panel: pd.DataFrame,
    *,
    signal_names: Sequence[str],
    long_gross: float,
    short_gross: float,
    required_nonzero_names: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for signal in signal_names:
        signal_panel = panel.dropna(subset=[signal]).copy()
        if signal_panel.empty:
            continue
        signal_panel["_score"] = signal_panel[signal]
        grouped = signal_panel.groupby(["variant", "session_date"], sort=True)
        for (variant, session_date), group in grouped:
            book_positions, diagnostic, skip = _construct_one_session(
                group,
                variant=variant,
                session_date=session_date,
                signal=signal,
                long_gross=long_gross,
                short_gross=short_gross,
                required_nonzero_names=required_nonzero_names,
                target_names_per_side=target_names_per_side,
                max_names_per_side=max_names_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                beta_match_tolerance=beta_match_tolerance,
            )
            positions.extend(book_positions)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            if skip is not None:
                skipped.append(skip)
    return (
        pd.DataFrame(positions, columns=_position_columns()),
        pd.DataFrame(diagnostics, columns=_diagnostic_columns()),
        pd.DataFrame(skipped, columns=_skipped_columns()),
    )


def _construct_one_session(
    group: pd.DataFrame,
    *,
    variant: str,
    session_date: str,
    signal: str,
    long_gross: float,
    short_gross: float,
    required_nonzero_names: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    eligible = group.dropna(
        subset=[
            "_score",
            "beta",
            "forward_return_5d",
            "benchmark_forward_return_5d",
            "forward_beta_residual_return_5d",
        ]
    ).copy()
    eligible = eligible.drop_duplicates("symbol", keep="last")
    minimum_candidate_names = max(target_names_per_side, required_nonzero_names)
    candidates_per_side = min(max_names_per_side, len(eligible) // 2)
    if candidates_per_side < minimum_candidate_names:
        return [], None, _skip_row(
            session_date=session_date,
            variant=variant,
            signal=signal,
            eligible_names=len(eligible),
            skip_reason="insufficient_non_overlapping_candidates",
        )

    long_candidates = eligible.sort_values(["_score", "symbol"], ascending=[False, True]).head(
        candidates_per_side
    )
    short_candidates = eligible.sort_values(["_score", "symbol"], ascending=[True, True]).head(
        candidates_per_side
    )
    overlap = set(long_candidates["symbol"]).intersection(set(short_candidates["symbol"]))
    if overlap:
        return [], None, _skip_row(
            session_date=session_date,
            variant=variant,
            signal=signal,
            eligible_names=len(eligible),
            skip_reason="long_short_symbol_overlap",
        )

    beta_target, skip_reason = _shared_beta_target(
        long_candidates["beta"].to_numpy(dtype=float),
        short_candidates["beta"].to_numpy(dtype=float),
        max_single_name_side_weight=max_single_name_side_weight,
    )
    if beta_target is None:
        return [], None, _skip_row(
            session_date=session_date,
            variant=variant,
            signal=signal,
            eligible_names=len(eligible),
            skip_reason=skip_reason or "beta_range_no_overlap",
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
        return [], None, _skip_row(
            session_date=session_date,
            variant=variant,
            signal=signal,
            eligible_names=len(eligible),
            skip_reason=f"weight_solver_failed:{exc}",
        )

    long_count = int((long_weights > 1e-10).sum())
    short_count = int((short_weights > 1e-10).sum())
    if long_count < required_nonzero_names or short_count < required_nonzero_names:
        return [], None, _skip_row(
            session_date=session_date,
            variant=variant,
            signal=signal,
            eligible_names=len(eligible),
            skip_reason="insufficient_nonzero_names_after_beta_match",
        )

    long_beta = float(np.dot(long_weights, long_candidates["beta"].to_numpy(dtype=float)))
    short_beta = float(np.dot(short_weights, short_candidates["beta"].to_numpy(dtype=float)))
    net_beta = float(long_gross * long_beta - short_gross * short_beta)
    beta_match_error = float(long_beta - short_beta)
    if abs(net_beta) > beta_match_tolerance:
        return [], None, _skip_row(
            session_date=session_date,
            variant=variant,
            signal=signal,
            eligible_names=len(eligible),
            skip_reason="beta_match_tolerance_breach",
        )

    book_positions = [
        *_position_rows(
            long_candidates,
            weights=long_weights,
            side="long",
            side_gross=long_gross,
            variant=variant,
            session_date=session_date,
            signal=signal,
        ),
        *_position_rows(
            short_candidates,
            weights=short_weights,
            side="short",
            side_gross=short_gross,
            variant=variant,
            session_date=session_date,
            signal=signal,
        ),
    ]
    long_return = float(
        long_gross * np.dot(long_weights, long_candidates["forward_return_5d"].to_numpy(dtype=float))
    )
    short_return = float(
        -short_gross
        * np.dot(short_weights, short_candidates["forward_return_5d"].to_numpy(dtype=float))
    )
    beta_residual_spread_return = float(
        long_gross
        * np.dot(
            long_weights,
            long_candidates["forward_beta_residual_return_5d"].to_numpy(dtype=float),
        )
        - short_gross
        * np.dot(
            short_weights,
            short_candidates["forward_beta_residual_return_5d"].to_numpy(dtype=float),
        )
    )
    max_abs_position_weight = max(abs(row["signed_weight"]) for row in book_positions)
    diagnostic = {
        "session_date": session_date,
        "variant": variant,
        "signal": signal,
        "construction_status": "constructed",
        "eligible_names": int(len(eligible)),
        "long_count": long_count,
        "short_count": short_count,
        "long_beta": long_beta,
        "short_beta": short_beta,
        "beta_target": float(beta_target),
        "beta_match_error": beta_match_error,
        "net_beta": net_beta,
        "long_gross": float(long_gross),
        "short_gross": float(short_gross),
        "gross_exposure": float(long_gross + short_gross),
        "net_exposure": float(long_gross - short_gross),
        "max_abs_position_weight": float(max_abs_position_weight),
        "long_return": long_return,
        "short_return": short_return,
        "spread_return": float(long_return + short_return),
        "beta_residual_spread_return": beta_residual_spread_return,
        "benchmark_return": float(eligible["benchmark_forward_return_5d"].iloc[0]),
        "test_window_used": False,
    }
    return book_positions, diagnostic, None


def _shared_beta_target(
    long_betas: np.ndarray,
    short_betas: np.ndarray,
    *,
    max_single_name_side_weight: float,
) -> tuple[float | None, str | None]:
    try:
        long_low, long_high = _feasible_beta_range(long_betas, max_single_name_side_weight)
        short_low, short_high = _feasible_beta_range(short_betas, max_single_name_side_weight)
    except ValueError as exc:
        return None, f"beta_range_failed:{exc}"
    overlap_low = max(long_low, short_low)
    overlap_high = min(long_high, short_high)
    if overlap_low > overlap_high:
        return None, "beta_range_no_overlap"
    preferred = float((np.mean(long_betas) + np.mean(short_betas)) / 2.0)
    return float(np.clip(preferred, overlap_low, overlap_high)), None


def _feasible_beta_range(betas: np.ndarray, max_weight: float) -> tuple[float, float]:
    return (
        _extreme_beta(betas, max_weight=max_weight, ascending=True),
        _extreme_beta(betas, max_weight=max_weight, ascending=False),
    )


def _extreme_beta(betas: np.ndarray, *, max_weight: float, ascending: bool) -> float:
    clean = np.asarray(betas, dtype=float)
    if clean.size == 0 or np.any(~np.isfinite(clean)):
        raise ValueError("invalid_beta_values")
    ordered = np.sort(clean)
    if not ascending:
        ordered = ordered[::-1]
    remaining = 1.0
    beta_sum = 0.0
    for beta in ordered:
        weight = min(max_weight, remaining)
        beta_sum += weight * beta
        remaining -= weight
        if remaining <= 1e-12:
            return float(beta_sum)
    raise ValueError("insufficient_weight_capacity")


def _weights_to_target_beta(
    betas: np.ndarray,
    *,
    target_beta: float,
    max_weight: float,
) -> np.ndarray:
    clean = np.asarray(betas, dtype=float)
    if clean.size == 0 or np.any(~np.isfinite(clean)):
        raise ValueError("invalid_beta_values")
    if clean.size * max_weight < 1.0 - 1e-12:
        raise ValueError("insufficient_weight_capacity")
    weights = np.full(clean.size, 1.0 / clean.size)
    if weights.max() > max_weight + 1e-12:
        raise ValueError("equal_weight_exceeds_single_name_cap")
    current = float(np.dot(weights, clean))
    if abs(current - target_beta) <= 1e-10:
        return weights

    if current < target_beta:
        donors = np.argsort(clean)
        receivers = np.argsort(clean)[::-1]
    else:
        donors = np.argsort(clean)[::-1]
        receivers = np.argsort(clean)

    for donor in donors:
        if abs(float(np.dot(weights, clean)) - target_beta) <= 1e-10:
            break
        for receiver in receivers:
            residual = target_beta - float(np.dot(weights, clean))
            if abs(residual) <= 1e-10:
                break
            beta_delta = clean[receiver] - clean[donor]
            if abs(beta_delta) <= 1e-12 or residual * beta_delta <= 0:
                continue
            capacity = max_weight - weights[receiver]
            available = weights[donor]
            if capacity <= 1e-12 or available <= 1e-12:
                continue
            amount = min(available, capacity, abs(residual / beta_delta))
            if amount <= 1e-12:
                continue
            weights[donor] -= amount
            weights[receiver] += amount

    matched_beta = float(np.dot(weights, clean))
    if abs(matched_beta - target_beta) > 1e-7:
        raise ValueError("target_beta_not_reached")
    if abs(float(weights.sum()) - 1.0) > 1e-10:
        raise ValueError("side_weights_do_not_sum_to_one")
    if weights.min() < -1e-10 or weights.max() > max_weight + 1e-10:
        raise ValueError("side_weight_bounds_breached")
    weights[np.abs(weights) < 1e-12] = 0.0
    return weights


def _position_rows(
    candidates: pd.DataFrame,
    *,
    weights: np.ndarray,
    side: str,
    side_gross: float,
    variant: str,
    session_date: str,
    signal: str,
) -> list[dict[str, Any]]:
    sign = 1.0 if side == "long" else -1.0
    rows = []
    for (_, row), side_weight in zip(candidates.iterrows(), weights):
        if side_weight <= 1e-12:
            continue
        rows.append(
            {
                "session_date": session_date,
                "variant": variant,
                "signal": signal,
                "side": side,
                "symbol": row["symbol"],
                "score": float(row["_score"]),
                "beta": float(row["beta"]),
                "side_weight": float(side_weight),
                "signed_weight": float(sign * side_gross * side_weight),
                "forward_return_5d": float(row["forward_return_5d"]),
                "forward_beta_residual_return_5d": float(
                    row["forward_beta_residual_return_5d"]
                ),
                "benchmark_forward_return_5d": float(row["benchmark_forward_return_5d"]),
                "test_window_used": False,
            }
        )
    return rows


def _portfolio_summary(diagnostics: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "signal"]
    if diagnostics.empty and skipped.empty:
        return pd.DataFrame(columns=_summary_columns())
    requested = _requested_counts(diagnostics, skipped)
    if diagnostics.empty:
        summary = requested.copy()
        summary["constructed_sessions"] = 0
        summary["construction_rate"] = 0.0
        for column in _summary_metric_columns():
            summary[column] = np.nan
        summary["test_window_used"] = False
        return summary[_summary_columns()]

    rows = []
    for (variant, signal), group in diagnostics.groupby(keys, sort=False):
        abs_net_beta = group["net_beta"].abs()
        rows.append(
            {
                "variant": variant,
                "signal": signal,
                "constructed_sessions": int(len(group)),
                "mean_spread_return": float(group["spread_return"].mean()),
                "spread_hit_rate": float((group["spread_return"] > 0).mean()),
                "mean_beta_residual_spread_return": float(
                    group["beta_residual_spread_return"].mean()
                ),
                "mean_long_return": float(group["long_return"].mean()),
                "mean_short_return": float(group["short_return"].mean()),
                "mean_abs_net_beta": float(abs_net_beta.mean()),
                "max_abs_net_beta": float(abs_net_beta.max()),
                "mean_abs_beta_match_error": float(group["beta_match_error"].abs().mean()),
                "median_long_count": float(group["long_count"].median()),
                "median_short_count": float(group["short_count"].median()),
                "median_max_abs_position_weight": float(
                    group["max_abs_position_weight"].median()
                ),
                "test_window_used": False,
            }
        )
    summary = pd.DataFrame(rows)
    summary = requested.merge(summary, on=keys, how="left")
    summary["constructed_sessions"] = summary["constructed_sessions"].fillna(0).astype(int)
    summary["construction_rate"] = (
        summary["constructed_sessions"] / summary["requested_sessions"].replace(0, np.nan)
    ).fillna(0.0)
    summary["test_window_used"] = summary["test_window_used"].fillna(False)
    return summary[_summary_columns()].sort_values(
        ["variant", "mean_spread_return", "signal"],
        ascending=[True, False, True],
    )


def _requested_counts(diagnostics: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not diagnostics.empty:
        frames.append(diagnostics[["variant", "signal", "session_date"]])
    if not skipped.empty:
        frames.append(skipped[["variant", "signal", "session_date"]])
    requested = pd.concat(frames, ignore_index=True)
    return (
        requested.groupby(["variant", "signal"], sort=False)
        .agg(requested_sessions=("session_date", "nunique"))
        .reset_index()
    )


def _portfolio_memo(
    summary: pd.DataFrame,
    *,
    signal_names: Sequence[str],
    long_gross: float,
    short_gross: float,
    min_names_per_side: int,
    target_names_per_side: int,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
    required_nonzero_names: int,
) -> str:
    lines = [
        "# Pure Alpha Phase 4 Beta-Matched Portfolio Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only portfolio-construction diagnostic. It converts Phase 3 "
        "scores into beta-matched long-short books, but it is not yet the Phase 5 "
        "backtest artifact contract and does not inspect the test lockbox.",
        "",
        "## Construction",
        "",
        f"- signals: `{', '.join(signal_names)}`",
        f"- long gross: `{long_gross}`",
        f"- short gross: `{short_gross}`",
        f"- min names per side: `{min_names_per_side}`",
        f"- target names per side: `{target_names_per_side}`",
        f"- max candidates per side: `{max_names_per_side}`",
        f"- required nonzero names per side after cap: `{required_nonzero_names}`",
        f"- max single-name side weight: `{max_single_name_side_weight}`",
        f"- net beta tolerance: `{beta_match_tolerance}`",
        "",
        "## Summary Snapshot",
        "",
        "| Variant | Signal | Constructed | Rate | Mean Spread | Mean Abs Net Beta | Hit Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.head(24).iterrows():
        lines.append(
            f"| {row['variant']} | {row['signal']} | {row['constructed_sessions']} | "
            f"{row['construction_rate']} | {row['mean_spread_return']} | "
            f"{row['mean_abs_net_beta']} | {row['spread_hit_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Long and short side weights each sum to one side before gross scaling.",
            "- Beta matching is ex-ante and uses the Phase 2 lagged beta panel carried in Phase 3.",
            "- Sessions with infeasible beta overlap or insufficient candidates are skipped explicitly.",
            "- Costs, borrow, turnover, and non-overlapping backtest accounting belong to Phase 5.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _skip_row(
    *,
    session_date: str,
    variant: str,
    signal: str,
    eligible_names: int,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "variant": variant,
        "signal": signal,
        "eligible_names": int(eligible_names),
        "skip_reason": skip_reason,
        "test_window_used": False,
    }


def _position_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "signal",
        "side",
        "symbol",
        "score",
        "beta",
        "side_weight",
        "signed_weight",
        "forward_return_5d",
        "forward_beta_residual_return_5d",
        "benchmark_forward_return_5d",
        "test_window_used",
    ]


def _diagnostic_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "signal",
        "construction_status",
        "eligible_names",
        "long_count",
        "short_count",
        "long_beta",
        "short_beta",
        "beta_target",
        "beta_match_error",
        "net_beta",
        "long_gross",
        "short_gross",
        "gross_exposure",
        "net_exposure",
        "max_abs_position_weight",
        "long_return",
        "short_return",
        "spread_return",
        "beta_residual_spread_return",
        "benchmark_return",
        "test_window_used",
    ]


def _skipped_columns() -> list[str]:
    return ["session_date", "variant", "signal", "eligible_names", "skip_reason", "test_window_used"]


def _summary_columns() -> list[str]:
    return [
        "variant",
        "signal",
        "requested_sessions",
        "constructed_sessions",
        "construction_rate",
        *_summary_metric_columns(),
        "test_window_used",
    ]


def _summary_metric_columns() -> list[str]:
    return [
        "mean_spread_return",
        "spread_hit_rate",
        "mean_beta_residual_spread_return",
        "mean_long_return",
        "mean_short_return",
        "mean_abs_net_beta",
        "max_abs_net_beta",
        "mean_abs_beta_match_error",
        "median_long_count",
        "median_short_count",
        "median_max_abs_position_weight",
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
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4 beta-matched portfolio constructor."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--signal", action="append", dest="signals")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--long-gross", type=float, default=1.0)
    parser.add_argument("--short-gross", type=float, default=1.0)
    parser.add_argument("--min-names-per-side", type=int, default=10)
    parser.add_argument("--target-names-per-side", type=int, default=20)
    parser.add_argument("--max-names-per-side", type=int, default=30)
    parser.add_argument("--max-single-name-side-weight", type=float, default=0.05)
    parser.add_argument("--beta-match-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = build_phase4_portfolio_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        signal_names=tuple(args.signals or DEFAULT_SIGNALS),
        variant_names=tuple(args.variants) if args.variants else None,
        long_gross=args.long_gross,
        short_gross=args.short_gross,
        min_names_per_side=args.min_names_per_side,
        target_names_per_side=args.target_names_per_side,
        max_names_per_side=args.max_names_per_side,
        max_single_name_side_weight=args.max_single_name_side_weight,
        beta_match_tolerance=args.beta_match_tolerance,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
