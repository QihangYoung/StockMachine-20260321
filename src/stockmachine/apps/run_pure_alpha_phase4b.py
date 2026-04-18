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
DEFAULT_PHASE4_POSITIONS = (
    RESEARCH_ROOT
    / "phase4_beta_matched_portfolios_20260418"
    / "phase4_beta_matched_positions_validation.csv.gz"
)
DEFAULT_PHASE4_SKIPPED = (
    RESEARCH_ROOT
    / "phase4_beta_matched_portfolios_20260418"
    / "phase4_skipped_sessions_validation.csv"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4b_short_book_diagnosis_20260418"
SIGNAL_COLUMNS = (
    "random_control",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "liquidity_rank_score",
    "transparent_composite",
)
BASE_COLUMNS = (
    "session_date",
    "variant",
    "symbol",
    "beta",
    "forward_return_5d",
    "forward_beta_residual_return_5d",
)


def build_phase4b_short_book_diagnostics(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    phase4_positions_path: str | Path = DEFAULT_PHASE4_POSITIONS,
    phase4_skipped_path: str | Path = DEFAULT_PHASE4_SKIPPED,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    signal_names: Sequence[str] = SIGNAL_COLUMNS,
    variant_names: Sequence[str] | None = None,
    candidate_count: int = 30,
    primary_phase4_signal: str = "reversal_5d",
) -> dict[str, Any]:
    """Diagnose why the validation short book is not yet contributing."""

    if not signal_names:
        raise ValueError("At least one signal name is required.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_panel = _load_signal_panel(
        signal_panel_path=signal_panel_path,
        signal_names=signal_names,
        variant_names=variant_names,
    )
    equal_weight_daily = _equal_weight_signal_side_diagnostics(
        signal_panel,
        signal_names=signal_names,
        candidate_count=candidate_count,
    )
    phase4_weighted_daily = _phase4_weighted_side_diagnostics(
        phase4_positions_path=phase4_positions_path,
        variant_names=variant_names,
    )
    side_daily = pd.concat([equal_weight_daily, phase4_weighted_daily], ignore_index=True)
    side_summary = _side_summary(side_daily)
    failure_matrix = _short_failure_matrix(
        side_summary,
        primary_phase4_signal=primary_phase4_signal,
    )
    skip_summary = _skip_summary(
        phase4_skipped_path=phase4_skipped_path,
        variant_names=variant_names,
    )

    daily_path = output_dir / "phase4b_short_book_side_diagnostics_validation.csv"
    summary_path = output_dir / "phase4b_short_book_side_summary_validation.csv"
    matrix_path = output_dir / "phase4b_short_failure_matrix_validation.csv"
    skip_path = output_dir / "phase4b_phase4_skip_summary_validation.csv"
    memo_path = output_dir / "phase4b_short_book_diagnosis_memo.md"
    rollup_path = output_dir / "phase4b_short_book_diagnosis_rollup.json"

    side_daily.to_csv(daily_path, index=False)
    side_summary.to_csv(summary_path, index=False)
    failure_matrix.to_csv(matrix_path, index=False)
    skip_summary.to_csv(skip_path, index=False)
    memo_path.write_text(
        _diagnosis_memo(
            side_summary,
            failure_matrix,
            skip_summary,
            candidate_count=candidate_count,
            primary_phase4_signal=primary_phase4_signal,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "phase4_positions_path": Path(phase4_positions_path).as_posix(),
        "phase4_skipped_path": Path(phase4_skipped_path).as_posix(),
        "signal_names": list(signal_names),
        "variant_names": sorted(signal_panel["variant"].astype(str).unique()) if not signal_panel.empty else [],
        "candidate_count": int(candidate_count),
        "primary_phase4_signal": primary_phase4_signal,
        "validation_start": _first_or_none(side_daily["session_date"]) if not side_daily.empty else None,
        "validation_end": _last_or_none(side_daily["session_date"]) if not side_daily.empty else None,
        "signal_panel_rows_loaded": int(len(signal_panel)),
        "side_diagnostic_rows": int(len(side_daily)),
        "side_summary_rows": int(len(side_summary)),
        "failure_matrix_rows": int(len(failure_matrix)),
        "skip_summary_rows": int(len(skip_summary)),
        "side_diagnostics_artifact": daily_path.as_posix(),
        "side_summary_artifact": summary_path.as_posix(),
        "failure_matrix_artifact": matrix_path.as_posix(),
        "skip_summary_artifact": skip_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is diagnosis only; it does not promote a new signal or freeze a candidate.",
            "Equal-weight candidate diagnostics are not tradable portfolio returns.",
            "Phase 4 weighted diagnostics inherit overlapping h5 forward labels.",
            "No transaction costs, borrow costs, turnover costs, or test-window performance are computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


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
    for column in [*BASE_COLUMNS[3:], *signal_names]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(
        subset=[
            "beta",
            "forward_return_5d",
            "forward_beta_residual_return_5d",
        ]
    )
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _equal_weight_signal_side_diagnostics(
    panel: pd.DataFrame,
    *,
    signal_names: Sequence[str],
    candidate_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = panel.groupby(["variant", "session_date"], sort=True)
    for (variant, session_date), group in grouped:
        for signal in signal_names:
            signal_group = group.dropna(subset=[signal]).drop_duplicates("symbol", keep="last")
            if len(signal_group) < candidate_count * 2:
                continue
            long_candidates = signal_group.sort_values(
                [signal, "symbol"],
                ascending=[False, True],
            ).head(candidate_count)
            short_candidates = signal_group.sort_values(
                [signal, "symbol"],
                ascending=[True, True],
            ).head(candidate_count)
            rows.append(
                _side_row(
                    long_candidates,
                    source="equal_weight_candidates",
                    variant=variant,
                    session_date=session_date,
                    signal=signal,
                    side="long",
                    score_column=signal,
                    weighted=False,
                )
            )
            rows.append(
                _side_row(
                    short_candidates,
                    source="equal_weight_candidates",
                    variant=variant,
                    session_date=session_date,
                    signal=signal,
                    side="short",
                    score_column=signal,
                    weighted=False,
                )
            )
    return pd.DataFrame(rows, columns=_side_daily_columns())


def _phase4_weighted_side_diagnostics(
    *,
    phase4_positions_path: str | Path,
    variant_names: Sequence[str] | None,
) -> pd.DataFrame:
    positions = pd.read_csv(phase4_positions_path, low_memory=False)
    if positions.empty:
        return pd.DataFrame(columns=_side_daily_columns())
    required = {
        "session_date",
        "variant",
        "signal",
        "side",
        "symbol",
        "score",
        "beta",
        "side_weight",
        "forward_return_5d",
        "forward_beta_residual_return_5d",
    }
    missing = sorted(required.difference(positions.columns))
    if missing:
        raise ValueError(f"Phase 4 positions are missing required columns: {missing}")
    positions["session_date"] = pd.to_datetime(positions["session_date"]).dt.date.astype(str)
    positions["variant"] = positions["variant"].astype(str)
    if variant_names is not None:
        positions = positions[positions["variant"].isin(set(variant_names))].copy()
    for column in [
        "score",
        "beta",
        "side_weight",
        "forward_return_5d",
        "forward_beta_residual_return_5d",
    ]:
        positions[column] = pd.to_numeric(positions[column], errors="coerce")
    rows = []
    grouped = positions.dropna(subset=["side_weight"]).groupby(
        ["variant", "session_date", "signal", "side"],
        sort=True,
    )
    for (variant, session_date, signal, side), group in grouped:
        rows.append(
            _side_row(
                group,
                source="phase4_weighted_positions",
                variant=variant,
                session_date=session_date,
                signal=signal,
                side=side,
                score_column="score",
                weighted=True,
            )
        )
    return pd.DataFrame(rows, columns=_side_daily_columns())


def _side_row(
    frame: pd.DataFrame,
    *,
    source: str,
    variant: str,
    session_date: str,
    signal: str,
    side: str,
    score_column: str,
    weighted: bool,
) -> dict[str, Any]:
    sign = 1.0 if side == "long" else -1.0
    weights = _normalized_weights(frame["side_weight"]) if weighted else None
    score = _weighted_mean(frame[score_column], weights)
    beta = _weighted_mean(frame["beta"], weights)
    raw_return = _weighted_mean(frame["forward_return_5d"], weights)
    residual_return = _weighted_mean(frame["forward_beta_residual_return_5d"], weights)
    return {
        "session_date": session_date,
        "variant": variant,
        "signal": signal,
        "source": source,
        "side": side,
        "names": int(len(frame)),
        "mean_score": score,
        "mean_beta": beta,
        "candidate_forward_return_5d": raw_return,
        "side_contribution_return": sign * raw_return,
        "candidate_forward_beta_residual_return_5d": residual_return,
        "side_contribution_beta_residual_return": sign * residual_return,
        "max_side_weight": float(frame["side_weight"].max()) if weighted else np.nan,
        "test_window_used": False,
    }


def _normalized_weights(series: pd.Series) -> np.ndarray:
    weights = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def _weighted_mean(series: pd.Series, weights: np.ndarray | None) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    if weights is None:
        return float(np.mean(values[mask])) if mask.any() else float("nan")
    clean_weights = weights[mask]
    if clean_weights.sum() <= 0:
        return float("nan")
    return float(np.dot(values[mask], clean_weights / clean_weights.sum()))


def _safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.median())


def _side_summary(side_daily: pd.DataFrame) -> pd.DataFrame:
    if side_daily.empty:
        return pd.DataFrame(columns=_side_summary_columns())
    rows = []
    grouped = side_daily.groupby(["source", "variant", "signal", "side"], sort=True)
    for (source, variant, signal, side), group in grouped:
        contribution = group["side_contribution_return"]
        residual_contribution = group["side_contribution_beta_residual_return"]
        rows.append(
            {
                "source": source,
                "variant": variant,
                "signal": signal,
                "side": side,
                "sessions": int(len(group)),
                "mean_names": float(group["names"].mean()),
                "mean_score": float(group["mean_score"].mean()),
                "mean_beta": float(group["mean_beta"].mean()),
                "mean_candidate_forward_return_5d": float(
                    group["candidate_forward_return_5d"].mean()
                ),
                "mean_side_contribution_return": float(contribution.mean()),
                "side_contribution_hit_rate": float((contribution > 0).mean()),
                "mean_candidate_forward_beta_residual_return_5d": float(
                    group["candidate_forward_beta_residual_return_5d"].mean()
                ),
                "mean_side_contribution_beta_residual_return": float(
                    residual_contribution.mean()
                ),
                "side_contribution_beta_residual_hit_rate": float(
                    (residual_contribution > 0).mean()
                ),
                "median_max_side_weight": _safe_median(group["max_side_weight"]),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_side_summary_columns())


def _short_failure_matrix(
    side_summary: pd.DataFrame,
    *,
    primary_phase4_signal: str,
) -> pd.DataFrame:
    if side_summary.empty:
        return pd.DataFrame(columns=_failure_matrix_columns())
    equal = side_summary[side_summary["source"] == "equal_weight_candidates"].copy()
    weighted = side_summary[side_summary["source"] == "phase4_weighted_positions"].copy()
    equal_short = equal[equal["side"] == "short"].copy()
    equal_long = equal[equal["side"] == "long"].copy()
    rows = []
    for _, short_row in equal_short.iterrows():
        long_row = _matching_row(equal_long, short_row["variant"], short_row["signal"])
        weighted_short = _matching_row(weighted[weighted["side"] == "short"], short_row["variant"], short_row["signal"])
        weighted_long = _matching_row(weighted[weighted["side"] == "long"], short_row["variant"], short_row["signal"])
        rows.append(
            {
                "variant": short_row["variant"],
                "signal": short_row["signal"],
                "equal_short_sessions": int(short_row["sessions"]),
                "equal_short_candidate_forward_return_5d": short_row[
                    "mean_candidate_forward_return_5d"
                ],
                "equal_short_contribution_return": short_row[
                    "mean_side_contribution_return"
                ],
                "equal_short_contribution_beta_residual_return": short_row[
                    "mean_side_contribution_beta_residual_return"
                ],
                "equal_short_hit_rate": short_row["side_contribution_hit_rate"],
                "equal_long_contribution_beta_residual_return": _value_or_nan(
                    long_row,
                    "mean_side_contribution_beta_residual_return",
                ),
                "equal_long_hit_rate": _value_or_nan(long_row, "side_contribution_hit_rate"),
                "equal_spread_beta_residual_return": _value_or_nan(
                    long_row,
                    "mean_side_contribution_beta_residual_return",
                )
                + short_row["mean_side_contribution_beta_residual_return"],
                "phase4_weighted_short_contribution_return": _value_or_nan(
                    weighted_short,
                    "mean_side_contribution_return",
                ),
                "phase4_weighted_short_contribution_beta_residual_return": _value_or_nan(
                    weighted_short,
                    "mean_side_contribution_beta_residual_return",
                ),
                "phase4_weighted_long_contribution_beta_residual_return": _value_or_nan(
                    weighted_long,
                    "mean_side_contribution_beta_residual_return",
                ),
                "diagnosis_label": _diagnosis_label(
                    short_row,
                    long_row,
                    weighted_short,
                    primary_phase4_signal=primary_phase4_signal,
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_failure_matrix_columns()).sort_values(
        ["variant", "equal_short_contribution_beta_residual_return", "signal"],
        ascending=[True, False, True],
    )


def _matching_row(frame: pd.DataFrame, variant: str, signal: str) -> pd.Series | None:
    match = frame[(frame["variant"] == variant) & (frame["signal"] == signal)]
    if match.empty:
        return None
    return match.iloc[0]


def _value_or_nan(row: pd.Series | None, column: str) -> float:
    if row is None:
        return float("nan")
    return float(row[column])


def _diagnosis_label(
    short_row: pd.Series,
    long_row: pd.Series | None,
    weighted_short: pd.Series | None,
    *,
    primary_phase4_signal: str,
) -> str:
    equal_short_residual = float(short_row["mean_side_contribution_beta_residual_return"])
    equal_long_residual = _value_or_nan(long_row, "mean_side_contribution_beta_residual_return")
    weighted_short_residual = _value_or_nan(
        weighted_short,
        "mean_side_contribution_beta_residual_return",
    )
    if equal_short_residual < 0 and equal_long_residual > 0:
        return "short_selection_fails_before_beta_matching"
    if short_row["signal"] == primary_phase4_signal and equal_short_residual > 0 and weighted_short_residual < 0:
        return "beta_matching_or_weighting_dilutes_short"
    if equal_short_residual > 0:
        return "short_signal_candidate_viable"
    return "both_legs_weak_or_unclear"


def _skip_summary(
    *,
    phase4_skipped_path: str | Path,
    variant_names: Sequence[str] | None,
) -> pd.DataFrame:
    path = Path(phase4_skipped_path)
    if not path.exists():
        return pd.DataFrame(columns=_skip_summary_columns())
    skipped = pd.read_csv(path)
    if skipped.empty:
        return pd.DataFrame(columns=_skip_summary_columns())
    skipped["variant"] = skipped["variant"].astype(str)
    skipped["signal"] = skipped["signal"].astype(str)
    skipped["skip_reason"] = skipped["skip_reason"].astype(str)
    if variant_names is not None:
        skipped = skipped[skipped["variant"].isin(set(variant_names))].copy()
    summary = (
        skipped.groupby(["variant", "signal", "skip_reason"], sort=True)
        .agg(skipped_sessions=("session_date", "nunique"))
        .reset_index()
    )
    summary["test_window_used"] = False
    return summary[_skip_summary_columns()]


def _diagnosis_memo(
    side_summary: pd.DataFrame,
    failure_matrix: pd.DataFrame,
    skip_summary: pd.DataFrame,
    *,
    candidate_count: int,
    primary_phase4_signal: str,
) -> str:
    lines = [
        "# Pure Alpha Phase 4B Short Book Diagnosis Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only diagnostic pause between Phase 4 and Phase 5. "
        "It asks why the short book is not contributing before converting the "
        "portfolio into a formal backtest artifact.",
        "",
        "## Method",
        "",
        f"- equal-weight candidate count per side: `{candidate_count}`",
        f"- primary Phase 4 constructed signal: `{primary_phase4_signal}`",
        "- equal-weight diagnostics test whether the signal's raw short candidates work before beta matching;",
        "- Phase 4 weighted diagnostics test whether the constructed short book works after beta matching;",
        "- raw and beta-residual contribution are reported separately.",
        "",
        "## Short-Side Signal Snapshot",
        "",
        "| Variant | Signal | Equal Short Residual Contribution | Equal Short Hit Rate | Label |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in failure_matrix.head(36).iterrows():
        lines.append(
            f"| {row['variant']} | {row['signal']} | "
            f"{row['equal_short_contribution_beta_residual_return']} | "
            f"{row['equal_short_hit_rate']} | {row['diagnosis_label']} |"
        )
    lines.extend(
        [
            "",
            "## Phase 4 Weighted Reversal Snapshot",
            "",
            "| Variant | Side | Mean Residual Contribution | Hit Rate |",
            "|---|---|---:|---:|",
        ]
    )
    weighted = side_summary[
        (side_summary["source"] == "phase4_weighted_positions")
        & (side_summary["signal"] == primary_phase4_signal)
    ]
    for _, row in weighted.iterrows():
        lines.append(
            f"| {row['variant']} | {row['side']} | "
            f"{row['mean_side_contribution_beta_residual_return']} | "
            f"{row['side_contribution_beta_residual_hit_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Phase 4 Skip Summary",
            "",
            "| Variant | Signal | Reason | Sessions |",
            "|---|---|---|---:|",
        ]
    )
    for _, row in skip_summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['signal']} | {row['skip_reason']} | "
            f"{row['skipped_sessions']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This memo diagnoses failure modes; it does not choose a final short signal.",
            "- Any asymmetric long/short design suggested here must go through a new validation-only construction pass.",
            "- Costs, borrow, turnover, and non-overlapping accounting remain outside this diagnostic step.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _side_daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "signal",
        "source",
        "side",
        "names",
        "mean_score",
        "mean_beta",
        "candidate_forward_return_5d",
        "side_contribution_return",
        "candidate_forward_beta_residual_return_5d",
        "side_contribution_beta_residual_return",
        "max_side_weight",
        "test_window_used",
    ]


def _side_summary_columns() -> list[str]:
    return [
        "source",
        "variant",
        "signal",
        "side",
        "sessions",
        "mean_names",
        "mean_score",
        "mean_beta",
        "mean_candidate_forward_return_5d",
        "mean_side_contribution_return",
        "side_contribution_hit_rate",
        "mean_candidate_forward_beta_residual_return_5d",
        "mean_side_contribution_beta_residual_return",
        "side_contribution_beta_residual_hit_rate",
        "median_max_side_weight",
        "test_window_used",
    ]


def _failure_matrix_columns() -> list[str]:
    return [
        "variant",
        "signal",
        "equal_short_sessions",
        "equal_short_candidate_forward_return_5d",
        "equal_short_contribution_return",
        "equal_short_contribution_beta_residual_return",
        "equal_short_hit_rate",
        "equal_long_contribution_beta_residual_return",
        "equal_long_hit_rate",
        "equal_spread_beta_residual_return",
        "phase4_weighted_short_contribution_return",
        "phase4_weighted_short_contribution_beta_residual_return",
        "phase4_weighted_long_contribution_beta_residual_return",
        "diagnosis_label",
        "test_window_used",
    ]


def _skip_summary_columns() -> list[str]:
    return ["variant", "signal", "skip_reason", "skipped_sessions", "test_window_used"]


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
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 4B short-book diagnosis.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--phase4-positions-path", default=str(DEFAULT_PHASE4_POSITIONS))
    parser.add_argument("--phase4-skipped-path", default=str(DEFAULT_PHASE4_SKIPPED))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--signal", action="append", dest="signals")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--primary-phase4-signal", default="reversal_5d")
    args = parser.parse_args(argv)

    result = build_phase4b_short_book_diagnostics(
        signal_panel_path=args.signal_panel_path,
        phase4_positions_path=args.phase4_positions_path,
        phase4_skipped_path=args.phase4_skipped_path,
        output_root=args.output_root,
        signal_names=tuple(args.signals or SIGNAL_COLUMNS),
        variant_names=tuple(args.variants) if args.variants else None,
        candidate_count=args.candidate_count,
        primary_phase4_signal=args.primary_phase4_signal,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
