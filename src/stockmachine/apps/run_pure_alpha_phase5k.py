from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import _fmt_float, _fmt_pct, _table
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT


DEFAULT_PHASE5F_ROOT = RESEARCH_ROOT / "phase5f_overlay_combo_on_turnover_aware_20260427"
DEFAULT_PHASE5J_ROOT = RESEARCH_ROOT / "phase5j_whitebox_state_activation_20260428"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5k_sota_role_framework_20260428"

DEFAULT_PHASE5F_METRICS = DEFAULT_PHASE5F_ROOT / "phase5f_metrics.csv"
DEFAULT_PHASE5F_STRICT_CURVE = DEFAULT_PHASE5F_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_daily_curve.csv"
DEFAULT_PHASE5J_METRICS = DEFAULT_PHASE5J_ROOT / "phase5j_metrics.csv"
DEFAULT_PHASE5J_STRICT_CURVE = DEFAULT_PHASE5J_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_daily_curve.csv"
DEFAULT_PHASE5J_ACTIVATION = DEFAULT_PHASE5J_ROOT / "phase5j_activation_summary.csv"

REFERENCE_SERIES = "current sota"
WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)


def build_phase5k_sota_role_framework_artifacts(
    *,
    phase5f_metrics_path: str | Path = DEFAULT_PHASE5F_METRICS,
    phase5f_strict_curve_path: str | Path = DEFAULT_PHASE5F_STRICT_CURVE,
    phase5j_metrics_path: str | Path = DEFAULT_PHASE5J_METRICS,
    phase5j_strict_curve_path: str | Path = DEFAULT_PHASE5J_STRICT_CURVE,
    phase5j_activation_path: str | Path = DEFAULT_PHASE5J_ACTIVATION,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_f = pd.read_csv(phase5f_metrics_path)
    metrics_j = pd.read_csv(phase5j_metrics_path)
    curve_f = pd.read_csv(phase5f_strict_curve_path, parse_dates=["return_date"])
    curve_j = pd.read_csv(phase5j_strict_curve_path, parse_dates=["return_date"])
    activation = pd.read_csv(phase5j_activation_path)

    selected = _select_shortlist(metrics_f=metrics_f, metrics_j=metrics_j, activation=activation)
    window_table = _build_window_table(curve_f=curve_f, curve_j=curve_j, selected=selected)
    scorecard = _build_scorecard(selected=selected, window_table=window_table)
    decisions = _build_decisions(scorecard)

    selected_path = output_dir / "phase5k_selected_variants.csv"
    window_path = output_dir / "phase5k_window_comparison.csv"
    scorecard_path = output_dir / "phase5k_role_scorecard.csv"
    decision_path = output_dir / "phase5k_role_decisions.csv"
    memo_path = output_dir / "phase5k_sota_role_framework_memo.md"
    rollup_path = output_dir / "phase5k_rollup.json"

    selected.to_csv(selected_path, index=False)
    window_table.to_csv(window_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    decisions.to_csv(decision_path, index=False)
    memo_path.write_text(
        _memo(
            selected=selected,
            window_table=window_table,
            scorecard=scorecard,
            decisions=decisions,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "phase5f_metrics_path": Path(phase5f_metrics_path).as_posix(),
        "phase5f_strict_curve_path": Path(phase5f_strict_curve_path).as_posix(),
        "phase5j_metrics_path": Path(phase5j_metrics_path).as_posix(),
        "phase5j_strict_curve_path": Path(phase5j_strict_curve_path).as_posix(),
        "phase5j_activation_path": Path(phase5j_activation_path).as_posix(),
        "selected_artifact": selected_path.as_posix(),
        "window_comparison_artifact": window_path.as_posix(),
        "scorecard_artifact": scorecard_path.as_posix(),
        "decisions_artifact": decision_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "reference_series": REFERENCE_SERIES,
        "windows": [{"window": w, "start": s, "end": e} for w, s, e in WINDOWS],
        "method": "unified_role_framework_for_sota_and_state_variants",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "Role assignment is a research governance heuristic, not formal statistical proof.",
            "State variants are chosen from the Phase5J family by best Sharpe within the soft and hard families.",
            "Dual overlay is compared on the same stress windows using its Phase5F strict path.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _select_shortlist(
    *,
    metrics_f: pd.DataFrame,
    metrics_j: pd.DataFrame,
    activation: pd.DataFrame,
) -> pd.DataFrame:
    metrics_f = metrics_f[metrics_f["series"].notna()].copy()
    metrics_j = metrics_j[metrics_j["series"].notna()].copy()

    current = metrics_j.loc[metrics_j["series"].eq(REFERENCE_SERIES)].copy()
    dual = metrics_f.loc[metrics_f["series"].eq("dual overlay")].copy()
    soft_family = metrics_j[metrics_j["series"].astype(str).str.startswith("state soft ")].copy()
    hard_family = metrics_j[metrics_j["series"].astype(str).str.startswith("state hard ")].copy()

    best_soft = soft_family.sort_values(
        ["sharpe_no_rf", "annualized_return"],
        ascending=[False, False],
    ).head(1)
    best_hard = hard_family.sort_values(
        ["sharpe_no_rf", "annualized_return"],
        ascending=[False, False],
    ).head(1)

    out = pd.concat([current, best_soft, best_hard, dual], ignore_index=True)
    out = out[
        [
            "series",
            "portfolio",
            "long_score",
            "short_selector",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "max_drawdown",
            "rolling_60_positive_rate",
            "corr_to_spy",
            "mean_daily_return_bps",
            "aggregate_turnover_mean",
            "mean_abs_net_beta",
        ]
    ].copy()
    out["source_run"] = out["series"].map(
        {
            REFERENCE_SERIES: "phase5j",
            "dual overlay": "phase5f",
        }
    ).fillna("phase5j")
    out["activation_rate"] = np.nan
    gate_rate = activation.loc[
        activation["gate"].eq("gate_score050_or_spy20m03")
        & activation["window"].isna(),
        "activation_rate_full",
    ]
    activation_rate = float(gate_rate.iloc[0]) if not gate_rate.empty else np.nan
    out.loc[out["series"].astype(str).str.contains(r"score050\+spy20", regex=True), "activation_rate"] = activation_rate
    out.loc[out["series"].eq("dual overlay"), "activation_rate"] = 1.0
    out.loc[out["series"].eq(REFERENCE_SERIES), "activation_rate"] = 1.0
    out["activation_mode"] = out["series"].map(
        {
            REFERENCE_SERIES: "always_on",
            "dual overlay": "always_on",
        }
    ).fillna("state_dependent")
    return out.reset_index(drop=True)


def _build_window_table(
    *,
    curve_f: pd.DataFrame,
    curve_j: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_curve = curve_j[curve_j["portfolio"].astype(str).eq("current_sota")].copy()
    reference_windows = _window_returns_for_portfolio(reference_curve)
    reference_map = {
        row["window"]: row["compound_return"]
        for row in reference_windows.to_dict(orient="records")
    }
    for row in selected.itertuples(index=False):
        portfolio = str(row.portfolio)
        source = str(row.source_run)
        curve = curve_j if source == "phase5j" else curve_f
        sub = curve[curve["portfolio"].astype(str).eq(portfolio)].copy()
        if sub.empty:
            continue
        summary = _window_returns_for_portfolio(sub)
        summary["series"] = row.series
        summary["portfolio"] = portfolio
        summary["source_run"] = source
        summary["improvement_vs_current_sota"] = summary["window"].map(reference_map)
        summary["improvement_vs_current_sota"] = (
            summary["compound_return"] - summary["improvement_vs_current_sota"]
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _window_returns_for_portfolio(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = frame.sort_values("return_date").copy()
    for window, start, end in WINDOWS:
        subset = frame[
            (frame["return_date"] >= pd.Timestamp(start))
            & (frame["return_date"] <= pd.Timestamp(end))
        ]
        if subset.empty:
            continue
        rows.append(
            {
                "window": window,
                "start": start,
                "end": end,
                "compound_return": float((1.0 + subset["gross_return"].astype(float)).prod() - 1.0),
                "hit_rate_daily": float((subset["gross_return"].astype(float) > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_scorecard(
    *,
    selected: pd.DataFrame,
    window_table: pd.DataFrame,
) -> pd.DataFrame:
    reference = selected.loc[selected["series"].eq(REFERENCE_SERIES)].iloc[0]
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        if row.series == REFERENCE_SERIES:
            rows.append(
                {
                    "series": row.series,
                    "role_bucket": "reference_lead",
                    "annualized_return_delta": 0.0,
                    "annualized_vol_delta": 0.0,
                    "sharpe_delta": 0.0,
                    "max_drawdown_improvement": 0.0,
                    "rolling_60_positive_delta": 0.0,
                    "mean_stress_improvement": 0.0,
                    "min_stress_improvement": 0.0,
                    "improved_windows": 0,
                    "total_windows": int(len(WINDOWS)),
                    "activation_rate": float(row.activation_rate),
                    "activation_mode": row.activation_mode,
                    "score": 0,
                }
            )
            continue

        stress = window_table[window_table["series"].eq(row.series)].copy()
        ann_delta = float(row.annualized_return - reference["annualized_return"])
        vol_delta = float(row.annualized_vol - reference["annualized_vol"])
        sharpe_delta = float(row.sharpe_no_rf - reference["sharpe_no_rf"])
        dd_improvement = float(abs(reference["max_drawdown"]) - abs(row.max_drawdown))
        rolling_delta = float(row.rolling_60_positive_rate - reference["rolling_60_positive_rate"])
        mean_stress = float(stress["improvement_vs_current_sota"].mean())
        min_stress = float(stress["improvement_vs_current_sota"].min())
        improved_windows = int((stress["improvement_vs_current_sota"] > 0).sum())
        score = _score_variant(
            ann_delta=ann_delta,
            sharpe_delta=sharpe_delta,
            dd_improvement=dd_improvement,
            rolling_delta=rolling_delta,
            improved_windows=improved_windows,
            total_windows=int(len(stress)),
            activation_rate=float(row.activation_rate),
            activation_mode=str(row.activation_mode),
        )
        rows.append(
            {
                "series": row.series,
                "role_bucket": _role_bucket(
                    ann_delta=ann_delta,
                    sharpe_delta=sharpe_delta,
                    dd_improvement=dd_improvement,
                    improved_windows=improved_windows,
                    total_windows=int(len(stress)),
                    activation_mode=str(row.activation_mode),
                ),
                "annualized_return_delta": ann_delta,
                "annualized_vol_delta": vol_delta,
                "sharpe_delta": sharpe_delta,
                "max_drawdown_improvement": dd_improvement,
                "rolling_60_positive_delta": rolling_delta,
                "mean_stress_improvement": mean_stress,
                "min_stress_improvement": min_stress,
                "improved_windows": improved_windows,
                "total_windows": int(len(stress)),
                "activation_rate": float(row.activation_rate),
                "activation_mode": row.activation_mode,
                "score": int(score),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["score", "sharpe_delta", "max_drawdown_improvement"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _score_variant(
    *,
    ann_delta: float,
    sharpe_delta: float,
    dd_improvement: float,
    rolling_delta: float,
    improved_windows: int,
    total_windows: int,
    activation_rate: float,
    activation_mode: str,
) -> int:
    score = 0
    if ann_delta >= -0.015:
        score += 2
    elif ann_delta >= -0.03:
        score += 1
    if sharpe_delta >= 0.0:
        score += 2
    elif sharpe_delta >= -0.03:
        score += 1
    if dd_improvement >= 0.015:
        score += 2
    elif dd_improvement >= 0.0075:
        score += 1
    if improved_windows >= total_windows - 1:
        score += 2
    elif improved_windows >= total_windows - 2:
        score += 1
    if rolling_delta >= 0.0:
        score += 1
    if activation_mode == "state_dependent":
        if 0.15 <= activation_rate <= 0.40:
            score += 2
        elif activation_rate <= 0.50:
            score += 1
    else:
        score += 1
    return score


def _role_bucket(
    *,
    ann_delta: float,
    sharpe_delta: float,
    dd_improvement: float,
    improved_windows: int,
    total_windows: int,
    activation_mode: str,
) -> str:
    if (
        activation_mode == "state_dependent"
        and ann_delta >= -0.015
        and sharpe_delta >= 0.0
        and dd_improvement >= 0.01
        and improved_windows >= total_windows - 1
    ):
        return "balanced_challenger"
    if dd_improvement >= 0.02 and improved_windows >= total_windows - 1:
        return "defensive_variant"
    return "reference_or_secondary"


def _build_decisions(scorecard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in scorecard.itertuples(index=False):
        if row.role_bucket == "reference_lead":
            action = "Keep as current alpha lead reference."
        elif row.series == "state soft score050+spy20":
            action = (
                "Closest balanced alternative, but keep secondary for now because it still "
                "underperforms the lead in 2 of 6 registered stress windows."
            )
        elif row.role_bucket == "balanced_challenger":
            action = "Promote as balanced challenger; strongest candidate to test against the current lead."
        elif row.role_bucket == "defensive_variant":
            action = "Keep as defensive variant for smoother product paths and lower drawdown."
        else:
            action = "Keep as secondary/reference only; not preferred over the lead or main challenger."
        rows.append(
            {
                "series": row.series,
                "role_bucket": row.role_bucket,
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def _memo(
    *,
    selected: pd.DataFrame,
    window_table: pd.DataFrame,
    scorecard: pd.DataFrame,
    decisions: pd.DataFrame,
) -> str:
    display_selected = selected.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
    ):
        if column in display_selected.columns:
            display_selected[column] = display_selected[column].map(_fmt_pct)
    for column in (
        "sharpe_no_rf",
        "corr_to_spy",
        "mean_daily_return_bps",
        "aggregate_turnover_mean",
        "mean_abs_net_beta",
        "activation_rate",
    ):
        if column in display_selected.columns:
            display_selected[column] = display_selected[column].map(
                lambda v: _fmt_float(float(v)) if pd.notna(v) else ""
            )

    display_windows = window_table.copy()
    if not display_windows.empty:
        display_windows["compound_return"] = display_windows["compound_return"].map(_fmt_pct)
        display_windows["improvement_vs_current_sota"] = display_windows[
            "improvement_vs_current_sota"
        ].map(_fmt_pct)
        display_windows["hit_rate_daily"] = display_windows["hit_rate_daily"].map(_fmt_pct)

    display_scorecard = scorecard.copy()
    for column in (
        "annualized_return_delta",
        "annualized_vol_delta",
        "max_drawdown_improvement",
        "rolling_60_positive_delta",
        "mean_stress_improvement",
        "min_stress_improvement",
    ):
        if column in display_scorecard.columns:
            display_scorecard[column] = display_scorecard[column].map(_fmt_pct)
    for column in ("sharpe_delta", "activation_rate"):
        if column in display_scorecard.columns:
            display_scorecard[column] = display_scorecard[column].map(
                lambda v: _fmt_float(float(v)) if pd.notna(v) else ""
            )

    lines = [
        "# Phase5K Unified Role Framework",
        "",
        "## Goal",
        "",
        "- Put the current SOTA, the best white-box state-dependent long variants, and dual overlay on one table.",
        "- Judge not just 'which has the highest backtest mean', but which role each variant deserves.",
        "",
        "## Shortlist",
        "",
        _table(display_selected),
        "",
        "## Stress Windows",
        "",
        _table(display_windows),
        "",
        "## Role Scorecard",
        "",
        _table(display_scorecard),
        "",
        "## Decisions",
        "",
        _table(decisions),
        "",
        "## Reading",
        "",
        "- `current sota` stays the reference lead because it still has the highest annualized return while keeping a strong Sharpe.",
        "- `state soft score050+spy20` is the closest balanced alternative: modest alpha tax, better Sharpe, materially better drawdown, and only one-third activation frequency, but it still loses on 2 of the 6 registered stress windows, so it stays secondary for now.",
        "- `state hard score050+spy20` looks like the strongest defensive white-box variant: best Sharpe / lowest beta-like co-movement, but meaningfully more alpha tax than the soft challenger.",
        "- `dual overlay` remains a defensive product variant, but it is now dominated by the state-hard white-box version on the 'defense per unit alpha tax' trade-off.",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified role framework for current SOTA, state variants, and dual overlay.",
    )
    parser.add_argument("--phase5f-metrics-path", default=str(DEFAULT_PHASE5F_METRICS))
    parser.add_argument("--phase5f-strict-curve-path", default=str(DEFAULT_PHASE5F_STRICT_CURVE))
    parser.add_argument("--phase5j-metrics-path", default=str(DEFAULT_PHASE5J_METRICS))
    parser.add_argument("--phase5j-strict-curve-path", default=str(DEFAULT_PHASE5J_STRICT_CURVE))
    parser.add_argument("--phase5j-activation-path", default=str(DEFAULT_PHASE5J_ACTIVATION))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)

    rollup = build_phase5k_sota_role_framework_artifacts(
        phase5f_metrics_path=args.phase5f_metrics_path,
        phase5f_strict_curve_path=args.phase5f_strict_curve_path,
        phase5j_metrics_path=args.phase5j_metrics_path,
        phase5j_strict_curve_path=args.phase5j_strict_curve_path,
        phase5j_activation_path=args.phase5j_activation_path,
        output_root=args.output_root,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
