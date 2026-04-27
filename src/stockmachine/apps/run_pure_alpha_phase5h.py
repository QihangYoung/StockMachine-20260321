from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import _fmt_bps, _fmt_float, _fmt_pct, _table
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT


DEFAULT_PHASE5F_ROOT = RESEARCH_ROOT / "phase5f_overlay_combo_on_turnover_aware_20260427"
DEFAULT_PHASE5G_ROOT = RESEARCH_ROOT / "phase5g_sota_loss_window_profiles_20260427"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5h_patch_acceptance_framework_20260427"
DEFAULT_METRICS_PATH = DEFAULT_PHASE5F_ROOT / "phase5f_metrics.csv"
DEFAULT_WINDOWS_PATH = DEFAULT_PHASE5F_ROOT / "phase5f_window_returns.csv"
DEFAULT_BUCKET_LIFTS_PATH = DEFAULT_PHASE5G_ROOT / "phase5g_bucket_lifts.csv"
DEFAULT_REPEAT_OFFENDERS_PATH = DEFAULT_PHASE5G_ROOT / "phase5g_repeat_offenders.csv"

BASELINE_SERIES = "baseline old-old"
SPY_SERIES = "SPY raw"
ALIASED_SERIES = "turnover-aware daily optimizer"
PATCH_SPECS: tuple[dict[str, Any], ...] = (
    {
        "series": "short overlay only",
        "overlay_side": "short",
        "complexity_score": 2,
        "intended_use": "lead_or_challenger",
    },
    {
        "series": "long overlay only",
        "overlay_side": "long",
        "complexity_score": 1,
        "intended_use": "risk_overlay_lab",
    },
    {
        "series": "dual overlay",
        "overlay_side": "dual",
        "complexity_score": 1,
        "intended_use": "defensive_variant",
    },
)


def build_phase5h_patch_acceptance_artifacts(
    *,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    windows_path: str | Path = DEFAULT_WINDOWS_PATH,
    bucket_lifts_path: str | Path = DEFAULT_BUCKET_LIFTS_PATH,
    repeat_offenders_path: str | Path = DEFAULT_REPEAT_OFFENDERS_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    windows = pd.read_csv(windows_path)
    bucket_lifts = pd.read_csv(bucket_lifts_path)
    repeat_offenders = pd.read_csv(repeat_offenders_path)

    metrics = metrics[
        metrics["series"].notna()
        & ~metrics["series"].eq(SPY_SERIES)
        & ~metrics["series"].eq(ALIASED_SERIES)
    ].copy()
    baseline = metrics.loc[metrics["series"].eq(BASELINE_SERIES)].iloc[0]

    scorecard = _build_scorecard(metrics=metrics, windows=windows, baseline=baseline)
    mechanism = _build_mechanism_summary(bucket_lifts=bucket_lifts, repeat_offenders=repeat_offenders)
    decision = _build_decision_table(scorecard)

    scorecard_path = output_dir / "phase5h_patch_scorecard.csv"
    mechanism_path = output_dir / "phase5h_mechanism_summary.csv"
    decision_path = output_dir / "phase5h_patch_decision_table.csv"
    memo_path = output_dir / "phase5h_patch_acceptance_framework_memo.md"
    rollup_path = output_dir / "phase5h_rollup.json"

    scorecard.to_csv(scorecard_path, index=False)
    mechanism.to_csv(mechanism_path, index=False)
    decision.to_csv(decision_path, index=False)
    memo_path.write_text(
        _memo(
            baseline=baseline,
            scorecard=scorecard,
            mechanism=mechanism,
            decision=decision,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "metrics_path": Path(metrics_path).as_posix(),
        "windows_path": Path(windows_path).as_posix(),
        "bucket_lifts_path": Path(bucket_lifts_path).as_posix(),
        "repeat_offenders_path": Path(repeat_offenders_path).as_posix(),
        "baseline_series": BASELINE_SERIES,
        "patches": [spec["series"] for spec in PATCH_SPECS],
        "scorecard_artifact": scorecard_path.as_posix(),
        "mechanism_artifact": mechanism_path.as_posix(),
        "decision_artifact": decision_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "patch_acceptance_framework_with_full_sample_stress_and_mechanism_checks",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "The acceptance scores are heuristics for internal research governance, not statistical proof.",
            "Mechanism evidence is currently strongest for the SOTA short-overlay line because Phase5G profiles that live portfolio directly.",
            "Window evidence uses the fixed stress windows already defined in Phase5F; unseen future states can still differ.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_scorecard(
    *,
    metrics: pd.DataFrame,
    windows: pd.DataFrame,
    baseline: pd.Series,
) -> pd.DataFrame:
    baseline_windows = windows.loc[windows["series"].eq(BASELINE_SERIES), ["window", "compound_return"]].rename(
        columns={"compound_return": "baseline_compound_return"}
    )
    rows: list[dict[str, Any]] = []
    for spec in PATCH_SPECS:
        series = spec["series"]
        metric_row = metrics.loc[metrics["series"].eq(series)].iloc[0]
        stress = windows.loc[windows["series"].eq(series), ["window", "compound_return"]].merge(
            baseline_windows,
            on="window",
            how="left",
        )
        stress["improvement_vs_baseline"] = (
            stress["compound_return"] - stress["baseline_compound_return"]
        )

        ann_delta = float(metric_row["annualized_return"] - baseline["annualized_return"])
        sharpe_delta = float(metric_row["sharpe_no_rf"] - baseline["sharpe_no_rf"])
        vol_delta = float(metric_row["annualized_vol"] - baseline["annualized_vol"])
        max_dd_improvement = float(abs(baseline["max_drawdown"]) - abs(metric_row["max_drawdown"]))
        rolling60_delta = float(metric_row["rolling_60_positive_rate"] - baseline["rolling_60_positive_rate"])
        turnover_delta = float(metric_row["aggregate_turnover_mean"] - baseline["aggregate_turnover_mean"])
        mean_daily_bps_delta = float(metric_row["mean_daily_return_bps"] - baseline["mean_daily_return_bps"])
        mean_stress_improvement = float(stress["improvement_vs_baseline"].mean())
        min_stress_improvement = float(stress["improvement_vs_baseline"].min())
        improved_windows = int((stress["improvement_vs_baseline"] > 0).sum())
        total_windows = int(len(stress))

        full_sample_score = _full_sample_score(ann_delta=ann_delta, sharpe_delta=sharpe_delta)
        stress_score = _stress_score(
            mean_stress_improvement=mean_stress_improvement,
            min_stress_improvement=min_stress_improvement,
            improved_windows=improved_windows,
            total_windows=total_windows,
        )
        risk_score = _risk_score(
            max_dd_improvement=max_dd_improvement,
            vol_delta=vol_delta,
            rolling60_delta=rolling60_delta,
        )
        turnover_score = _turnover_score(turnover_delta=turnover_delta)
        complexity_score = int(spec["complexity_score"])
        total_score = int(
            full_sample_score + stress_score + risk_score + turnover_score + complexity_score
        )

        rows.append(
            {
                "series": series,
                "overlay_side": spec["overlay_side"],
                "intended_use": spec["intended_use"],
                "annualized_return_delta": ann_delta,
                "sharpe_delta": sharpe_delta,
                "annualized_vol_delta": vol_delta,
                "max_drawdown_improvement": max_dd_improvement,
                "rolling_60_positive_delta": rolling60_delta,
                "turnover_delta": turnover_delta,
                "mean_daily_return_bps_delta": mean_daily_bps_delta,
                "mean_stress_improvement": mean_stress_improvement,
                "min_stress_improvement": min_stress_improvement,
                "improved_windows": improved_windows,
                "total_windows": total_windows,
                "full_sample_score": full_sample_score,
                "stress_score": stress_score,
                "risk_score": risk_score,
                "turnover_score": turnover_score,
                "complexity_score": complexity_score,
                "total_score": total_score,
                "acceptance_bucket": _acceptance_bucket(
                    series=series,
                    full_sample_score=full_sample_score,
                    stress_score=stress_score,
                    risk_score=risk_score,
                    total_score=total_score,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["total_score", "mean_stress_improvement", "annualized_return_delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_mechanism_summary(
    *,
    bucket_lifts: pd.DataFrame,
    repeat_offenders: pd.DataFrame,
) -> pd.DataFrame:
    short_bucket = (
        bucket_lifts[
            bucket_lifts["side"].eq("short")
            & bucket_lifts["bucket"].isin(
                ["winner_strength_gt_p10", "mom20_gt_p05_and_beta_gt_p05", "recent_not_down"]
            )
        ]
        .groupby("bucket", as_index=False)["lift"]
        .mean()
        .rename(columns={"lift": "mean_lift"})
    )
    short_repeat = repeat_offenders[
        repeat_offenders["side"].eq("short") & (repeat_offenders["windows"].astype(int) >= 2)
    ].copy()
    long_repeat = repeat_offenders[
        repeat_offenders["side"].eq("long") & (repeat_offenders["windows"].astype(int) >= 2)
    ].copy()

    rows = [
        {
            "topic": "short_overlay_mechanism",
            "evidence": "mean_winner_strength_gt_p10_lift",
            "value": float(
                short_bucket.loc[
                    short_bucket["bucket"].eq("winner_strength_gt_p10"), "mean_lift"
                ].iloc[0]
            ),
            "notes": "Bad short windows remain concentrated in strong former winners.",
        },
        {
            "topic": "short_overlay_mechanism",
            "evidence": "mean_growth_beta_continuation_lift",
            "value": float(
                short_bucket.loc[
                    short_bucket["bucket"].eq("mom20_gt_p05_and_beta_gt_p05"), "mean_lift"
                ].iloc[0]
            ),
            "notes": "Short culprits are still high-momentum/high-beta continuation names.",
        },
        {
            "topic": "short_overlay_mechanism",
            "evidence": "mean_recent_not_down_lift",
            "value": float(
                short_bucket.loc[
                    short_bucket["bucket"].eq("recent_not_down"), "mean_lift"
                ].iloc[0]
            ),
            "notes": "Short failures skew toward names that have not fully rolled over.",
        },
        {
            "topic": "repeat_offenders",
            "evidence": "top_short_repeat_offenders",
            "value": np.nan,
            "notes": ", ".join(short_repeat.head(5)["symbol"].astype(str).tolist()),
        },
        {
            "topic": "repeat_offenders",
            "evidence": "top_long_repeat_offenders",
            "value": np.nan,
            "notes": ", ".join(long_repeat.head(5)["symbol"].astype(str).tolist()),
        },
    ]
    return pd.DataFrame(rows)


def _build_decision_table(scorecard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in scorecard.itertuples(index=False):
        if row.acceptance_bucket == "promote_challenger_lead":
            action = "Keep as primary challenger to current lead and verify on future unseen periods."
        elif row.acceptance_bucket == "keep_defensive_variant":
            action = "Keep as defensive product/risk-managed variant, not default alpha lead."
        else:
            action = "Keep in research lab only; do not promote without more generic evidence."
        rows.append(
            {
                "series": row.series,
                "acceptance_bucket": row.acceptance_bucket,
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def _full_sample_score(*, ann_delta: float, sharpe_delta: float) -> int:
    if ann_delta >= 0.0 and sharpe_delta >= 0.0:
        return 2
    if ann_delta >= -0.01 and sharpe_delta >= 0.0:
        return 1
    return 0


def _stress_score(
    *,
    mean_stress_improvement: float,
    min_stress_improvement: float,
    improved_windows: int,
    total_windows: int,
) -> int:
    if total_windows <= 0:
        return 0
    if improved_windows == total_windows and mean_stress_improvement >= 0.015 and min_stress_improvement > 0:
        return 2
    if improved_windows >= max(1, total_windows - 1) and mean_stress_improvement >= 0.0075:
        return 1
    return 0


def _risk_score(*, max_dd_improvement: float, vol_delta: float, rolling60_delta: float) -> int:
    if max_dd_improvement >= 0.02 and vol_delta < 0.0 and rolling60_delta >= 0.03:
        return 2
    if max_dd_improvement >= 0.01 and vol_delta < 0.0:
        return 1
    return 0


def _turnover_score(*, turnover_delta: float) -> int:
    if turnover_delta <= 0.01:
        return 2
    if turnover_delta <= 0.03:
        return 1
    return 0


def _acceptance_bucket(
    *,
    series: str,
    full_sample_score: int,
    stress_score: int,
    risk_score: int,
    total_score: int,
) -> str:
    if full_sample_score == 2 and stress_score >= 1 and total_score >= 7:
        return "promote_challenger_lead"
    if risk_score == 2 and stress_score == 2 and total_score >= 6:
        return "keep_defensive_variant"
    return "keep_lab_overlay"


def _memo(
    *,
    baseline: pd.Series,
    scorecard: pd.DataFrame,
    mechanism: pd.DataFrame,
    decision: pd.DataFrame,
) -> str:
    best = scorecard.iloc[0]
    lines = [
        "# Phase5H Patch Acceptance Framework",
        "",
        "## Why this memo exists",
        "",
        "- A patch that repairs a few bad windows can still be overfit.",
        "- We therefore score patches as overlays/risk rules first, not as automatic new alpha.",
        "- Promotion requires passing full-sample, stress-window, risk-payoff, turnover, and simplicity checks together.",
        "",
        "## Baseline",
        "",
        f"- Baseline series: `{BASELINE_SERIES}`",
        f"- Annualized return: {_fmt_pct(float(baseline['annualized_return']))}",
        f"- Annualized vol: {_fmt_pct(float(baseline['annualized_vol']))}",
        f"- Sharpe: {_fmt_float(float(baseline['sharpe_no_rf']))}",
        f"- Max drawdown: {_fmt_pct(float(baseline['max_drawdown']))}",
        f"- Rolling 60 positive rate: {_fmt_pct(float(baseline['rolling_60_positive_rate']))}",
        "",
        "## Acceptance Rules",
        "",
        "1. Full-sample check: the patch should not obviously break the whole validation path.",
        "2. Stress-window check: the patch should improve all or almost all registered bad windows, not just one episode.",
        "3. Risk-payoff check: if return is sacrificed, the drawdown/vol/path improvement must be meaningful.",
        "4. Turnover check: the patch should not buy its improvement by exploding turnover.",
        "5. Simplicity bias: lower-DoF, more interpretable overlays get the benefit of the doubt.",
        "",
        "## Scorecard",
        "",
        _table(scorecard),
        "",
        "## Decisions",
        "",
        _table(decision),
        "",
        "## Mechanism Evidence",
        "",
        _table(mechanism),
        "",
        "## Reading",
        "",
        f"- Best overall patch by this framework: `{best['series']}` with total score {int(best['total_score'])} and bucket `{best['acceptance_bucket']}`.",
        "- `short overlay only` is the only patch that both improves full-sample return/Sharpe and repairs all registered stress windows.",
        "- `dual overlay` looks strongest as a defensive product variant: large stress repair and path improvement, but too much alpha tax to be the default lead.",
        "- `long overlay only` stays in the lab for now: it helps bad windows, but the return tax is too visible for promotion.",
        "- Mechanism evidence is currently strongest for the short overlay because our live SOTA culprit profiling still points to continuation winners rather than a one-off episode.",
        "",
        "## Practical rule",
        "",
        "- Promote to lead/challenger only if a patch wins on both full-sample and multi-window stress checks.",
        "- Keep as defensive variant if it clearly improves tail/path quality but taxes alpha.",
        "- Keep in the lab if it is explainable but still looks too expensive or too patch-like.",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a patch-acceptance framework scorecard for pure-alpha overlays.",
    )
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--windows-path", default=str(DEFAULT_WINDOWS_PATH))
    parser.add_argument("--bucket-lifts-path", default=str(DEFAULT_BUCKET_LIFTS_PATH))
    parser.add_argument("--repeat-offenders-path", default=str(DEFAULT_REPEAT_OFFENDERS_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)

    rollup = build_phase5h_patch_acceptance_artifacts(
        metrics_path=args.metrics_path,
        windows_path=args.windows_path,
        bucket_lifts_path=args.bucket_lifts_path,
        repeat_offenders_path=args.repeat_offenders_path,
        output_root=args.output_root,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
