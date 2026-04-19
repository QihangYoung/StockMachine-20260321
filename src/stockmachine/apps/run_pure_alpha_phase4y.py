from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4f import (
    BENCHMARK_FORWARD_COLUMN,
    _load_posterior_panel,
)
from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
    _add_residual_targets,
)
from stockmachine.apps.run_pure_alpha_phase4j import _add_selector_scores, _selector_catalog
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_SHORT_SELECTOR,
    DEFAULT_SHORT_VARIANT,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4y_short_selector_failure_diagnostics_20260419"
DEFAULT_TARGETS = (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
)
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


def build_phase4y_short_selector_failure_diagnostics(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    lead_selector: str = DEFAULT_SHORT_SELECTOR,
    residual_targets: Sequence[str] = DEFAULT_TARGETS,
    candidate_count: int = 30,
    loser_quantile: float = 0.20,
    score_bins: int = 10,
    min_regression_rows: int = 80,
) -> dict[str, Any]:
    """Diagnose short-selector failures by window, component, and market state."""

    _validate_settings(
        residual_targets=residual_targets,
        candidate_count=candidate_count,
        loser_quantile=loser_quantile,
        score_bins=score_bins,
        min_regression_rows=min_regression_rows,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_posterior_panel(
        signal_panel_path,
        variant_names=(short_variant,),
        target_column=TARGET_COLUMN,
    )
    panel = _add_residual_targets(panel, min_regression_rows=min_regression_rows)
    panel = _add_all_selector_scores(panel)
    panel = _add_market_state(panel)

    daily = _selector_daily_diagnostics(
        panel,
        residual_targets=residual_targets,
        candidate_count=candidate_count,
        loser_quantile=loser_quantile,
    )
    window_summary = _window_summary(daily)
    state_summary = _state_summary(daily)
    bin_summary = _score_bin_summary(
        panel,
        selectors=[name for name, _ in _selector_catalog()],
        residual_targets=residual_targets,
        score_bins=score_bins,
    )
    lead_window = window_summary[
        (window_summary["selector"] == lead_selector)
        & (window_summary["residual_target"] == TARGET_BETA_RESIDUAL)
    ].copy()
    memo = _memo(
        lead_window,
        window_summary,
        state_summary,
        bin_summary,
        short_variant=short_variant,
        lead_selector=lead_selector,
        candidate_count=candidate_count,
        score_bins=score_bins,
    )

    daily_path = output_dir / "phase4y_short_selector_daily_validation.csv"
    window_path = output_dir / "phase4y_short_selector_window_summary_validation.csv"
    state_path = output_dir / "phase4y_short_selector_state_summary_validation.csv"
    bin_path = output_dir / "phase4y_short_selector_score_bin_summary_validation.csv"
    memo_path = output_dir / "phase4y_short_selector_failure_memo.md"
    rollup_path = output_dir / "phase4y_short_selector_failure_rollup.json"

    daily.to_csv(daily_path, index=False)
    window_summary.to_csv(window_path, index=False)
    state_summary.to_csv(state_path, index=False)
    bin_summary.to_csv(bin_path, index=False)
    memo_path.write_text(memo, encoding="utf-8")

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "short_variant": short_variant,
        "lead_selector": lead_selector,
        "residual_targets": list(residual_targets),
        "candidate_count": int(candidate_count),
        "loser_quantile": float(loser_quantile),
        "score_bins": int(score_bins),
        "min_regression_rows": int(min_regression_rows),
        "panel_rows": int(len(panel)),
        "daily_rows": int(len(daily)),
        "window_summary_rows": int(len(window_summary)),
        "state_summary_rows": int(len(state_summary)),
        "score_bin_summary_rows": int(len(bin_summary)),
        "daily_artifact": daily_path.as_posix(),
        "window_summary_artifact": window_path.as_posix(),
        "state_summary_artifact": state_path.as_posix(),
        "score_bin_summary_artifact": bin_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "short_selector_window_component_market_state_diagnostics",
        "limitations": [
            "This is a standalone selector diagnostic, not a beta-matched portfolio backtest.",
            "The h10 signal panel keeps historical 5d column names, but values are the h10 probe values.",
            "Future market states use forward benchmark returns for diagnosis only and are not tradable gates.",
            "No transaction costs, borrow costs, or test-window performance are computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    residual_targets: Sequence[str],
    candidate_count: int,
    loser_quantile: float,
    score_bins: int,
    min_regression_rows: int,
) -> None:
    if not residual_targets:
        raise ValueError("At least one residual target is required.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")
    if score_bins < 2:
        raise ValueError("score_bins must be at least 2.")
    if min_regression_rows <= 0:
        raise ValueError("min_regression_rows must be positive.")


def _add_all_selector_scores(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frames.append(_add_selector_scores(group.drop_duplicates("symbol", keep="last")))
    return pd.concat(frames, ignore_index=True) if frames else panel


def _add_market_state(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    session_market = (
        frame[["session_date", BENCHMARK_FORWARD_COLUMN]]
        .drop_duplicates("session_date")
        .dropna(subset=[BENCHMARK_FORWARD_COLUMN])
    )
    stress_cutoff = float(session_market[BENCHMARK_FORWARD_COLUMN].quantile(0.20))
    rebound_cutoff = float(session_market[BENCHMARK_FORWARD_COLUMN].quantile(0.80))

    def tail_state(value: float) -> str:
        if value <= stress_cutoff:
            return "future_stress"
        if value >= rebound_cutoff:
            return "future_rebound"
        return "future_normal"

    session_market["future_market_direction"] = np.where(
        session_market[BENCHMARK_FORWARD_COLUMN] < 0.0,
        "future_down",
        "future_up",
    )
    session_market["future_market_tail_state"] = session_market[
        BENCHMARK_FORWARD_COLUMN
    ].map(tail_state)
    return frame.merge(
        session_market[
            [
                "session_date",
                "future_market_direction",
                "future_market_tail_state",
            ]
        ],
        on="session_date",
        how="left",
    )


def _selector_daily_diagnostics(
    panel: pd.DataFrame,
    *,
    residual_targets: Sequence[str],
    candidate_count: int,
    loser_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selector_catalog = _selector_catalog()
    for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True):
        frame = group.drop_duplicates("symbol", keep="last")
        if len(frame) < candidate_count * 2:
            continue
        market_direction = _first_string(frame, "future_market_direction")
        market_tail_state = _first_string(frame, "future_market_tail_state")
        benchmark_forward = _first_float(frame, BENCHMARK_FORWARD_COLUMN)
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
            universe_mean_target = float(target_frame[target_name].mean())
            universe_negative_share = float((target_frame[target_name] < 0).mean())
            for selector_name, selector_label in selector_catalog:
                subset = target_frame.dropna(subset=[selector_name]).copy()
                if len(subset) < candidate_count * 2:
                    continue
                selected = subset.sort_values(
                    [selector_name, "symbol"],
                    ascending=[False, True],
                ).head(candidate_count)
                target = selected[target_name]
                raw = selected["forward_return_5d"]
                rank_ic = subset[selector_name].corr(subset[target_name], method="spearman")
                selected_mean_target = float(target.mean())
                rows.append(
                    {
                        "session_date": session_date,
                        "variant": variant,
                        "residual_target": target_name,
                        "selector": selector_name,
                        "selector_label": selector_label,
                        "future_market_direction": market_direction,
                        "future_market_tail_state": market_tail_state,
                        "benchmark_forward_return_h10": benchmark_forward,
                        "eligible_names": int(len(subset)),
                        "selected_names": int(len(selected)),
                        "universe_mean_target_return": universe_mean_target,
                        "universe_negative_target_share": universe_negative_share,
                        "selected_mean_target_return": selected_mean_target,
                        "short_contribution_target": float(-selected_mean_target),
                        "selected_negative_target_share": float((target < 0).mean()),
                        "selected_positive_target_share": float((target > 0).mean()),
                        "selected_mean_forward_return_h10": float(raw.mean()),
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
                        "short_edge_vs_universe": float(
                            universe_mean_target - selected_mean_target
                        ),
                        "rank_ic_selector_vs_target": float(rank_ic),
                        "rank_ic_good": bool(rank_ic < 0) if pd.notna(rank_ic) else False,
                        "test_window_used": False,
                    }
                )
    return pd.DataFrame(rows, columns=_daily_columns())


def _window_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_window_summary_columns())
    rows = []
    working = daily.copy()
    working["session_date_dt"] = pd.to_datetime(working["session_date"])
    for window_name, start, end in WINDOWS:
        mask = (working["session_date_dt"] >= pd.Timestamp(start)) & (
            working["session_date_dt"] <= pd.Timestamp(end)
        )
        window = working.loc[mask]
        if window.empty:
            continue
        for keys, group in window.groupby(["variant", "residual_target", "selector"], sort=True):
            rows.append(
                _summary_row(
                    scope_type="window",
                    scope_value=window_name,
                    start=start,
                    end=end,
                    keys=keys,
                    group=group,
                )
            )
    return pd.DataFrame(rows, columns=_window_summary_columns())


def _state_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_state_summary_columns())
    rows = []
    for state_column in ("future_market_direction", "future_market_tail_state"):
        for keys, group in daily.groupby(
            ["variant", "residual_target", "selector", state_column], sort=True
        ):
            variant, target, selector, state = keys
            summary = _summary_row(
                scope_type=state_column,
                scope_value=state,
                start="",
                end="",
                keys=(variant, target, selector),
                group=group,
            )
            rows.append(summary)
    return pd.DataFrame(rows, columns=_state_summary_columns())


def _score_bin_summary(
    panel: pd.DataFrame,
    *,
    selectors: Sequence[str],
    residual_targets: Sequence[str],
    score_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    working = panel.copy()
    working["session_date_dt"] = pd.to_datetime(working["session_date"])
    for window_name, start, end in WINDOWS:
        window = working[
            (working["session_date_dt"] >= pd.Timestamp(start))
            & (working["session_date_dt"] <= pd.Timestamp(end))
        ]
        if window.empty:
            continue
        for (variant, session_date), group in window.groupby(
            ["variant", "session_date"], sort=True
        ):
            for selector in selectors:
                if selector not in group.columns:
                    continue
                base = group.dropna(subset=[selector]).copy()
                if len(base) < score_bins * 2:
                    continue
                base["score_bin"] = _score_bins(base[selector], score_bins)
                for target_name in residual_targets:
                    subset = base.dropna(subset=[target_name, "score_bin"])
                    if subset.empty:
                        continue
                    for score_bin, bin_group in subset.groupby("score_bin", sort=True):
                        rows.append(
                            {
                                "window": window_name,
                                "start": start,
                                "end": end,
                                "variant": variant,
                                "session_date": session_date,
                                "residual_target": target_name,
                                "selector": selector,
                                "score_bin": int(score_bin),
                                "names": int(len(bin_group)),
                                "mean_target_return": float(bin_group[target_name].mean()),
                                "short_contribution_target": float(
                                    -bin_group[target_name].mean()
                                ),
                                "negative_target_share": float(
                                    (bin_group[target_name] < 0).mean()
                                ),
                                "test_window_used": False,
                            }
                        )
    daily_bins = pd.DataFrame(rows, columns=_score_bin_daily_columns())
    if daily_bins.empty:
        return pd.DataFrame(columns=_score_bin_summary_columns())
    summary = (
        daily_bins.groupby(
            ["window", "start", "end", "variant", "residual_target", "selector", "score_bin"],
            sort=True,
        )
        .agg(
            sessions=("session_date", "nunique"),
            mean_names=("names", "mean"),
            mean_target_return=("mean_target_return", "mean"),
            mean_short_contribution_target=("short_contribution_target", "mean"),
            mean_negative_target_share=("negative_target_share", "mean"),
            test_window_used=("test_window_used", "any"),
        )
        .reset_index()
    )
    return summary[_score_bin_summary_columns()]


def _score_bins(values: pd.Series, score_bins: int) -> pd.Series:
    ranks = values.rank(method="first", ascending=True)
    bins = pd.qcut(ranks, q=score_bins, labels=False, duplicates="drop")
    return bins.astype(float) + 1.0


def _summary_row(
    *,
    scope_type: str,
    scope_value: str,
    start: str,
    end: str,
    keys: tuple[Any, Any, Any],
    group: pd.DataFrame,
) -> dict[str, Any]:
    variant, target, selector = keys
    contribution = group["short_contribution_target"]
    oracle_gap = group["oracle_short_contribution_target"] - contribution
    rank_ic = group["rank_ic_selector_vs_target"]
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "start": start,
        "end": end,
        "variant": variant,
        "residual_target": target,
        "selector": selector,
        "selector_label": group["selector_label"].iloc[0],
        "sessions": int(len(group)),
        "mean_benchmark_forward_return_h10": float(
            group["benchmark_forward_return_h10"].mean()
        ),
        "mean_universe_target_return": float(group["universe_mean_target_return"].mean()),
        "mean_short_contribution_target": float(contribution.mean()),
        "short_target_hit_rate": float((contribution > 0).mean()),
        "mean_selected_negative_target_share": float(
            group["selected_negative_target_share"].mean()
        ),
        "mean_selected_positive_target_share": float(
            group["selected_positive_target_share"].mean()
        ),
        "mean_short_edge_vs_universe": float(group["short_edge_vs_universe"].mean()),
        "mean_short_contribution_return": float(group["short_contribution_return"].mean()),
        "mean_oracle_short_contribution_target": float(
            group["oracle_short_contribution_target"].mean()
        ),
        "mean_oracle_gap_target": float(oracle_gap.mean()),
        "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
        "mean_loser_quantile_capture_rate": float(
            group["loser_quantile_capture_rate"].mean()
        ),
        "mean_rank_ic_selector_vs_target": float(rank_ic.mean()),
        "rank_ic_good_share": float(group["rank_ic_good"].mean()),
        "rank_ic_bad_share": float((rank_ic > 0).mean()),
        "test_window_used": bool(group["test_window_used"].any()),
    }


def _memo(
    lead_window: pd.DataFrame,
    window_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    bin_summary: pd.DataFrame,
    *,
    short_variant: str,
    lead_selector: str,
    candidate_count: int,
    score_bins: int,
) -> str:
    lines = [
        "# Phase4Y Short Selector Failure Diagnostics",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- short universe variant: `{short_variant}`;",
        f"- lead selector: `{lead_selector}`;",
        f"- selected names per standalone diagnostic: `{candidate_count}`;",
        f"- score bins per session: `{score_bins}`;",
        "- horizon: h10 validation panel; legacy columns still include `5d` in names;",
        "- lockbox/test window: not used.",
        "",
        "## Lead Selector Window Summary",
        "",
        "| window | sessions | short contribution target | hit rate | selected positive share | rank IC | bad IC share | oracle gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in lead_window.sort_values(["scope_value"]).iterrows():
        lines.append(
            f"| {row['scope_value']} | {int(row['sessions'])} | "
            f"{_fmt_bps(row['mean_short_contribution_target'])} | "
            f"{_fmt_pct(row['short_target_hit_rate'])} | "
            f"{_fmt_pct(row['mean_selected_positive_target_share'])} | "
            f"{_fmt_float(row['mean_rank_ic_selector_vs_target'])} | "
            f"{_fmt_pct(row['rank_ic_bad_share'])} | "
            f"{_fmt_bps(row['mean_oracle_gap_target'])} |"
        )

    component = window_summary[
        (window_summary["residual_target"] == TARGET_BETA_RESIDUAL)
        & (window_summary["scope_value"].isin(["2019_mid", "2019_may_drawdown"]))
    ].copy()
    component = component.sort_values(
        ["scope_value", "mean_short_contribution_target", "selector"],
        ascending=[True, False, True],
    )
    lines.extend(
        [
            "",
            "## 2019 Component Ranking",
            "",
            "| window | selector | short contribution target | hit rate | rank IC | positive selected share |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in component.iterrows():
        lines.append(
            f"| {row['scope_value']} | {row['selector']} | "
            f"{_fmt_bps(row['mean_short_contribution_target'])} | "
            f"{_fmt_pct(row['short_target_hit_rate'])} | "
            f"{_fmt_float(row['mean_rank_ic_selector_vs_target'])} | "
            f"{_fmt_pct(row['mean_selected_positive_target_share'])} |"
        )

    state = state_summary[
        (state_summary["residual_target"] == TARGET_BETA_RESIDUAL)
        & (state_summary["selector"] == lead_selector)
    ].copy()
    state = state.sort_values(["scope_type", "scope_value"])
    lines.extend(
        [
            "",
            "## Lead Selector Market-State Summary",
            "",
            "| state type | state | sessions | short contribution target | hit rate | rank IC | positive selected share |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in state.iterrows():
        lines.append(
            f"| {row['scope_type']} | {row['scope_value']} | {int(row['sessions'])} | "
            f"{_fmt_bps(row['mean_short_contribution_target'])} | "
            f"{_fmt_pct(row['short_target_hit_rate'])} | "
            f"{_fmt_float(row['mean_rank_ic_selector_vs_target'])} | "
            f"{_fmt_pct(row['mean_selected_positive_target_share'])} |"
        )

    lead_bins = bin_summary[
        (bin_summary["selector"] == lead_selector)
        & (bin_summary["residual_target"] == TARGET_BETA_RESIDUAL)
        & (bin_summary["window"].isin(["full", "2019_mid", "2019_may_drawdown"]))
    ].copy()
    lead_bins = lead_bins.sort_values(["window", "score_bin"])
    lines.extend(
        [
            "",
            "## Lead Selector Score-Bin Shape",
            "",
            "Higher score bins are names the short selector prefers more. For a useful short selector, higher bins should have lower future residual return.",
            "",
            "| window | score bin | short contribution target | negative target share |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in lead_bins.iterrows():
        lines.append(
            f"| {row['window']} | {int(row['score_bin'])} | "
            f"{_fmt_bps(row['mean_short_contribution_target'])} | "
            f"{_fmt_pct(row['mean_negative_target_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive `short contribution target` means selected shorts had negative future residual return; negative values mean the short selector picked names that rose on the residual target.",
            "- Negative rank IC is good for a short selector because higher selector score should correspond to lower future residual return.",
            "- `future_market_*` states use forward benchmark returns for diagnostics only; they are not tradable gates.",
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
        "future_market_direction",
        "future_market_tail_state",
        "benchmark_forward_return_h10",
        "eligible_names",
        "selected_names",
        "universe_mean_target_return",
        "universe_negative_target_share",
        "selected_mean_target_return",
        "short_contribution_target",
        "selected_negative_target_share",
        "selected_positive_target_share",
        "selected_mean_forward_return_h10",
        "short_contribution_return",
        "oracle_short_contribution_target",
        "oracle_overlap_rate",
        "loser_quantile_capture_rate",
        "short_edge_vs_universe",
        "rank_ic_selector_vs_target",
        "rank_ic_good",
        "test_window_used",
    ]


def _window_summary_columns() -> list[str]:
    return [
        "scope_type",
        "scope_value",
        "start",
        "end",
        "variant",
        "residual_target",
        "selector",
        "selector_label",
        "sessions",
        "mean_benchmark_forward_return_h10",
        "mean_universe_target_return",
        "mean_short_contribution_target",
        "short_target_hit_rate",
        "mean_selected_negative_target_share",
        "mean_selected_positive_target_share",
        "mean_short_edge_vs_universe",
        "mean_short_contribution_return",
        "mean_oracle_short_contribution_target",
        "mean_oracle_gap_target",
        "mean_oracle_overlap_rate",
        "mean_loser_quantile_capture_rate",
        "mean_rank_ic_selector_vs_target",
        "rank_ic_good_share",
        "rank_ic_bad_share",
        "test_window_used",
    ]


def _state_summary_columns() -> list[str]:
    return _window_summary_columns()


def _score_bin_daily_columns() -> list[str]:
    return [
        "window",
        "start",
        "end",
        "variant",
        "session_date",
        "residual_target",
        "selector",
        "score_bin",
        "names",
        "mean_target_return",
        "short_contribution_target",
        "negative_target_share",
        "test_window_used",
    ]


def _score_bin_summary_columns() -> list[str]:
    return [
        "window",
        "start",
        "end",
        "variant",
        "residual_target",
        "selector",
        "score_bin",
        "sessions",
        "mean_names",
        "mean_target_return",
        "mean_short_contribution_target",
        "mean_negative_target_share",
        "test_window_used",
    ]


def _first_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""


def _first_float(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def _fmt_bps(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 10000:.2f} bps"


def _fmt_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase4Y short selector failure diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--short-variant", default=DEFAULT_SHORT_VARIANT)
    parser.add_argument("--lead-selector", default=DEFAULT_SHORT_SELECTOR)
    parser.add_argument("--residual-target", action="append", dest="residual_targets")
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    args = parser.parse_args(argv)

    result = build_phase4y_short_selector_failure_diagnostics(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        short_variant=args.short_variant,
        lead_selector=args.lead_selector,
        residual_targets=tuple(args.residual_targets or DEFAULT_TARGETS),
        candidate_count=args.candidate_count,
        loser_quantile=args.loser_quantile,
        score_bins=args.score_bins,
        min_regression_rows=args.min_regression_rows,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
