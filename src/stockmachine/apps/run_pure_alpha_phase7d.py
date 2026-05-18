"""Phase7D signed graph-prior reversal payoff experiment.

Phase7D keeps the target as reversal payoff:

    long recent losers - short recent winners

but allows the graph to choose signed exposure. Positive exposure trades
reversal, zero exposure stays flat, and negative exposure trades continuation.
This is validation-only and does not read the test lockbox.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_SEC_SUBMISSIONS_DIR,
)
from stockmachine.apps.run_pure_alpha_phase7b import (
    DEFAULT_END,
    DEFAULT_EVAL_START,
    DEFAULT_FUNDAMENTAL_PANEL,
    DEFAULT_NON_PRICE_PANEL,
    DEFAULT_QUANTILE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SIZE_PANEL,
    DEFAULT_START,
    DEFAULT_VARIANT,
    EdgeSpec,
    _basket_spread,
    _hhi,
    _load_joined_panel,
    _markdown_table,
    _rolling_zscore,
    _safe_corr,
    _sector_concentration_gap,
    _t_stat,
    _weighted_mean,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7d_signed_graph_reversal_payoff_20260517"
DEFAULT_WALK_FORWARD_WINDOW = 252
DEFAULT_WALK_FORWARD_MIN_HISTORY = 252


@dataclass(frozen=True)
class ScoreSpec:
    score_column: str
    label: str


SIGNED_EDGE_SPECS = (
    EdgeSpec(
        "reversal_dislocation",
        "mispricing_convergence",
        1.0,
        0.25,
        "overreaction",
        "Large recent winner-loser spread can create room for reversal.",
    ),
    EdgeSpec(
        "fundamental_fragility_spread",
        "loser_falling_knife_risk",
        -1.0,
        0.28,
        "fundamental_stress",
        "Fragile loser baskets are more likely continuation than reversal.",
    ),
    EdgeSpec(
        "liquidity_fragility",
        "loser_liquidity_stress",
        -1.0,
        0.16,
        "liquidity",
        "Less-liquid loser baskets are exposed to forced selling.",
    ),
    EdgeSpec(
        "beta_fragility",
        "risk_off_fragility",
        -1.0,
        0.12,
        "risk_exposure",
        "High-beta losers are vulnerable in risk-off windows.",
    ),
    EdgeSpec(
        "trend_dominance",
        "winner_return_continuation",
        -1.0,
        0.18,
        "style_persistence",
        "Intermediate winner-minus-loser momentum implies continuation risk.",
    ),
    EdgeSpec(
        "capital_concentration",
        "capital_concentration_regime",
        -1.0,
        0.16,
        "fund_flow_proxy",
        "Size-weighted strength over equal-weight strength implies concentration.",
    ),
    EdgeSpec(
        "breadth_weakness",
        "narrow_market_leadership",
        -1.0,
        0.10,
        "market_breadth",
        "Weak breadth makes winner continuation more plausible.",
    ),
    EdgeSpec(
        "sector_shock_concentration",
        "sector_specific_loser_shock",
        -1.0,
        0.10,
        "industry_graph",
        "Sector-concentrated losers may reflect real sector deterioration.",
    ),
    EdgeSpec(
        "winner_fundamental_resilience",
        "winner_fundamental_upgrade",
        -1.0,
        0.12,
        "fundamental_strength",
        "Winner baskets with better fundamentals support rational continuation.",
    ),
    EdgeSpec(
        "winner_profit_resilience",
        "winner_fundamental_upgrade",
        -1.0,
        0.10,
        "fundamental_strength",
        "Winner baskets with less profit stress support rational continuation.",
    ),
    EdgeSpec(
        "loser_fundamental_stress",
        "loser_fundamental_deterioration",
        -1.0,
        0.12,
        "fundamental_stress",
        "Losers with worse absolute fundamental stress are risky reversal longs.",
    ),
    EdgeSpec(
        "winner_size_leadership",
        "mega_cap_leadership",
        -1.0,
        0.08,
        "capital_concentration",
        "Winner baskets led by larger firms can reflect persistent leadership.",
    ),
    EdgeSpec(
        "winner_liquidity_leadership",
        "institutional_winner_accumulation",
        -1.0,
        0.08,
        "liquidity",
        "More liquid winners can absorb sustained institutional demand.",
    ),
)

SIGNED_CORE_SOURCES = frozenset(
    {
        "fundamental_fragility_spread",
        "reversal_dislocation",
        "liquidity_fragility",
        "beta_fragility",
    }
)

CONCENTRATION_SOURCES = frozenset(
    {
        "trend_dominance",
        "capital_concentration",
        "breadth_weakness",
        "winner_fundamental_resilience",
        "winner_profit_resilience",
        "winner_size_leadership",
        "winner_liquidity_leadership",
    }
)

SCORE_SPECS = (
    ScoreSpec("signed_core_score", "reliable-core signed graph"),
    ScoreSpec("signed_full_graph_score", "expanded signed graph"),
    ScoreSpec("concentration_signed_score", "concentration and winner-upgrade signed graph"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase7D signed graph-prior reversal payoff test."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--fundamental-panel-path", default=str(DEFAULT_FUNDAMENTAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-bottom-quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--walk-forward-window", type=int, default=DEFAULT_WALK_FORWARD_WINDOW)
    parser.add_argument(
        "--walk-forward-min-history",
        type=int,
        default=DEFAULT_WALK_FORWARD_MIN_HISTORY,
    )
    args = parser.parse_args(argv)

    rollup = build_phase7d_signed_graph_reversal_payoff(
        signal_panel_path=args.signal_panel_path,
        size_panel_path=args.size_panel_path,
        non_price_panel_path=args.non_price_panel_path,
        fundamental_panel_path=args.fundamental_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        top_bottom_quantile=args.top_bottom_quantile,
        rolling_window=args.rolling_window,
        walk_forward_window=args.walk_forward_window,
        walk_forward_min_history=args.walk_forward_min_history,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7d_signed_graph_reversal_payoff(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    size_panel_path: str | Path = DEFAULT_SIZE_PANEL,
    non_price_panel_path: str | Path = DEFAULT_NON_PRICE_PANEL,
    fundamental_panel_path: str | Path = DEFAULT_FUNDAMENTAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    top_bottom_quantile: float = DEFAULT_QUANTILE,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    walk_forward_window: int = DEFAULT_WALK_FORWARD_WINDOW,
    walk_forward_min_history: int = DEFAULT_WALK_FORWARD_MIN_HISTORY,
) -> dict[str, Any]:
    if walk_forward_window < 2:
        raise ValueError("walk_forward_window must be at least 2.")
    if walk_forward_min_history < 2:
        raise ValueError("walk_forward_min_history must be at least 2.")
    if walk_forward_min_history > walk_forward_window:
        raise ValueError("walk_forward_min_history cannot exceed walk_forward_window.")

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_joined_panel(
        signal_panel_path=signal_panel_path,
        size_panel_path=size_panel_path,
        non_price_panel_path=non_price_panel_path,
        fundamental_panel_path=fundamental_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        variant=variant,
        start=start,
        end=end,
    )
    daily = _build_signed_daily_states(panel, top_bottom_quantile=top_bottom_quantile)
    scored = _score_signed_graph(daily, rolling_window=rolling_window)
    scored = _add_walk_forward_thresholds(
        scored,
        window=walk_forward_window,
        min_history=walk_forward_min_history,
    )
    eval_frame = scored[
        scored["session_date"].ge(eval_start) & scored["session_date"].le(end)
    ].copy()

    decision_metrics = _decision_metrics(eval_frame)
    action_metrics = _action_metrics(eval_frame)
    score_bins = _score_bins(eval_frame)
    contribution_diagnostics = _contribution_diagnostics(eval_frame)
    nodes = _graph_nodes()
    edges = _graph_edges()

    daily_path = output_dir / "phase7d_daily_signed_states.csv"
    nodes_path = output_dir / "phase7d_graph_nodes.csv"
    edges_path = output_dir / "phase7d_graph_edges.csv"
    decision_path = output_dir / "phase7d_decision_metrics.csv"
    action_path = output_dir / "phase7d_action_metrics.csv"
    bins_path = output_dir / "phase7d_score_bins.csv"
    contributions_path = output_dir / "phase7d_contribution_diagnostics.csv"
    memo_path = output_dir / "phase7d_signed_graph_reversal_payoff_memo.md"
    rollup_path = output_dir / "phase7d_rollup.json"

    scored.to_csv(daily_path, index=False)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    decision_metrics.to_csv(decision_path, index=False)
    action_metrics.to_csv(action_path, index=False)
    score_bins.to_csv(bins_path, index=False)
    contribution_diagnostics.to_csv(contributions_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            rolling_window=rolling_window,
            walk_forward_window=walk_forward_window,
            walk_forward_min_history=walk_forward_min_history,
            panel=panel,
            decision_metrics=decision_metrics,
            action_metrics=action_metrics,
            score_bins=score_bins,
            contributions=contribution_diagnostics,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7d_signed_graph_reversal_payoff",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "rolling_window": int(rolling_window),
        "walk_forward_window": int(walk_forward_window),
        "walk_forward_min_history": int(walk_forward_min_history),
        "rows": {
            "panel": int(len(panel)),
            "daily": int(len(scored)),
            "eval_daily": int(len(eval_frame)),
            "nodes": int(len(nodes)),
            "edges": int(len(edges)),
        },
        "outputs": {
            "daily_signed_states": daily_path.as_posix(),
            "graph_nodes": nodes_path.as_posix(),
            "graph_edges": edges_path.as_posix(),
            "decision_metrics": decision_path.as_posix(),
            "action_metrics": action_path.as_posix(),
            "score_bins": bins_path.as_posix(),
            "contribution_diagnostics": contributions_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_signed_daily_states(
    panel: pd.DataFrame,
    *,
    top_bottom_quantile: float,
) -> pd.DataFrame:
    frame = panel.copy()
    frame = frame.dropna(subset=[TARGET_COLUMN, "forward_return_5d", "beta"])
    rows: list[dict[str, Any]] = []
    for session_date, group in frame.groupby("session_date", sort=True):
        group = group.dropna(subset=["reversal_5d", TARGET_COLUMN, "return_5d"]).copy()
        if len(group) < 100:
            continue
        low = group["reversal_5d"].quantile(top_bottom_quantile)
        high = group["reversal_5d"].quantile(1.0 - top_bottom_quantile)
        losers = group[group["reversal_5d"] >= high].copy()
        winners = group[group["reversal_5d"] <= low].copy()
        if losers.empty or winners.empty:
            continue

        payoff = float(losers[TARGET_COLUMN].mean() - winners[TARGET_COLUMN].mean())
        row = {
            "session_date": str(session_date),
            "names": int(len(group)),
            "loser_count": int(len(losers)),
            "winner_count": int(len(winners)),
            "reversal_payoff": payoff,
            "reversal_payoff_bps": payoff * 10000.0,
            "reversal_payoff_positive": bool(payoff > 0),
            "loser_forward_residual_mean": float(losers[TARGET_COLUMN].mean()),
            "winner_forward_residual_mean": float(winners[TARGET_COLUMN].mean()),
            "market_breadth_5d": float((group["return_5d"] > 0).mean()),
            "market_equal_weight_return_5d": float(group["return_5d"].mean()),
            "market_size_weight_return_5d": _weighted_mean(group, "return_5d", "market_cap"),
            "reversal_dislocation": float(winners["return_5d"].mean() - losers["return_5d"].mean()),
            "trend_dominance": float(winners["momentum_20d"].mean() - losers["momentum_20d"].mean()),
            "liquidity_fragility": float(winners["log_adv"].mean() - losers["log_adv"].mean()),
            "beta_fragility": float(losers["beta"].mean() - winners["beta"].mean()),
            "small_loser_tilt": float(winners["market_cap_log_z"].mean() - losers["market_cap_log_z"].mean()),
            "fundamental_fragility_spread": _basket_spread(
                losers,
                winners,
                "fundamental_fragility_score",
            ),
            "fundamental_leverage_spread": _basket_spread(
                losers,
                winners,
                "fundamental_leverage_pressure_score",
            ),
            "fundamental_profit_stress_spread": _basket_spread(
                losers,
                winners,
                "fundamental_profit_stress_score",
            ),
            "filing_red_flag_spread": _basket_spread(losers, winners, "filing_red_flag_score"),
            "insider_support_spread": _basket_spread(losers, winners, "insider_net_buy_score"),
            "sector_shock_concentration": _sector_concentration_gap(group, losers),
            "winner_sector_concentration": _winner_sector_concentration_gap(group, winners),
            "test_window_used": False,
        }
        row["capital_concentration"] = (
            row["market_size_weight_return_5d"] - row["market_equal_weight_return_5d"]
            if np.isfinite(row["market_size_weight_return_5d"])
            else np.nan
        )
        row["breadth_weakness"] = 0.5 - row["market_breadth_5d"]
        row["winner_fundamental_resilience"] = _relative_resilience(
            group,
            winners,
            "fundamental_fragility_score",
        )
        row["winner_profit_resilience"] = _relative_resilience(
            group,
            winners,
            "fundamental_profit_stress_score",
        )
        row["loser_fundamental_stress"] = _relative_stress(
            group,
            losers,
            "fundamental_fragility_score",
        )
        row["loser_profit_stress"] = _relative_stress(
            group,
            losers,
            "fundamental_profit_stress_score",
        )
        row["winner_size_leadership"] = _basket_minus_universe(
            group,
            winners,
            "market_cap_log_z",
        )
        row["winner_liquidity_leadership"] = _basket_minus_universe(
            group,
            winners,
            "log_adv",
        )
        row["winner_momentum_leadership"] = _basket_minus_universe(
            group,
            winners,
            "momentum_20d",
        )
        row["loser_momentum_weakness"] = _universe_minus_basket(
            group,
            losers,
            "momentum_20d",
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)


def _score_signed_graph(
    daily: pd.DataFrame,
    *,
    rolling_window: int,
) -> pd.DataFrame:
    frame = daily.copy().sort_values("session_date").reset_index(drop=True)
    full_score = pd.Series(0.0, index=frame.index)
    core_score = pd.Series(0.0, index=frame.index)
    concentration_score = pd.Series(0.0, index=frame.index)

    for edge in SIGNED_EDGE_SPECS:
        z_col = f"z_{edge.source}"
        contrib_col = f"signed_contrib_{edge.source}"
        frame[z_col] = _rolling_zscore(frame[edge.source], window=rolling_window)
        contribution = edge.sign * edge.weight * frame[z_col].fillna(0.0)
        frame[contrib_col] = contribution
        full_score = full_score + contribution
        if edge.source in SIGNED_CORE_SOURCES:
            core_score = core_score + contribution
        if edge.source in CONCENTRATION_SOURCES:
            concentration_score = concentration_score + contribution

    frame["signed_full_graph_score"] = _normalize_score(full_score, SIGNED_EDGE_SPECS)
    frame["signed_core_score"] = _normalize_score(
        core_score,
        [edge for edge in SIGNED_EDGE_SPECS if edge.source in SIGNED_CORE_SOURCES],
    )
    frame["concentration_signed_score"] = _normalize_score(
        concentration_score,
        [edge for edge in SIGNED_EDGE_SPECS if edge.source in CONCENTRATION_SOURCES],
    )
    for spec in SCORE_SPECS:
        frame[f"{spec.score_column}_probability"] = 1.0 / (
            1.0 + np.exp(-frame[spec.score_column])
        )
    return frame


def _add_walk_forward_thresholds(
    frame: pd.DataFrame,
    *,
    window: int,
    min_history: int,
) -> pd.DataFrame:
    out = frame.copy().sort_values("session_date").reset_index(drop=True)
    for spec in SCORE_SPECS:
        score = pd.to_numeric(out[spec.score_column], errors="coerce")
        historical_score = score.shift(1)
        rolling = historical_score.rolling(window, min_periods=min_history)
        out[f"{spec.score_column}_wf_q20"] = rolling.quantile(0.20)
        out[f"{spec.score_column}_wf_q33"] = rolling.quantile(1.0 / 3.0)
        out[f"{spec.score_column}_wf_q67"] = rolling.quantile(2.0 / 3.0)
        out[f"{spec.score_column}_wf_q80"] = rolling.quantile(0.80)
    return out


def _decision_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_score_metric_row(frame, "always_on", pd.Series(True, index=frame.index), "")]
    for spec in SCORE_SPECS:
        score = frame[spec.score_column]
        rows.extend(
            [
                _score_metric_row(frame, f"{spec.score_column}_positive", score > 0, spec.score_column),
                _score_metric_row(
                    frame,
                    f"{spec.score_column}_walk_forward_top_tercile",
                    score.ge(frame[f"{spec.score_column}_wf_q67"])
                    & frame[f"{spec.score_column}_wf_q67"].notna(),
                    spec.score_column,
                ),
                _score_metric_row(
                    frame,
                    f"{spec.score_column}_walk_forward_bottom_tercile",
                    score.le(frame[f"{spec.score_column}_wf_q33"])
                    & frame[f"{spec.score_column}_wf_q33"].notna(),
                    spec.score_column,
                ),
            ]
        )
    out = pd.DataFrame(rows)
    for spec in SCORE_SPECS:
        out[f"{spec.score_column}_spearman_vs_payoff"] = _safe_corr(
            frame[spec.score_column],
            frame["reversal_payoff"],
            method="spearman",
        )
    out["test_window_used"] = False
    return out


def _action_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SCORE_SPECS:
        ready = frame[f"{spec.score_column}_wf_q67"].notna() & frame[
            f"{spec.score_column}_wf_q33"
        ].notna()
        ready_frame = frame[ready].copy()
        if ready_frame.empty:
            continue
        score = ready_frame[spec.score_column]
        q33 = ready_frame[f"{spec.score_column}_wf_q33"]
        q67 = ready_frame[f"{spec.score_column}_wf_q67"]
        q20 = ready_frame[f"{spec.score_column}_wf_q20"]
        q80 = ready_frame[f"{spec.score_column}_wf_q80"]
        top = score.ge(q67)
        bottom = score.le(q33)
        extreme_top = score.ge(q80)
        extreme_bottom = score.le(q20)

        actions: dict[str, pd.Series] = {
            f"{spec.score_column}__always_on_ready": pd.Series(1.0, index=ready_frame.index),
            f"{spec.score_column}__stop_bottom_tercile": (~bottom).astype(float),
            f"{spec.score_column}__top_tercile_only": top.astype(float),
            f"{spec.score_column}__reverse_bottom_tercile_only": -bottom.astype(float),
            f"{spec.score_column}__signed_top_bottom_tercile": top.astype(float)
            - bottom.astype(float),
            f"{spec.score_column}__signed_extreme_quintile": extreme_top.astype(float)
            - extreme_bottom.astype(float),
            f"{spec.score_column}__soft_stop_125_075_025": _piecewise_action(
                top,
                bottom,
                top_value=1.25,
                middle_value=0.75,
                bottom_value=0.25,
            ),
            f"{spec.score_column}__soft_signed_125_075_neg025": _piecewise_action(
                top,
                bottom,
                top_value=1.25,
                middle_value=0.75,
                bottom_value=-0.25,
            ),
        }
        always_mean = float(ready_frame["reversal_payoff_bps"].mean())
        for rule, action in actions.items():
            rows.append(
                _action_metric_row(
                    ready_frame,
                    score_column=spec.score_column,
                    rule=rule,
                    action=action,
                    always_ready_mean_bps=always_mean,
                    total_eval_sessions=len(frame),
                )
            )
    out = pd.DataFrame(rows)
    out["test_window_used"] = False
    return out


def _score_metric_row(
    frame: pd.DataFrame,
    rule: str,
    mask: pd.Series,
    score_column: str,
) -> dict[str, Any]:
    selected = frame[mask.fillna(False)].copy()
    payoff = selected["reversal_payoff"].astype(float)
    all_payoff = frame["reversal_payoff"].astype(float)
    return {
        "rule": rule,
        "sessions": int(len(selected)),
        "coverage": float(len(selected) / len(frame)) if len(frame) else np.nan,
        "mean_payoff_bps": float(payoff.mean() * 10000.0) if len(payoff) else np.nan,
        "median_payoff_bps": float(payoff.median() * 10000.0) if len(payoff) else np.nan,
        "hit_rate": float((payoff > 0).mean()) if len(payoff) else np.nan,
        "t_stat": _t_stat(payoff),
        "mean_payoff_lift_vs_always_bps": float(
            (payoff.mean() - all_payoff.mean()) * 10000.0
        )
        if len(payoff)
        else np.nan,
        "score_mean": float(selected[score_column].mean())
        if score_column and len(selected)
        else np.nan,
    }


def _action_metric_row(
    frame: pd.DataFrame,
    *,
    score_column: str,
    rule: str,
    action: pd.Series,
    always_ready_mean_bps: float,
    total_eval_sessions: int,
) -> dict[str, Any]:
    aligned_action = action.reindex(frame.index).fillna(0.0).astype(float)
    signed_payoff_bps = aligned_action * frame["reversal_payoff_bps"].astype(float)
    active = aligned_action.abs() > 0.0
    reverse = aligned_action < 0.0
    long_reversal = aligned_action > 0.0
    gross_exposure = aligned_action.abs()
    return {
        "score": score_column,
        "rule": rule,
        "ready_sessions": int(len(frame)),
        "total_eval_sessions": int(total_eval_sessions),
        "ready_coverage_of_eval": float(len(frame) / total_eval_sessions)
        if total_eval_sessions
        else np.nan,
        "active_sessions": int(active.sum()),
        "long_reversal_sessions": int(long_reversal.sum()),
        "reverse_continuation_sessions": int(reverse.sum()),
        "flat_sessions": int((~active).sum()),
        "active_coverage_of_ready": float(active.mean()) if len(active) else np.nan,
        "mean_gross_exposure": float(gross_exposure.mean()) if len(gross_exposure) else np.nan,
        "calendar_mean_signed_payoff_bps": float(signed_payoff_bps.mean()),
        "active_mean_signed_payoff_bps": float(signed_payoff_bps[active].mean())
        if active.any()
        else np.nan,
        "active_hit_rate": float((signed_payoff_bps[active] > 0).mean())
        if active.any()
        else np.nan,
        "calendar_t_stat": _t_stat(signed_payoff_bps / 10000.0),
        "active_t_stat": _t_stat((signed_payoff_bps[active] / 10000.0))
        if active.any()
        else np.nan,
        "always_on_ready_mean_payoff_bps": float(always_ready_mean_bps),
        "calendar_lift_vs_always_ready_bps": float(
            signed_payoff_bps.mean() - always_ready_mean_bps
        ),
        "reverse_underlying_reversal_mean_bps": float(
            frame.loc[reverse, "reversal_payoff_bps"].mean()
        )
        if reverse.any()
        else np.nan,
        "reverse_action_mean_payoff_bps": float(signed_payoff_bps[reverse].mean())
        if reverse.any()
        else np.nan,
    }


def _score_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SCORE_SPECS:
        valid = frame.dropna(subset=[spec.score_column, "reversal_payoff"]).copy()
        if len(valid) < 10:
            continue
        valid["score_bin"] = pd.qcut(
            valid[spec.score_column],
            q=min(5, len(valid)),
            labels=False,
            duplicates="drop",
        )
        for score_bin, group in valid.groupby("score_bin", sort=True):
            payoff = group["reversal_payoff"].astype(float)
            rows.append(
                {
                    "score": spec.score_column,
                    "score_bin": int(score_bin),
                    "sessions": int(len(group)),
                    "score_min": float(group[spec.score_column].min()),
                    "score_max": float(group[spec.score_column].max()),
                    "score_mean": float(group[spec.score_column].mean()),
                    "mean_payoff_bps": float(payoff.mean() * 10000.0),
                    "hit_rate": float((payoff > 0).mean()),
                    "t_stat": _t_stat(payoff),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _contribution_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edge in SIGNED_EDGE_SPECS:
        contribution = frame[f"signed_contrib_{edge.source}"]
        rows.append(
            {
                "source": edge.source,
                "mechanism": edge.mechanism,
                "sign_to_reversal_payoff": edge.sign,
                "weight": edge.weight,
                "mean_contribution": float(contribution.mean()),
                "mean_abs_contribution": float(contribution.abs().mean()),
                "spearman_vs_payoff": _safe_corr(
                    contribution,
                    frame["reversal_payoff"],
                    method="spearman",
                ),
                "pearson_vs_payoff": _safe_corr(
                    contribution,
                    frame["reversal_payoff"],
                    method="pearson",
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_vs_payoff", ascending=False)


def _graph_nodes() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "node": "ReversalPayoff",
            "node_type": "target",
            "description": "Future loser-minus-winner beta-residual payoff.",
        }
    ]
    for edge in SIGNED_EDGE_SPECS:
        rows.append(
            {
                "node": edge.source,
                "node_type": "structural_feature",
                "description": edge.description,
            }
        )
        rows.append(
            {
                "node": edge.mechanism,
                "node_type": "mechanism",
                "description": edge.channel,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("node").reset_index(drop=True)


def _graph_edges() -> pd.DataFrame:
    rows = []
    for edge in SIGNED_EDGE_SPECS:
        rows.append(
            {
                "source": edge.source,
                "target": edge.mechanism,
                "edge_type": "feature_to_mechanism",
                "sign_to_reversal_payoff": edge.sign,
                "weight": edge.weight,
                "channel": edge.channel,
                "description": edge.description,
            }
        )
        rows.append(
            {
                "source": edge.mechanism,
                "target": "ReversalPayoff",
                "edge_type": "mechanism_to_target",
                "sign_to_reversal_payoff": edge.sign,
                "weight": edge.weight,
                "channel": edge.channel,
                "description": edge.description,
            }
        )
    return pd.DataFrame(rows)


def _piecewise_action(
    top: pd.Series,
    bottom: pd.Series,
    *,
    top_value: float,
    middle_value: float,
    bottom_value: float,
) -> pd.Series:
    action = pd.Series(middle_value, index=top.index, dtype=float)
    action.loc[top] = top_value
    action.loc[bottom] = bottom_value
    return action


def _normalize_score(score: pd.Series, edges: Sequence[EdgeSpec]) -> pd.Series:
    scale = float(np.sqrt(sum(edge.weight**2 for edge in edges)))
    return score / scale if scale > 0 else score


def _basket_minus_universe(group: pd.DataFrame, basket: pd.DataFrame, column: str) -> float:
    return _numeric_mean(basket, column) - _numeric_mean(group, column)


def _universe_minus_basket(group: pd.DataFrame, basket: pd.DataFrame, column: str) -> float:
    return _numeric_mean(group, column) - _numeric_mean(basket, column)


def _relative_resilience(group: pd.DataFrame, basket: pd.DataFrame, stress_column: str) -> float:
    return _numeric_mean(group, stress_column) - _numeric_mean(basket, stress_column)


def _relative_stress(group: pd.DataFrame, basket: pd.DataFrame, stress_column: str) -> float:
    return _numeric_mean(basket, stress_column) - _numeric_mean(group, stress_column)


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean())


def _winner_sector_concentration_gap(group: pd.DataFrame, winners: pd.DataFrame) -> float:
    universe_hhi = _hhi(group["sic2_sector"].astype(str))
    winner_hhi = _hhi(winners["sic2_sector"].astype(str))
    return float(winner_hhi - universe_hhi)


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    top_bottom_quantile: float,
    rolling_window: int,
    walk_forward_window: int,
    walk_forward_min_history: int,
    panel: pd.DataFrame,
    decision_metrics: pd.DataFrame,
    action_metrics: pd.DataFrame,
    score_bins: pd.DataFrame,
    contributions: pd.DataFrame,
) -> str:
    lines = [
        "# Phase7D Signed Graph Reversal Payoff Results",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- universe variant: `{variant}`",
        f"- data window: `{start}` through `{end}`",
        f"- evaluation start: `{eval_start}`",
        f"- reversal basket quantile: `{top_bottom_quantile}`",
        f"- rolling z-score window: `{rolling_window}` sessions",
        f"- walk-forward threshold window: `{walk_forward_window}` sessions",
        f"- walk-forward min history: `{walk_forward_min_history}` sessions",
        f"- joined panel rows: `{len(panel)}`",
        "",
        "## Target And Action",
        "",
        "`reversal_payoff` remains the target: long recent losers minus short recent winners.",
        "Positive action trades reversal. Negative action trades continuation.",
        "",
        "## Decision Metrics",
        "",
        _markdown_table(decision_metrics),
        "",
        "## Action Metrics",
        "",
        _markdown_table(action_metrics),
        "",
        "## Score Bins",
        "",
        _markdown_table(score_bins),
        "",
        "## Contribution Diagnostics",
        "",
        _markdown_table(contributions),
        "",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
