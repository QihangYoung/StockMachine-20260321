"""Phase7C repaired graph-prior reversal payoff experiment.

Phase7C keeps Phase7B's daily state construction but repairs the graph shape and
adds a conservative edge-reliability audit. It is validation-only and does not
read the test lockbox.
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

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_SEC_SUBMISSIONS_DIR,
)
from stockmachine.apps.run_pure_alpha_phase7b import (
    DEFAULT_END,
    DEFAULT_EVAL_START,
    DEFAULT_FUNDAMENTAL_PANEL,
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    DEFAULT_NON_PRICE_PANEL,
    DEFAULT_QUANTILE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SIZE_PANEL,
    DEFAULT_START,
    DEFAULT_VARIANT,
    EdgeSpec,
    _build_daily_graph_states,
    _load_joined_panel,
    _markdown_table,
    _rolling_zscore,
    _safe_corr,
    _t_stat,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7c_repaired_graph_reversal_payoff_20260517"
DEFAULT_RELIABILITY_WINDOW = 252
DEFAULT_STRUCTURAL_SHARE = 0.80
DEFAULT_WALK_FORWARD_WINDOW = 252
DEFAULT_WALK_FORWARD_MIN_HISTORY = 60


REPAIRED_EDGE_SPECS = (
    EdgeSpec(
        "reversal_dislocation",
        "mispricing_convergence",
        1.0,
        0.25,
        "overreaction",
        "Large recent winner-loser spread can create room for reversal.",
    ),
    EdgeSpec(
        "insider_support_spread",
        "insider_support",
        1.0,
        0.12,
        "insider_behavior",
        "Loser insider support relative to winners favors rebound credibility.",
    ),
    EdgeSpec(
        "fundamental_fragility_spread",
        "fundamental_loser_risk",
        -1.0,
        0.30,
        "fundamental_stress",
        "Fragile loser baskets are more likely falling knives than mispricing.",
    ),
    EdgeSpec(
        "filing_red_flag_spread",
        "filing_quality_risk",
        -1.0,
        0.12,
        "filing_quality",
        "Red-flag filing pressure weakens the reversal prior.",
    ),
    EdgeSpec(
        "trend_dominance",
        "trend_continuation_risk",
        -1.0,
        0.20,
        "style_persistence",
        "Intermediate winner-minus-loser momentum implies continuation risk.",
    ),
    EdgeSpec(
        "capital_concentration",
        "capital_concentration_regime",
        -1.0,
        0.20,
        "fund_flow_proxy",
        "Size-weighted strength over equal-weight strength implies concentration.",
    ),
    EdgeSpec(
        "liquidity_fragility",
        "liquidity_stress",
        -1.0,
        0.18,
        "liquidity",
        "Less-liquid loser baskets are more exposed to forced selling.",
    ),
    EdgeSpec(
        "beta_fragility",
        "risk_off_fragility",
        -1.0,
        0.12,
        "risk_exposure",
        "Higher beta losers are vulnerable in risk-off windows.",
    ),
    EdgeSpec(
        "sector_shock_concentration",
        "sector_shock_risk",
        -1.0,
        0.12,
        "industry_graph",
        "Sector-concentrated loser baskets may reflect real sector shocks.",
    ),
    EdgeSpec(
        "breadth_weakness",
        "risk_off_fragility",
        -1.0,
        0.12,
        "market_breadth",
        "Weak breadth makes loser rebound less trustworthy.",
    ),
)


@dataclass(frozen=True)
class ScoreSpec:
    score_column: str
    label: str


SCORE_SPECS = (
    ScoreSpec("repaired_graph_prior_score", "repaired graph prior"),
    ScoreSpec("reliability_80_20_score", "80/20 reliability-audited graph"),
    ScoreSpec("positive_reliability_score", "positive-reliability edge subset"),
    ScoreSpec("reliable_core_score", "stable structural core edge subset"),
)

RELIABLE_CORE_SOURCES = frozenset(
    {
        "fundamental_fragility_spread",
        "reversal_dislocation",
        "liquidity_fragility",
        "beta_fragility",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase7C repaired graph-prior reversal payoff test."
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
    parser.add_argument("--reliability-window", type=int, default=DEFAULT_RELIABILITY_WINDOW)
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    parser.add_argument("--structural-share", type=float, default=DEFAULT_STRUCTURAL_SHARE)
    parser.add_argument(
        "--walk-forward-window",
        type=int,
        default=DEFAULT_WALK_FORWARD_WINDOW,
    )
    parser.add_argument(
        "--walk-forward-min-history",
        type=int,
        default=DEFAULT_WALK_FORWARD_MIN_HISTORY,
    )
    args = parser.parse_args(argv)

    rollup = build_phase7c_repaired_graph_reversal_payoff(
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
        reliability_window=args.reliability_window,
        holding_period_sessions=args.holding_period_sessions,
        structural_share=args.structural_share,
        walk_forward_window=args.walk_forward_window,
        walk_forward_min_history=args.walk_forward_min_history,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7c_repaired_graph_reversal_payoff(
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
    reliability_window: int = DEFAULT_RELIABILITY_WINDOW,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
    structural_share: float = DEFAULT_STRUCTURAL_SHARE,
    walk_forward_window: int = DEFAULT_WALK_FORWARD_WINDOW,
    walk_forward_min_history: int = DEFAULT_WALK_FORWARD_MIN_HISTORY,
) -> dict[str, Any]:
    if not 0.0 <= structural_share <= 1.0:
        raise ValueError("structural_share must be in [0, 1].")
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
    daily = _build_daily_graph_states(panel, top_bottom_quantile=top_bottom_quantile)
    scored = _score_repaired_graph(
        daily,
        rolling_window=rolling_window,
        reliability_window=reliability_window,
        holding_period_sessions=holding_period_sessions,
        structural_share=structural_share,
    )
    scored = _add_walk_forward_thresholds(
        scored,
        window=walk_forward_window,
        min_history=walk_forward_min_history,
    )
    eval_frame = scored[
        scored["session_date"].ge(eval_start) & scored["session_date"].le(end)
    ].copy()
    metrics = _decision_metrics(eval_frame)
    bins = _score_bins(eval_frame)
    contributions = _contribution_diagnostics(eval_frame)
    reliability = _edge_reliability_summary(eval_frame)
    nodes = _graph_nodes()
    edges = _graph_edges()

    daily_path = output_dir / "phase7c_daily_graph_states.csv"
    nodes_path = output_dir / "phase7c_graph_nodes.csv"
    edges_path = output_dir / "phase7c_graph_edges.csv"
    metrics_path = output_dir / "phase7c_decision_metrics.csv"
    bins_path = output_dir / "phase7c_score_bins.csv"
    contributions_path = output_dir / "phase7c_contribution_diagnostics.csv"
    reliability_path = output_dir / "phase7c_edge_reliability_summary.csv"
    memo_path = output_dir / "phase7c_repaired_graph_reversal_payoff_memo.md"
    rollup_path = output_dir / "phase7c_rollup.json"

    scored.to_csv(daily_path, index=False)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bins.to_csv(bins_path, index=False)
    contributions.to_csv(contributions_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            rolling_window=rolling_window,
            reliability_window=reliability_window,
            holding_period_sessions=holding_period_sessions,
            structural_share=structural_share,
            walk_forward_window=walk_forward_window,
            walk_forward_min_history=walk_forward_min_history,
            panel=panel,
            metrics=metrics,
            bins=bins,
            contributions=contributions,
            reliability=reliability,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7c_repaired_graph_reversal_payoff",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "rolling_window": int(rolling_window),
        "reliability_window": int(reliability_window),
        "holding_period_sessions": int(holding_period_sessions),
        "structural_share": float(structural_share),
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
            "daily_graph_states": daily_path.as_posix(),
            "graph_nodes": nodes_path.as_posix(),
            "graph_edges": edges_path.as_posix(),
            "decision_metrics": metrics_path.as_posix(),
            "score_bins": bins_path.as_posix(),
            "contribution_diagnostics": contributions_path.as_posix(),
            "edge_reliability_summary": reliability_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _score_repaired_graph(
    daily: pd.DataFrame,
    *,
    rolling_window: int,
    reliability_window: int,
    holding_period_sessions: int,
    structural_share: float,
) -> pd.DataFrame:
    frame = daily.copy().sort_values("session_date").reset_index(drop=True)
    data_share = 1.0 - structural_share
    scale = float(np.sqrt(sum(edge.weight**2 for edge in REPAIRED_EDGE_SPECS)))
    prior_score = pd.Series(0.0, index=frame.index)
    reliability_score = pd.Series(0.0, index=frame.index)
    positive_reliability_score = pd.Series(0.0, index=frame.index)
    reliable_core_score = pd.Series(0.0, index=frame.index)

    known_payoff = frame["reversal_payoff"].shift(holding_period_sessions)
    frame["known_lagged_payoff"] = known_payoff
    frame["known_lagged_roll60_payoff"] = known_payoff.rolling(60, min_periods=20).mean()

    for edge in REPAIRED_EDGE_SPECS:
        z_col = f"z_{edge.source}"
        contrib_col = f"contrib_{edge.source}"
        reliability_col = f"reliability_corr_{edge.source}"
        reliability_contrib_col = f"reliability_contrib_{edge.source}"
        positive_contrib_col = f"positive_reliability_contrib_{edge.source}"

        frame[z_col] = _rolling_zscore(frame[edge.source], window=rolling_window)
        prior_contribution = edge.sign * edge.weight * frame[z_col].fillna(0.0)
        shifted_prior_contribution = prior_contribution.shift(holding_period_sessions)
        reliability_corr = shifted_prior_contribution.rolling(
            reliability_window,
            min_periods=max(40, min(reliability_window, 120)),
        ).corr(known_payoff)
        reliability_corr = reliability_corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        reliability_corr = reliability_corr.clip(-1.0, 1.0)

        reliability_multiplier = structural_share + data_share * reliability_corr
        reliability_contribution = prior_contribution * reliability_multiplier
        positive_reliability_contribution = prior_contribution.where(reliability_corr >= 0.0, 0.0)

        frame[contrib_col] = prior_contribution
        frame[reliability_col] = reliability_corr
        frame[reliability_contrib_col] = reliability_contribution
        frame[positive_contrib_col] = positive_reliability_contribution
        prior_score = prior_score + prior_contribution
        reliability_score = reliability_score + reliability_contribution
        positive_reliability_score = positive_reliability_score + positive_reliability_contribution
        if edge.source in RELIABLE_CORE_SOURCES:
            reliable_core_score = reliable_core_score + prior_contribution

    frame["repaired_graph_prior_score"] = prior_score / scale if scale > 0 else prior_score
    frame["reliability_80_20_score"] = reliability_score / scale if scale > 0 else reliability_score
    frame["positive_reliability_score"] = (
        positive_reliability_score / scale if scale > 0 else positive_reliability_score
    )
    core_scale = float(
        np.sqrt(sum(edge.weight**2 for edge in REPAIRED_EDGE_SPECS if edge.source in RELIABLE_CORE_SOURCES))
    )
    frame["reliable_core_score"] = (
        reliable_core_score / core_scale if core_scale > 0 else reliable_core_score
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
        out[f"{spec.score_column}_wf_q33"] = rolling.quantile(1.0 / 3.0)
        out[f"{spec.score_column}_wf_q67"] = rolling.quantile(2.0 / 3.0)
    return out


def _decision_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_metric_row(frame, "always_on", pd.Series(True, index=frame.index), "")]
    for spec in SCORE_SPECS:
        score = frame[spec.score_column]
        lower_threshold = frame[f"{spec.score_column}_wf_q33"]
        upper_threshold = frame[f"{spec.score_column}_wf_q67"]
        rows.extend(
            [
                _metric_row(frame, f"{spec.score_column}_positive", score > 0, spec.score_column),
                _metric_row(
                    frame,
                    f"{spec.score_column}_walk_forward_top_tercile",
                    score.ge(upper_threshold) & upper_threshold.notna(),
                    spec.score_column,
                ),
                _metric_row(
                    frame,
                    f"{spec.score_column}_walk_forward_bottom_tercile",
                    score.le(lower_threshold) & lower_threshold.notna(),
                    spec.score_column,
                ),
            ]
        )
    rows.append(
        _metric_row(
            frame,
            "lagged_roll60_positive",
            frame["known_lagged_roll60_payoff"] > 0,
            "known_lagged_roll60_payoff",
        )
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


def _metric_row(
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
    for edge in REPAIRED_EDGE_SPECS:
        for label, column in (
            ("prior", f"contrib_{edge.source}"),
            ("reliability_80_20", f"reliability_contrib_{edge.source}"),
            ("positive_reliability", f"positive_reliability_contrib_{edge.source}"),
        ):
            contribution = frame[column]
            rows.append(
                {
                    "score_type": label,
                    "source": edge.source,
                    "mechanism": edge.mechanism,
                    "sign": edge.sign,
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
    return pd.DataFrame(rows).sort_values(
        ["score_type", "mean_abs_contribution"],
        ascending=[True, False],
    )


def _edge_reliability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edge in REPAIRED_EDGE_SPECS:
        values = frame[f"reliability_corr_{edge.source}"].astype(float)
        rows.append(
            {
                "source": edge.source,
                "mechanism": edge.mechanism,
                "sign": edge.sign,
                "weight": edge.weight,
                "mean_reliability_corr": float(values.mean()),
                "median_reliability_corr": float(values.median()),
                "positive_reliability_rate": float((values >= 0).mean()),
                "p10_reliability_corr": float(values.quantile(0.10)),
                "p90_reliability_corr": float(values.quantile(0.90)),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_reliability_corr", ascending=False)


def _graph_nodes() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "node": "ReversalPayoff",
            "node_type": "target",
            "description": "Future loser-minus-winner beta-residual payoff.",
        }
    ]
    for edge in REPAIRED_EDGE_SPECS:
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
    for edge in REPAIRED_EDGE_SPECS:
        rows.append(
            {
                "source": edge.source,
                "target": edge.mechanism,
                "edge_type": "feature_to_mechanism",
                "sign": edge.sign,
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
                "sign": edge.sign,
                "weight": edge.weight,
                "channel": edge.channel,
                "description": edge.description,
            }
        )
    return pd.DataFrame(rows)


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    top_bottom_quantile: float,
    rolling_window: int,
    reliability_window: int,
    holding_period_sessions: int,
    structural_share: float,
    walk_forward_window: int,
    walk_forward_min_history: int,
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    contributions: pd.DataFrame,
    reliability: pd.DataFrame,
) -> str:
    lines = [
        "# Phase7C Repaired Graph Reversal Payoff Results",
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
        f"- reliability window: `{reliability_window}` sessions",
        f"- known payoff lag: `{holding_period_sessions}` sessions",
        f"- structural share in audited score: `{structural_share}`",
        f"- walk-forward tercile window: `{walk_forward_window}` sessions",
        f"- walk-forward tercile min history: `{walk_forward_min_history}` sessions",
        f"- joined panel rows: `{len(panel)}`",
        "",
        "## Decision Metrics",
        "",
        "Top/bottom tercile rules use walk-forward score thresholds computed from prior daily scores only.",
        "",
        _markdown_table(metrics),
        "",
        "## Score Bins",
        "",
        _markdown_table(bins),
        "",
        "## Edge Reliability",
        "",
        _markdown_table(reliability),
        "",
        "## Contribution Diagnostics",
        "",
        _markdown_table(contributions),
    ]
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
