"""Phase7B graph-prior reversal payoff state experiment.

This validation-only app asks whether an explicit structural graph prior can
rank future reversal payoff states. It does not read the test lockbox and does
not construct a promoted trading overlay.
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
    _load_sec_sic_map,
)


PROJECT = "us_equities_pure_alpha_h5"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7b_graph_reversal_payoff_20260517"
DEFAULT_VARIANT = "top1000_clean_core_beta_full"
DEFAULT_START = "2014-08-05"
DEFAULT_EVAL_START = "2014-11-11"
DEFAULT_END = "2019-12-31"
DEFAULT_QUANTILE = 0.20
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_HOLDING_PERIOD_SESSIONS = 10

DEFAULT_SIZE_PANEL = (
    RESEARCH_ROOT
    / "phase6j_true_size_mvp_20260508"
    / "phase6j_true_size_panel_validation.csv.gz"
)
DEFAULT_NON_PRICE_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase1_top1000_two_factor_build_20260428"
    / "non_price_two_factor_panel.csv.gz"
)
DEFAULT_FUNDAMENTAL_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase4_companyfacts_fundamentals_20260505"
    / "companyfacts_fundamental_panel.csv.gz"
)

PRICE_COLUMNS = (
    "variant",
    "session_date",
    "symbol",
    "trailing_median_dollar_volume_20",
    "liquidity_rank",
    "beta",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "forward_return_5d",
    "benchmark_forward_return_5d",
    TARGET_COLUMN,
)
SIZE_COLUMNS = (
    "variant",
    "session_date",
    "symbol",
    "market_cap",
    "market_cap_log_z",
    "market_cap_percentile",
)
NON_PRICE_COLUMNS = (
    "session_date",
    "symbol",
    "filing_red_flag_score",
    "insider_net_buy_score",
    "insider_buy_events_20d",
    "insider_sell_events_20d",
    "insider_buy_intensity_60d",
    "insider_sell_intensity_60d",
)
FUNDAMENTAL_COLUMNS = (
    "session_date",
    "symbol",
    "fundamental_leverage_pressure_score",
    "fundamental_profit_stress_score",
    "fundamental_fragility_score",
)


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    mechanism: str
    sign: float
    weight: float
    channel: str
    description: str


EDGE_SPECS = (
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
        "structural_loser_filter",
        1.0,
        0.20,
        "insider_behavior",
        "Loser insider support relative to winners favors rebound credibility.",
    ),
    EdgeSpec(
        "fundamental_fragility_spread",
        "structural_loser_filter",
        -1.0,
        0.30,
        "fundamental_stress",
        "Fragile loser baskets are more likely falling knives than mispricing.",
    ),
    EdgeSpec(
        "filing_red_flag_spread",
        "structural_loser_filter",
        -1.0,
        0.15,
        "filing_quality",
        "Red-flag filing pressure weakens the reversal prior.",
    ),
    EdgeSpec(
        "trend_dominance",
        "trend_continuation",
        -1.0,
        0.25,
        "style_persistence",
        "Intermediate winner-minus-loser momentum implies continuation risk.",
    ),
    EdgeSpec(
        "capital_concentration",
        "capital_concentration",
        -1.0,
        0.20,
        "fund_flow_proxy",
        "Size-weighted strength over equal-weight strength implies concentration.",
    ),
    EdgeSpec(
        "liquidity_fragility",
        "liquidity_stress",
        -1.0,
        0.20,
        "liquidity",
        "Less-liquid loser baskets are more exposed to forced selling.",
    ),
    EdgeSpec(
        "beta_fragility",
        "risk_off",
        -1.0,
        0.15,
        "risk_exposure",
        "Higher beta losers are vulnerable in risk-off windows.",
    ),
    EdgeSpec(
        "sector_shock_concentration",
        "sector_shock",
        -1.0,
        0.15,
        "industry_graph",
        "Sector-concentrated loser baskets may reflect real sector shocks.",
    ),
    EdgeSpec(
        "breadth_weakness",
        "risk_off",
        -1.0,
        0.15,
        "market_breadth",
        "Weak breadth makes loser rebound less trustworthy.",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7B graph-prior reversal payoff test.")
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
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    args = parser.parse_args(argv)

    rollup = build_phase7b_graph_reversal_payoff(
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
        holding_period_sessions=args.holding_period_sessions,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7b_graph_reversal_payoff(
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
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
    if not 0.0 < top_bottom_quantile < 0.5:
        raise ValueError("top_bottom_quantile must be in (0, 0.5).")
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
    scored = _score_daily_graph_states(
        daily,
        rolling_window=rolling_window,
        holding_period_sessions=holding_period_sessions,
    )
    eval_frame = scored[
        scored["session_date"].ge(eval_start) & scored["session_date"].le(end)
    ].copy()
    metrics = _decision_metrics(eval_frame)
    bins = _score_bins(eval_frame)
    contributions = _contribution_diagnostics(eval_frame)
    nodes = _graph_nodes()
    edges = _graph_edges()

    daily_path = output_dir / "phase7b_daily_graph_states.csv"
    nodes_path = output_dir / "phase7b_graph_nodes.csv"
    edges_path = output_dir / "phase7b_graph_edges.csv"
    metrics_path = output_dir / "phase7b_decision_metrics.csv"
    bins_path = output_dir / "phase7b_score_bins.csv"
    contributions_path = output_dir / "phase7b_contribution_diagnostics.csv"
    memo_path = output_dir / "phase7b_graph_reversal_payoff_memo.md"
    rollup_path = output_dir / "phase7b_rollup.json"

    scored.to_csv(daily_path, index=False)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bins.to_csv(bins_path, index=False)
    contributions.to_csv(contributions_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            rolling_window=rolling_window,
            holding_period_sessions=holding_period_sessions,
            panel=panel,
            metrics=metrics,
            bins=bins,
            contributions=contributions,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7b_graph_reversal_payoff",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "rolling_window": int(rolling_window),
        "holding_period_sessions": int(holding_period_sessions),
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
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_joined_panel(
    *,
    signal_panel_path: str | Path,
    size_panel_path: str | Path,
    non_price_panel_path: str | Path,
    fundamental_panel_path: str | Path,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    variant: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    price = _load_price_panel(signal_panel_path, variant=variant, start=start, end=end)
    size = _load_size_panel(size_panel_path, variant=variant, start=start, end=end)
    non_price = _load_optional_panel(non_price_panel_path, NON_PRICE_COLUMNS, start=start, end=end)
    fundamentals = _load_optional_panel(
        fundamental_panel_path,
        FUNDAMENTAL_COLUMNS,
        start=start,
        end=end,
    )
    sic = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)

    panel = (
        price.merge(size, on=["variant", "session_date", "symbol"], how="left")
        .merge(non_price, on=["session_date", "symbol"], how="left")
        .merge(fundamentals, on=["session_date", "symbol"], how="left")
        .merge(sic[["symbol", "sic2_sector", "sic4_industry"]], on="symbol", how="left")
    )
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["return_5d"] = -panel["reversal_5d"].astype(float)
    panel["log_adv"] = np.log(
        panel["trailing_median_dollar_volume_20"].astype(float).where(
            panel["trailing_median_dollar_volume_20"].astype(float) > 0
        )
    )
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _load_price_panel(path: str | Path, *, variant: str, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(PRICE_COLUMNS), low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["variant"] = frame["variant"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[
        frame["variant"].eq(variant)
        & frame["session_date"].ge(start)
        & frame["session_date"].le(end)
    ].copy()
    for column in PRICE_COLUMNS:
        if column not in {"variant", "session_date", "symbol"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["reversal_5d"])
    frame = frame.dropna(subset=[TARGET_COLUMN, "forward_return_5d", "beta"])
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _load_size_panel(path: str | Path, *, variant: str, start: str, end: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=SIZE_COLUMNS)
    chunks = []
    for chunk in pd.read_csv(path, usecols=list(SIZE_COLUMNS), chunksize=500_000):
        chunk["session_date"] = pd.to_datetime(chunk["session_date"]).dt.date.astype(str)
        chunk["variant"] = chunk["variant"].astype(str)
        chunk["symbol"] = chunk["symbol"].astype(str)
        chunk = chunk[
            chunk["variant"].eq(variant)
            & chunk["session_date"].ge(start)
            & chunk["session_date"].le(end)
        ].copy()
        if not chunk.empty:
            chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=SIZE_COLUMNS)
    for column in ("market_cap", "market_cap_log_z", "market_cap_percentile"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(["variant", "session_date", "symbol"], keep="last")


def _load_optional_panel(
    path: str | Path,
    columns: Sequence[str],
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, usecols=list(columns), low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[frame["session_date"].ge(start) & frame["session_date"].le(end)].copy()
    for column in columns:
        if column not in {"session_date", "symbol"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(["session_date", "symbol"], keep="last")


def _build_daily_graph_states(
    panel: pd.DataFrame,
    *,
    top_bottom_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_date, group in panel.groupby("session_date", sort=True):
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
            "test_window_used": False,
        }
        row["capital_concentration"] = (
            row["market_size_weight_return_5d"] - row["market_equal_weight_return_5d"]
            if np.isfinite(row["market_size_weight_return_5d"])
            else np.nan
        )
        row["breadth_weakness"] = 0.5 - row["market_breadth_5d"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)


def _score_daily_graph_states(
    daily: pd.DataFrame,
    *,
    rolling_window: int,
    holding_period_sessions: int,
) -> pd.DataFrame:
    frame = daily.copy().sort_values("session_date").reset_index(drop=True)
    feature_names = [edge.source for edge in EDGE_SPECS]
    for feature in feature_names:
        frame[f"z_{feature}"] = _rolling_zscore(frame[feature], window=rolling_window)

    score = pd.Series(0.0, index=frame.index)
    random_score = pd.Series(0.0, index=frame.index)
    rng = np.random.default_rng(260321)
    random_signs = {edge.source: float(rng.choice([-1.0, 1.0])) for edge in EDGE_SPECS}
    for edge in EDGE_SPECS:
        contribution = edge.sign * edge.weight * frame[f"z_{edge.source}"].fillna(0.0)
        frame[f"contrib_{edge.source}"] = contribution
        score = score + contribution
        random_score = random_score + random_signs[edge.source] * edge.weight * frame[
            f"z_{edge.source}"
        ].fillna(0.0)
    scale = float(np.sqrt(sum(edge.weight**2 for edge in EDGE_SPECS)))
    frame["graph_prior_score"] = score / scale if scale > 0 else score
    frame["random_sign_graph_score"] = random_score / scale if scale > 0 else random_score
    frame["graph_prior_probability"] = 1.0 / (1.0 + np.exp(-frame["graph_prior_score"]))
    frame["known_lagged_payoff"] = _known_lagged_series(
        frame["reversal_payoff"],
        holding_period_sessions=holding_period_sessions,
    )
    frame["known_lagged_roll60_payoff"] = (
        frame["known_lagged_payoff"].rolling(60, min_periods=20).mean()
    )
    return frame


def _decision_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _metric_row(frame, "always_on", pd.Series(True, index=frame.index), score_column=""),
        _metric_row(
            frame,
            "graph_prior_positive",
            frame["graph_prior_score"] > 0,
            score_column="graph_prior_score",
        ),
        _metric_row(
            frame,
            "graph_prior_top_tercile",
            frame["graph_prior_score"] >= frame["graph_prior_score"].quantile(2.0 / 3.0),
            score_column="graph_prior_score",
        ),
        _metric_row(
            frame,
            "graph_prior_bottom_tercile",
            frame["graph_prior_score"] <= frame["graph_prior_score"].quantile(1.0 / 3.0),
            score_column="graph_prior_score",
        ),
        _metric_row(
            frame,
            "lagged_roll60_positive",
            frame["known_lagged_roll60_payoff"] > 0,
            score_column="known_lagged_roll60_payoff",
        ),
        _metric_row(
            frame,
            "graph_and_lagged_positive",
            (frame["graph_prior_score"] > 0) & (frame["known_lagged_roll60_payoff"] > 0),
            score_column="graph_prior_score",
        ),
        _metric_row(
            frame,
            "random_sign_graph_positive",
            frame["random_sign_graph_score"] > 0,
            score_column="random_sign_graph_score",
        ),
    ]
    out = pd.DataFrame(rows)
    score_corr = _safe_corr(frame["graph_prior_score"], frame["reversal_payoff"], method="spearman")
    random_corr = _safe_corr(
        frame["random_sign_graph_score"],
        frame["reversal_payoff"],
        method="spearman",
    )
    out["graph_score_spearman_vs_payoff"] = score_corr
    out["random_score_spearman_vs_payoff"] = random_corr
    out["test_window_used"] = False
    return out


def _metric_row(
    frame: pd.DataFrame,
    rule: str,
    mask: pd.Series,
    *,
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
    valid = frame.dropna(subset=["graph_prior_score", "reversal_payoff"]).copy()
    if len(valid) < 10:
        return pd.DataFrame()
    valid["score_bin"] = pd.qcut(
        valid["graph_prior_score"],
        q=min(5, len(valid)),
        labels=False,
        duplicates="drop",
    )
    rows = []
    for score_bin, group in valid.groupby("score_bin", sort=True):
        payoff = group["reversal_payoff"].astype(float)
        rows.append(
            {
                "score_bin": int(score_bin),
                "sessions": int(len(group)),
                "score_min": float(group["graph_prior_score"].min()),
                "score_max": float(group["graph_prior_score"].max()),
                "score_mean": float(group["graph_prior_score"].mean()),
                "mean_payoff_bps": float(payoff.mean() * 10000.0),
                "hit_rate": float((payoff > 0).mean()),
                "t_stat": _t_stat(payoff),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _contribution_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edge in EDGE_SPECS:
        contribution = frame[f"contrib_{edge.source}"]
        rows.append(
            {
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
    return pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False)


def _graph_nodes() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "node": "ReversalPayoff",
            "node_type": "target",
            "description": "Future loser-minus-winner beta-residual payoff.",
        }
    ]
    for edge in EDGE_SPECS:
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
    for edge in EDGE_SPECS:
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


def _weighted_mean(frame: pd.DataFrame, value_column: str, weight_column: str) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _basket_spread(losers: pd.DataFrame, winners: pd.DataFrame, column: str) -> float:
    if column not in losers or column not in winners:
        return np.nan
    loser_values = pd.to_numeric(losers[column], errors="coerce")
    winner_values = pd.to_numeric(winners[column], errors="coerce")
    return float(loser_values.mean() - winner_values.mean())


def _sector_concentration_gap(group: pd.DataFrame, losers: pd.DataFrame) -> float:
    universe_hhi = _hhi(group["sic2_sector"].astype(str))
    loser_hhi = _hhi(losers["sic2_sector"].astype(str))
    return float(loser_hhi - universe_hhi)


def _hhi(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    shares = values.value_counts(normalize=True)
    return float((shares**2).sum())


def _rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=max(20, min(window, 60))).mean().shift(1)
    std = values.rolling(window, min_periods=max(20, min(window, 60))).std(ddof=0).shift(1)
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _known_lagged_series(
    series: pd.Series,
    *,
    holding_period_sessions: int,
) -> pd.Series:
    return series.shift(holding_period_sessions)


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["left"].std(ddof=0) == 0 or frame["right"].std(ddof=0) == 0:
        return np.nan
    return float(frame["left"].corr(frame["right"], method=method))


def _t_stat(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.nan
    return float(values.mean() / std * np.sqrt(len(values)))


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    top_bottom_quantile: float,
    rolling_window: int,
    holding_period_sessions: int,
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    contributions: pd.DataFrame,
) -> str:
    lines = [
        "# Phase7B Graph-Prior Reversal Payoff Results",
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
        f"- lag for known payoff baseline: `{holding_period_sessions}` sessions",
        f"- joined panel rows: `{len(panel)}`",
        "",
        "## Decision Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Graph Score Bins",
        "",
        _markdown_table(bins),
        "",
        "## Contribution Diagnostics",
        "",
        _markdown_table(contributions),
        "",
        "## Reading",
        "",
        "The primary check is whether higher graph-prior score bins have higher realized",
        "reversal payoff. If `graph_prior_positive` improves mean payoff or hit rate",
        "against `always_on`, the graph is useful as a soft activation candidate.",
        "If the random-sign graph is similar or better, the structural prior is not yet",
        "earning its keep.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
