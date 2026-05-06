from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import _add_former_winner_scores
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SHORT_VARIANT,
    _load_panel,
)
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_SHORT_OVERLAY


PROJECT_ID = "us_equities_pure_alpha_h5"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5n_non_price_state_factor_bootstrap_20260428"
DEFAULT_LONG_NON_PRICE_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase1_top1000_two_factor_build_20260428"
    / "non_price_two_factor_panel.csv.gz"
)
DEFAULT_SHORT_NON_PRICE_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase1_adv30m_two_factor_build_20260419"
    / "non_price_two_factor_panel.csv.gz"
)
DEFAULT_PRICE_STATE_FRAME = (
    RESEARCH_ROOT
    / "phase5j_whitebox_state_activation_20260428"
    / "phase5j_state_frame.csv"
)

LONG_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)
SHORT_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
)

LONG_RAW_COLUMNS = (
    "filing_red_flag_events_20d",
    "red_8k_events_60d",
    "periodic_delay_events_252d",
    "days_since_last_filing_red_flag",
    "insider_sell_events_20d",
    "insider_sell_intensity_60d",
    "insider_net_buy_score",
)
SHORT_RAW_COLUMNS = (
    "filing_red_flag_events_20d",
    "red_8k_events_60d",
    "periodic_delay_events_252d",
    "days_since_last_filing_red_flag",
    "insider_buy_events_20d",
    "insider_cluster_buy_events_60d",
    "insider_buy_intensity_60d",
    "days_since_last_insider_buy",
    "insider_net_buy_score",
)


def build_non_price_phase5n_state_factor_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    long_non_price_panel_path: str | Path = DEFAULT_LONG_NON_PRICE_PANEL,
    short_non_price_panel_path: str | Path = DEFAULT_SHORT_NON_PRICE_PANEL,
    price_state_frame_path: str | Path = DEFAULT_PRICE_STATE_FRAME,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    long_score: str = DEFAULT_LONG_SCORE,
    short_selector: str = DEFAULT_SHORT_OVERLAY,
    candidate_count: int = 80,
    trigger_threshold: float = 0.75,
    recent_filing_days: int = 20,
    recent_insider_buy_days: int = 20,
) -> dict[str, Any]:
    """Build validation-only non-price state-factor diagnostics aligned to alpha Phase5M."""

    _validate_settings(
        candidate_count=candidate_count,
        trigger_threshold=trigger_threshold,
        recent_filing_days=recent_filing_days,
        recent_insider_buy_days=recent_insider_buy_days,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)

    long_candidates = _build_candidates(
        panel=panel,
        variant=long_variant,
        score_column=long_score,
        side="long",
        candidate_count=candidate_count,
    )
    short_candidates = _build_candidates(
        panel=panel,
        variant=short_variant,
        score_column=short_selector,
        side="short",
        candidate_count=candidate_count,
    )

    long_non_price = _load_non_price_percentiles(
        Path(long_non_price_panel_path),
        raw_columns=LONG_RAW_COLUMNS,
        inverse_columns=("days_since_last_filing_red_flag", "insider_net_buy_score"),
    )
    short_non_price = _load_non_price_percentiles(
        Path(short_non_price_panel_path),
        raw_columns=SHORT_RAW_COLUMNS,
        inverse_columns=("days_since_last_filing_red_flag", "days_since_last_insider_buy"),
    )

    long_scored = _score_long_candidates(
        candidates=long_candidates,
        non_price=long_non_price,
        recent_filing_days=recent_filing_days,
    )
    short_scored = _score_short_candidates(
        candidates=short_candidates,
        non_price=short_non_price,
        recent_filing_days=recent_filing_days,
        recent_insider_buy_days=recent_insider_buy_days,
    )
    candidate_frame = pd.concat([long_scored, short_scored], ignore_index=True).sort_values(
        ["session_date", "side", "candidate_rank", "symbol"]
    )

    price_state = _load_price_state_frame(Path(price_state_frame_path))
    state_frame = _build_state_frame(
        long_candidates=long_scored,
        short_candidates=short_scored,
        trigger_threshold=trigger_threshold,
        price_state=price_state,
    )
    culprit_summary = _culprit_attribution_summary(
        long_candidates=long_scored,
        short_candidates=short_scored,
        trigger_threshold=trigger_threshold,
    )
    window_state_summary = _window_state_summary(state_frame)
    factor_schema = _factor_schema(
        candidate_frame=candidate_frame,
        long_trigger_threshold=trigger_threshold,
        short_trigger_threshold=trigger_threshold,
    )
    factor_overlap = _factor_overlap_summary(state_frame)
    factor_verdict = _factor_verdict_summary(candidate_frame, culprit_summary)
    basket_state_verdict = _basket_state_verdict(window_state_summary)
    stock_examples = _stock_examples(
        long_candidates=long_scored,
        short_candidates=short_scored,
        trigger_threshold=trigger_threshold,
    )

    paths = _write_outputs(
        output_dir=output_dir,
        candidate_frame=candidate_frame,
        state_frame=state_frame,
        culprit_summary=culprit_summary,
        window_state_summary=window_state_summary,
        factor_schema=factor_schema,
        factor_overlap=factor_overlap,
        factor_verdict=factor_verdict,
        basket_state_verdict=basket_state_verdict,
        stock_examples=stock_examples,
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "project_id": PROJECT_ID,
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "long_non_price_panel_path": Path(long_non_price_panel_path).as_posix(),
        "short_non_price_panel_path": Path(short_non_price_panel_path).as_posix(),
        "price_state_frame_path": Path(price_state_frame_path).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "long_score": long_score,
        "short_selector": short_selector,
        "candidate_count": int(candidate_count),
        "trigger_threshold": float(trigger_threshold),
        "recent_filing_days": int(recent_filing_days),
        "recent_insider_buy_days": int(recent_insider_buy_days),
        "candidate_rows": int(len(candidate_frame)),
        "state_rows": int(len(state_frame)),
        "culprit_rows": int(len(culprit_summary)),
        "schema_rows": int(len(factor_schema)),
        "factor_verdict_rows": int(len(factor_verdict)),
        "basket_state_verdict_rows": int(len(basket_state_verdict)),
        "artifacts": paths,
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "non_price_state_factor_bootstrap_for_selector_validity_diagnosis",
        "limitations": [
            "This phase evaluates state observability, not standalone alpha monetization.",
            "Only existing SEC filing and insider channels are used in the MVP.",
            "Analyst revisions, borrow, short-interest, and structured fundamentals are not yet wired.",
            "All outputs are validation-only and do not open the test lockbox.",
        ],
    }
    rollup_path = output_dir / "phase5n_non_price_state_factor_rollup.json"
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    memo_path = output_dir / "phase5n_non_price_state_factor_memo.md"
    memo_path.write_text(
        _memo(
            rollup=rollup,
            state_frame=state_frame,
            culprit_summary=culprit_summary,
            window_state_summary=window_state_summary,
            factor_overlap=factor_overlap,
            factor_schema=factor_schema,
            factor_verdict=factor_verdict,
            basket_state_verdict=basket_state_verdict,
        ),
        encoding="utf-8",
    )
    rollup["artifacts"]["rollup"] = rollup_path.as_posix()
    rollup["artifacts"]["memo"] = memo_path.as_posix()
    return rollup


def _validate_settings(
    *,
    candidate_count: int,
    trigger_threshold: float,
    recent_filing_days: int,
    recent_insider_buy_days: int,
) -> None:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if not 0.0 < trigger_threshold < 1.0:
        raise ValueError("trigger_threshold must be between 0 and 1.")
    if recent_filing_days <= 0:
        raise ValueError("recent_filing_days must be positive.")
    if recent_insider_buy_days <= 0:
        raise ValueError("recent_insider_buy_days must be positive.")


def _build_candidates(
    *,
    panel: pd.DataFrame,
    variant: str,
    score_column: str,
    side: str,
    candidate_count: int,
) -> pd.DataFrame:
    subset = panel[panel["variant"].astype(str).eq(variant)].copy()
    rows: list[pd.DataFrame] = []
    for session_date, group in subset.groupby("session_date", sort=True):
        frame = group.dropna(
            subset=[
                score_column,
                "forward_beta_residual_return_5d",
                "forward_return_5d",
                "benchmark_forward_return_5d",
            ]
        ).drop_duplicates("symbol", keep="last")
        if len(frame) < candidate_count:
            continue
        top = frame.sort_values([score_column, "symbol"], ascending=[False, True]).head(candidate_count)
        top = top.copy()
        top["candidate_rank"] = np.arange(1, len(top) + 1, dtype=int)
        top["side"] = side
        top["base_score_column"] = score_column
        rows.append(
            top[
                [
                    "session_date",
                    "variant",
                    "symbol",
                    "beta",
                    score_column,
                    "forward_beta_residual_return_5d",
                    "forward_return_5d",
                    "benchmark_forward_return_5d",
                    "candidate_rank",
                    "side",
                    "base_score_column",
                ]
            ].rename(columns={score_column: "base_score"})
        )
    if not rows:
        raise ValueError(f"No candidate rows built for variant={variant} side={side}.")
    return pd.concat(rows, ignore_index=True).sort_values(
        ["session_date", "candidate_rank", "symbol"]
    )


def _load_non_price_percentiles(
    path: Path,
    *,
    raw_columns: Sequence[str],
    inverse_columns: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Non-price panel not found: {path}")
    usecols = ["session_date", "symbol", *raw_columns]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for column in raw_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame[f"{column}_pct"] = frame.groupby("session_date")[column].rank(
            method="average",
            pct=True,
        )
    for column in inverse_columns:
        inverse_name = f"{column}_inverse_raw"
        pct_name = f"{column}_inverse_pct"
        frame[inverse_name] = -frame[column]
        frame[pct_name] = frame.groupby("session_date")[inverse_name].rank(
            method="average",
            pct=True,
        )
    return frame


def _score_long_candidates(
    *,
    candidates: pd.DataFrame,
    non_price: pd.DataFrame,
    recent_filing_days: int,
) -> pd.DataFrame:
    merged = candidates.merge(non_price, on=["session_date", "symbol"], how="left", validate="one_to_one")
    merged["recent_filing_red_flag"] = (
        merged["days_since_last_filing_red_flag"].le(float(recent_filing_days))
        & merged["filing_red_flag_events_20d"].gt(0)
    ).astype(float)
    merged["long_filing_distress_score"] = _mean_available(
        [
            merged["filing_red_flag_events_20d_pct"],
            merged["red_8k_events_60d_pct"],
            merged["periodic_delay_events_252d_pct"],
            merged["days_since_last_filing_red_flag_inverse_pct"],
            merged["recent_filing_red_flag"],
        ]
    )
    merged["long_insider_sell_pressure_score"] = _mean_available(
        [
            merged["insider_sell_events_20d_pct"],
            merged["insider_sell_intensity_60d_pct"],
            merged["insider_net_buy_score_inverse_pct"],
        ]
    )
    merged["long_non_price_distress_score"] = _mean_available(
        [
            merged["long_filing_distress_score"],
            merged["long_insider_sell_pressure_score"],
        ]
    )
    merged["trigger_long_filing_distress"] = merged["long_filing_distress_score"].ge(0.75)
    merged["trigger_long_insider_sell_pressure"] = merged["long_insider_sell_pressure_score"].ge(0.75)
    merged["trigger_long_non_price_distress"] = merged["long_non_price_distress_score"].ge(0.75)
    merged["window_side"] = "long"
    return merged


def _score_short_candidates(
    *,
    candidates: pd.DataFrame,
    non_price: pd.DataFrame,
    recent_filing_days: int,
    recent_insider_buy_days: int,
) -> pd.DataFrame:
    merged = candidates.merge(non_price, on=["session_date", "symbol"], how="left", validate="one_to_one")
    merged["recent_filing_red_flag"] = (
        merged["days_since_last_filing_red_flag"].le(float(recent_filing_days))
        & merged["filing_red_flag_events_20d"].gt(0)
    ).astype(float)
    merged["recent_insider_buy"] = (
        merged["days_since_last_insider_buy"].le(float(recent_insider_buy_days))
        & merged["insider_buy_events_20d"].gt(0)
    ).astype(float)
    merged["short_insider_buy_support_score"] = _mean_available(
        [
            merged["insider_buy_events_20d_pct"],
            merged["insider_cluster_buy_events_60d_pct"],
            merged["insider_buy_intensity_60d_pct"],
            merged["insider_net_buy_score_pct"],
            merged["days_since_last_insider_buy_inverse_pct"],
            merged["recent_insider_buy"],
        ]
    )
    merged["short_filing_confirmation_score"] = _mean_available(
        [
            merged["filing_red_flag_events_20d_pct"],
            merged["red_8k_events_60d_pct"],
            merged["periodic_delay_events_252d_pct"],
            merged["days_since_last_filing_red_flag_inverse_pct"],
            merged["recent_filing_red_flag"],
        ]
    )
    merged["short_non_price_improver_score"] = merged["short_insider_buy_support_score"]
    merged["trigger_short_insider_buy_support"] = merged["short_insider_buy_support_score"].ge(0.75)
    merged["trigger_short_recent_buy"] = merged["recent_insider_buy"].ge(1.0)
    merged["trigger_short_cluster_buy"] = merged["insider_cluster_buy_events_60d"].gt(0)
    merged["trigger_short_non_price_improver"] = merged["short_non_price_improver_score"].ge(0.75)
    merged["trigger_short_filing_confirmation"] = merged["short_filing_confirmation_score"].ge(0.75)
    merged["window_side"] = "short"
    return merged


def _load_price_state_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price state frame not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    return frame


def _build_state_frame(
    *,
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    trigger_threshold: float,
    price_state: pd.DataFrame,
) -> pd.DataFrame:
    long_inputs = long_candidates.assign(
        _long_filing_distress_hi=long_candidates["long_filing_distress_score"]
        .ge(trigger_threshold)
        .astype(float),
        _long_insider_sell_hi=long_candidates["long_insider_sell_pressure_score"]
        .ge(trigger_threshold)
        .astype(float),
        _long_non_price_distress_hi=long_candidates["long_non_price_distress_score"]
        .ge(trigger_threshold)
        .astype(float),
    )
    long_state = (
        long_inputs.groupby("session_date", sort=True)
        .agg(
            long_candidate_count=("long_non_price_distress_score", "size"),
            share_long_filing_distress_hi=("_long_filing_distress_hi", "mean"),
            share_long_insider_sell_hi=("_long_insider_sell_hi", "mean"),
            share_long_non_price_distress_hi=("_long_non_price_distress_hi", "mean"),
            mean_long_non_price_distress_score=("long_non_price_distress_score", "mean"),
        )
        .reset_index()
    )
    long_state["phi_long_non_price"] = long_state[
        [
            "share_long_filing_distress_hi",
            "share_long_insider_sell_hi",
            "share_long_non_price_distress_hi",
        ]
    ].mean(axis=1)

    short_inputs = short_candidates.assign(
        _short_insider_buy_support_hi=short_candidates["short_insider_buy_support_score"]
        .ge(trigger_threshold)
        .astype(float),
        _short_recent_buy=short_candidates["recent_insider_buy"].ge(1.0).astype(float),
        _short_cluster_buy=short_candidates["insider_cluster_buy_events_60d"].gt(0).astype(float),
        _short_non_price_improver_hi=short_candidates["short_non_price_improver_score"]
        .ge(trigger_threshold)
        .astype(float),
        _short_filing_confirmation_hi=short_candidates["short_filing_confirmation_score"]
        .ge(trigger_threshold)
        .astype(float),
    )
    short_state = (
        short_inputs.groupby("session_date", sort=True)
        .agg(
            short_candidate_count=("short_non_price_improver_score", "size"),
            share_short_insider_buy_support_hi=("_short_insider_buy_support_hi", "mean"),
            share_short_recent_buy=("_short_recent_buy", "mean"),
            share_short_cluster_buy=("_short_cluster_buy", "mean"),
            share_short_non_price_improver_hi=("_short_non_price_improver_hi", "mean"),
            share_short_filing_confirmation_hi=("_short_filing_confirmation_hi", "mean"),
            mean_short_non_price_improver_score=("short_non_price_improver_score", "mean"),
        )
        .reset_index()
    )
    short_state["phi_short_non_price"] = short_state[
        [
            "share_short_insider_buy_support_hi",
            "share_short_recent_buy",
            "share_short_cluster_buy",
            "share_short_non_price_improver_hi",
        ]
    ].mean(axis=1)

    state = long_state.merge(short_state, on="session_date", how="outer", validate="one_to_one")
    state = state.merge(price_state, on="session_date", how="left")
    state["phi_non_price_gap"] = state["phi_short_non_price"] - state["phi_long_non_price"]
    state["gate_phi_long_np_050"] = state["phi_long_non_price"].gt(0.50)
    state["gate_phi_short_np_050"] = state["phi_short_non_price"].gt(0.50)
    return state.sort_values("session_date").reset_index(drop=True)


def _culprit_attribution_summary(
    *,
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    trigger_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    long_specs = (
        ("long_filing_distress_score", "trigger_long_filing_distress"),
        ("long_insider_sell_pressure_score", "trigger_long_insider_sell_pressure"),
        ("long_non_price_distress_score", "trigger_long_non_price_distress"),
    )
    short_specs = (
        ("short_insider_buy_support_score", "trigger_short_insider_buy_support"),
        ("short_non_price_improver_score", "trigger_short_non_price_improver"),
        ("short_filing_confirmation_score", "trigger_short_filing_confirmation"),
    )

    for window, start, end in LONG_WINDOWS:
        subset = long_candidates[
            (long_candidates["session_date"] >= start) & (long_candidates["session_date"] <= end)
        ].copy()
        rows.extend(
            _window_rows(
                subset,
                window=window,
                side="long",
                factor_specs=long_specs,
                trigger_threshold=trigger_threshold,
            )
        )
    for window, start, end in SHORT_WINDOWS:
        subset = short_candidates[
            (short_candidates["session_date"] >= start) & (short_candidates["session_date"] <= end)
        ].copy()
        rows.extend(
            _window_rows(
                subset,
                window=window,
                side="short",
                factor_specs=short_specs,
                trigger_threshold=trigger_threshold,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["side", "window", "factor_name"]).reset_index(drop=True)


def _window_rows(
    frame: pd.DataFrame,
    *,
    window: str,
    side: str,
    factor_specs: Sequence[tuple[str, str]],
    trigger_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for factor_name, trigger_col in factor_specs:
        subset = frame.dropna(subset=[factor_name, "forward_beta_residual_return_5d"]).copy()
        if subset.empty:
            continue
        triggered = subset[subset[trigger_col].astype(bool)]
        rest = subset[~subset[trigger_col].astype(bool)]
        if triggered.empty or rest.empty:
            label = "insufficient_trigger_rows"
            edge = float("nan")
            target_support = False
        else:
            trig_mean = float(triggered["forward_beta_residual_return_5d"].mean())
            rest_mean = float(rest["forward_beta_residual_return_5d"].mean())
            edge = trig_mean - rest_mean
            target_support = edge < 0 if side == "long" else edge > 0
            label = "culprit_supported" if target_support else "culprit_not_supported"
        rows.append(
            {
                "window": window,
                "side": side,
                "factor_name": factor_name,
                "trigger_threshold": float(trigger_threshold),
                "candidate_rows": int(len(subset)),
                "trigger_rows": int(len(triggered)),
                "trigger_share": float(len(triggered) / len(subset)),
                "trigger_mean_factor_score": float(triggered[factor_name].mean())
                if not triggered.empty
                else float("nan"),
                "trigger_mean_forward_beta_residual_return": float(
                    triggered["forward_beta_residual_return_5d"].mean()
                )
                if not triggered.empty
                else float("nan"),
                "rest_mean_forward_beta_residual_return": float(
                    rest["forward_beta_residual_return_5d"].mean()
                )
                if not rest.empty
                else float("nan"),
                "trigger_edge_vs_rest": edge,
                "trigger_negative_share": float(
                    (triggered["forward_beta_residual_return_5d"] < 0).mean()
                )
                if not triggered.empty
                else float("nan"),
                "rest_negative_share": float((rest["forward_beta_residual_return_5d"] < 0).mean())
                if not rest.empty
                else float("nan"),
                "label": label,
                "test_window_used": False,
            }
        )
    return rows


def _window_state_summary(state_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side, windows in (("long", LONG_WINDOWS), ("short", SHORT_WINDOWS)):
        factor_col = "phi_long_non_price" if side == "long" else "phi_short_non_price"
        full_mean = float(state_frame[factor_col].mean())
        for window, start, end in windows:
            subset = state_frame[
                (state_frame["session_date"] >= start) & (state_frame["session_date"] <= end)
            ].copy()
            rows.append(
                {
                    "window": window,
                    "side": side,
                    "sessions": int(len(subset)),
                    "phi_mean": float(subset[factor_col].mean()) if not subset.empty else float("nan"),
                    "phi_median": float(subset[factor_col].median()) if not subset.empty else float("nan"),
                    "phi_edge_vs_full_mean": (
                        float(subset[factor_col].mean()) - full_mean if not subset.empty else float("nan")
                    ),
                    "price_state_mean": float(subset.get("whitebox_state_score_a", pd.Series(dtype=float)).mean())
                    if not subset.empty
                    else float("nan"),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _factor_schema(
    *,
    candidate_frame: pd.DataFrame,
    long_trigger_threshold: float,
    short_trigger_threshold: float,
) -> pd.DataFrame:
    long_cov = float(
        candidate_frame[candidate_frame["side"].eq("long")]["long_non_price_distress_score"].notna().mean()
    )
    short_cov = float(
        candidate_frame[candidate_frame["side"].eq("short")]["short_non_price_improver_score"].notna().mean()
    )
    rows = [
        {
            "factor_name": "long_filing_distress_score",
            "economic_mechanism": "Recent filing red flags and periodic delays increase the odds that a loser is distress, not oversold.",
            "update_frequency": "daily_from_sec_filing_alignment",
            "point_in_time_availability": "available_next_session_after_filing_date",
            "coverage_on_candidate_panel": long_cov,
            "expected_help_on_phi_long": "high",
            "expected_help_on_phi_short": "low",
            "stock_level_use": f"veto_or_penalize_long_candidates_above_{long_trigger_threshold:.2f}",
            "basket_level_use": "share of long candidate basket with fresh filing distress",
            "known_failure_modes": "Routine filing activity and stale history can look scary without current deterioration.",
        },
        {
            "factor_name": "long_insider_sell_pressure_score",
            "economic_mechanism": "Heavy recent insider selling weakens the case that a loser is merely temporarily oversold.",
            "update_frequency": "daily_from_form4_alignment",
            "point_in_time_availability": "available_next_session_after_form4_filing",
            "coverage_on_candidate_panel": long_cov,
            "expected_help_on_phi_long": "medium_high",
            "expected_help_on_phi_short": "low",
            "stock_level_use": f"penalize_long_candidates_above_{long_trigger_threshold:.2f}",
            "basket_level_use": "share of long candidate basket with insider-backed selling pressure",
            "known_failure_modes": "Routine diversification sales are noisy and can overstate true distress.",
        },
        {
            "factor_name": "long_non_price_distress_score",
            "economic_mechanism": "Combine filing distress and insider selling to detect falling knives that price-only reversal may misread.",
            "update_frequency": "daily_composite_from_sec_channels",
            "point_in_time_availability": "next_session_after_underlying_sec_events",
            "coverage_on_candidate_panel": long_cov,
            "expected_help_on_phi_long": "high",
            "expected_help_on_phi_short": "low",
            "stock_level_use": f"candidate-level_long_fragility_score_above_{long_trigger_threshold:.2f}",
            "basket_level_use": "phi_long_non_price observable component",
            "known_failure_modes": "Can miss fundamental deterioration that is not yet reflected in SEC activity.",
        },
        {
            "factor_name": "short_insider_buy_support_score",
            "economic_mechanism": "Fresh cluster buying and strong insider demand suggest a winner may be a genuine improver, not just a crowded squeeze candidate.",
            "update_frequency": "daily_from_form4_alignment",
            "point_in_time_availability": "available_next_session_after_form4_filing",
            "coverage_on_candidate_panel": short_cov,
            "expected_help_on_phi_long": "low",
            "expected_help_on_phi_short": "high",
            "stock_level_use": f"avoid_or_delay_shorts_above_{short_trigger_threshold:.2f}",
            "basket_level_use": "share of short candidate basket with insider-backed continuation risk",
            "known_failure_modes": "Insider buying is sparse and may not capture institutionally driven genuine improvers.",
        },
        {
            "factor_name": "short_filing_confirmation_score",
            "economic_mechanism": "Recent hard filing stress can confirm that an apparent winner is actually fragile enough to short.",
            "update_frequency": "daily_from_sec_filing_alignment",
            "point_in_time_availability": "available_next_session_after_filing_date",
            "coverage_on_candidate_panel": short_cov,
            "expected_help_on_phi_long": "low",
            "expected_help_on_phi_short": "confirmation_not_phi",
            "stock_level_use": f"short_confirmation_overlay_above_{short_trigger_threshold:.2f}",
            "basket_level_use": "share of short candidate basket with negative SEC confirmation",
            "known_failure_modes": "Useful for confirmation, but not a direct observable of genuine improvement risk.",
        },
        {
            "factor_name": "short_non_price_improver_score",
            "economic_mechanism": "Use insider-backed support as a first non-price observable for Phi_short, where dangerous shorts are genuine improvers rather than fading momentum names.",
            "update_frequency": "daily_composite_from_form4_alignment",
            "point_in_time_availability": "next_session_after_underlying_form4_events",
            "coverage_on_candidate_panel": short_cov,
            "expected_help_on_phi_long": "low",
            "expected_help_on_phi_short": "high",
            "stock_level_use": f"short_side_genuine_improver_flag_above_{short_trigger_threshold:.2f}",
            "basket_level_use": "phi_short_non_price observable component",
            "known_failure_modes": "MVP relies heavily on insider data because revisions, borrow, and short-interest channels are not yet wired.",
        },
    ]
    return pd.DataFrame(rows)


def _factor_overlap_summary(state_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = (
        ("phi_long_non_price", "whitebox_state_score_a"),
        ("phi_short_non_price", "whitebox_state_score_a"),
        ("phi_non_price_gap", "whitebox_state_score_a"),
        ("phi_long_non_price", "whitebox_spy_ret20"),
        ("phi_short_non_price", "whitebox_spy_ret20"),
    )
    for left, right in specs:
        subset = state_frame[[left, right]].dropna()
        rows.append(
            {
                "left_series": left,
                "right_series": right,
                "rows": int(len(subset)),
                "pearson_corr": float(subset[left].corr(subset[right], method="pearson"))
                if len(subset) >= 2
                else float("nan"),
                "spearman_corr": float(subset[left].corr(subset[right], method="spearman"))
                if len(subset) >= 2
                else float("nan"),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _factor_verdict_summary(candidate_frame: pd.DataFrame, culprit_summary: pd.DataFrame) -> pd.DataFrame:
    trigger_specs = (
        ("long", "long_filing_distress_score", "trigger_long_filing_distress"),
        ("long", "long_insider_sell_pressure_score", "trigger_long_insider_sell_pressure"),
        ("long", "long_non_price_distress_score", "trigger_long_non_price_distress"),
        ("short", "short_insider_buy_support_score", "trigger_short_insider_buy_support"),
        ("short", "short_filing_confirmation_score", "trigger_short_filing_confirmation"),
        ("short", "short_non_price_improver_score", "trigger_short_non_price_improver"),
    )
    rows: list[dict[str, Any]] = []
    for side, factor_name, trigger_col in trigger_specs:
        side_frame = candidate_frame[candidate_frame["side"].eq(side)].copy()
        tested = culprit_summary[culprit_summary["factor_name"].eq(factor_name)].copy()
        candidate_rows = int(len(side_frame))
        trigger_rows = int(side_frame[trigger_col].sum()) if trigger_col in side_frame.columns else 0
        tested_windows = int(len(tested))
        supported_windows = int(tested["label"].eq("culprit_supported").sum()) if tested_windows else 0
        support_rate = (
            float(supported_windows / tested_windows) if tested_windows else float("nan")
        )
        mean_window_edge = float(tested["trigger_edge_vs_rest"].mean()) if tested_windows else float("nan")
        if tested_windows == 0:
            verdict = "not_tested"
        elif supported_windows >= 2:
            verdict = "promising_mvp"
        elif supported_windows == 1:
            verdict = "partial_support"
        else:
            verdict = "not_supported_yet"
        rows.append(
            {
                "side": side,
                "factor_name": factor_name,
                "candidate_rows": candidate_rows,
                "trigger_rows": trigger_rows,
                "overall_trigger_share": float(trigger_rows / candidate_rows) if candidate_rows else float("nan"),
                "tested_windows": tested_windows,
                "supported_windows": supported_windows,
                "support_rate": support_rate,
                "mean_window_edge_vs_rest": mean_window_edge,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _basket_state_verdict(window_state_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side, subset in window_state_summary.groupby("side", sort=True):
        registered_windows = int(len(subset))
        windows_with_phi_rise = int(subset["phi_edge_vs_full_mean"].gt(0.0).sum())
        support_rate = (
            float(windows_with_phi_rise / registered_windows) if registered_windows else float("nan")
        )
        mean_phi_edge = float(subset["phi_edge_vs_full_mean"].mean()) if registered_windows else float("nan")
        required_windows = max(1, (registered_windows + 1) // 2)
        if windows_with_phi_rise >= required_windows and mean_phi_edge > 0.0:
            verdict = "directional_mvp"
        elif windows_with_phi_rise > 0:
            verdict = "mixed_signal"
        else:
            verdict = "not_ready"
        rows.append(
            {
                "side": side,
                "registered_windows": registered_windows,
                "windows_with_phi_rise": windows_with_phi_rise,
                "window_support_rate": support_rate,
                "mean_phi_edge_vs_full_mean": mean_phi_edge,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _stock_examples(
    *,
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    trigger_threshold: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    long_specs = (
        ("long_filing_distress_score", "long_culprit"),
        ("long_insider_sell_pressure_score", "long_culprit"),
        ("long_non_price_distress_score", "long_culprit"),
    )
    for window, start, end in LONG_WINDOWS:
        base_subset = long_candidates[
            (long_candidates["session_date"] >= start) & (long_candidates["session_date"] <= end)
        ].copy()
        for score_col, example_role in long_specs:
            subset = base_subset[base_subset[score_col].ge(trigger_threshold)].copy()
            if subset.empty:
                continue
            top = (
                subset.sort_values(
                    [score_col, "session_date", "candidate_rank"],
                    ascending=[False, True, True],
                )
                .head(10)
                .copy()
            )
            top["window"] = window
            top["example_role"] = example_role
            top["focus_factor"] = score_col
            top["factor_score"] = top[score_col]
            rows.append(
                top[
                    [
                        "window",
                        "example_role",
                        "focus_factor",
                        "session_date",
                        "symbol",
                        "candidate_rank",
                        "base_score",
                        "factor_score",
                        "forward_beta_residual_return_5d",
                    ]
                ]
            )
    short_specs = (
        ("short_filing_confirmation_score", "short_confirmation"),
        ("short_insider_buy_support_score", "short_culprit"),
        ("short_non_price_improver_score", "short_culprit"),
    )
    for window, start, end in SHORT_WINDOWS:
        base_subset = short_candidates[
            (short_candidates["session_date"] >= start) & (short_candidates["session_date"] <= end)
        ].copy()
        for score_col, example_role in short_specs:
            subset = base_subset[base_subset[score_col].ge(trigger_threshold)].copy()
            if subset.empty:
                continue
            top = (
                subset.sort_values(
                    [score_col, "session_date", "candidate_rank"],
                    ascending=[False, True, True],
                )
                .head(10)
                .copy()
            )
            top["window"] = window
            top["example_role"] = example_role
            top["focus_factor"] = score_col
            top["factor_score"] = top[score_col]
            rows.append(
                top[
                    [
                        "window",
                        "example_role",
                        "focus_factor",
                        "session_date",
                        "symbol",
                        "candidate_rank",
                        "base_score",
                        "factor_score",
                        "forward_beta_residual_return_5d",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_outputs(
    *,
    output_dir: Path,
    candidate_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    culprit_summary: pd.DataFrame,
    window_state_summary: pd.DataFrame,
    factor_schema: pd.DataFrame,
    factor_overlap: pd.DataFrame,
    factor_verdict: pd.DataFrame,
    basket_state_verdict: pd.DataFrame,
    stock_examples: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "candidate_frame": output_dir / "phase5n_candidate_state_stock_validation.csv.gz",
        "state_frame": output_dir / "phase5n_non_price_state_frame.csv",
        "culprit_summary": output_dir / "phase5n_culprit_attribution_summary.csv",
        "window_state_summary": output_dir / "phase5n_window_state_summary.csv",
        "factor_schema": output_dir / "phase5n_factor_schema.csv",
        "factor_overlap": output_dir / "phase5n_factor_overlap_summary.csv",
        "factor_verdict": output_dir / "phase5n_factor_verdict_summary.csv",
        "basket_state_verdict": output_dir / "phase5n_basket_state_verdict_summary.csv",
        "stock_examples": output_dir / "phase5n_stock_examples.csv",
    }
    candidate_frame.to_csv(paths["candidate_frame"], index=False, compression="gzip")
    state_frame.to_csv(paths["state_frame"], index=False)
    culprit_summary.to_csv(paths["culprit_summary"], index=False)
    window_state_summary.to_csv(paths["window_state_summary"], index=False)
    factor_schema.to_csv(paths["factor_schema"], index=False)
    factor_overlap.to_csv(paths["factor_overlap"], index=False)
    factor_verdict.to_csv(paths["factor_verdict"], index=False)
    basket_state_verdict.to_csv(paths["basket_state_verdict"], index=False)
    stock_examples.to_csv(paths["stock_examples"], index=False)
    return {key: value.as_posix() for key, value in paths.items()}


def _memo(
    *,
    rollup: Mapping[str, Any],
    state_frame: pd.DataFrame,
    culprit_summary: pd.DataFrame,
    window_state_summary: pd.DataFrame,
    factor_overlap: pd.DataFrame,
    factor_schema: pd.DataFrame,
    factor_verdict: pd.DataFrame,
    basket_state_verdict: pd.DataFrame,
) -> str:
    lines = [
        "# Phase5N Non-Price State Factor Bootstrap Memo",
        "",
        f"Generated: {rollup['created_at_utc']}",
        "",
        "## Scope",
        "",
        "Validation-only MVP for the Phase5M non-price request.",
        "This phase does not build a new alpha model. It builds state observables for selector validity diagnosis.",
        "",
        "## Candidate Alignment",
        "",
        f"- long basket: `{rollup['long_variant']}` top `{rollup['candidate_count']}` by `{rollup['long_score']}`",
        f"- short basket: `{rollup['short_variant']}` top `{rollup['candidate_count']}` by `{rollup['short_selector']}`",
        "- long-side non-price observables: filing distress, insider sell pressure",
        "- short-side non-price observables: insider buy support, filing confirmation",
        "",
        "## Daily State Summary",
        "",
        _table(
            state_frame[
                [
                    "phi_long_non_price",
                    "phi_short_non_price",
                    "phi_non_price_gap",
                    "whitebox_state_score_a",
                ]
            ]
            .agg(["mean", "median"])
            .reset_index()
            .rename(columns={"index": "stat"})
        ),
        "",
        "## Registered Window Summary",
        "",
        _table(window_state_summary),
        "",
        "## Basket-State Verdict",
        "",
        _table(basket_state_verdict),
        "",
        "## Culprit Attribution",
        "",
        _table(culprit_summary),
        "",
        "## Factor Verdict",
        "",
        _table(factor_verdict),
        "",
        "## Overlap With Price State",
        "",
        _table(factor_overlap),
        "",
        "## Factor Schema",
        "",
        _table(factor_schema),
        "",
        "## Read",
        "",
        "- The success criterion here is not standalone IC. It is whether the non-price factor triggers line up with known bad windows and dangerous names inside the candidate baskets.",
        "- `phi_long_non_price` should rise when long candidates look more like distress than oversold losers.",
        "- `phi_short_non_price` should rise when short candidates look more like genuine improvers than crowding-only winners.",
        "- `short_filing_confirmation_score` is included as a confirmation layer, but it is not treated as a direct observable of `Phi_short,t`.",
        "- In this MVP, long-side stock-level diagnostics are more promising than basket-level `phi` aggregation, while short-side `Phi_short` observability is still underpowered without revisions / borrow / short-interest channels.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(_fmt_float_like)
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def _mean_available(series_list: Sequence[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def _fmt_float_like(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if abs(number) >= 1000 or (abs(number) > 0 and abs(number) < 1e-4):
        return f"{number:.3e}"
    return f"{number:.6g}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--long-non-price-panel-path", default=str(DEFAULT_LONG_NON_PRICE_PANEL))
    parser.add_argument("--short-non-price-panel-path", default=str(DEFAULT_SHORT_NON_PRICE_PANEL))
    parser.add_argument("--price-state-frame-path", default=str(DEFAULT_PRICE_STATE_FRAME))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--long-variant", default=DEFAULT_LONG_VARIANT)
    parser.add_argument("--short-variant", default=DEFAULT_SHORT_VARIANT)
    parser.add_argument("--long-score", default=DEFAULT_LONG_SCORE)
    parser.add_argument("--short-selector", default=DEFAULT_SHORT_OVERLAY)
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--trigger-threshold", type=float, default=0.75)
    parser.add_argument("--recent-filing-days", type=int, default=20)
    parser.add_argument("--recent-insider-buy-days", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = build_non_price_phase5n_state_factor_artifacts(
        signal_panel_path=args.signal_panel_path,
        long_non_price_panel_path=args.long_non_price_panel_path,
        short_non_price_panel_path=args.short_non_price_panel_path,
        price_state_frame_path=args.price_state_frame_path,
        output_root=args.output_root,
        long_variant=args.long_variant,
        short_variant=args.short_variant,
        long_score=args.long_score,
        short_selector=args.short_selector,
        candidate_count=args.candidate_count,
        trigger_threshold=args.trigger_threshold,
        recent_filing_days=args.recent_filing_days,
        recent_insider_buy_days=args.recent_insider_buy_days,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
