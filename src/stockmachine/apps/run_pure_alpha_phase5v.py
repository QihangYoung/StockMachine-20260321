from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import _add_former_winner_scores
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4af import _add_falling_knife_overlay_scores
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_VARIANT,
    _load_panel,
    _load_sec_sic_map,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5r import _add_price_only_falling_knife_scores


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5v_fk_exposure_control_20260507"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_SOTA_POSITIONS = (
    RESEARCH_ROOT
    / "phase5f_overlay_combo_on_turnover_aware_20260427"
    / "phase5f_positions_validation.csv.gz"
)
DEFAULT_SOTA_STRICT_SLEEVES = (
    RESEARCH_ROOT
    / "phase5f_overlay_combo_on_turnover_aware_20260427"
    / "strict_h10_rebuild"
    / "phase4z_strict_h10_sleeve_returns.csv.gz"
)
DEFAULT_BASE_PORTFOLIO = "short_overlay_only"

SCALE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "portfolio": "current_sota",
        "series": "current sota",
        "kind": "identity",
        "threshold": np.nan,
        "floor": 1.0,
    },
    {
        "portfolio": "fk_linear_t085_floor075",
        "series": "fk linear t085 floor075",
        "kind": "linear",
        "threshold": 0.85,
        "floor": 0.75,
    },
    {
        "portfolio": "fk_linear_t085_floor050",
        "series": "fk linear t085 floor050",
        "kind": "linear",
        "threshold": 0.85,
        "floor": 0.50,
    },
    {
        "portfolio": "fk_linear_t080_floor050",
        "series": "fk linear t080 floor050",
        "kind": "linear",
        "threshold": 0.80,
        "floor": 0.50,
    },
    {
        "portfolio": "fk_hard_half_t085",
        "series": "fk hard half t085",
        "kind": "hard_floor",
        "threshold": 0.85,
        "floor": 0.50,
    },
    {
        "portfolio": "fk_hard_off_t0875",
        "series": "fk hard off t0875",
        "kind": "hard_floor",
        "threshold": 0.875,
        "floor": 0.00,
    },
    {
        "portfolio": "fk_hard_off_t085",
        "series": "fk hard off t085",
        "kind": "hard_floor",
        "threshold": 0.85,
        "floor": 0.00,
    },
)

PLOT_COLORS = {
    "current sota": "#2563eb",
    "fk linear t085 floor075": "#0f766e",
    "fk linear t085 floor050": "#0891b2",
    "fk linear t080 floor050": "#155e75",
    "fk hard half t085": "#c2410c",
    "fk hard off t0875": "#9333ea",
    "fk hard off t085": "#991b1b",
    "SPY raw": "#6b7280",
}


def build_phase5v_fk_exposure_control_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    sota_positions_path: str | Path = DEFAULT_SOTA_POSITIONS,
    sota_strict_sleeves_path: str | Path = DEFAULT_SOTA_STRICT_SLEEVES,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    base_portfolio: str = DEFAULT_BASE_PORTFOLIO,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_positions = _load_base_positions(sota_positions_path, base_portfolio=base_portfolio)
    panel = _price_state_panel(
        signal_panel_path=signal_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        long_variant=long_variant,
        short_variant=short_variant,
    )
    sleeve_exposure = _sleeve_price_state_exposure(base_positions, panel=panel)
    scaled_positions, scale_diagnostics = _scaled_positions(base_positions, sleeve_exposure)
    sleeve_tail_bins, sleeve_tail_thresholds = _sleeve_tail_diagnostics(
        sleeve_exposure=sleeve_exposure,
        strict_sleeves_path=sota_strict_sleeves_path,
        base_portfolio=base_portfolio,
    )

    positions_path = output_dir / "phase5v_scaled_positions_validation.csv.gz"
    scale_path = output_dir / "phase5v_sleeve_scale_diagnostics.csv"
    exposure_path = output_dir / "phase5v_sota_sleeve_fk_exposure.csv"
    tail_bins_path = output_dir / "phase5v_sleeve_tail_bins.csv"
    tail_thresholds_path = output_dir / "phase5v_sleeve_tail_thresholds.csv"
    scaled_positions.to_csv(positions_path, index=False, compression="gzip")
    scale_diagnostics.to_csv(scale_path, index=False)
    sleeve_exposure.to_csv(exposure_path, index=False)
    sleeve_tail_bins.to_csv(tail_bins_path, index=False)
    sleeve_tail_thresholds.to_csv(tail_thresholds_path, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=positions_path,
        output_root=strict_root,
        phase3_rollup_path=phase3_rollup_path,
        existing_curve_path=None,
        holding_period_sessions=holding_period_sessions,
        lead_portfolio="current_sota",
    )
    strict_curve_path = Path(strict_root) / "phase4z_strict_h10_daily_curve.csv"
    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])
    benchmark_calendar = strict_curve["return_date"].drop_duplicates().sort_values().to_frame(
        name="session_date"
    )
    strict_metrics = _strict_metric_summary(strict_curve)
    target_turnover = _target_turnover_summary(scaled_positions)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=benchmark_calendar,
        holding_period_sessions=holding_period_sessions,
    )
    scale_summary = _scale_summary(scale_diagnostics)
    metrics = _build_metrics(
        strict_metrics=strict_metrics,
        strict_curve=strict_curve,
        target_turnover=target_turnover,
        aggregate_turnover=aggregate_turnover,
        scale_summary=scale_summary,
    )
    comparison_curve = _comparison_curve(strict_curve)

    metrics_path = output_dir / "phase5v_metrics.csv"
    comparison_curve_path = output_dir / "phase5v_comparison_curve.csv"
    plot_path = output_dir / "phase5v_fk_exposure_control_plot.png"
    memo_path = output_dir / "phase5v_fk_exposure_control_memo.md"
    rollup_path = output_dir / "phase5v_rollup.json"
    metrics.to_csv(metrics_path, index=False)
    comparison_curve.to_csv(comparison_curve_path, index=False)
    _plot_comparison(comparison_curve, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            sleeve_tail_bins=sleeve_tail_bins,
            sleeve_tail_thresholds=sleeve_tail_thresholds,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "sota_positions_path": Path(sota_positions_path).as_posix(),
        "sota_strict_sleeves_path": Path(sota_strict_sleeves_path).as_posix(),
        "base_portfolio": base_portfolio,
        "scale_specs": list(SCALE_SPECS),
        "artifacts": {
            "positions": positions_path.as_posix(),
            "scale_diagnostics": scale_path.as_posix(),
            "sleeve_fk_exposure": exposure_path.as_posix(),
            "tail_bins": tail_bins_path.as_posix(),
            "tail_thresholds": tail_thresholds_path.as_posix(),
            "strict_curve": strict_curve_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "comparison_curve": comparison_curve_path.as_posix(),
            "plot": plot_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "strict_rollup": strict_rollup,
        "method": "scale_entire_beta_matched_sleeve_by_selected_long_falling_knife_exposure",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This experiment changes exposure, not selector ranking.",
            "Scaling long and short sleeves together preserves sleeve-level beta matching.",
            "No transaction costs, borrow costs, or financing costs are charged.",
            "Price falling-knife score is a tail-intensity signal, not a directional left-tail signal.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_base_positions(path: str | Path, *, base_portfolio: str) -> pd.DataFrame:
    usecols = [
        "session_date",
        "portfolio",
        "long_variant",
        "short_variant",
        "side",
        "symbol",
        "score_column",
        "score",
        "beta",
        "sic2_sector",
        "sic4_industry",
        "sic_description",
        "side_weight",
        "signed_weight",
        "test_window_used",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame[frame["portfolio"].astype(str).eq(base_portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No base positions found for portfolio {base_portfolio}.")
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    if frame["test_window_used"].map(_is_true).any():
        raise ValueError("Base positions include test-window rows; refusing to continue.")
    return frame.reset_index(drop=True)


def _price_state_panel(
    *,
    signal_panel_path: str | Path,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    long_variant: str,
    short_variant: str,
) -> pd.DataFrame:
    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = _add_falling_knife_overlay_scores(panel)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_price_only_falling_knife_scores(panel, long_variant=long_variant)
    return panel


def _sleeve_price_state_exposure(positions: pd.DataFrame, *, panel: pd.DataFrame) -> pd.DataFrame:
    factors = panel[
        [
            "session_date",
            "symbol",
            "price_falling_knife_score",
            "price_still_falling_score",
            "price_stabilization_score",
            "price_fk_stillfall_penalty",
            "price_fk_stab_relief_penalty",
        ]
    ].drop_duplicates(["session_date", "symbol"])
    longs = positions[positions["side"].eq("long")].copy()
    merged = longs.merge(factors, on=["session_date", "symbol"], how="left")
    score_columns = [
        "price_falling_knife_score",
        "price_still_falling_score",
        "price_stabilization_score",
        "price_fk_stillfall_penalty",
        "price_fk_stab_relief_penalty",
    ]
    for column in score_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for session_date, group in merged.groupby("session_date", sort=True):
        weights = group["side_weight"].astype(float)
        denom = float(weights.sum())
        if denom <= 0:
            continue
        row: dict[str, Any] = {
            "session_date": session_date,
            "long_names": int(len(group)),
            "long_side_weight_sum": denom,
        }
        for column in score_columns:
            row[f"weighted_{column}"] = float(np.dot(weights, group[column]) / denom)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)


def _scaled_positions(
    positions: pd.DataFrame,
    sleeve_exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale_by_date = {
        str(row.session_date): float(row.weighted_price_falling_knife_score)
        for row in sleeve_exposure.itertuples(index=False)
    }
    frames: list[pd.DataFrame] = []
    scale_rows: list[dict[str, Any]] = []
    for spec in SCALE_SPECS:
        portfolio = str(spec["portfolio"])
        frame = positions.copy()
        scales = frame["session_date"].map(
            lambda value: _scale_for_score(scale_by_date.get(str(value), 0.0), spec)
        )
        frame["portfolio"] = portfolio
        frame["side_weight"] = frame["side_weight"].astype(float) * scales.astype(float)
        frame["signed_weight"] = frame["signed_weight"].astype(float) * scales.astype(float)
        frames.append(frame)
        for session_date, scale in (
            frame[["session_date"]]
            .drop_duplicates()
            .assign(scale=lambda df: df["session_date"].map(
                lambda value: _scale_for_score(scale_by_date.get(str(value), 0.0), spec)
            ))
            .itertuples(index=False)
        ):
            fk = scale_by_date.get(str(session_date), np.nan)
            scale_rows.append(
                {
                    "portfolio": portfolio,
                    "series": str(spec["series"]),
                    "session_date": session_date,
                    "weighted_price_falling_knife_score": fk,
                    "scale": float(scale),
                    "target_gross_after_scale": float(2.0 * scale),
                    "kind": str(spec["kind"]),
                    "threshold": float(spec["threshold"])
                    if not pd.isna(spec["threshold"])
                    else np.nan,
                    "floor": float(spec["floor"]),
                    "test_window_used": False,
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(scale_rows)


def _scale_for_score(score: float, spec: dict[str, Any]) -> float:
    kind = str(spec["kind"])
    if kind == "identity":
        return 1.0
    threshold = float(spec["threshold"])
    floor = float(spec["floor"])
    if kind == "hard_floor":
        return floor if score >= threshold else 1.0
    if kind == "linear":
        if score <= threshold:
            return 1.0
        if threshold >= 1.0:
            return floor
        severity = min(1.0, max(0.0, (score - threshold) / (1.0 - threshold)))
        return max(floor, 1.0 - (1.0 - floor) * severity)
    raise ValueError(f"Unsupported scale kind: {kind}")


def _sleeve_tail_diagnostics(
    *,
    sleeve_exposure: pd.DataFrame,
    strict_sleeves_path: str | Path,
    base_portfolio: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sleeves = pd.read_csv(
        strict_sleeves_path,
        parse_dates=["sleeve_session_date", "return_date"],
    )
    sleeves = sleeves[
        sleeves["portfolio"].astype(str).eq(base_portfolio)
        & sleeves["complete_sleeve"].map(_is_true)
    ].copy()
    sleeve_returns = (
        sleeves.groupby("sleeve_session_date", as_index=False)
        .agg(
            h10_return=("sleeve_gross_return", lambda s: float((1.0 + s.astype(float)).prod() - 1.0)),
            h10_simple_sum=("sleeve_gross_return", "sum"),
            h10_long_sum=("sleeve_long_return", "sum"),
            h10_short_sum=("sleeve_short_return", "sum"),
            days=("sleeve_gross_return", "size"),
        )
        .rename(columns={"sleeve_session_date": "session_date"})
    )
    sleeve_returns["session_date"] = pd.to_datetime(sleeve_returns["session_date"]).dt.date.astype(str)
    exposure = sleeve_exposure.copy()
    exposure["session_date"] = pd.to_datetime(exposure["session_date"]).dt.date.astype(str)
    merged = sleeve_returns.merge(exposure, on="session_date", how="inner")
    merged["fk_quintile"] = pd.qcut(
        merged["weighted_price_falling_knife_score"],
        q=5,
        labels=False,
        duplicates="drop",
    )
    bins = (
        merged.groupby("fk_quintile", as_index=False)
        .agg(
            sleeves=("h10_return", "size"),
            mean_fk=("weighted_price_falling_knife_score", "mean"),
            mean_h10_return=("h10_return", "mean"),
            median_h10_return=("h10_return", "median"),
            left_tail_2pct_rate=("h10_return", lambda s: float((s < -0.02).mean())),
            right_tail_2pct_rate=("h10_return", lambda s: float((s > 0.02).mean())),
            left_tail_5pct_rate=("h10_return", lambda s: float((s < -0.05).mean())),
            right_tail_5pct_rate=("h10_return", lambda s: float((s > 0.05).mean())),
        )
        .sort_values("fk_quintile")
    )
    thresholds = []
    total_sleeves = int(len(merged))
    for threshold in (0.75, 0.80, 0.825, 0.85, 0.875, 0.90):
        for bucket, mask in (
            ("high_fk", merged["weighted_price_falling_knife_score"] >= threshold),
            ("low_fk", merged["weighted_price_falling_knife_score"] < threshold),
        ):
            subset = merged[mask].copy()
            thresholds.append(
                _tail_row(
                    subset,
                    threshold=threshold,
                    bucket=bucket,
                    total_sleeves=total_sleeves,
                )
            )
    return bins, pd.DataFrame(thresholds)


def _tail_row(
    subset: pd.DataFrame,
    *,
    threshold: float,
    bucket: str,
    total_sleeves: int,
) -> dict[str, Any]:
    if subset.empty:
        return {
            "threshold": threshold,
            "bucket": bucket,
            "sleeves": 0,
            "share_sleeves": 0.0,
            "mean_fk": np.nan,
            "mean_h10_return": np.nan,
            "median_h10_return": np.nan,
            "left_tail_2pct_rate": np.nan,
            "right_tail_2pct_rate": np.nan,
            "left_tail_5pct_rate": np.nan,
            "right_tail_5pct_rate": np.nan,
            "sum_h10_return": 0.0,
            "test_window_used": False,
        }
    total = len(subset)
    denominator = max(1, int(total_sleeves))
    return {
        "threshold": threshold,
        "bucket": bucket,
        "sleeves": int(total),
        "share_sleeves": float(total / denominator),
        "mean_fk": float(subset["weighted_price_falling_knife_score"].mean()),
        "mean_h10_return": float(subset["h10_return"].mean()),
        "median_h10_return": float(subset["h10_return"].median()),
        "left_tail_2pct_rate": float((subset["h10_return"] < -0.02).mean()),
        "right_tail_2pct_rate": float((subset["h10_return"] > 0.02).mean()),
        "left_tail_5pct_rate": float((subset["h10_return"] < -0.05).mean()),
        "right_tail_5pct_rate": float((subset["h10_return"] > 0.05).mean()),
        "sum_h10_return": float(subset["h10_return"].sum()),
        "test_window_used": False,
    }


def _scale_summary(scale_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in scale_diagnostics.groupby("portfolio", sort=True):
        rows.append(
            {
                "portfolio": portfolio,
                "mean_scale": float(group["scale"].mean()),
                "median_scale": float(group["scale"].median()),
                "scale_lt_one_rate": float((group["scale"] < 1.0).mean()),
                "scale_eq_zero_rate": float((group["scale"] <= 1e-12).mean()),
                "mean_target_gross_after_scale": float(group["target_gross_after_scale"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_metrics(
    *,
    strict_metrics: pd.DataFrame,
    strict_curve: pd.DataFrame,
    target_turnover: pd.DataFrame,
    aggregate_turnover: pd.DataFrame,
    scale_summary: pd.DataFrame,
) -> pd.DataFrame:
    sweep = (
        strict_metrics.merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(scale_summary, on="portfolio", how="left")
    )
    rows: list[dict[str, Any]] = []
    for spec in SCALE_SPECS:
        portfolio = str(spec["portfolio"])
        row = sweep[sweep["portfolio"].astype(str).eq(portfolio)]
        if row.empty:
            continue
        item = row.iloc[0]
        returns = strict_curve[strict_curve["portfolio"].astype(str).eq(portfolio)][
            "gross_return"
        ].astype(float)
        rows.append(
            {
                "series": str(spec["series"]),
                "portfolio": portfolio,
                "annualized_return": float(item["strict_annualized_return"]),
                "annualized_vol": float(item["strict_annualized_vol"]),
                "sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                "max_drawdown": float(item["strict_max_drawdown"]),
                "rolling_60_positive_rate": float(item["strict_rolling_60_positive_rate"]),
                "corr_to_spy": float(item["strict_corr_to_spy"]),
                "final_equity": float(item["strict_final_equity"]),
                "mean_daily_return_bps": float(returns.mean() * 10000.0),
                "target_turnover_mean": float(item["target_turnover_mean"]),
                "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                "aggregate_positions_mean": float(item["aggregate_positions_mean"]),
                "aggregate_gross_mean": float(item["aggregate_gross_mean"]),
                "mean_scale": float(item["mean_scale"]),
                "scale_lt_one_rate": float(item["scale_lt_one_rate"]),
                "scale_eq_zero_rate": float(item["scale_eq_zero_rate"]),
                "mean_target_gross_after_scale": float(item["mean_target_gross_after_scale"]),
                "test_window_used": False,
            }
        )
    benchmark = strict_curve.sort_values(["return_date", "portfolio"]).drop_duplicates("return_date")
    spy_returns = benchmark["benchmark_oto_return"].astype(float)
    spy_equity = (1.0 + spy_returns).cumprod()
    spy_drawdown = spy_equity / spy_equity.cummax() - 1.0
    spy_rolling_60 = spy_equity / spy_equity.shift(60) - 1.0
    rows.append(
        {
            "series": "SPY raw",
            "portfolio": "SPY",
            "annualized_return": float(spy_equity.iloc[-1] ** (252.0 / len(spy_equity)) - 1.0),
            "annualized_vol": float(spy_returns.std(ddof=1) * np.sqrt(252.0)),
            "sharpe_no_rf": float(
                spy_returns.mean() / spy_returns.std(ddof=1) * np.sqrt(252.0)
                if spy_returns.std(ddof=1) > 0
                else np.nan
            ),
            "max_drawdown": float(spy_drawdown.min()),
            "rolling_60_positive_rate": float((spy_rolling_60 > 0).mean()),
            "corr_to_spy": 1.0,
            "final_equity": float(spy_equity.iloc[-1]),
            "mean_daily_return_bps": float(spy_returns.mean() * 10000.0),
            "target_turnover_mean": np.nan,
            "aggregate_turnover_mean": np.nan,
            "aggregate_positions_mean": np.nan,
            "aggregate_gross_mean": np.nan,
            "mean_scale": np.nan,
            "scale_lt_one_rate": np.nan,
            "scale_eq_zero_rate": np.nan,
            "mean_target_gross_after_scale": np.nan,
            "test_window_used": False,
        }
    )
    order = {str(spec["series"]): i for i, spec in enumerate(SCALE_SPECS)}
    order["SPY raw"] = len(order)
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["series"].map(order)
    return frame.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def _comparison_curve(strict_curve: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        strict_curve.sort_values(["return_date", "portfolio"])
        .drop_duplicates("return_date")[
            ["return_date", "benchmark_oto_return", "benchmark_equity", "benchmark_drawdown"]
        ]
        .copy()
    )
    benchmark["benchmark_rolling_60_return"] = (
        benchmark["benchmark_equity"].astype(float)
        / benchmark["benchmark_equity"].astype(float).shift(60)
        - 1.0
    )
    out = benchmark
    for spec in SCALE_SPECS:
        portfolio = str(spec["portfolio"])
        key = _series_key(str(spec["series"]))
        subset = strict_curve[strict_curve["portfolio"].astype(str).eq(portfolio)].copy()
        subset = subset.rename(
            columns={
                "gross_return": f"{key}__return",
                "equity": f"{key}__equity",
                "drawdown": f"{key}__drawdown",
                "rolling_60_return": f"{key}__rolling_60_return",
            }
        )
        out = out.merge(
            subset[
                [
                    "return_date",
                    f"{key}__return",
                    f"{key}__equity",
                    f"{key}__drawdown",
                    f"{key}__rolling_60_return",
                ]
            ],
            on="return_date",
            how="left",
        )
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


def _plot_comparison(curve: pd.DataFrame, *, output_path: str | Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8, 1.8]},
    )
    ax1, ax2, ax3 = axes
    dates = curve["return_date"]
    for spec in SCALE_SPECS:
        label = str(spec["series"])
        key = _series_key(label)
        color = PLOT_COLORS[label]
        ax1.plot(dates, curve[f"{key}__equity"], label=label, color=color, linewidth=1.45)
        ax2.plot(dates, curve[f"{key}__drawdown"] * 100.0, color=color, linewidth=1.15)
        ax3.plot(dates, curve[f"{key}__rolling_60_return"] * 100.0, color=color, linewidth=1.15)
    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.25,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.0,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.0,
    )
    ax1.set_title("Phase5V Falling-Knife Exposure Control")
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax3.set_xlabel("Date")
    for ax in axes:
        ax.grid(True, alpha=0.22)
    ax1.axhline(1.0, color="#374151", linestyle="--", linewidth=0.8)
    ax2.axhline(0.0, color="#374151", linestyle="--", linewidth=0.8)
    ax3.axhline(0.0, color="#374151", linestyle="--", linewidth=0.8)
    ax1.legend(loc="best", fontsize=9)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    sleeve_tail_bins: pd.DataFrame,
    sleeve_tail_thresholds: pd.DataFrame,
    plot_path: Path,
) -> str:
    display_metrics = metrics.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
        "scale_lt_one_rate",
        "scale_eq_zero_rate",
    ):
        display_metrics[column] = display_metrics[column].map(_fmt_pct_like)
    for column in (
        "sharpe_no_rf",
        "corr_to_spy",
        "final_equity",
        "mean_daily_return_bps",
        "target_turnover_mean",
        "aggregate_turnover_mean",
        "aggregate_positions_mean",
        "aggregate_gross_mean",
        "mean_scale",
        "mean_target_gross_after_scale",
    ):
        display_metrics[column] = display_metrics[column].map(_fmt_float_like)

    display_bins = sleeve_tail_bins.copy()
    for column in (
        "mean_h10_return",
        "median_h10_return",
        "left_tail_2pct_rate",
        "right_tail_2pct_rate",
        "left_tail_5pct_rate",
        "right_tail_5pct_rate",
    ):
        display_bins[column] = display_bins[column].map(_fmt_pct_like)
    display_bins["mean_fk"] = display_bins["mean_fk"].map(_fmt_float_like)

    display_thresholds = sleeve_tail_thresholds.copy()
    for column in (
        "share_sleeves",
        "mean_h10_return",
        "median_h10_return",
        "left_tail_2pct_rate",
        "right_tail_2pct_rate",
        "left_tail_5pct_rate",
        "right_tail_5pct_rate",
    ):
        display_thresholds[column] = display_thresholds[column].map(_fmt_pct_like)
    for column in ("mean_fk", "sum_h10_return"):
        display_thresholds[column] = display_thresholds[column].map(_fmt_float_like)

    lines = [
        "# Phase5V Falling-Knife Exposure Control",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "- Start from the current SOTA target positions.",
        "- Do not change selectors or optimizer-selected names.",
        "- Compute selected-long weighted price falling-knife exposure per decision sleeve.",
        "- Scale both long and short sides of the whole sleeve by the same multiplier.",
        "- This preserves sleeve-level beta matching while reducing gross exposure in high-tail states.",
        "- Validation only; no test-window rows are used.",
        "",
        "## Metrics",
        "",
        _text_table(display_metrics),
        "",
        "## Original SOTA Sleeve Returns by Falling-Knife Quintile",
        "",
        _text_table(display_bins),
        "",
        "## Threshold Tail Diagnostic",
        "",
        _text_table(display_thresholds),
        "",
        "## Interpretation",
        "",
        "High falling-knife exposure behaves more like a tail-intensity signal than a left-tail "
        "direction signal. In validation, the highest falling-knife sleeves have both higher "
        "left-tail incidence and substantially higher right-tail incidence. Therefore, turning "
        "off high-FK sleeves removes bad tails but also removes many of the strategy's best rebounds.",
        "",
        "## Plot",
        "",
        f"![Phase5V comparison]({plot_path.as_posix()})",
        "",
    ]
    return "\n".join(lines)


def _series_key(series: str) -> str:
    return str(series).replace(" ", "_").replace("-", "_").replace(".", "p")


def _is_true(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "t", "yes", "y", "1"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase5V falling-knife exposure control.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--sota-positions-path", default=str(DEFAULT_SOTA_POSITIONS))
    parser.add_argument("--sota-strict-sleeves-path", default=str(DEFAULT_SOTA_STRICT_SLEEVES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--base-portfolio", default=DEFAULT_BASE_PORTFOLIO)
    args = parser.parse_args(argv)
    rollup = build_phase5v_fk_exposure_control_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        sota_positions_path=args.sota_positions_path,
        sota_strict_sleeves_path=args.sota_strict_sleeves_path,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        base_portfolio=args.base_portfolio,
    )
    print(json.dumps({"ok": True, "result": rollup}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
