from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4aa import _add_former_winner_scores
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4af import _add_falling_knife_overlay_scores
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_VARIANT,
    _add_residual_targets,
    _load_panel,
    _load_sec_sic_map,
    _skip_row,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _construct_one_session_turnover_aware,
    _diagnostic_summary,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_SHORT_OVERLAY


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5j_whitebox_state_activation_20260428"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_TURNOVER_PENALTY = 0.005
DEFAULT_BENCHMARK_SYMBOL = "SPY"
CURRENT_SOTA_SERIES = "current sota"
STATE_SCORE_THRESHOLD = 0.50
MARKET_RET20_THRESHOLD = -0.03

STATE_SOFT_SCORE = "long_reversal_soft_structural_weak_overlay"
STATE_HARD_SCORE = "long_reversal_top70_veto_structural_weak_t050"
SCORE_A_COLUMN = "whitebox_state_score_a"
SPY_RET20_COLUMN = "whitebox_spy_ret20"

LONG_STATE_SCORE_COLUMNS = {
    "long_state_soft_score050": {
        "gate_col": "gate_score050",
        "on_score": STATE_SOFT_SCORE,
    },
    "long_state_hard_score050": {
        "gate_col": "gate_score050",
        "on_score": STATE_HARD_SCORE,
    },
    "long_state_soft_score050_or_spy20m03": {
        "gate_col": "gate_score050_or_spy20m03",
        "on_score": STATE_SOFT_SCORE,
    },
    "long_state_hard_score050_or_spy20m03": {
        "gate_col": "gate_score050_or_spy20m03",
        "on_score": STATE_HARD_SCORE,
    },
}

COMBO_SPECS: tuple[dict[str, str], ...] = (
    {
        "portfolio": "current_sota",
        "series": CURRENT_SOTA_SERIES,
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "state_soft_score050",
        "series": "state soft score050",
        "long_score": "long_state_soft_score050",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "state_hard_score050",
        "series": "state hard score050",
        "long_score": "long_state_hard_score050",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "state_soft_score050_or_spy20m03",
        "series": "state soft score050+spy20",
        "long_score": "long_state_soft_score050_or_spy20m03",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "state_hard_score050_or_spy20m03",
        "series": "state hard score050+spy20",
        "long_score": "long_state_hard_score050_or_spy20m03",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
)

PLOT_COLORS = {
    CURRENT_SOTA_SERIES: "#2563eb",
    "state soft score050": "#0f766e",
    "state hard score050": "#c2410c",
    "state soft score050+spy20": "#7c3aed",
    "state hard score050+spy20": "#b45309",
    "SPY raw": "#6b7280",
}

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)


def build_phase5j_whitebox_state_activation_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = _add_falling_knife_overlay_scores(panel)
    state_frame = _build_whitebox_state_frame(
        panel=panel,
        long_variant=long_variant,
        benchmark_symbol=benchmark_symbol,
    )
    panel = _add_state_activated_long_scores(panel, state_frame=state_frame)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )

    positions, diagnostics, skipped = _construct_combo_positions(
        panel,
        long_variant=long_variant,
        short_variant=short_variant,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        soft_group_penalty=soft_group_penalty,
        turnover_penalty=turnover_penalty,
    )

    positions_path = output_dir / "phase5j_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5j_daily_validation.csv"
    skipped_path = output_dir / "phase5j_skipped_validation.csv"
    state_path = output_dir / "phase5j_state_frame.csv"
    activation_path = output_dir / "phase5j_activation_summary.csv"
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    state_frame.to_csv(state_path, index=False)
    _activation_summary(state_frame).to_csv(activation_path, index=False)

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

    benchmark_calendar = (
        strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    target_turnover = _target_turnover_summary(positions)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=benchmark_calendar.to_frame(name="session_date"),
        holding_period_sessions=holding_period_sessions,
    )
    strict_metrics = _strict_metric_summary(strict_curve)
    diagnostic_summary = _diagnostic_summary(diagnostics)

    sweep = (
        strict_metrics.merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(diagnostic_summary, on="portfolio", how="left")
    )
    sweep["series"] = sweep["portfolio"].map(_portfolio_to_series)
    sweep["long_score"] = sweep["portfolio"].map(_portfolio_to_long_score)
    sweep["short_selector"] = sweep["portfolio"].map(_portfolio_to_short_selector)
    sweep["turnover_penalty"] = float(turnover_penalty)

    metrics = _build_metrics(sweep=sweep, strict_curve=strict_curve, turnover_penalty=turnover_penalty)
    windows = _window_return_summary(strict_curve)
    comparison_curve = _build_comparison_curve(strict_curve)

    sweep_path = output_dir / "phase5j_sweep.csv"
    metrics_path = output_dir / "phase5j_metrics.csv"
    windows_path = output_dir / "phase5j_window_returns.csv"
    comparison_curve_path = output_dir / "phase5j_comparison_curve.csv"
    plot_path = output_dir / "phase5j_comparison_plot.png"
    memo_path = output_dir / "phase5j_whitebox_state_activation_memo.md"
    rollup_path = output_dir / "phase5j_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    windows.to_csv(windows_path, index=False)
    comparison_curve.to_csv(comparison_curve_path, index=False)
    _plot_comparison(comparison_curve, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            sweep=sweep,
            windows=windows,
            state_frame=state_frame,
            plot_path=plot_path,
            turnover_penalty=turnover_penalty,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "phase3_rollup_path": Path(phase3_rollup_path).as_posix(),
        "positions_artifact": positions_path.as_posix(),
        "daily_artifact": diagnostics_path.as_posix(),
        "skipped_artifact": skipped_path.as_posix(),
        "state_frame_artifact": state_path.as_posix(),
        "activation_summary_artifact": activation_path.as_posix(),
        "strict_rollup": strict_rollup,
        "sweep_artifact": sweep_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "window_returns_artifact": windows_path.as_posix(),
        "comparison_curve_artifact": comparison_curve_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "combo_specs": list(COMBO_SPECS),
        "state_score_threshold": STATE_SCORE_THRESHOLD,
        "market_ret20_threshold": MARKET_RET20_THRESHOLD,
        "turnover_penalty": float(turnover_penalty),
        "method": "whitebox_state_dependent_long_activation_on_current_sota",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_whitebox_state_frame(
    *,
    panel: pd.DataFrame,
    long_variant: str,
    benchmark_symbol: str,
) -> pd.DataFrame:
    long_panel = panel[panel["variant"].astype(str).eq(long_variant)].copy()
    long_panel["session_date"] = pd.to_datetime(long_panel["session_date"])
    rows: list[dict[str, Any]] = []
    for session_date, group in long_panel.groupby("session_date", sort=True):
        group = group.dropna(
            subset=[
                DEFAULT_LONG_SCORE,
                "falling_knife_structural_weakness",
                "beta_residual_momentum_20d_z",
                "momentum_60d_z",
                "beta_z",
            ]
        ).copy()
        if len(group) < 80:
            continue
        top = group.sort_values([DEFAULT_LONG_SCORE, "symbol"], ascending=[False, True]).head(80)
        share_struct_075 = float((top["falling_knife_structural_weakness"] > 0.75).mean())
        share_resid_bad = float((top["beta_residual_momentum_20d_z"] < -0.5).mean())
        share_beta_mom60_bad = float(
            ((top["beta_z"] > 0.5) & (top["momentum_60d_z"] < -0.5)).mean()
        )
        rows.append(
            {
                "session_date": session_date,
                "share_struct_075": share_struct_075,
                "share_resid_bad": share_resid_bad,
                "share_beta_mom60_bad": share_beta_mom60_bad,
                SCORE_A_COLUMN: float(
                    np.mean([share_struct_075, share_resid_bad, share_beta_mom60_bad])
                ),
            }
        )
    state = pd.DataFrame(rows)
    if state.empty:
        raise ValueError("No white-box state rows could be computed.")

    prices = _load_adjusted_prices(
        daily_globs=DEFAULT_BENCHMARK_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        symbols=(benchmark_symbol,),
        end_date="2019-12-31",
    )
    spy = prices[prices["symbol"].astype(str).eq(benchmark_symbol)].copy().sort_values("session_date")
    spy["session_date"] = pd.to_datetime(spy["session_date"])
    spy[SPY_RET20_COLUMN] = spy["adjusted_open"] / spy["adjusted_open"].shift(20) - 1.0
    state = state.merge(spy[["session_date", SPY_RET20_COLUMN]], on="session_date", how="left")
    state["gate_score050"] = state[SCORE_A_COLUMN] > STATE_SCORE_THRESHOLD
    state["gate_score050_or_spy20m03"] = state["gate_score050"] | (
        state[SPY_RET20_COLUMN] < MARKET_RET20_THRESHOLD
    )
    return state.sort_values("session_date").reset_index(drop=True)


def _add_state_activated_long_scores(
    panel: pd.DataFrame,
    *,
    state_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = panel.merge(state_frame, on="session_date", how="left")
    for new_score, spec in LONG_STATE_SCORE_COLUMNS.items():
        gate = merged[spec["gate_col"]].fillna(False)
        on_score = merged[spec["on_score"]]
        baseline = merged[DEFAULT_LONG_SCORE]
        merged[new_score] = baseline.where(~gate, on_score)
    return merged


def _activation_summary(state_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gate_col in ("gate_score050", "gate_score050_or_spy20m03"):
        rows.append(
            {
                "gate": gate_col,
                "activation_rate_full": float(state_frame[gate_col].astype(float).mean()),
            }
        )
        for window, start, end in WINDOWS:
            subset = state_frame[
                (state_frame["session_date"] >= pd.Timestamp(start))
                & (state_frame["session_date"] <= pd.Timestamp(end))
            ]
            rows.append(
                {
                    "gate": gate_col,
                    "window": window,
                    "start": start,
                    "end": end,
                    "activation_rate": float(subset[gate_col].astype(float).mean()) if not subset.empty else np.nan,
                    "mean_state_score": float(subset[SCORE_A_COLUMN].mean()) if not subset.empty else np.nan,
                    "mean_spy_ret20": float(subset[SPY_RET20_COLUMN].mean()) if not subset.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _construct_combo_positions(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    short_variant: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    previous_by_portfolio: dict[str, dict[str, dict[str, float]]] = {
        str(spec["portfolio"]): {"long": {}, "short": {}} for spec in COMBO_SPECS
    }

    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        previous_side = previous_by_portfolio[portfolio]
        for session_date in sessions:
            long_group = groups.get((long_variant, session_date))
            short_group = groups.get((short_variant, session_date))
            if long_group is None or short_group is None:
                skipped.append(_skip_row(session_date, portfolio, 0, 0, "missing_group"))
                continue
            book, diagnostic, skip, updated = _construct_one_session_turnover_aware(
                long_group,
                short_group,
                session_date=session_date,
                portfolio=portfolio,
                long_variant=long_variant,
                short_variant=short_variant,
                long_score=str(spec["long_score"]),
                short_selector=str(spec["short_selector"]),
                soft_group="sic2_sector",
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
                turnover_penalty=turnover_penalty,
                previous_weights=previous_side,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostic["series"] = str(spec["series"])
                diagnostics.append(diagnostic)
            if skip is not None:
                skip["series"] = str(spec["series"])
                skipped.append(skip)
            previous_side = updated
        previous_by_portfolio[portfolio] = previous_side

    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _portfolio_to_series(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["series"])
    return str(portfolio)


def _series_key(series: str) -> str:
    return str(series).replace(" ", "_").replace("-", "_").replace("+", "_plus_")


def _portfolio_to_long_score(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["long_score"])
    return ""


def _portfolio_to_short_selector(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["short_selector"])
    return ""


def _build_metrics(
    *,
    sweep: pd.DataFrame,
    strict_curve: pd.DataFrame,
    turnover_penalty: float,
) -> pd.DataFrame:
    ordered = [_portfolio_to_series(str(spec["portfolio"])) for spec in COMBO_SPECS]
    rows: list[dict[str, Any]] = []
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        row = sweep[sweep["portfolio"].eq(portfolio)]
        if row.empty:
            continue
        item = row.iloc[0]
        rows.append(
            {
                "series": str(spec["series"]),
                "portfolio": portfolio,
                "long_score": str(spec["long_score"]),
                "short_selector": str(spec["short_selector"]),
                "annualized_return": float(item["strict_annualized_return"]),
                "annualized_vol": float(item["strict_annualized_vol"]),
                "sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                "max_drawdown": float(item["strict_max_drawdown"]),
                "rolling_60_positive_rate": float(item["strict_rolling_60_positive_rate"]),
                "corr_to_spy": float(item["strict_corr_to_spy"]),
                "final_equity": float(item["strict_final_equity"]),
                "mean_daily_return_bps": float(
                    strict_curve[strict_curve["portfolio"].eq(portfolio)]["gross_return"].mean() * 10000.0
                ),
                "target_turnover_mean": float(item["target_turnover_mean"]),
                "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                "aggregate_positions_mean": float(item["aggregate_positions_mean"]),
                "aggregate_gross_mean": float(item["aggregate_gross_mean"]),
                "mean_abs_net_beta": float(item["mean_abs_net_beta"]),
                "mean_long_count": float(item["mean_long_count"]),
                "mean_short_count": float(item["mean_short_count"]),
                "turnover_penalty": float(turnover_penalty),
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
            "long_score": "",
            "short_selector": "",
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
            "mean_abs_net_beta": np.nan,
            "mean_long_count": np.nan,
            "mean_short_count": np.nan,
            "turnover_penalty": np.nan,
            "test_window_used": False,
        }
    )
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["series"].map(
        {name: i for i, name in enumerate([*ordered, "SPY raw"])}
    )
    return frame.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def _window_return_summary(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for portfolio, group in strict_curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date").copy()
        for window, start, end in WINDOWS:
            subset = frame[
                (frame["return_date"] >= pd.Timestamp(start))
                & (frame["return_date"] <= pd.Timestamp(end))
            ]
            if subset.empty:
                continue
            compound = float((1.0 + subset["gross_return"].astype(float)).prod() - 1.0)
            rows.append(
                {
                    "portfolio": portfolio,
                    "series": _portfolio_to_series(portfolio),
                    "window": window,
                    "start": start,
                    "end": end,
                    "compound_return": compound,
                    "hit_rate_daily": float((subset["gross_return"].astype(float) > 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "portfolio"]).reset_index(drop=True)


def _build_comparison_curve(strict_curve: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        strict_curve.sort_values(["return_date", "portfolio"])
        .drop_duplicates("return_date")[
            [
                "return_date",
                "benchmark_oto_return",
                "benchmark_equity",
                "benchmark_drawdown",
            ]
        ]
        .copy()
    )
    benchmark["benchmark_rolling_60_return"] = (
        benchmark["benchmark_equity"].astype(float)
        / benchmark["benchmark_equity"].astype(float).shift(60)
        - 1.0
    )
    out = benchmark
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        series = _series_key(str(spec["series"]))
        subset = strict_curve[strict_curve["portfolio"].eq(portfolio)].copy()
        subset = subset.rename(
            columns={
                "gross_return": f"{series}__return",
                "equity": f"{series}__equity",
                "drawdown": f"{series}__drawdown",
                "rolling_60_return": f"{series}__rolling_60_return",
            }
        )
        out = out.merge(
            subset[
                [
                    "return_date",
                    f"{series}__return",
                    f"{series}__equity",
                    f"{series}__drawdown",
                    f"{series}__rolling_60_return",
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

    for spec in COMBO_SPECS:
        label = str(spec["series"])
        key = _series_key(label)
        color = PLOT_COLORS[label]
        ax1.plot(dates, curve[f"{key}__equity"], label=label, color=color, linewidth=2.0)
        ax2.plot(dates, curve[f"{key}__drawdown"] * 100.0, color=color, linewidth=1.8)
        ax3.plot(dates, curve[f"{key}__rolling_60_return"] * 100.0, color=color, linewidth=1.8)

    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.8,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.6,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.6,
    )

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax3.set_xlabel("Date")
    ax1.set_title("Phase5J White-Box Long State Activation on Current SOTA (Validation Only)")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
    windows: pd.DataFrame,
    state_frame: pd.DataFrame,
    plot_path: Path,
    turnover_penalty: float,
) -> str:
    display_metrics = metrics.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
    ):
        if column in display_metrics.columns:
            display_metrics[column] = display_metrics[column].map(_fmt_pct_like)
    for column in (
        "sharpe_no_rf",
        "corr_to_spy",
        "mean_daily_return_bps",
        "target_turnover_mean",
        "aggregate_turnover_mean",
        "aggregate_positions_mean",
        "aggregate_gross_mean",
        "mean_abs_net_beta",
        "mean_long_count",
        "mean_short_count",
        "turnover_penalty",
    ):
        if column in display_metrics.columns:
            display_metrics[column] = display_metrics[column].map(_fmt_float_like)

    display_windows = windows.copy()
    if not display_windows.empty:
        display_windows["compound_return"] = display_windows["compound_return"].map(_fmt_pct_like)
        display_windows["hit_rate_daily"] = display_windows["hit_rate_daily"].map(_fmt_pct_like)

    shortlist = sweep[
        [
            "series",
            "long_score",
            "strict_annualized_return",
            "strict_sharpe_no_rf",
            "strict_max_drawdown",
            "aggregate_turnover_mean",
            "mean_abs_net_beta",
        ]
    ].copy()
    shortlist["strict_annualized_return"] = shortlist["strict_annualized_return"].map(_fmt_pct_like)
    shortlist["strict_max_drawdown"] = shortlist["strict_max_drawdown"].map(_fmt_pct_like)
    for column in ("strict_sharpe_no_rf", "aggregate_turnover_mean", "mean_abs_net_beta"):
        shortlist[column] = shortlist[column].map(_fmt_float_like)

    state_summary = pd.DataFrame(
        [
            {
                "metric": "state_score_mean",
                "value": float(state_frame[SCORE_A_COLUMN].mean()),
            },
            {
                "metric": "gate_score050_rate",
                "value": float(state_frame["gate_score050"].astype(float).mean()),
            },
            {
                "metric": "gate_score050_or_spy20m03_rate",
                "value": float(state_frame["gate_score050_or_spy20m03"].astype(float).mean()),
            },
            {
                "metric": "spy_ret20_mean",
                "value": float(state_frame[SPY_RET20_COLUMN].mean()),
            },
        ]
    )
    state_summary["value"] = state_summary["value"].map(_fmt_float_like)

    lines = [
        "# Phase5J White-Box Long State Activation on Current SOTA",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Baseline short side is fixed to `{DEFAULT_SHORT_OVERLAY}`.",
        f"- Flexible schedule stays fixed: turnover-aware daily optimizer with `lambda = {turnover_penalty:.4f}`.",
        f"- White-box state score = mean(`share_struct_075`, `share_resid_bad`, `share_beta_mom60_bad`) on the top-80 long reversal candidates.",
        f"- Market stress add-on = `{DEFAULT_BENCHMARK_SYMBOL}` 20-session return < {MARKET_RET20_THRESHOLD:.2%}.",
        f"- Activation threshold = `state_score > {STATE_SCORE_THRESHOLD:.2f}`.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## State Summary",
        "",
        _text_table(state_summary),
        "",
        "## Metrics",
        "",
        _text_table(display_metrics),
        "",
        "## Selector Summary",
        "",
        _text_table(shortlist),
        "",
        "## Stress Windows",
        "",
        _text_table(display_windows),
        "",
        "## Plot",
        "",
        f"![Phase5J comparison]({plot_path.as_posix()})",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run white-box state-dependent long activation on top of the current SOTA."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    args = parser.parse_args(argv)

    build_phase5j_whitebox_state_activation_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        turnover_penalty=args.turnover_penalty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
