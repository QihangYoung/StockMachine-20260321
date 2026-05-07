"""Phase6E hard ADV-floor experiment for current SOTA.

The goal is not to find a new alpha. It tests whether a point-in-time
liquidity floor can remove the Phase6D low-ADV tail while preserving the
existing SOTA economics and beta-neutral construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _construct_one_session_turnover_aware,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5f import (
    DEFAULT_SHORT_OVERLAY,
    DEFAULT_TURNOVER_PENALTY,
)
from stockmachine.apps.run_pure_alpha_phase6 import (
    _build_backtest_records,
    _build_daily_aggregate_positions,
    _load_strict_curve,
)
from stockmachine.apps.run_pure_alpha_phase6d import (
    CAPITAL_LEVELS,
    _build_low_adv_gate_summary,
    _build_order_liquidity,
    _build_position_liquidity,
    _load_liquidity,
)


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT_PROJECT = RESEARCH_ROOT
PHASE5F_ROOT = RESEARCH_ROOT_PROJECT / "phase5f_overlay_combo_on_turnover_aware_20260427"
EXISTING_SOTA_POSITIONS = PHASE5F_ROOT / "phase5f_positions_validation.csv.gz"
OUTDIR = RESEARCH_ROOT_PROJECT / "phase6e_adv_floor_hardening_20260507"
STRICT_ROOT = OUTDIR / "strict_h10_rebuild"
COMBINED_POSITIONS_PATH = OUTDIR / "phase6e_positions_validation.csv.gz"

ADV_FLOORS: tuple[float, ...] = (0.0, 1_000_000.0, 5_000_000.0, 10_000_000.0, 30_000_000.0)
SOTA_SOURCE_PORTFOLIO = "short_overlay_only"
CAPITAL_USD = 500_000


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio: str
    series: str
    adv_floor_usd: float


def _floor_label(floor: float) -> str:
    if floor <= 0:
        return "0"
    if floor >= 1_000_000:
        return f"{int(floor / 1_000_000)}m"
    return str(int(floor))


def _portfolio_for_floor(floor: float) -> str:
    return f"adv_floor_{_floor_label(floor)}"


def _series_for_floor(floor: float) -> str:
    if floor <= 0:
        return "current SOTA"
    return f"ADV floor ${_floor_label(floor).upper()}"


def _load_feature_panel(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
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
    return _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )


def _construct_one_floor(
    panel: pd.DataFrame,
    spec: PortfolioSpec,
    *,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = panel.copy()
    if spec.adv_floor_usd > 0:
        eligible = eligible[
            eligible["trailing_median_dollar_volume_20"].fillna(0.0) >= spec.adv_floor_usd
        ].copy()

    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in eligible.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    previous_side: dict[str, dict[str, float]] = {"long": {}, "short": {}}

    for session_date in sessions:
        long_group = groups.get((long_variant, session_date))
        short_group = groups.get((short_variant, session_date))
        if long_group is None or short_group is None:
            skipped.append(_skip_row(session_date, spec.portfolio, 0, 0, "missing_group"))
            continue
        book, diagnostic, skip, updated = _construct_one_session_turnover_aware(
            long_group,
            short_group,
            session_date=session_date,
            portfolio=spec.portfolio,
            long_variant=long_variant,
            short_variant=short_variant,
            long_score=DEFAULT_LONG_SCORE,
            short_selector=DEFAULT_SHORT_OVERLAY,
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
            diagnostic["series"] = spec.series
            diagnostic["adv_floor_usd"] = spec.adv_floor_usd
            diagnostics.append(diagnostic)
        if skip is not None:
            skip["series"] = spec.series
            skip["adv_floor_usd"] = spec.adv_floor_usd
            skipped.append(skip)
        previous_side = updated

    positions_frame = pd.DataFrame(positions)
    if not positions_frame.empty:
        positions_frame["series"] = spec.series
        positions_frame["adv_floor_usd"] = spec.adv_floor_usd
    return positions_frame, pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _load_existing_sota_as_floor0() -> pd.DataFrame:
    frame = pd.read_csv(EXISTING_SOTA_POSITIONS, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq(SOTA_SOURCE_PORTFOLIO)].copy()
    frame["portfolio"] = _portfolio_for_floor(0.0)
    frame["series"] = _series_for_floor(0.0)
    frame["adv_floor_usd"] = 0.0
    return frame


def _build_positions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _load_feature_panel()
    all_positions = [_load_existing_sota_as_floor0()]
    all_diagnostics: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    for floor in ADV_FLOORS:
        if floor <= 0:
            continue
        spec = PortfolioSpec(
            portfolio=_portfolio_for_floor(floor),
            series=_series_for_floor(floor),
            adv_floor_usd=floor,
        )
        positions, diagnostics, skipped = _construct_one_floor(panel, spec)
        all_positions.append(positions)
        all_diagnostics.append(diagnostics)
        all_skipped.append(skipped)

    positions = pd.concat(all_positions, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    skipped = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    for frame in (positions, diagnostics, skipped):
        if not frame.empty and "session_date" in frame:
            frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d")
    return positions, diagnostics, skipped


def _build_capacity_for_portfolio(
    positions_path: Path,
    strict_curve_path: Path,
    portfolio: str,
    liquidity: pd.DataFrame,
) -> dict[str, Any]:
    curve = _load_strict_curve(strict_curve_path, portfolio=portfolio)
    positions = pd.read_csv(positions_path, parse_dates=["session_date"])
    positions = positions[positions["portfolio"].astype(str).eq(portfolio)].copy()
    calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    aggregate_positions, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
    )
    turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=calendar.to_frame(name="session_date"),
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
    )
    turnover_curve = turnover[turnover["portfolio"].astype(str).eq(portfolio)]
    # _aggregate_turnover_summary is summary-level, so use the Phase6D order diff for
    # capacity while keeping strict metrics from the rebuilt h10 path.
    inputs = type(
        "Inputs",
        (),
        {
            "aggregate_positions": aggregate_positions,
            "strict_curve": curve,
            "records": _build_backtest_records(
                curve=curve,
                turnover=_empty_turnover_curve(curve),
                daily_exposure=daily_exposure,
            ),
            "liquidity": liquidity,
        },
    )()
    position_liq = _build_position_liquidity(inputs)
    orders, _ = _build_order_liquidity(inputs)
    low_adv, _ = _build_low_adv_gate_summary(position_liq, orders)
    adv_1m = low_adv.loc[low_adv["adv_floor_usd"] == 1_000_000.0].iloc[0]
    return {
        "portfolio": portfolio,
        "positions_rows": int(len(positions)),
        "mean_abs_net_beta": float(
            (positions["signed_weight"].astype(float) * positions["beta"].astype(float))
            .groupby(positions["session_date"])
            .sum()
            .abs()
            .mean()
        ),
        "position_p99_adv_participation_500k": float(
            np.nanquantile(position_liq["position_adv_share"] * CAPITAL_USD, 0.99)
        ),
        "position_max_adv_participation_500k": float(
            np.nanmax(position_liq["position_adv_share"] * CAPITAL_USD)
        ),
        "order_p99_adv_participation_500k": float(
            np.nanquantile(orders["order_adv_share"] * CAPITAL_USD, 0.99)
        ),
        "order_max_adv_participation_500k": float(
            np.nanmax(orders["order_adv_share"] * CAPITAL_USD)
        ),
        "position_abs_weight_share_below_1m_adv": float(
            adv_1m["position_abs_weight_share_below_floor"]
        ),
        "order_abs_delta_share_below_1m_adv": float(
            adv_1m["order_abs_delta_share_below_floor"]
        ),
    }


def _empty_turnover_curve(curve: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "return_date": curve["return_date"],
            "h10_turnover": 0.0,
            "h10_positions": curve.get("positions", pd.Series(np.nan, index=curve.index)),
        }
    )


def _build_metrics(strict_curve: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    strict_metrics = _strict_metric_summary(strict_curve)
    target_turnover = _target_turnover_summary(positions)
    benchmark_calendar = strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=COMBINED_POSITIONS_PATH,
        benchmark_returns=benchmark_calendar.to_frame(name="session_date"),
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
    )
    floors = pd.DataFrame(
        [
            {
                "portfolio": _portfolio_for_floor(floor),
                "series": _series_for_floor(floor),
                "adv_floor_usd": floor,
            }
            for floor in ADV_FLOORS
        ]
    )
    return (
        floors.merge(strict_metrics, on="portfolio", how="left")
        .merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .sort_values("adv_floor_usd")
        .reset_index(drop=True)
    )


def _build_curve_export(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for floor in ADV_FLOORS:
        portfolio = _portfolio_for_floor(floor)
        sub = strict_curve[strict_curve["portfolio"].astype(str).eq(portfolio)].copy()
        if sub.empty:
            continue
        sub["series"] = _series_for_floor(floor)
        sub["equity"] = (1.0 + sub["gross_return"].astype(float)).cumprod()
        sub["drawdown"] = sub["equity"] / sub["equity"].cummax() - 1.0
        rows.append(
            sub[
                [
                    "return_date",
                    "portfolio",
                    "series",
                    "gross_return",
                    "benchmark_oto_return",
                    "equity",
                    "drawdown",
                    "test_window_used",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot(curve: pd.DataFrame, metrics: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {
        "current SOTA": "#2563eb",
        "ADV floor $1M": "#0f766e",
        "ADV floor $5M": "#c2410c",
        "ADV floor $10M": "#7c3aed",
        "ADV floor $30M": "#6b7280",
    }
    for series, group in curve.groupby("series", sort=False):
        axes[0].plot(group["return_date"], group["equity"], label=series, color=colors.get(series))
        axes[1].plot(group["return_date"], group["drawdown"] * 100, label=series, color=colors.get(series))
    axes[0].set_title("Phase6E Hard ADV-Floor SOTA Variants (Validation Only)")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    # Keep the metrics argument intentional; this catches empty metrics during manual runs.
    if metrics.empty:
        raise ValueError("No Phase6E metrics were generated.")


def _write_memo(metrics: pd.DataFrame, capacity: pd.DataFrame, outpath: Path) -> None:
    view = metrics.merge(capacity, on="portfolio", how="left")
    cols = [
        "series",
        "adv_floor_usd",
        "strict_annualized_return",
        "strict_sharpe_no_rf",
        "strict_max_drawdown",
        "aggregate_turnover_mean",
        "position_abs_weight_share_below_1m_adv",
        "order_max_adv_participation_500k",
    ]
    table = view[cols].copy()
    text = ["# Phase6E Hard ADV-Floor Memo", ""]
    text.append("Scope: validation-window only; test window remains unused.")
    text.append("")
    text.append("This experiment keeps the current SOTA selector and optimizer logic fixed, then adds a hard point-in-time ADV floor before portfolio construction. It tests whether the Phase6D low-ADV tail can be removed without materially damaging the strategy.")
    text.append("")
    text.append("## Results")
    text.append("")
    text.append("```text")
    text.append(table.to_string(index=False))
    text.append("```")
    text.append("")
    text.append("## Interpretation")
    text.append("")
    text.append("- If a low ADV floor preserves return and Sharpe while reducing max ADV participation, it is a clean production hardening candidate.")
    text.append("- If performance falls sharply, the current SOTA is leaning on less-liquid long-book names and the universe definition must be revisited before paper trading.")
    text.append("- This is still not a full production cost model: borrow fees, broker routing, and margin liquidation rules remain separate Phase6 gates.")
    outpath.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    strict_curve_path = STRICT_ROOT / "phase4z_strict_h10_daily_curve.csv"
    if COMBINED_POSITIONS_PATH.exists() and strict_curve_path.exists():
        positions = pd.read_csv(COMBINED_POSITIONS_PATH, parse_dates=["session_date"])
        strict_rollup = {
            "reused_existing_artifacts": True,
            "positions_path": COMBINED_POSITIONS_PATH.as_posix(),
            "strict_curve_path": strict_curve_path.as_posix(),
        }
    else:
        positions, diagnostics, skipped = _build_positions()
        positions.to_csv(COMBINED_POSITIONS_PATH, index=False, compression="gzip")
        diagnostics.to_csv(OUTDIR / "phase6e_daily_validation.csv", index=False)
        skipped.to_csv(OUTDIR / "phase6e_skipped_validation.csv", index=False)

        strict_rollup = build_phase4z_strict_horizon_daily_paths(
            positions_path=COMBINED_POSITIONS_PATH,
            output_root=STRICT_ROOT,
            phase3_rollup_path=DEFAULT_PHASE3_H10_ROLLUP,
            existing_curve_path=None,
            holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
            lead_portfolio=_portfolio_for_floor(1_000_000.0),
        )
    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])
    metrics = _build_metrics(strict_curve, positions)
    curve = _build_curve_export(strict_curve)

    liquidity = _load_liquidity()
    capacity_rows = [
        _build_capacity_for_portfolio(
            COMBINED_POSITIONS_PATH,
            strict_curve_path,
            _portfolio_for_floor(floor),
            liquidity,
        )
        for floor in ADV_FLOORS
    ]
    capacity = pd.DataFrame(capacity_rows)

    metrics.to_csv(OUTDIR / "phase6e_adv_floor_metrics.csv", index=False)
    capacity.to_csv(OUTDIR / "phase6e_adv_floor_capacity_500k.csv", index=False)
    curve.to_csv(OUTDIR / "phase6e_adv_floor_curve.csv", index=False)
    _plot(curve, metrics, OUTDIR / "phase6e_adv_floor_plot.png")
    _write_memo(metrics, capacity, OUTDIR / "phase6e_adv_floor_memo.md")

    print(f"Wrote Phase6E ADV-floor artifacts to {OUTDIR}")
    print(metrics.to_string(index=False))
    print(capacity.to_string(index=False))
    print(strict_rollup)


if __name__ == "__main__":
    main()
