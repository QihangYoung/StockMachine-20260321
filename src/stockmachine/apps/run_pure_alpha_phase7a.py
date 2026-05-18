"""Phase7A one-time test-lockbox evaluation for the pure-alpha product candidate.

This app opens the test lockbox under a frozen protocol. It does not select
parameters on the test window. The only primary candidate is the validation-
selected `adv_floor_1m` production candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase0 import build_pit_universe_feasibility
from stockmachine.apps.run_pure_alpha_phase1 import build_phase1_universe_artifacts
from stockmachine.apps.run_pure_alpha_phase2 import build_phase2_beta_artifacts
from stockmachine.apps.run_pure_alpha_phase3 import build_phase3_signal_artifacts
from stockmachine.apps.run_pure_alpha_phase4z import build_phase4z_strict_horizon_daily_paths
from stockmachine.apps.run_pure_alpha_phase5b import _build_h10_turnover_curve
from stockmachine.apps.run_pure_alpha_phase6 import (
    _build_backtest_records,
    _build_daily_aggregate_positions,
    _daily_path_metrics,
    _load_full_positions,
    _load_strict_curve,
    _rolling_positive_rate,
)
from stockmachine.apps.run_pure_alpha_phase6e import PortfolioSpec, _construct_one_floor, _load_feature_panel


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT / "research"
OUTDIR = RESEARCH_ROOT / "phase7a_test_lockbox_eval_20260508"

TEST_WARMUP_START = "2019-01-01"
TEST_START = "2020-01-02"
TEST_END = "2026-04-16"
HOLDING_PERIOD_SESSIONS = 10
PORTFOLIO = "test_adv_floor_1m"
SERIES = "Test lockbox ADV $1M product candidate"
ADV_FLOOR_USD = 1_000_000.0
CAPITAL_USD = 500_000

PHASE0_ROOT = OUTDIR / "phase0_pit_universe_test_warmup"
PHASE0_FILTERED_MEMBERSHIP = OUTDIR / "phase0_membership_test_only.csv.gz"
PHASE1_ROOT = OUTDIR / "phase1_universe_test"
PHASE2_ROOT = OUTDIR / "phase2_beta_test"
PHASE3_ROOT = OUTDIR / "phase3_signals_h10_test"
POSITIONS_PATH = OUTDIR / "phase7a_positions_test.csv.gz"
STRICT_ROOT = OUTDIR / "strict_h10_test"


@dataclass(frozen=True)
class NetScenario:
    scenario: str
    turnover_cost_bps: float
    borrow_cost_bps_annual: float = 0.0
    debit_rate_annual: float = 0.0
    debit_nav: float = 0.0
    debit_day_share: float = 0.0
    note: str = ""


SCENARIOS = (
    NetScenario("gross_no_cost", 0.0, note="Research upper bound."),
    NetScenario(
        "alpaca_base_tc4_etb_no_debit",
        4.0,
        note="Primary base case: 4 bps traded-notional cost, ETB borrow fee 0, no persistent debit.",
    ),
    NetScenario(
        "borrow_200bps",
        4.0,
        borrow_cost_bps_annual=200.0,
        note="Stress: 200 bps annual fee on all short gross.",
    ),
    NetScenario(
        "borrow_500bps",
        4.0,
        borrow_cost_bps_annual=500.0,
        note="Stress: 500 bps annual fee on all short gross.",
    ),
    NetScenario(
        "transient_debit_moderate",
        4.0,
        debit_rate_annual=0.0475,
        debit_nav=0.20,
        debit_day_share=0.10,
        note="Stress: 20% NAV debit on top 10% turnover days at 4.75%.",
    ),
)


def _write_protocol(path: Path) -> None:
    text = f"""# Phase7A Test Lockbox Protocol

Date: 2026-05-08

This protocol freezes the one-time test-lockbox evaluation before test performance is read.

## Frozen Candidate

- Candidate: `{PORTFOLIO}` / validation-selected `$1M ADV floor` product candidate.
- Long selector: `reversal_5d`.
- Short selector: `short_hybrid_soft_fw_overlay`.
- Construction: turnover-aware optimizer, exact ex-ante beta match, SIC2 soft neutral, rolling h10.
- Universe: validation-selected long `top1000_clean_core_beta_full`, short `adv30m_clean_core_beta_full`, with hard `$1M` ADV floor before optimization.

## Test Window

- Warmup data may start at `{TEST_WARMUP_START}` for lagged liquidity.
- Test performance starts at `{TEST_START}` and ends at `{TEST_END}` subject to available bars and h10 active-sleeve alignment.
- No test-window result may be used to alter selector, ADV floor, turnover penalty, overlay, or universe.

## Primary Metrics

- Gross h10 rolling daily path.
- Alpaca base-case net path: `4 bps turnover cost + ETB borrow fee 0 + no persistent debit`.
- Report annualized return, volatility, Sharpe, max drawdown, SPY correlation, realized beta, rolling positive rates, and calendar-year returns.

## Secondary Stress

- Borrow 200 bps and 500 bps on all short gross.
- Transient debit stress.
- These are explanatory stress results only; they do not change the candidate.

## Lockbox Rule

This is a one-time test evaluation. If the result fails, return to validation research with a new written plan before any future test read.
"""
    path.write_text(text, encoding="utf-8")


def _filter_phase0_membership() -> Path:
    source = PHASE0_ROOT / "provisional_current_top1000_lagged_liquidity_membership_validation.csv.gz"
    if not source.exists():
        build_pit_universe_feasibility(
            output_root=PHASE0_ROOT,
            validation_start=TEST_WARMUP_START,
            validation_end=TEST_END,
        )
    membership = pd.read_csv(source)
    membership["session_date"] = pd.to_datetime(membership["session_date"]).dt.date.astype(str)
    membership = membership[membership["session_date"] >= TEST_START].copy()
    if membership.empty:
        raise ValueError("Filtered test membership is empty.")
    membership.to_csv(PHASE0_FILTERED_MEMBERSHIP, index=False, compression="gzip")
    return PHASE0_FILTERED_MEMBERSHIP


def _build_test_inputs() -> tuple[Path, Path, Path]:
    membership_path = _filter_phase0_membership()
    phase1_membership = PHASE1_ROOT / "phase1_candidate_universe_membership_validation.csv.gz"
    if not phase1_membership.exists():
        build_phase1_universe_artifacts(
            membership_path=membership_path,
            output_root=PHASE1_ROOT,
        )
    beta_panel = PHASE2_ROOT / "phase2_beta_panel_validation.csv.gz"
    if not beta_panel.exists():
        build_phase2_beta_artifacts(
            membership_path=phase1_membership,
            output_root=PHASE2_ROOT,
        )
    signal_panel = PHASE3_ROOT / "phase3_baseline_signal_panel_validation.csv.gz"
    if not signal_panel.exists():
        build_phase3_signal_artifacts(
            membership_path=phase1_membership,
            beta_panel_path=beta_panel,
            output_root=PHASE3_ROOT,
            holding_period_sessions=HOLDING_PERIOD_SESSIONS,
        )
    return phase1_membership, beta_panel, signal_panel


def _build_positions(signal_panel_path: Path) -> Path:
    if POSITIONS_PATH.exists():
        return POSITIONS_PATH
    panel = _load_feature_panel(signal_panel_path=signal_panel_path)
    spec = PortfolioSpec(
        portfolio=PORTFOLIO,
        series=SERIES,
        adv_floor_usd=ADV_FLOOR_USD,
    )
    positions, diagnostics, skipped = _construct_one_floor(panel, spec)
    for frame in (positions, diagnostics, skipped):
        if not frame.empty and "session_date" in frame:
            frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    if positions.empty:
        raise ValueError("No test positions were generated.")
    positions.to_csv(POSITIONS_PATH, index=False, compression="gzip")
    diagnostics.to_csv(OUTDIR / "phase7a_daily_test.csv", index=False)
    skipped.to_csv(OUTDIR / "phase7a_skipped_test.csv", index=False)
    return POSITIONS_PATH


def _build_strict_curve(positions_path: Path) -> Path:
    curve_path = STRICT_ROOT / "phase4z_strict_h10_daily_curve.csv"
    if curve_path.exists():
        return curve_path
    build_phase4z_strict_horizon_daily_paths(
        positions_path=positions_path,
        output_root=STRICT_ROOT,
        phase3_rollup_path=PHASE3_ROOT / "phase3_baseline_signal_rollup.json",
        existing_curve_path=None,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
        validation_price_end=TEST_END,
        lead_portfolio=PORTFOLIO,
    )
    return curve_path


def _records(positions_path: Path, strict_curve_path: Path) -> pd.DataFrame:
    curve = _load_strict_curve(strict_curve_path, portfolio=PORTFOLIO)
    positions = _load_full_positions(positions_path, portfolio=PORTFOLIO)
    calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    turnover = _build_h10_turnover_curve(
        positions_path=positions_path,
        portfolio=PORTFOLIO,
        benchmark_calendar=calendar,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
    )
    _, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
    )
    records = _build_backtest_records(curve=curve, turnover=turnover, daily_exposure=daily_exposure)
    return records


def _debit_mask(turnover: pd.Series, share: float) -> pd.Series:
    if share <= 0:
        return pd.Series(False, index=turnover.index)
    threshold = turnover.astype(float).quantile(1.0 - share)
    return turnover.astype(float) >= threshold


def _scenario_returns(records: pd.DataFrame, scenario: NetScenario) -> pd.Series:
    gross = records["gross_return"].astype(float)
    if scenario.scenario == "gross_no_cost":
        return gross
    turnover = records["turnover"].astype(float).fillna(0.0)
    short_gross = records["short_gross"].astype(float).fillna(0.0)
    debit = pd.Series(0.0, index=records.index)
    if scenario.debit_day_share > 0 and scenario.debit_nav > 0:
        debit.loc[_debit_mask(turnover, scenario.debit_day_share)] = scenario.debit_nav
    return (
        gross
        - turnover * scenario.turnover_cost_bps / 10000.0
        - short_gross * scenario.borrow_cost_bps_annual / 10000.0 / 252.0
        - debit * scenario.debit_rate_annual / 252.0
    )


def _scenario_metrics(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    gross_metrics = _daily_path_metrics(records["gross_return"], benchmark=records["benchmark_return"])
    for scenario in SCENARIOS:
        returns = _scenario_returns(records, scenario)
        metrics = _daily_path_metrics(returns, benchmark=records["benchmark_return"])
        metrics.update(
            {
                "portfolio": PORTFOLIO,
                "series": SERIES,
                "scenario": scenario.scenario,
                "turnover_cost_bps": scenario.turnover_cost_bps,
                "borrow_cost_bps_annual": scenario.borrow_cost_bps_annual,
                "debit_rate_annual": scenario.debit_rate_annual,
                "debit_nav": scenario.debit_nav,
                "debit_day_share": scenario.debit_day_share,
                "annualized_return_haircut_vs_gross": float(
                    gross_metrics["annualized_return"] - metrics["annualized_return"]
                ),
                "annualized_pnl_usd_at_500k": float(metrics["annualized_return"] * CAPITAL_USD),
                "rolling_10_positive_rate": _rolling_positive_rate(returns, 10),
                "rolling_60_positive_rate": _rolling_positive_rate(returns, 60),
                "rolling_252_positive_rate": _rolling_positive_rate(returns, 252),
                "mean_turnover": float(records["turnover"].astype(float).mean()),
                "mean_gross_exposure": float(records["gross_exposure"].astype(float).mean()),
                "mean_short_gross": float(records["short_gross"].astype(float).mean()),
                "note": scenario.note,
            }
        )
        rows.append(metrics)
        curve = records[["entry_date", "benchmark_return"]].copy()
        curve["scenario"] = scenario.scenario
        curve["daily_return"] = returns
        curve["equity"] = (1.0 + returns).cumprod()
        curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
        curves.append(curve)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def _yearly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    frame = curves.copy()
    frame["year"] = pd.to_datetime(frame["entry_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for (scenario, year), group in frame.groupby(["scenario", "year"], sort=True):
        ret = float((1.0 + group["daily_return"].astype(float)).prod() - 1.0)
        rows.append({"scenario": scenario, "year": int(year), "return": ret, "sessions": int(len(group))})
    return pd.DataFrame(rows)


def _plot(curves: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {
        "gross_no_cost": "#111827",
        "alpaca_base_tc4_etb_no_debit": "#0f766e",
        "borrow_200bps": "#2563eb",
        "borrow_500bps": "#c2410c",
        "transient_debit_moderate": "#7c3aed",
    }
    for scenario, group in curves.groupby("scenario", sort=False):
        axes[0].plot(
            pd.to_datetime(group["entry_date"]),
            group["equity"],
            label=scenario,
            color=colors.get(scenario),
        )
        axes[1].plot(
            pd.to_datetime(group["entry_date"]),
            group["drawdown"] * 100.0,
            label=scenario,
            color=colors.get(scenario),
        )
    axes[0].set_title("Phase7A Test Lockbox Paths")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_memo(metrics: pd.DataFrame, yearly: pd.DataFrame, outpath: Path) -> None:
    display = metrics[
        [
            "scenario",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "max_drawdown",
            "corr_to_spy",
            "realized_beta_to_spy",
            "rolling_60_positive_rate",
            "annualized_pnl_usd_at_500k",
        ]
    ].copy()
    yearly_pivot = yearly.pivot(index="year", columns="scenario", values="return").reset_index()
    lines = [
        "# Phase7A Test Lockbox Evaluation",
        "",
        "This is the first test-lockbox read for the validation-selected pure-alpha product candidate. No test result is used for parameter selection in this app.",
        "",
        f"- Test window requested: `{TEST_START}` through `{TEST_END}`.",
        f"- Candidate: `{PORTFOLIO}` (`$1M ADV floor`).",
        "- Primary net scenario: `alpaca_base_tc4_etb_no_debit`.",
        "",
        "## Scenario Metrics",
        "",
        "```text",
        display.to_string(index=False),
        "```",
        "",
        "## Calendar-Year Returns",
        "",
        "```text",
        yearly_pivot.to_string(index=False),
        "```",
        "",
        "## Interpretation Rules",
        "",
        "- This test read is final for the current frozen candidate.",
        "- Do not use these results to tune ADV floor, selector, overlay, or turnover penalty.",
        "- If the result is not acceptable, write a new validation-only research plan before any future test read.",
        "- Remaining structural caveat: the universe is still current-top1000 bootstrap, not survivorship-bias-free institutional coverage.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _write_protocol(OUTDIR / "phase7a_test_lockbox_protocol.md")
    _, _, signal_panel = _build_test_inputs()
    positions_path = _build_positions(signal_panel)
    strict_curve_path = _build_strict_curve(positions_path)
    records = _records(positions_path, strict_curve_path)
    metrics, curves = _scenario_metrics(records)
    yearly = _yearly_returns(curves)

    records.to_csv(OUTDIR / "phase7a_test_records.csv", index=False)
    metrics.to_csv(OUTDIR / "phase7a_test_metrics.csv", index=False)
    curves.to_csv(OUTDIR / "phase7a_test_curves.csv", index=False)
    yearly.to_csv(OUTDIR / "phase7a_test_yearly_returns.csv", index=False)
    _plot(curves, OUTDIR / "phase7a_test_lockbox_plot.png")
    _write_memo(metrics, yearly, OUTDIR / "phase7a_test_lockbox_memo.md")

    print(f"Wrote Phase7A test lockbox artifacts to {OUTDIR}")
    print(metrics.to_string(index=False))
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
