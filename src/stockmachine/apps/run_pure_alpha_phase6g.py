"""Phase6G Alpaca-realistic net model for pure-alpha candidates.

This phase replaces the coarse "gross above 100% NAV pays margin interest"
stress with a more realistic decomposition:

* turnover/slippage cost on traded notional;
* borrow fee stress on short gross;
* transient debit-balance stress on a small share of high-turnover days.

Validation window only. Test-window performance remains unused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase6 import _daily_path_metrics, _rolling_positive_rate
from stockmachine.apps.run_pure_alpha_phase6f import (
    CAPITAL_USD,
    PORTFOLIOS,
    _records_for_portfolio,
)


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT / "research"
OUTDIR = RESEARCH_ROOT / "phase6g_alpaca_realistic_net_model_20260507"
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class CostScenario:
    scenario: str
    turnover_cost_bps: float
    borrow_cost_bps_annual_on_short_gross: float
    debit_balance_rate_annual: float
    debit_balance_nav: float
    debit_event_day_share: float
    note: str


SCENARIOS: tuple[CostScenario, ...] = (
    CostScenario(
        "gross_no_cost",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        "No implementation costs; research-only upper bound.",
    ),
    CostScenario(
        "alpaca_base_tc4_etb_no_debit",
        4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        "Base case: 4 bps traded-notional cost, ETB borrow fee assumed zero, no overnight debit balance.",
    ),
    CostScenario(
        "transient_debit_mild",
        4.0,
        0.0,
        0.0475,
        0.10,
        0.05,
        "Debit stress: 10% NAV debit on top 5% turnover days at 4.75% annualized.",
    ),
    CostScenario(
        "transient_debit_moderate",
        4.0,
        0.0,
        0.0475,
        0.20,
        0.10,
        "Debit stress: 20% NAV debit on top 10% turnover days at 4.75% annualized.",
    ),
    CostScenario(
        "borrow_mild_50bps",
        4.0,
        50.0,
        0.0,
        0.0,
        0.0,
        "Borrow stress: 50 bps annualized on total short gross.",
    ),
    CostScenario(
        "borrow_moderate_200bps",
        4.0,
        200.0,
        0.0,
        0.0,
        0.0,
        "Borrow stress: 200 bps annualized on total short gross.",
    ),
    CostScenario(
        "borrow_htb_stress_500bps",
        4.0,
        500.0,
        0.0,
        0.0,
        0.0,
        "HTB stress: 500 bps annualized on total short gross.",
    ),
    CostScenario(
        "combined_moderate",
        4.0,
        200.0,
        0.0475,
        0.20,
        0.10,
        "Moderate combined stress: tc4 + borrow 200 bps + transient debit moderate.",
    ),
    CostScenario(
        "combined_harsh",
        8.0,
        500.0,
        0.0625,
        0.30,
        0.20,
        "Harsh combined stress: tc8 + borrow 500 bps + 30% NAV debit on top 20% turnover days at 6.25%.",
    ),
)


def _debit_event_mask(turnover: pd.Series, share: float) -> pd.Series:
    if share <= 0:
        return pd.Series(False, index=turnover.index)
    share = min(max(float(share), 0.0), 1.0)
    if share >= 1:
        return pd.Series(True, index=turnover.index)
    threshold = turnover.astype(float).quantile(1.0 - share)
    return turnover.astype(float) >= threshold


def _apply_cost_scenario(records: pd.DataFrame, scenario: CostScenario) -> pd.DataFrame:
    out = records.copy()
    turnover = out["turnover"].astype(float).fillna(0.0)
    short_gross = out["short_gross"].astype(float).fillna(0.0)
    debit_mask = _debit_event_mask(turnover, scenario.debit_event_day_share)
    debit_nav = pd.Series(0.0, index=out.index)
    debit_nav.loc[debit_mask] = float(scenario.debit_balance_nav)

    out["scenario"] = scenario.scenario
    out["turnover_cost"] = turnover * scenario.turnover_cost_bps / 10000.0
    out["borrow_cost"] = (
        short_gross * scenario.borrow_cost_bps_annual_on_short_gross / 10000.0 / TRADING_DAYS
    )
    out["debit_cost"] = debit_nav * scenario.debit_balance_rate_annual / TRADING_DAYS
    out["net_return"] = (
        out["gross_return"].astype(float)
        - out["turnover_cost"]
        - out["borrow_cost"]
        - out["debit_cost"]
    )
    out["debit_event"] = debit_mask
    out["debit_balance_nav"] = debit_nav
    return out


def _scenario_metrics(
    records: pd.DataFrame,
    scenario_records: pd.DataFrame,
    *,
    portfolio: str,
    label: str,
    scenario: CostScenario,
    gross_ann_return: float,
) -> dict[str, object]:
    metrics = _daily_path_metrics(
        scenario_records["net_return"],
        benchmark=scenario_records["benchmark_return"],
    )
    returns = scenario_records["net_return"].astype(float)
    metrics.update(
        {
            "portfolio": portfolio,
            "label": label,
            "scenario": scenario.scenario,
            "turnover_cost_bps": scenario.turnover_cost_bps,
            "borrow_cost_bps_annual_on_short_gross": scenario.borrow_cost_bps_annual_on_short_gross,
            "debit_balance_rate_annual": scenario.debit_balance_rate_annual,
            "debit_balance_nav": scenario.debit_balance_nav,
            "debit_event_day_share_assumption": scenario.debit_event_day_share,
            "debit_event_days_realized": int(scenario_records["debit_event"].sum()),
            "debit_event_day_share_realized": float(scenario_records["debit_event"].mean()),
            "mean_turnover": float(records["turnover"].astype(float).mean()),
            "mean_short_gross": float(records["short_gross"].astype(float).mean()),
            "mean_daily_turnover_cost_bps_nav": float(
                scenario_records["turnover_cost"].mean() * 10000.0
            ),
            "mean_daily_borrow_cost_bps_nav": float(
                scenario_records["borrow_cost"].mean() * 10000.0
            ),
            "mean_daily_debit_cost_bps_nav": float(
                scenario_records["debit_cost"].mean() * 10000.0
            ),
            "approx_annual_turnover_cost_pct_nav": float(
                scenario_records["turnover_cost"].mean() * TRADING_DAYS
            ),
            "approx_annual_borrow_cost_pct_nav": float(
                scenario_records["borrow_cost"].mean() * TRADING_DAYS
            ),
            "approx_annual_debit_cost_pct_nav": float(
                scenario_records["debit_cost"].mean() * TRADING_DAYS
            ),
            "annualized_return_haircut_vs_gross": float(
                gross_ann_return - metrics["annualized_return"]
            ),
            "annualized_pnl_usd": float(metrics["annualized_return"] * CAPITAL_USD),
            "rolling_10_positive_rate": _rolling_positive_rate(returns, 10),
            "rolling_60_positive_rate": _rolling_positive_rate(returns, 60),
            "rolling_252_positive_rate": _rolling_positive_rate(returns, 252),
            "note": scenario.note,
        }
    )
    return metrics


def _build_all_scenarios() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []
    records_by_portfolio: dict[str, pd.DataFrame] = {}

    for portfolio, label in PORTFOLIOS:
        records = _records_for_portfolio(portfolio)
        records_by_portfolio[portfolio] = records
        gross_metrics = _daily_path_metrics(records["gross_return"], benchmark=records["benchmark_return"])
        gross_ann_return = float(gross_metrics["annualized_return"])
        for scenario in SCENARIOS:
            scenario_records = _apply_cost_scenario(records, scenario)
            metric_rows.append(
                _scenario_metrics(
                    records,
                    scenario_records,
                    portfolio=portfolio,
                    label=label,
                    scenario=scenario,
                    gross_ann_return=gross_ann_return,
                )
            )
            curve = scenario_records[
                [
                    "entry_date",
                    "gross_return",
                    "net_return",
                    "turnover_cost",
                    "borrow_cost",
                    "debit_cost",
                    "benchmark_return",
                ]
            ].copy()
            curve["portfolio"] = portfolio
            curve["label"] = label
            curve["scenario"] = scenario.scenario
            curve["equity"] = (1.0 + curve["net_return"].astype(float)).cumprod()
            curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
            curve_rows.append(curve)

    return (
        pd.DataFrame(metric_rows).sort_values(["portfolio", "scenario"]).reset_index(drop=True),
        pd.concat(curve_rows, ignore_index=True),
        records_by_portfolio,
    )


def _build_candidate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "gross_no_cost",
        "alpaca_base_tc4_etb_no_debit",
        "transient_debit_moderate",
        "borrow_moderate_200bps",
        "combined_moderate",
        "combined_harsh",
    ]
    rows: list[dict[str, object]] = []
    for portfolio, label in PORTFOLIOS:
        sub = metrics[metrics["portfolio"].eq(portfolio)]
        row: dict[str, object] = {"portfolio": portfolio, "label": label}
        for scenario in scenarios:
            item = sub[sub["scenario"].eq(scenario)].iloc[0]
            row[f"{scenario}_ann_return"] = float(item["annualized_return"])
            row[f"{scenario}_sharpe"] = float(item["sharpe_no_rf"])
            row[f"{scenario}_max_dd"] = float(item["max_drawdown"])
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(curves: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    color_by_portfolio = {
        "adv_floor_0": "#2563eb",
        "adv_floor_1m": "#0f766e",
        "adv_floor_5m": "#c2410c",
    }
    base = curves[curves["scenario"].eq("alpaca_base_tc4_etb_no_debit")]
    for portfolio, label in PORTFOLIOS:
        group = base[base["portfolio"].eq(portfolio)]
        axes[0].plot(
            pd.to_datetime(group["entry_date"]),
            group["equity"],
            label=f"{label} base-case",
            color=color_by_portfolio[portfolio],
        )
    candidate = curves[
        curves["portfolio"].eq("adv_floor_1m")
        & curves["scenario"].isin(
            [
                "gross_no_cost",
                "alpaca_base_tc4_etb_no_debit",
                "transient_debit_moderate",
                "borrow_moderate_200bps",
                "combined_moderate",
                "combined_harsh",
            ]
        )
    ]
    styles = {
        "gross_no_cost": ("#111827", "-"),
        "alpaca_base_tc4_etb_no_debit": ("#0f766e", "-"),
        "transient_debit_moderate": ("#2563eb", "--"),
        "borrow_moderate_200bps": ("#7c3aed", "--"),
        "combined_moderate": ("#c2410c", "-."),
        "combined_harsh": ("#6b7280", ":"),
    }
    for scenario, group in candidate.groupby("scenario", sort=False):
        color, linestyle = styles[scenario]
        axes[1].plot(
            pd.to_datetime(group["entry_date"]),
            group["equity"],
            label=scenario,
            color=color,
            linestyle=linestyle,
        )

    axes[0].set_title("Phase6G Alpaca Base-Case Net Paths")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_title("Production Candidate ADV $1M: Stress Decomposition")
    axes[1].set_ylabel("Growth of $1")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_memo(metrics: pd.DataFrame, summary: pd.DataFrame, outpath: Path) -> None:
    candidate = metrics[
        metrics["portfolio"].eq("adv_floor_1m")
        & metrics["scenario"].isin(
            [
                "gross_no_cost",
                "alpaca_base_tc4_etb_no_debit",
                "transient_debit_moderate",
                "borrow_moderate_200bps",
                "combined_moderate",
                "combined_harsh",
            ]
        )
    ].copy()
    display = candidate[
        [
            "scenario",
            "annualized_return",
            "sharpe_no_rf",
            "max_drawdown",
            "annualized_pnl_usd",
            "approx_annual_turnover_cost_pct_nav",
            "approx_annual_borrow_cost_pct_nav",
            "approx_annual_debit_cost_pct_nav",
        ]
    ]
    lines = [
        "# Phase6G Alpaca-Realistic Net Model",
        "",
        "Scope: validation-window only. Test-window performance remains unused.",
        "",
        "This phase separates implementation cost into turnover/slippage, borrow fee stress, and transient debit-balance stress. Margin interest is not charged mechanically on gross exposure above 100% NAV.",
        "",
        "## Production Candidate Focus",
        "",
        "The current preferred product candidate remains `adv_floor_1m`: it preserves most of the research SOTA while removing the low-ADV order tail.",
        "",
        "```text",
        display.to_string(index=False),
        "```",
        "",
        "## Candidate Comparison",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Base case is `4 bps turnover cost + ETB borrow fee 0 + no overnight debit balance`.",
        "- Transient debit stress is intentionally small because debit balance should be temporary if the long book stays below 100% NAV.",
        "- Borrow/HTB stress matters much more than debit stress; point-in-time borrow fee data remains the main production-cost unknown.",
        "- If Alpaca paper trading confirms no persistent debit balance and shorts are mostly ETB, the product candidate economics are closer to the base-case line than to the old 4.75% gross-above-100% stress.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    metrics, curves, _ = _build_all_scenarios()
    summary = _build_candidate_summary(metrics)
    metrics.to_csv(OUTDIR / "phase6g_alpaca_realistic_scenarios.csv", index=False)
    summary.to_csv(OUTDIR / "phase6g_candidate_net_summary.csv", index=False)
    curves.to_csv(OUTDIR / "phase6g_scenario_daily_curve.csv", index=False)
    _plot(curves, OUTDIR / "phase6g_alpaca_realistic_net_plot.png")
    _write_memo(metrics, summary, OUTDIR / "phase6g_alpaca_realistic_net_memo.md")

    print(f"Wrote Phase6G Alpaca-realistic net artifacts to {OUTDIR}")
    print(summary.to_string(index=False))
    focus = metrics[
        metrics["portfolio"].eq("adv_floor_1m")
        & metrics["scenario"].isin(
            [
                "gross_no_cost",
                "alpaca_base_tc4_etb_no_debit",
                "transient_debit_moderate",
                "borrow_moderate_200bps",
                "combined_moderate",
                "combined_harsh",
            ]
        )
    ]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
