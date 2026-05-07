"""Phase6F product-candidate comparison.

Compares the research SOTA with hard-ADV-floor production candidates under
gross and simple cost/financing scenarios. Validation window only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase5b import _build_h10_turnover_curve
from stockmachine.apps.run_pure_alpha_phase6 import (
    _build_backtest_records,
    _build_daily_aggregate_positions,
    _build_financing_borrow_stress,
    _daily_path_metrics,
    _load_strict_curve,
)


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT / "research"
PHASE6E_ROOT = RESEARCH_ROOT / "phase6e_adv_floor_hardening_20260507"
POSITIONS_PATH = PHASE6E_ROOT / "phase6e_positions_validation.csv.gz"
STRICT_CURVE_PATH = PHASE6E_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_daily_curve.csv"
PHASE6E_CAPACITY_PATH = PHASE6E_ROOT / "phase6e_adv_floor_capacity_500k.csv"
OUTDIR = RESEARCH_ROOT / "phase6f_product_candidate_comparison_20260507"

PORTFOLIOS: tuple[tuple[str, str], ...] = (
    ("adv_floor_0", "research_sota_no_adv_floor"),
    ("adv_floor_1m", "production_candidate_adv_floor_1m"),
    ("adv_floor_5m", "production_candidate_adv_floor_5m"),
)
HOLDING_PERIOD_SESSIONS = 10
CAPITAL_USD = 500_000


def _scenario_name(row: pd.Series) -> str:
    tx = float(row["transaction_cost_bps_per_traded_notional"])
    borrow = float(row["borrow_cost_bps_annual_on_short_gross"])
    margin = float(row["margin_cost_bps_annual_on_gross_above_100pct_nav"])
    if tx == 0 and borrow == 0 and margin == 0:
        return "gross_no_cost"
    return f"tc{tx:.0f}bps_margin{margin / 100:.2f}pct_borrow{borrow / 100:.1f}pct"


def _records_for_portfolio(portfolio: str) -> pd.DataFrame:
    curve = _load_strict_curve(STRICT_CURVE_PATH, portfolio=portfolio)
    positions = pd.read_csv(POSITIONS_PATH, parse_dates=["session_date"])
    positions = positions[positions["portfolio"].astype(str).eq(portfolio)].copy()
    calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    turnover = _build_h10_turnover_curve(
        positions_path=POSITIONS_PATH,
        portfolio=portfolio,
        benchmark_calendar=calendar,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
    )
    _, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
    )
    return _build_backtest_records(curve=curve, turnover=turnover, daily_exposure=daily_exposure)


def _build_cost_metrics() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[pd.DataFrame] = []
    records_by_portfolio: dict[str, pd.DataFrame] = {}
    for portfolio, label in PORTFOLIOS:
        records = _records_for_portfolio(portfolio)
        records_by_portfolio[portfolio] = records
        metrics = _build_financing_borrow_stress(
            records,
            transaction_cost_levels_bps=(0.0, 4.0),
            borrow_cost_levels_bps_annual=(0.0, 200.0),
            margin_cost_levels_bps_annual=(0.0, 475.0),
        )
        metrics["portfolio"] = portfolio
        metrics["label"] = label
        metrics["scenario"] = metrics.apply(_scenario_name, axis=1)
        metrics["annualized_pnl_usd"] = metrics["annualized_return"] * CAPITAL_USD
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True), records_by_portfolio


def _build_summary(cost_metrics: pd.DataFrame) -> pd.DataFrame:
    capacity = pd.read_csv(PHASE6E_CAPACITY_PATH)
    rows: list[dict[str, object]] = []
    for portfolio, label in PORTFOLIOS:
        sub = cost_metrics[cost_metrics["portfolio"].eq(portfolio)].copy()
        cap = capacity[capacity["portfolio"].eq(portfolio)].iloc[0]

        def _ann(scenario: str) -> float:
            return float(sub.loc[sub["scenario"].eq(scenario), "annualized_return"].iloc[0])

        def _sharpe(scenario: str) -> float:
            return float(sub.loc[sub["scenario"].eq(scenario), "sharpe_no_rf"].iloc[0])

        gross = _ann("gross_no_cost")
        tx4 = _ann("tc4bps_margin0.00pct_borrow0.0pct")
        tx4_margin = _ann("tc4bps_margin4.75pct_borrow0.0pct")
        tx4_margin_borrow = _ann("tc4bps_margin4.75pct_borrow2.0pct")
        rows.append(
            {
                "portfolio": portfolio,
                "label": label,
                "gross_annualized_return": gross,
                "gross_sharpe": _sharpe("gross_no_cost"),
                "tc4_annualized_return": tx4,
                "alpaca_base_case_annualized_return": tx4,
                "debit_balance_stress475_annualized_return": tx4_margin,
                "debit_balance_stress475_borrow200_annualized_return": tx4_margin_borrow,
                "tc4_haircut": gross - tx4,
                "debit_balance_stress475_haircut": gross - tx4_margin,
                "debit_balance_stress475_borrow200_haircut": gross - tx4_margin_borrow,
                "annualized_pnl_usd_gross": gross * CAPITAL_USD,
                "annualized_pnl_usd_alpaca_base_case": tx4 * CAPITAL_USD,
                "annualized_pnl_usd_debit_balance_stress475": tx4_margin * CAPITAL_USD,
                "aggregate_turnover_mean": float(sub["mean_daily_transaction_cost_bps"].max() / 4.0)
                if 4.0 in sub["transaction_cost_bps_per_traded_notional"].values
                else np.nan,
                "order_max_adv_participation_500k": float(
                    cap["order_max_adv_participation_500k"]
                ),
                "position_max_adv_participation_500k": float(
                    cap["position_max_adv_participation_500k"]
                ),
                "position_abs_weight_share_below_1m_adv": float(
                    cap["position_abs_weight_share_below_1m_adv"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _net_returns(records: pd.DataFrame, *, tx_bps: float, margin_bps: float, borrow_bps: float) -> pd.Series:
    gross = records["gross_return"].astype(float)
    turnover = records["turnover"].astype(float).fillna(0.0)
    short_gross = records["short_gross"].astype(float).fillna(0.0)
    excess_gross = (records["gross_exposure"].astype(float).fillna(0.0) - 1.0).clip(lower=0.0)
    return (
        gross
        - turnover * tx_bps / 10000.0
        - short_gross * borrow_bps / 10000.0 / 252.0
        - excess_gross * margin_bps / 10000.0 / 252.0
    )


def _plot(records_by_portfolio: dict[str, pd.DataFrame], outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {
        "adv_floor_0": "#2563eb",
        "adv_floor_1m": "#0f766e",
        "adv_floor_5m": "#c2410c",
    }
    for portfolio, label in PORTFOLIOS:
        records = records_by_portfolio[portfolio]
        dates = pd.to_datetime(records["entry_date"])
        gross_equity = (1.0 + records["gross_return"].astype(float)).cumprod()
        axes[0].plot(dates, gross_equity, label=f"{label} gross", color=colors[portfolio])
        if portfolio == "adv_floor_1m":
            net = _net_returns(records, tx_bps=4.0, margin_bps=475.0, borrow_bps=0.0)
            axes[0].plot(
                dates,
                (1.0 + net).cumprod(),
                label="adv_floor_1m tc4+margin net",
                color="#111827",
                linestyle="--",
            )
        drawdown = gross_equity / gross_equity.cummax() - 1.0
        axes[1].plot(dates, drawdown * 100, label=label, color=colors[portfolio])

    axes[0].set_title("Phase6F Product Candidate Comparison (Validation Only)")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Gross drawdown %")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_memo(summary: pd.DataFrame, outpath: Path) -> None:
    display = summary[
        [
            "label",
            "gross_annualized_return",
            "gross_sharpe",
            "tc4_annualized_return",
            "alpaca_base_case_annualized_return",
            "debit_balance_stress475_annualized_return",
            "order_max_adv_participation_500k",
            "position_max_adv_participation_500k",
        ]
    ].copy()
    text = [
        "# Phase6F Product Candidate Comparison",
        "",
        "Scope: validation-window only. Test remains unused.",
        "",
        "The research SOTA remains the highest gross-return variant, but the ADV-floor candidates are more realistic production candidates because they remove the low-liquidity order tail found in Phase6D.",
        "",
        "Cost convention: Alpaca base case here means 4 bps turnover cost, zero borrow fee for ETB names, and zero margin-interest cost when the account has no overnight debit balance. The 4.75% line is retained only as a debit-balance stress scenario, not as a mechanical cost on gross exposure above 100% NAV.",
        "",
        "```text",
        display.to_string(index=False),
        "```",
        "",
        "Recommended naming:",
        "",
        "- Research SOTA: `adv_floor_0` / no hard ADV floor.",
        "- Production candidate: `adv_floor_1m`, because it cuts worst order ADV participation from 66.7% to about 0.2% with only a small validation return haircut.",
        "- Conservative production candidate: `adv_floor_5m`, similar return and slightly lower worst position participation, but not clearly better enough to displace `$1m` yet.",
        "",
        "Remaining blockers: point-in-time borrow fees, broker-specific debit-balance/margin/liquidation model, and paper-trading order simulation.",
    ]
    outpath.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cost_metrics, records_by_portfolio = _build_cost_metrics()
    summary = _build_summary(cost_metrics)
    cost_metrics.to_csv(OUTDIR / "phase6f_candidate_cost_metrics.csv", index=False)
    summary.to_csv(OUTDIR / "phase6f_candidate_summary.csv", index=False)
    _plot(records_by_portfolio, OUTDIR / "phase6f_candidate_comparison_plot.png")
    _write_memo(summary, OUTDIR / "phase6f_product_candidate_memo.md")

    print(f"Wrote Phase6F product-candidate artifacts to {OUTDIR}")
    print(summary.to_string(index=False))
    for portfolio, records in records_by_portfolio.items():
        metrics = _daily_path_metrics(records["gross_return"], benchmark=records["benchmark_return"])
        print(portfolio, metrics)


if __name__ == "__main__":
    main()
