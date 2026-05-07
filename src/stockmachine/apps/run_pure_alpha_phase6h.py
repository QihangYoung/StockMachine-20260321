"""Phase6H borrow/HTB risk diagnostics for the product candidate.

This phase keeps the portfolio fixed and decomposes short-borrow risk into:

* current ETB/shortable proxy coverage;
* recurring short-name concentration;
* partial HTB fee stresses on subsets of the short book;
* all-short-book breakeven borrow fee.

Validation window only. The current ETB fields are explicitly treated as a
non-point-in-time proxy, not as production-ready borrow history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase5b import _build_h10_turnover_curve
from stockmachine.apps.run_pure_alpha_phase6 import (
    _build_daily_aggregate_positions,
    _daily_path_metrics,
    _load_strict_curve,
)
from stockmachine.apps.run_pure_alpha_phase6d import _load_liquidity
from stockmachine.apps.run_pure_alpha_phase6f import CAPITAL_USD


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT / "research"
PHASE6E_ROOT = RESEARCH_ROOT / "phase6e_adv_floor_hardening_20260507"
POSITIONS_PATH = PHASE6E_ROOT / "phase6e_positions_validation.csv.gz"
STRICT_CURVE_PATH = PHASE6E_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_daily_curve.csv"
OUTDIR = RESEARCH_ROOT / "phase6h_borrow_htb_diagnostics_20260507"

PORTFOLIO = "adv_floor_1m"
LABEL = "production_candidate_adv_floor_1m"
HOLDING_PERIOD_SESSIONS = 10
TURNOVER_COST_BPS = 4.0
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class BorrowScenario:
    scenario: str
    fee_bps_annual: float
    mode: str
    htb_share: float = 0.0
    top_symbol_count: int = 0
    note: str = ""


SCENARIOS: tuple[BorrowScenario, ...] = (
    BorrowScenario("base_etb_zero", 0.0, "all_short", note="Base case: ETB borrow fee assumed zero."),
    BorrowScenario("all_short_50bps", 50.0, "all_short", note="50 bps annual fee on all short gross."),
    BorrowScenario("all_short_200bps", 200.0, "all_short", note="200 bps annual fee on all short gross."),
    BorrowScenario("all_short_500bps", 500.0, "all_short", note="500 bps annual fee on all short gross."),
    BorrowScenario(
        "current_non_etb_500bps",
        500.0,
        "current_non_etb",
        note="500 bps only on rows not marked ETB by current proxy.",
    ),
    BorrowScenario(
        "largest_10pct_short_gross_500bps",
        500.0,
        "largest_daily_share",
        htb_share=0.10,
        note="Each day, the largest 10% of short gross pays 500 bps.",
    ),
    BorrowScenario(
        "largest_20pct_short_gross_500bps",
        500.0,
        "largest_daily_share",
        htb_share=0.20,
        note="Each day, the largest 20% of short gross pays 500 bps.",
    ),
    BorrowScenario(
        "largest_10pct_short_gross_1000bps",
        1000.0,
        "largest_daily_share",
        htb_share=0.10,
        note="Each day, the largest 10% of short gross pays 1000 bps.",
    ),
    BorrowScenario(
        "top20_recurring_symbols_500bps",
        500.0,
        "top_recurring_symbols",
        top_symbol_count=20,
        note="The top 20 recurring short names by aggregate short gross pay 500 bps.",
    ),
)


def _as_bool_nullable(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[text.isin(["true", "1", "yes", "y", "t"])] = True
    out.loc[text.isin(["false", "0", "no", "n", "f"])] = False
    return out


def _prepare_records_and_short_book() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = _load_strict_curve(STRICT_CURVE_PATH, portfolio=PORTFOLIO)
    positions = pd.read_csv(POSITIONS_PATH, parse_dates=["session_date"])
    positions = positions[positions["portfolio"].astype(str).eq(PORTFOLIO)].copy()
    calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    turnover = _build_h10_turnover_curve(
        positions_path=POSITIONS_PATH,
        portfolio=PORTFOLIO,
        benchmark_calendar=calendar,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
    )
    aggregate_positions, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=HOLDING_PERIOD_SESSIONS,
    )
    records = (
        curve.merge(turnover, on="return_date", how="left")
        .merge(daily_exposure, on="return_date", how="left")
        .sort_values("return_date")
        .reset_index(drop=True)
    )
    records["gross_return"] = records["gross_return"].astype(float)
    records["benchmark_return"] = records["benchmark_oto_return"].astype(float)
    records["turnover"] = records["h10_turnover"].astype(float).fillna(0.0)

    liquidity = _load_liquidity()
    borrow_proxy = liquidity[["symbol", "shortable_current", "easy_to_borrow_current"]].copy()
    borrow_proxy["shortable_proxy"] = _as_bool_nullable(borrow_proxy["shortable_current"])
    borrow_proxy["etb_proxy"] = _as_bool_nullable(borrow_proxy["easy_to_borrow_current"])
    borrow_proxy = (
        borrow_proxy.groupby("symbol", as_index=False)
        .agg(shortable_proxy=("shortable_proxy", "max"), etb_proxy=("etb_proxy", "max"))
    )
    liquidity_for_adv = liquidity[
        ["date", "symbol", "trailing_median_dollar_volume_20", "lagged_close"]
    ].copy()
    short_book = aggregate_positions[aggregate_positions["portfolio_weight"].astype(float) < 0].copy()
    short_book["date"] = pd.to_datetime(short_book["return_date"])
    short_book["short_weight"] = -short_book["portfolio_weight"].astype(float)
    short_book = short_book.merge(liquidity_for_adv, on=["date", "symbol"], how="left")
    short_book = short_book.merge(borrow_proxy, on="symbol", how="left")
    short_book["missing_borrow_proxy"] = short_book["etb_proxy"].isna()
    short_book["shortable_proxy"] = short_book["shortable_proxy"].fillna(False).astype(bool)
    short_book["etb_proxy"] = short_book["etb_proxy"].fillna(False).astype(bool)
    return records, short_book


def _daily_proxy_coverage(short_book: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in short_book.groupby("date", sort=True):
        gross = float(group["short_weight"].sum())
        if gross <= 0:
            continue
        rows.append(
            {
                "date": date,
                "short_gross": gross,
                "short_positions": int(len(group)),
                "etb_weighted_proxy": float(
                    (group["short_weight"] * group["etb_proxy"].astype(float)).sum() / gross
                ),
                "shortable_weighted_proxy": float(
                    (group["short_weight"] * group["shortable_proxy"].astype(float)).sum() / gross
                ),
                "missing_proxy_weight_share": float(
                    (group["short_weight"] * group["missing_borrow_proxy"].astype(float)).sum()
                    / gross
                ),
                "non_etb_weight_share": float(
                    (group["short_weight"] * (~group["etb_proxy"]).astype(float)).sum() / gross
                ),
                "non_shortable_weight_share": float(
                    (group["short_weight"] * (~group["shortable_proxy"]).astype(float)).sum()
                    / gross
                ),
            }
        )
    return pd.DataFrame(rows)


def _symbol_concentration(short_book: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        short_book.groupby("symbol", as_index=False)
        .agg(
            active_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            short_weight_sum=("short_weight", "sum"),
            mean_short_weight_when_active=("short_weight", "mean"),
            max_short_weight=("short_weight", "max"),
            min_adv=("trailing_median_dollar_volume_20", "min"),
            median_adv=("trailing_median_dollar_volume_20", "median"),
            etb_proxy_rate=("etb_proxy", "mean"),
            shortable_proxy_rate=("shortable_proxy", "mean"),
            missing_proxy_rate=("missing_borrow_proxy", "mean"),
        )
        .sort_values("short_weight_sum", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["short_weight_sum"].sum())
    grouped["short_weight_sum_share"] = grouped["short_weight_sum"] / total if total > 0 else np.nan
    grouped["approx_annual_cost_at_500bps_nav"] = (
        grouped["short_weight_sum"] / short_book["date"].nunique() * 0.05
    )
    grouped["approx_annual_cost_at_500bps_usd"] = (
        grouped["approx_annual_cost_at_500bps_nav"] * CAPITAL_USD
    )
    return grouped


def _selected_daily_short_weight(
    short_book: pd.DataFrame,
    scenario: BorrowScenario,
    top_symbols: set[str],
) -> pd.Series:
    if scenario.mode == "all_short":
        return short_book.groupby("date")["short_weight"].sum()
    if scenario.mode == "current_non_etb":
        selected = short_book[~short_book["etb_proxy"]].copy()
        return selected.groupby("date")["short_weight"].sum()
    if scenario.mode == "top_recurring_symbols":
        selected = short_book[short_book["symbol"].isin(top_symbols)].copy()
        return selected.groupby("date")["short_weight"].sum()
    if scenario.mode == "largest_daily_share":
        rows: list[dict[str, object]] = []
        for date, group in short_book.groupby("date", sort=True):
            target = float(group["short_weight"].sum() * scenario.htb_share)
            remaining = target
            selected = 0.0
            ordered = group.sort_values("short_weight", ascending=False)
            for weight in ordered["short_weight"].astype(float):
                if remaining <= 0:
                    break
                take = min(weight, remaining)
                selected += take
                remaining -= take
            rows.append({"date": date, "selected_short_weight": selected})
        return pd.DataFrame(rows).set_index("date")["selected_short_weight"]
    raise ValueError(f"Unknown borrow scenario mode: {scenario.mode}")


def _apply_borrow_scenarios(
    records: pd.DataFrame,
    short_book: pd.DataFrame,
    concentration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_symbols = set(concentration.head(20)["symbol"])
    base = records.copy()
    base["date"] = pd.to_datetime(base["return_date"])
    base["base_net_return"] = (
        base["gross_return"].astype(float)
        - base["turnover"].astype(float).fillna(0.0) * TURNOVER_COST_BPS / 10000.0
    )
    gross_metrics = _daily_path_metrics(base["gross_return"], benchmark=base["benchmark_return"])
    curve_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    all_dates = pd.Index(base["date"])

    for scenario in SCENARIOS:
        selected = _selected_daily_short_weight(short_book, scenario, top_symbols)
        selected = selected.reindex(all_dates, fill_value=0.0).astype(float).reset_index(drop=True)
        borrow_cost = selected * scenario.fee_bps_annual / 10000.0 / TRADING_DAYS
        net = base["base_net_return"].astype(float) - borrow_cost
        metrics = _daily_path_metrics(net, benchmark=base["benchmark_return"])
        metrics.update(
            {
                "portfolio": PORTFOLIO,
                "label": LABEL,
                "scenario": scenario.scenario,
                "fee_bps_annual": scenario.fee_bps_annual,
                "mode": scenario.mode,
                "htb_share": scenario.htb_share,
                "top_symbol_count": scenario.top_symbol_count,
                "mean_selected_short_weight": float(selected.mean()),
                "max_selected_short_weight": float(selected.max()),
                "mean_daily_borrow_cost_bps_nav": float(borrow_cost.mean() * 10000.0),
                "approx_annual_borrow_cost_pct_nav": float(borrow_cost.mean() * TRADING_DAYS),
                "annualized_return_haircut_vs_gross": float(
                    gross_metrics["annualized_return"] - metrics["annualized_return"]
                ),
                "annualized_pnl_usd": float(metrics["annualized_return"] * CAPITAL_USD),
                "note": scenario.note,
            }
        )
        metric_rows.append(metrics)
        curve = base[["date", "gross_return", "benchmark_return", "turnover"]].copy()
        curve["scenario"] = scenario.scenario
        curve["selected_short_weight"] = selected
        curve["borrow_cost"] = borrow_cost
        curve["net_return"] = net
        curve["equity"] = (1.0 + net).cumprod()
        curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
        curve_rows.append(curve)
    return pd.DataFrame(metric_rows), pd.concat(curve_rows, ignore_index=True)


def _breakeven_all_short_borrow(records: pd.DataFrame) -> pd.DataFrame:
    base = records.copy()
    base["base_net_return"] = (
        base["gross_return"].astype(float)
        - base["turnover"].astype(float).fillna(0.0) * TURNOVER_COST_BPS / 10000.0
    )
    short_gross = base["short_gross"].astype(float).fillna(0.0)

    def _ann_return(fee_bps: float) -> float:
        net = base["base_net_return"] - short_gross * fee_bps / 10000.0 / TRADING_DAYS
        return float(_daily_path_metrics(net, benchmark=base["benchmark_return"])["annualized_return"])

    lo, hi = 0.0, 5000.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _ann_return(mid) > 0:
            lo = mid
        else:
            hi = mid
    breakeven = (lo + hi) / 2.0
    rows = [
        {
            "metric": "all_short_borrow_fee_bps_to_zero_base_case_ann_return",
            "value": breakeven,
            "note": "Fee applied to all short gross after 4 bps turnover cost.",
        },
        {
            "metric": "all_short_borrow_fee_bps_to_cut_base_case_return_half",
            "value": breakeven / 2.0,
            "note": "Approximate midpoint because borrow cost is close to linear at these levels.",
        },
        {
            "metric": "base_case_ann_return_before_borrow",
            "value": _ann_return(0.0),
            "note": "4 bps turnover cost, no borrow fee.",
        },
        {
            "metric": "mean_short_gross",
            "value": float(short_gross.mean()),
            "note": "Average rolling h10 short gross.",
        },
    ]
    return pd.DataFrame(rows)


def _coverage_summary(daily_coverage: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "mean_etb_weighted_proxy",
            "value": float(daily_coverage["etb_weighted_proxy"].mean()),
            "note": "Current proxy only; not point-in-time.",
        },
        {
            "metric": "p10_etb_weighted_proxy",
            "value": float(daily_coverage["etb_weighted_proxy"].quantile(0.10)),
            "note": "Current proxy only; not point-in-time.",
        },
        {
            "metric": "mean_non_etb_weight_share",
            "value": float(daily_coverage["non_etb_weight_share"].mean()),
            "note": "Rows not marked ETB by current proxy.",
        },
        {
            "metric": "mean_missing_proxy_weight_share",
            "value": float(daily_coverage["missing_proxy_weight_share"].mean()),
            "note": "Missing current borrow proxy.",
        },
        {
            "metric": "top20_symbol_short_weight_share",
            "value": float(concentration.head(20)["short_weight_sum_share"].sum()),
            "note": "Concentration of recurring short exposure.",
        },
        {
            "metric": "top50_symbol_short_weight_share",
            "value": float(concentration.head(50)["short_weight_sum_share"].sum()),
            "note": "Concentration of recurring short exposure.",
        },
        {
            "metric": "unique_short_symbols",
            "value": float(concentration["symbol"].nunique()),
            "note": "Unique short symbols in rolling h10 aggregate book.",
        },
    ]
    return pd.DataFrame(rows)


def _worst_proxy_days(daily_coverage: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    return (
        daily_coverage.sort_values(
            ["missing_proxy_weight_share", "non_etb_weight_share", "short_gross"],
            ascending=[False, False, False],
        )
        .head(n)
        .reset_index(drop=True)
    )


def _plot(scenario_metrics: pd.DataFrame, scenario_curves: pd.DataFrame, outpath: Path) -> None:
    focus = scenario_metrics[
        scenario_metrics["scenario"].isin(
            [
                "base_etb_zero",
                "all_short_200bps",
                "current_non_etb_500bps",
                "largest_10pct_short_gross_500bps",
                "largest_20pct_short_gross_500bps",
                "top20_recurring_symbols_500bps",
            ]
        )
    ].copy()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    axes[0].barh(focus["scenario"], focus["annualized_return"] * 100, color="#0f766e")
    axes[0].set_title("Phase6H Borrow/HTB Scenario Annualized Return")
    axes[0].set_xlabel("Annualized return %")
    axes[0].grid(axis="x", alpha=0.25)

    curve_focus = scenario_curves[
        scenario_curves["scenario"].isin(
            [
                "base_etb_zero",
                "all_short_200bps",
                "largest_20pct_short_gross_500bps",
                "top20_recurring_symbols_500bps",
            ]
        )
    ].copy()
    colors = {
        "base_etb_zero": "#111827",
        "all_short_200bps": "#2563eb",
        "largest_20pct_short_gross_500bps": "#c2410c",
        "top20_recurring_symbols_500bps": "#7c3aed",
    }
    for scenario, group in curve_focus.groupby("scenario", sort=False):
        axes[1].plot(
            pd.to_datetime(group["date"]),
            group["equity"],
            label=scenario,
            color=colors.get(scenario),
        )
    axes[1].set_title("Borrow Stress Net Equity Paths")
    axes[1].set_ylabel("Growth of $1")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_memo(
    scenario_metrics: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    worst_days: pd.DataFrame,
    breakeven: pd.DataFrame,
    outpath: Path,
) -> None:
    display_scenarios = scenario_metrics[
        [
            "scenario",
            "annualized_return",
            "sharpe_no_rf",
            "max_drawdown",
            "mean_selected_short_weight",
            "approx_annual_borrow_cost_pct_nav",
            "annualized_pnl_usd",
        ]
    ].copy()
    top_names = concentration.head(20)[
        [
            "symbol",
            "active_days",
            "short_weight_sum_share",
            "mean_short_weight_when_active",
            "max_short_weight",
            "median_adv",
            "etb_proxy_rate",
            "missing_proxy_rate",
        ]
    ].copy()
    lines = [
        "# Phase6H Borrow/HTB Diagnostics",
        "",
        "Scope: validation-window only. Test-window performance remains unused.",
        "",
        "This phase keeps the `adv_floor_1m` product candidate fixed and studies short-borrow risk. Current `easy_to_borrow` / `shortable` fields are treated only as a non-point-in-time proxy.",
        "",
        "## Scenario Results",
        "",
        "```text",
        display_scenarios.to_string(index=False),
        "```",
        "",
        "## Proxy Coverage",
        "",
        "```text",
        coverage_summary.to_string(index=False),
        "```",
        "",
        "## Worst Proxy Days",
        "",
        "```text",
        worst_days.head(12).to_string(index=False),
        "```",
        "",
        "## Breakeven",
        "",
        "```text",
        breakeven.to_string(index=False),
        "```",
        "",
        "## Top Recurring Short Names",
        "",
        "```text",
        top_names.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Current ETB/shortable proxy coverage is joined at symbol level because the field is a current proxy, not a date-level historical borrow observation.",
        "- Current ETB/shortable proxy coverage is useful for QA, but it is not sufficient for production because it is not point-in-time.",
        "- Borrow cost becomes material if a large share of the short book is HTB. A small transient non-ETB slice is much less damaging.",
        "- The next data task should collect point-in-time borrow status and borrow fee at order time, ideally from the broker/paper-trading layer.",
        "- Before live trading, any non-shortable or missing-borrow-proxy name should be excluded from the short candidate pool or routed to a manual reject list.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    records, short_book = _prepare_records_and_short_book()
    daily_coverage = _daily_proxy_coverage(short_book)
    concentration = _symbol_concentration(short_book)
    coverage_summary = _coverage_summary(daily_coverage, concentration)
    worst_days = _worst_proxy_days(daily_coverage)
    scenario_metrics, scenario_curves = _apply_borrow_scenarios(
        records,
        short_book,
        concentration,
    )
    breakeven = _breakeven_all_short_borrow(records)

    daily_coverage.to_csv(OUTDIR / "phase6h_daily_borrow_proxy_coverage.csv", index=False)
    concentration.to_csv(OUTDIR / "phase6h_short_symbol_concentration.csv", index=False)
    coverage_summary.to_csv(OUTDIR / "phase6h_borrow_proxy_summary.csv", index=False)
    worst_days.to_csv(OUTDIR / "phase6h_worst_borrow_proxy_days.csv", index=False)
    scenario_metrics.to_csv(OUTDIR / "phase6h_borrow_scenario_metrics.csv", index=False)
    scenario_curves.to_csv(OUTDIR / "phase6h_borrow_scenario_curves.csv", index=False)
    breakeven.to_csv(OUTDIR / "phase6h_borrow_breakeven.csv", index=False)
    _plot(scenario_metrics, scenario_curves, OUTDIR / "phase6h_borrow_htb_plot.png")
    _write_memo(
        scenario_metrics,
        coverage_summary,
        concentration,
        worst_days,
        breakeven,
        OUTDIR / "phase6h_borrow_htb_memo.md",
    )

    print(f"Wrote Phase6H borrow/HTB diagnostics to {OUTDIR}")
    print(scenario_metrics.to_string(index=False))
    print(coverage_summary.to_string(index=False))
    print(breakeven.to_string(index=False))


if __name__ == "__main__":
    main()
