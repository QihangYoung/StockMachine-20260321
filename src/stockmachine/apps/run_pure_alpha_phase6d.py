"""Phase6D product feasibility packet for the US pure-alpha SOTA strategy.

This app keeps the strategy fixed and audits whether the current SOTA is
tradable at <= USD 500k NAV. It only uses validation-window artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    _load_full_positions,
    _load_strict_curve,
)


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT / "research"
PHASE3_PANEL = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_h10_probe_20260418"
    / "phase3_baseline_signal_panel_validation.csv.gz"
)
POSITIONS_PATH = (
    RESEARCH_ROOT
    / "phase5f_overlay_combo_on_turnover_aware_20260427"
    / "phase5f_positions_validation.csv.gz"
)
STRICT_CURVE_PATH = (
    RESEARCH_ROOT
    / "phase5f_overlay_combo_on_turnover_aware_20260427"
    / "strict_h10_rebuild"
    / "phase4z_strict_h10_daily_curve.csv"
)
OUTDIR = RESEARCH_ROOT / "phase6d_product_feasibility_20260507"

SOTA_PORTFOLIO = "short_overlay_only"
CAPITAL_LEVELS = (100_000, 250_000, 500_000)
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class FeasibilityInputs:
    positions: pd.DataFrame
    aggregate_positions: pd.DataFrame
    strict_curve: pd.DataFrame
    records: pd.DataFrame
    liquidity: pd.DataFrame


def _pctile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _format_pct(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.{decimals}f}%"


def _format_usd(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"${value:,.0f}"


def _load_liquidity() -> pd.DataFrame:
    usecols = [
        "session_date",
        "symbol",
        "trailing_median_dollar_volume_20",
        "lagged_close",
        "shortable_current",
        "easy_to_borrow_current",
    ]
    liq = pd.read_csv(PHASE3_PANEL, usecols=usecols, parse_dates=["session_date"])
    liq = liq.dropna(subset=["session_date", "symbol"])
    liq = (
        liq.groupby(["session_date", "symbol"], as_index=False)
        .agg(
            trailing_median_dollar_volume_20=("trailing_median_dollar_volume_20", "max"),
            lagged_close=("lagged_close", "last"),
            shortable_current=("shortable_current", "max"),
            easy_to_borrow_current=("easy_to_borrow_current", "max"),
        )
        .rename(columns={"session_date": "date"})
    )
    return liq


def _prepare_inputs() -> FeasibilityInputs:
    strict_curve = _load_strict_curve(STRICT_CURVE_PATH, portfolio=SOTA_PORTFOLIO)
    positions = _load_full_positions(POSITIONS_PATH, portfolio=SOTA_PORTFOLIO)
    calendar = strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    turnover_curve = _build_h10_turnover_curve(
        positions_path=POSITIONS_PATH,
        portfolio=SOTA_PORTFOLIO,
        benchmark_calendar=calendar,
        holding_period_sessions=10,
    )
    aggregate_positions, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=10,
    )
    records = _build_backtest_records(
        curve=strict_curve,
        turnover=turnover_curve,
        daily_exposure=daily_exposure,
    )
    liquidity = _load_liquidity()
    return FeasibilityInputs(
        positions=positions,
        aggregate_positions=aggregate_positions,
        strict_curve=strict_curve,
        records=records,
        liquidity=liquidity,
    )


def _build_position_liquidity(inputs: FeasibilityInputs) -> pd.DataFrame:
    pos = inputs.aggregate_positions.copy()
    pos["date"] = pd.to_datetime(pos["return_date"])
    pos = pos.merge(inputs.liquidity, on=["date", "symbol"], how="left")
    pos["abs_weight"] = pos["portfolio_weight"].abs()
    pos["position_adv_share"] = (
        pos["abs_weight"] / pos["trailing_median_dollar_volume_20"].replace(0, np.nan)
    )
    # position_adv_share is per USD of NAV. Multiply by NAV before reporting.
    return pos


def _build_order_liquidity(inputs: FeasibilityInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = inputs.aggregate_positions[["return_date", "symbol", "portfolio_weight"]].copy()
    pos["date"] = pd.to_datetime(pos["return_date"])
    dates = sorted(pos["date"].dropna().unique())
    prev: dict[str, float] = {}
    order_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []

    for date in dates:
        day = pos.loc[pos["date"] == date, ["symbol", "portfolio_weight"]]
        curr = dict(zip(day["symbol"], day["portfolio_weight"]))
        symbols = sorted(set(prev) | set(curr))
        total_turnover = 0.0
        for symbol in symbols:
            delta = float(curr.get(symbol, 0.0) - prev.get(symbol, 0.0))
            if abs(delta) < 1e-12:
                continue
            total_turnover += abs(delta)
            order_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "delta_weight": delta,
                    "abs_delta_weight": abs(delta),
                    "direction": "buy" if delta > 0 else "sell",
                }
            )
        turnover_rows.append({"date": date, "turnover_from_position_deltas": total_turnover})
        prev = curr

    orders = pd.DataFrame(order_rows)
    turnover = pd.DataFrame(turnover_rows)
    if orders.empty:
        return orders, turnover
    orders = orders.merge(inputs.liquidity, on=["date", "symbol"], how="left")
    orders["order_adv_share"] = (
        orders["abs_delta_weight"]
        / orders["trailing_median_dollar_volume_20"].replace(0, np.nan)
    )
    return orders, turnover


def _summarize_participation(
    data: pd.DataFrame,
    share_col: str,
    weight_col: str,
    capital_levels: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for capital in capital_levels:
        adv_share = data[share_col] * capital
        notional = data[weight_col] * capital
        rows.append(
            {
                "capital_usd": capital,
                "rows": int(len(data)),
                "coverage_rate": float(data["trailing_median_dollar_volume_20"].notna().mean()),
                "mean_notional_usd": float(notional.mean()),
                "p95_notional_usd": _pctile(notional, 0.95),
                "p99_notional_usd": _pctile(notional, 0.99),
                "max_notional_usd": float(notional.max()),
                "mean_adv_participation": float(adv_share.mean()),
                "p90_adv_participation": _pctile(adv_share, 0.90),
                "p95_adv_participation": _pctile(adv_share, 0.95),
                "p99_adv_participation": _pctile(adv_share, 0.99),
                "max_adv_participation": float(adv_share.max()),
                "share_rows_over_1pct_adv": float((adv_share > 0.01).mean()),
                "share_rows_over_5pct_adv": float((adv_share > 0.05).mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_low_adv_gate_summary(
    position_liq: pd.DataFrame,
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    offender_rows: list[pd.DataFrame] = []
    position_total_weight = float(position_liq["abs_weight"].sum())
    order_total_weight = float(orders["abs_delta_weight"].sum())

    for threshold in (1_000_000.0, 5_000_000.0, 10_000_000.0, 30_000_000.0):
        pos_mask = position_liq["trailing_median_dollar_volume_20"].fillna(0.0) < threshold
        ord_mask = orders["trailing_median_dollar_volume_20"].fillna(0.0) < threshold
        pos_bad = position_liq.loc[pos_mask].copy()
        ord_bad = orders.loc[ord_mask].copy()
        rows.append(
            {
                "adv_floor_usd": threshold,
                "position_row_share_below_floor": float(pos_mask.mean()),
                "position_abs_weight_share_below_floor": float(
                    pos_bad["abs_weight"].sum() / position_total_weight
                )
                if position_total_weight > 0
                else np.nan,
                "order_row_share_below_floor": float(ord_mask.mean()),
                "order_abs_delta_share_below_floor": float(
                    ord_bad["abs_delta_weight"].sum() / order_total_weight
                )
                if order_total_weight > 0
                else np.nan,
                "unique_position_symbols_below_floor": int(pos_bad["symbol"].nunique()),
                "unique_order_symbols_below_floor": int(ord_bad["symbol"].nunique()),
            }
        )
        if not pos_bad.empty:
            offenders = (
                pos_bad.groupby("symbol", as_index=False)
                .agg(
                    rows=("symbol", "size"),
                    first_date=("date", "min"),
                    last_date=("date", "max"),
                    abs_weight_sum=("abs_weight", "sum"),
                    min_adv=("trailing_median_dollar_volume_20", "min"),
                    median_adv=("trailing_median_dollar_volume_20", "median"),
                    max_position_weight=("abs_weight", "max"),
                )
                .sort_values(["abs_weight_sum", "rows"], ascending=False)
                .head(20)
            )
            offenders.insert(0, "adv_floor_usd", threshold)
            offender_rows.append(offenders)

    offender_table = pd.concat(offender_rows, ignore_index=True) if offender_rows else pd.DataFrame()
    return pd.DataFrame(rows), offender_table


def _build_capacity_summary(
    inputs: FeasibilityInputs,
    position_liq: pd.DataFrame,
    orders: pd.DataFrame,
    turnover_from_positions: pd.DataFrame,
) -> pd.DataFrame:
    daily = inputs.records.copy()
    turnover = turnover_from_positions.rename(columns={"turnover_from_position_deltas": "delta_turnover"})
    daily["date"] = pd.to_datetime(daily["entry_date"])
    daily = daily.merge(turnover, on="date", how="left")
    daily["delta_turnover"] = daily["delta_turnover"].fillna(daily["turnover"])
    recurring = daily[daily["date"] > daily["date"].min()].copy()
    fully_ramped = daily[daily["active_sleeves"].fillna(0).astype(float) >= 10.0].copy()

    rows: list[dict[str, object]] = []
    for capital in CAPITAL_LEVELS:
        rows.append(
            {
                "capital_usd": capital,
                "mean_gross_nav": float(daily["gross_exposure"].mean()),
                "p95_gross_nav": _pctile(daily["gross_exposure"], 0.95),
                "max_gross_nav": float(daily["gross_exposure"].max()),
                "mean_long_nav": float(daily["long_gross"].mean()),
                "mean_short_nav": float(daily["short_gross"].mean()),
                "mean_daily_turnover_nav": float(daily["delta_turnover"].mean()),
                "p95_daily_turnover_nav": _pctile(daily["delta_turnover"], 0.95),
                "max_daily_turnover_nav": float(daily["delta_turnover"].max()),
                "max_recurring_daily_turnover_nav": float(recurring["delta_turnover"].max()),
                "max_fully_ramped_daily_turnover_nav": float(fully_ramped["delta_turnover"].max()),
                "mean_daily_traded_usd": float(daily["delta_turnover"].mean() * capital),
                "p95_daily_traded_usd": float(_pctile(daily["delta_turnover"], 0.95) * capital),
                "max_daily_traded_usd": float(daily["delta_turnover"].max() * capital),
                "max_recurring_daily_traded_usd": float(recurring["delta_turnover"].max() * capital),
                "max_fully_ramped_daily_traded_usd": float(
                    fully_ramped["delta_turnover"].max() * capital
                ),
                "max_single_position_usd": float(position_liq["abs_weight"].max() * capital),
                "p99_single_order_usd": float(_pctile(orders["abs_delta_weight"], 0.99) * capital),
                "max_single_order_usd": float(orders["abs_delta_weight"].max() * capital),
                "position_p99_adv_participation": float(
                    _pctile(position_liq["position_adv_share"] * capital, 0.99)
                ),
                "position_max_adv_participation": float(
                    (position_liq["position_adv_share"] * capital).max()
                ),
                "order_p99_adv_participation": float(
                    _pctile(orders["order_adv_share"] * capital, 0.99)
                ),
                "order_max_adv_participation": float(
                    (orders["order_adv_share"] * capital).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_worst_rows(
    data: pd.DataFrame,
    share_col: str,
    weight_col: str,
    capital: int,
    n: int = 25,
) -> pd.DataFrame:
    out = data.copy()
    out["capital_usd"] = capital
    out["notional_usd"] = out[weight_col] * capital
    out["adv_participation"] = out[share_col] * capital
    cols = [
        "date",
        "symbol",
        "capital_usd",
        "notional_usd",
        "adv_participation",
        "trailing_median_dollar_volume_20",
        "lagged_close",
        "shortable_current",
        "easy_to_borrow_current",
    ]
    if "direction" in out.columns:
        cols.insert(2, "direction")
    return out.sort_values("adv_participation", ascending=False).head(n)[cols]


def _build_borrow_proxy_summary(position_liq: pd.DataFrame) -> pd.DataFrame:
    short_rows = position_liq[position_liq["portfolio_weight"] < 0].copy()
    abs_weight = short_rows["abs_weight"]
    total_abs = abs_weight.sum()

    def _weighted_mean(col: str) -> float:
        if total_abs <= 0 or col not in short_rows:
            return float("nan")
        values = short_rows[col].fillna(0).astype(float)
        return float((values * abs_weight).sum() / total_abs)

    rows = [
        {
            "metric": "short_rows",
            "value": float(len(short_rows)),
            "note": "Short-side aggregate position rows.",
        },
        {
            "metric": "short_notional_weighted_shortable_proxy",
            "value": _weighted_mean("shortable_current"),
            "note": "Current metadata proxy, not point-in-time borrow availability.",
        },
        {
            "metric": "short_notional_weighted_etb_proxy",
            "value": _weighted_mean("easy_to_borrow_current"),
            "note": "Current ETB proxy, not point-in-time borrow fee history.",
        },
        {
            "metric": "unique_short_symbols",
            "value": float(short_rows["symbol"].nunique()),
            "note": "Validation-window short symbols in rolling aggregate book.",
        },
    ]
    return pd.DataFrame(rows)


def _build_margin_stress(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shock in (0.05, 0.10, 0.20, 0.30):
        both_adverse = shock * (records["long_gross"] + records["short_gross"])
        long_down_only = shock * records["long_gross"]
        short_up_only = shock * records["short_gross"]
        rows.append(
            {
                "shock": shock,
                "mean_both_legs_adverse_loss_nav": float(both_adverse.mean()),
                "p95_both_legs_adverse_loss_nav": _pctile(both_adverse, 0.95),
                "max_both_legs_adverse_loss_nav": float(both_adverse.max()),
                "min_equity_after_max_both_adverse": float(1.0 - both_adverse.max()),
                "p95_long_down_only_loss_nav": _pctile(long_down_only, 0.95),
                "p95_short_up_only_loss_nav": _pctile(short_up_only, 0.95),
                "note": "Stress PnL proxy only; not a broker-specific margin-call model.",
            }
        )
    return pd.DataFrame(rows)


def _build_cost_scenarios(records: pd.DataFrame, capital: int) -> pd.DataFrame:
    raw = _build_financing_borrow_stress(
        records,
        transaction_cost_levels_bps=(0.0, 2.0, 4.0, 8.0),
        borrow_cost_levels_bps_annual=(0.0, 200.0),
        margin_cost_levels_bps_annual=(0.0, 300.0, 475.0),
    ).reset_index(drop=True)
    raw["scenario"] = raw.apply(
        lambda row: "gross_no_cost"
        if row["transaction_cost_bps_per_traded_notional"] == 0.0
        and row["borrow_cost_bps_annual_on_short_gross"] == 0.0
        and row["margin_cost_bps_annual_on_gross_above_100pct_nav"] == 0.0
        else (
            f"tc{row['transaction_cost_bps_per_traded_notional']:.0f}bps_"
            f"margin{row['margin_cost_bps_annual_on_gross_above_100pct_nav'] / 100:.2f}pct_"
            f"borrow{row['borrow_cost_bps_annual_on_short_gross'] / 100:.1f}pct"
        ),
        axis=1,
    )
    raw["capital_usd"] = capital
    raw["annualized_pnl_usd"] = raw["annualized_return"] * capital
    raw["annualized_vol_usd"] = raw["annualized_vol"] * capital
    gross_ann = float(raw.loc[raw["scenario"] == "gross_no_cost", "annualized_return"].iloc[0])
    raw["annualized_return_haircut_vs_gross"] = gross_ann - raw["annualized_return"]
    raw["annualized_cost_usd_vs_gross"] = raw["annualized_return_haircut_vs_gross"] * capital
    return raw.sort_values(
        [
            "transaction_cost_bps_per_traded_notional",
            "margin_cost_bps_annual_on_gross_above_100pct_nav",
            "borrow_cost_bps_annual_on_short_gross",
        ]
    ).reset_index(drop=True)


def _build_summary_metrics(inputs: FeasibilityInputs) -> pd.DataFrame:
    metrics = _daily_path_metrics(
        inputs.strict_curve["gross_return"],
        benchmark=inputs.strict_curve["benchmark_oto_return"],
    )
    exposure = inputs.records[
        ["gross_exposure", "long_gross", "short_gross", "turnover"]
    ].agg(["mean", "median", "max"])
    rows = [{"metric": k, "value": v, "note": "Gross daily validation path."} for k, v in metrics.items()]
    rows.extend(
        [
            {
                "metric": "mean_gross_exposure_nav",
                "value": float(exposure.loc["mean", "gross_exposure"]),
                "note": "Rolling h10 aggregate gross exposure.",
            },
            {
                "metric": "mean_long_exposure_nav",
                "value": float(exposure.loc["mean", "long_gross"]),
                "note": "Rolling h10 aggregate long exposure.",
            },
            {
                "metric": "mean_short_exposure_nav",
                "value": float(exposure.loc["mean", "short_gross"]),
                "note": "Rolling h10 aggregate short exposure.",
            },
            {
                "metric": "mean_turnover_nav_per_day",
                "value": float(exposure.loc["mean", "turnover"]),
                "note": "Daily rolling h10 aggregate turnover.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _plot_feasibility(
    capacity: pd.DataFrame,
    position_part: pd.DataFrame,
    order_part: pd.DataFrame,
    cost: pd.DataFrame,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Phase6D Product Feasibility: Current SOTA, Validation Only", fontsize=14)

    ax = axes[0, 0]
    ax.plot(
        capacity["capital_usd"],
        capacity["mean_daily_traded_usd"],
        marker="o",
        label="mean daily traded",
    )
    ax.plot(
        capacity["capital_usd"],
        capacity["p95_daily_traded_usd"],
        marker="o",
        label="p95 daily traded",
    )
    ax.set_title("Daily Trading Notional")
    ax.set_xlabel("NAV")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        position_part["capital_usd"],
        position_part["p99_adv_participation"] * 100,
        marker="o",
        label="position p99",
    )
    ax.plot(
        order_part["capital_usd"],
        order_part["p99_adv_participation"] * 100,
        marker="o",
        label="order p99",
    )
    ax.plot(
        order_part["capital_usd"],
        order_part["max_adv_participation"] * 100,
        marker="o",
        label="order max",
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="1% ADV")
    ax.set_title("ADV Participation")
    ax.set_xlabel("NAV")
    ax.set_ylabel("% of 20d median dollar ADV")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    focus = cost[
        (cost["borrow_cost_bps_annual_on_short_gross"] == 0.0)
        & (cost["margin_cost_bps_annual_on_gross_above_100pct_nav"].isin([0.0, 475.0]))
        & (cost["transaction_cost_bps_per_traded_notional"].isin([0.0, 4.0, 8.0]))
    ].copy()
    labels = [
        row["scenario"].replace("_", "\n")
        for _, row in focus.iterrows()
    ]
    ax.bar(range(len(focus)), focus["annualized_return"] * 100, color="#2a6f97")
    ax.set_xticks(range(len(focus)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Cost/Financing Sensitivity at $500k")
    ax.set_ylabel("Annualized return %")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.plot(
        capacity["capital_usd"],
        capacity["max_single_position_usd"],
        marker="o",
        label="max position",
    )
    ax.plot(
        capacity["capital_usd"],
        capacity["max_single_order_usd"],
        marker="o",
        label="max daily order",
    )
    ax.set_title("Single-Name Dollar Size")
    ax.set_xlabel("NAV")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_memo(
    outpath: Path,
    summary: pd.DataFrame,
    capacity: pd.DataFrame,
    position_part: pd.DataFrame,
    order_part: pd.DataFrame,
    cost: pd.DataFrame,
    borrow_proxy: pd.DataFrame,
    margin_stress: pd.DataFrame,
    low_adv: pd.DataFrame,
    low_adv_offenders: pd.DataFrame,
) -> None:
    metric = dict(zip(summary["metric"], summary["value"]))
    cap500 = capacity.loc[capacity["capital_usd"] == 500_000].iloc[0]
    pos500 = position_part.loc[position_part["capital_usd"] == 500_000].iloc[0]
    ord500 = order_part.loc[order_part["capital_usd"] == 500_000].iloc[0]
    gross = cost.loc[cost["scenario"] == "gross_no_cost"].iloc[0]
    tc4 = cost.loc[
        (cost["transaction_cost_bps_per_traded_notional"] == 4.0)
        & (cost["margin_cost_bps_annual_on_gross_above_100pct_nav"] == 0.0)
        & (cost["borrow_cost_bps_annual_on_short_gross"] == 0.0)
    ].iloc[0]
    tc4_margin = cost.loc[
        (cost["transaction_cost_bps_per_traded_notional"] == 4.0)
        & (cost["margin_cost_bps_annual_on_gross_above_100pct_nav"] == 475.0)
        & (cost["borrow_cost_bps_annual_on_short_gross"] == 0.0)
    ].iloc[0]
    etb = borrow_proxy.loc[
        borrow_proxy["metric"] == "short_notional_weighted_etb_proxy", "value"
    ].iloc[0]
    stress10 = margin_stress.loc[margin_stress["shock"] == 0.10].iloc[0]
    low_adv_1m = low_adv.loc[low_adv["adv_floor_usd"] == 1_000_000.0].iloc[0]
    top_offender = (
        low_adv_offenders.loc[low_adv_offenders["adv_floor_usd"] == 1_000_000.0].head(1)
        if not low_adv_offenders.empty
        else pd.DataFrame()
    )
    offender_text = "none"
    if not top_offender.empty:
        offender = top_offender.iloc[0]
        offender_text = (
            f"{offender['symbol']} "
            f"(min ADV {_format_usd(float(offender['min_adv']))}, "
            f"rows {int(offender['rows'])})"
        )

    text = f"""# Phase6D Product Feasibility Memo

Date: 2026-05-07

Scope: validation-window only. This memo keeps the current SOTA fixed and asks whether it is operationally feasible for accounts up to USD 500k. No test-window data is used.

## Strategy Under Review

- Strategy: Phase5F `short_overlay_only` / `current_sota`.
- Construction: rolling h10, exact ex-ante beta match, SIC2 soft neutral, turnover-aware optimizer.
- Gross path before costs: annualized return {_format_pct(float(metric['annualized_return']))}, annualized volatility {_format_pct(float(metric['annualized_vol']))}, Sharpe {float(metric['sharpe_no_rf']):.2f}, max drawdown {_format_pct(float(metric['max_drawdown']))}.
- Average rolling exposure: gross {_format_pct(float(metric['mean_gross_exposure_nav']))}, long {_format_pct(float(metric['mean_long_exposure_nav']))}, short {_format_pct(float(metric['mean_short_exposure_nav']))}.

## Main Takeaways

1. Typical capacity is not the binding constraint at <= USD 500k. At USD 500k NAV, mean daily traded notional is {_format_usd(float(cap500['mean_daily_traded_usd']))}, p95 daily traded notional is {_format_usd(float(cap500['p95_daily_traded_usd']))}, and the largest single-name aggregate position is {_format_usd(float(cap500['max_single_position_usd']))}.
2. ADV participation is tiny for the typical book but has a real data/universe tail. At USD 500k NAV, p99 position participation is {_format_pct(float(pos500['p99_adv_participation']))}, p99 single-order participation is {_format_pct(float(ord500['p99_adv_participation']))}, but worst observed single-order participation is {_format_pct(float(ord500['max_adv_participation']))}. The top low-ADV offender under a USD 1m ADV floor is {offender_text}.
3. The economic bottleneck is not market capacity; it is path quality after transaction cost, borrow availability, and any actual margin debit. A pure 4 bps/turnover charge lowers annualized return from {_format_pct(float(gross['annualized_return']))} to {_format_pct(float(tc4['annualized_return']))}. The 4.75% line is only a debit-balance stress proxy; if the long book is funded without an overnight debit balance, this stress is not an Alpaca base-case cost.
4. Borrow availability is still a production blocker because the current research data only has current shortable/ETB metadata. The notional-weighted current ETB proxy is {_format_pct(float(etb))}, but this is not point-in-time borrow availability or point-in-time borrow fee history.

## Capacity Detail at USD 500k

- Mean daily turnover: {_format_pct(float(cap500['mean_daily_turnover_nav']))} of NAV.
- P95 daily turnover: {_format_pct(float(cap500['p95_daily_turnover_nav']))} of NAV.
- Max daily turnover: {_format_pct(float(cap500['max_daily_turnover_nav']))} of NAV during the h10 ramp-up/build period.
- Max fully-ramped daily turnover after 10 active sleeves: {_format_pct(float(cap500['max_fully_ramped_daily_turnover_nav']))} of NAV.
- P99 single order: {_format_usd(float(cap500['p99_single_order_usd']))}.
- Max single order: {_format_usd(float(cap500['max_single_order_usd']))}.
- P99 order ADV participation: {_format_pct(float(ord500['p99_adv_participation']))}.
- Max order ADV participation: {_format_pct(float(ord500['max_adv_participation']))}.

## Low-ADV Tail

Under a USD 1m ADV floor, affected aggregate position rows are {_format_pct(float(low_adv_1m['position_row_share_below_floor']))} of rows and {_format_pct(float(low_adv_1m['position_abs_weight_share_below_floor']))} of aggregate absolute weight. This tail is small in aggregate, but it is not ignorable operationally because the worst names can dominate their own ADV. The correct production fix is a hard point-in-time ADV gate before optimization, not a post-trade exception.

## Margin / Leverage Stress

The strategy averages about 178% gross exposure, so the account is economically leveraged even when the long book is below 100% NAV. A simple stress proxy says that if both long and short books move against us by 10% on the same day, the p95 loss would be {_format_pct(float(stress10['p95_both_legs_adverse_loss_nav']))} of NAV. This is not a broker-specific margin-call model; it is a sanity check for equity cushion and tail sizing.

Alpaca margin interest should be charged only on an overnight debit balance, not mechanically on gross exposure above 100% NAV. Therefore the no-debit base case should use zero margin-interest cost; the 4.75% line is retained only as a conservative stress scenario.

## Production Gates

- OK for USD 500k typical market capacity, but not OK until low-ADV symbol/date leaks are gated point-in-time.
- Not OK to call production-ready until borrow availability and borrow fees are point-in-time.
- Not OK to rely on gross return alone; report net paths under explicit turnover, debit-balance, and borrow assumptions.
- Next recommended work: add a hard ADV floor before portfolio construction, implement point-in-time borrow/ETB collection, paper-trading order simulation, and broker-specific margin/liquidation stress.
"""
    outpath.write_text(text, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inputs = _prepare_inputs()
    position_liq = _build_position_liquidity(inputs)
    orders, turnover_from_positions = _build_order_liquidity(inputs)

    summary = _build_summary_metrics(inputs)
    position_part = _summarize_participation(
        position_liq,
        "position_adv_share",
        "abs_weight",
        CAPITAL_LEVELS,
    )
    order_part = _summarize_participation(
        orders,
        "order_adv_share",
        "abs_delta_weight",
        CAPITAL_LEVELS,
    )
    capacity = _build_capacity_summary(inputs, position_liq, orders, turnover_from_positions)
    low_adv, low_adv_offenders = _build_low_adv_gate_summary(position_liq, orders)
    worst_positions = _build_worst_rows(
        position_liq,
        "position_adv_share",
        "abs_weight",
        max(CAPITAL_LEVELS),
    )
    worst_orders = _build_worst_rows(
        orders,
        "order_adv_share",
        "abs_delta_weight",
        max(CAPITAL_LEVELS),
    )
    borrow_proxy = _build_borrow_proxy_summary(position_liq)
    margin_stress = _build_margin_stress(inputs.records)
    cost = _build_cost_scenarios(inputs.records, max(CAPITAL_LEVELS))

    summary.to_csv(OUTDIR / "phase6d_product_feasibility_summary.csv", index=False)
    capacity.to_csv(OUTDIR / "phase6d_capital_capacity_summary.csv", index=False)
    position_part.to_csv(OUTDIR / "phase6d_position_adv_participation_summary.csv", index=False)
    order_part.to_csv(OUTDIR / "phase6d_order_adv_participation_summary.csv", index=False)
    low_adv.to_csv(OUTDIR / "phase6d_low_adv_gate_summary.csv", index=False)
    low_adv_offenders.to_csv(OUTDIR / "phase6d_low_adv_offenders.csv", index=False)
    worst_positions.to_csv(OUTDIR / "phase6d_worst_position_participation.csv", index=False)
    worst_orders.to_csv(OUTDIR / "phase6d_worst_order_participation.csv", index=False)
    borrow_proxy.to_csv(OUTDIR / "phase6d_borrow_proxy_summary.csv", index=False)
    margin_stress.to_csv(OUTDIR / "phase6d_margin_stress.csv", index=False)
    cost.to_csv(OUTDIR / "phase6d_cost_financing_scenarios.csv", index=False)

    _plot_feasibility(
        capacity,
        position_part,
        order_part,
        cost,
        OUTDIR / "phase6d_product_feasibility_plot.png",
    )
    _write_memo(
        OUTDIR / "phase6d_product_feasibility_memo.md",
        summary,
        capacity,
        position_part,
        order_part,
        cost,
        borrow_proxy,
        margin_stress,
        low_adv,
        low_adv_offenders,
    )

    print(f"Wrote Phase6D product feasibility artifacts to {OUTDIR}")
    print(capacity.to_string(index=False))
    print(order_part.to_string(index=False))
    print(cost.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
