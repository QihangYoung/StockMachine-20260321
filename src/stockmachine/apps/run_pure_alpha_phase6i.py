"""Phase6I ETB-gated optimizer replay and forward borrow schema.

Alpaca cannot open new HTB shorts, so the production rule is not "pay HTB
fees if alpha is high enough." The production rule is:

* new/increased short exposure must be shortable and ETB at order time;
* existing shorts that transition away from ETB are monitored and de-risked;
* point-in-time borrow status must be collected going forward.

Historical ETB availability is not available in the research dataset. This app
therefore runs synthetic ETB-unavailability scenarios and emits the forward data
schema needed for paper/live collection. Validation window only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SHORT_VARIANT,
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
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_SHORT_OVERLAY, DEFAULT_TURNOVER_PENALTY
from stockmachine.apps.run_pure_alpha_phase6 import _build_daily_aggregate_positions, _load_strict_curve
from stockmachine.apps.run_pure_alpha_phase6e import _load_feature_panel
from stockmachine.apps.run_pure_alpha_phase6f import CAPITAL_USD
from stockmachine.apps.run_pure_alpha_phase6h import OUTDIR as PHASE6H_OUTDIR


PROJECT = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT / "research"
OUTDIR = RESEARCH_ROOT / "phase6i_etb_gated_optimizer_20260508"
STRICT_ROOT = OUTDIR / "strict_h10_rebuild"
POSITIONS_PATH = OUTDIR / "phase6i_positions_validation.csv.gz"
ADV_FLOOR_USD = 1_000_000.0

PHASE6E_ROOT = RESEARCH_ROOT / "phase6e_adv_floor_hardening_20260507"
BASELINE_POSITIONS = PHASE6E_ROOT / "phase6e_positions_validation.csv.gz"
TOP_SHORT_CONCENTRATION = PHASE6H_OUTDIR / "phase6h_short_symbol_concentration.csv"


@dataclass(frozen=True)
class GateScenario:
    portfolio: str
    series: str
    mode: str
    reject_share: float = 0.0
    top_symbol_count: int = 0
    note: str = ""


SCENARIOS: tuple[GateScenario, ...] = (
    GateScenario(
        portfolio="etb_baseline_adv1m",
        series="ETB baseline ADV $1M",
        mode="baseline_existing",
        note="Existing Phase6E production candidate, renamed for comparison.",
    ),
    GateScenario(
        portfolio="etb_block_top20_recurring",
        series="ETB gate: block top20 recurring shorts",
        mode="block_top_recurring",
        top_symbol_count=20,
        note="Synthetic stress: top 20 recurring short names are unavailable for new shorts.",
    ),
    GateScenario(
        portfolio="etb_random_10pct_symbol_day",
        series="ETB gate: random 10% symbol-day rejects",
        mode="random_symbol_day",
        reject_share=0.10,
        note="Synthetic stress: deterministic 10% of short symbol-days are unavailable.",
    ),
    GateScenario(
        portfolio="etb_random_20pct_symbol_day",
        series="ETB gate: random 20% symbol-day rejects",
        mode="random_symbol_day",
        reject_share=0.20,
        note="Synthetic stress: deterministic 20% of short symbol-days are unavailable.",
    ),
)


def _stable_reject_mask(frame: pd.DataFrame, reject_share: float) -> pd.Series:
    if reject_share <= 0:
        return pd.Series(False, index=frame.index)
    key = (
        pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d")
        + "|"
        + frame["symbol"].astype(str)
    )
    hashed = pd.util.hash_pandas_object(key, index=False).astype("uint64")
    threshold = int(reject_share * 10_000)
    return (hashed % 10_000) < threshold


def _load_top_recurring_short_symbols(count: int) -> set[str]:
    if not TOP_SHORT_CONCENTRATION.exists() or count <= 0:
        return set()
    frame = pd.read_csv(TOP_SHORT_CONCENTRATION)
    return set(frame.head(count)["symbol"].astype(str))


def _baseline_positions() -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_POSITIONS, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq("adv_floor_1m")].copy()
    frame["portfolio"] = "etb_baseline_adv1m"
    frame["series"] = "ETB baseline ADV $1M"
    frame["etb_gate_mode"] = "baseline_existing"
    frame["synthetic_etb_reject_share"] = 0.0
    return frame


def _apply_gate(panel: pd.DataFrame, scenario: GateScenario, top_symbols: set[str]) -> pd.DataFrame:
    eligible = panel[panel["trailing_median_dollar_volume_20"].fillna(0.0) >= ADV_FLOOR_USD].copy()
    if scenario.mode == "block_top_recurring":
        mask = (eligible["variant"].astype(str).eq(DEFAULT_SHORT_VARIANT)) & (
            eligible["symbol"].astype(str).isin(top_symbols)
        )
        return eligible[~mask].copy()
    if scenario.mode == "random_symbol_day":
        short_rows = eligible["variant"].astype(str).eq(DEFAULT_SHORT_VARIANT)
        reject = pd.Series(False, index=eligible.index)
        reject.loc[short_rows] = _stable_reject_mask(eligible.loc[short_rows], scenario.reject_share)
        return eligible[~reject].copy()
    raise ValueError(f"Unsupported gate mode for construction: {scenario.mode}")


def _construct_gated_positions(
    panel: pd.DataFrame,
    scenario: GateScenario,
    top_symbols: set[str],
    *,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = _apply_gate(panel, scenario, top_symbols)
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in eligible.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    previous_side: dict[str, dict[str, float]] = {"long": {}, "short": {}}
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for session_date in sessions:
        long_group = groups.get((DEFAULT_LONG_VARIANT, session_date))
        short_group = groups.get((DEFAULT_SHORT_VARIANT, session_date))
        if long_group is None or short_group is None:
            skipped.append(_skip_row(session_date, scenario.portfolio, 0, 0, "missing_group"))
            continue
        book, diagnostic, skip, updated = _construct_one_session_turnover_aware(
            long_group,
            short_group,
            session_date=session_date,
            portfolio=scenario.portfolio,
            long_variant=DEFAULT_LONG_VARIANT,
            short_variant=DEFAULT_SHORT_VARIANT,
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
        for row in book:
            row["series"] = scenario.series
            row["etb_gate_mode"] = scenario.mode
            row["synthetic_etb_reject_share"] = scenario.reject_share
        positions.extend(book)
        if diagnostic is not None:
            diagnostic["series"] = scenario.series
            diagnostic["etb_gate_mode"] = scenario.mode
            diagnostic["synthetic_etb_reject_share"] = scenario.reject_share
            diagnostics.append(diagnostic)
        if skip is not None:
            skip["series"] = scenario.series
            skip["etb_gate_mode"] = scenario.mode
            skip["synthetic_etb_reject_share"] = scenario.reject_share
            skipped.append(skip)
        previous_side = updated

    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _build_positions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _load_feature_panel()
    top_symbols = _load_top_recurring_short_symbols(20)
    all_positions = [_baseline_positions()]
    all_diagnostics: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        if scenario.mode == "baseline_existing":
            continue
        positions, diagnostics, skipped = _construct_gated_positions(panel, scenario, top_symbols)
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


def _build_metrics(strict_curve: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    strict_metrics = _strict_metric_summary(strict_curve)
    target_turnover = _target_turnover_summary(positions)
    calendar = strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=POSITIONS_PATH,
        benchmark_returns=calendar.to_frame(name="session_date"),
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
    )
    scenario_frame = pd.DataFrame(
        [
            {
                "portfolio": item.portfolio,
                "series": item.series,
                "etb_gate_mode": item.mode,
                "synthetic_etb_reject_share": item.reject_share,
                "note": item.note,
            }
            for item in SCENARIOS
        ]
    )
    return (
        scenario_frame.merge(strict_metrics, on="portfolio", how="left")
        .merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .reset_index(drop=True)
    )


def _build_curve_export(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    labels = {item.portfolio: item.series for item in SCENARIOS}
    for portfolio in labels:
        sub = strict_curve[strict_curve["portfolio"].astype(str).eq(portfolio)].copy()
        if sub.empty:
            continue
        sub["series"] = labels[portfolio]
        sub["equity"] = (1.0 + sub["gross_return"].astype(float)).cumprod()
        sub["drawdown"] = sub["equity"] / sub["equity"].cummax() - 1.0
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_short_action_audit(strict_curve: pd.DataFrame, positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = positions.copy()
    positions["session_date"] = pd.to_datetime(positions["session_date"])
    strict_curve = strict_curve.copy()
    strict_curve["return_date"] = pd.to_datetime(strict_curve["return_date"])
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        curve = strict_curve[strict_curve["portfolio"].astype(str).eq(scenario.portfolio)].copy()
        pos = positions[positions["portfolio"].astype(str).eq(scenario.portfolio)].copy()
        if curve.empty or pos.empty:
            continue
        calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
        aggregate, _ = _build_daily_aggregate_positions(
            pos,
            benchmark_calendar=calendar,
            holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
        )
        aggregate["date"] = pd.to_datetime(aggregate["return_date"])
        prev: dict[str, float] = {}
        action_rows: list[dict[str, object]] = []
        for date in sorted(aggregate["date"].unique()):
            day = aggregate.loc[aggregate["date"].eq(date), ["symbol", "portfolio_weight"]]
            curr = dict(zip(day["symbol"].astype(str), day["portfolio_weight"].astype(float)))
            for symbol in sorted(set(prev) | set(curr)):
                old_weight = float(prev.get(symbol, 0.0))
                new_weight = float(curr.get(symbol, 0.0))
                old_short = max(-old_weight, 0.0)
                new_short = max(-new_weight, 0.0)
                add_short = max(new_short - old_short, 0.0)
                reduce_short = max(old_short - new_short, 0.0)
                if add_short <= 1e-12 and reduce_short <= 1e-12:
                    continue
                action_rows.append(
                    {
                        "date": date,
                        "portfolio": scenario.portfolio,
                        "series": scenario.series,
                        "symbol": symbol,
                        "old_short_weight": old_short,
                        "new_short_weight": new_short,
                        "add_short_weight": add_short,
                        "reduce_short_weight": reduce_short,
                        "add_short_usd_at_500k": add_short * CAPITAL_USD,
                        "requires_etb_gate": bool(add_short > 1e-12),
                    }
                )
            prev = curr
        actions = pd.DataFrame(action_rows)
        if actions.empty:
            continue
        rows.append(actions)
        daily = actions.groupby("date", as_index=False).agg(
            daily_add_short_weight=("add_short_weight", "sum"),
            daily_reduce_short_weight=("reduce_short_weight", "sum"),
            add_order_count=("requires_etb_gate", "sum"),
            max_single_add_short_weight=("add_short_weight", "max"),
        )
        summaries.append(
            {
                "portfolio": scenario.portfolio,
                "series": scenario.series,
                "mean_daily_add_short_nav": float(daily["daily_add_short_weight"].mean()),
                "p95_daily_add_short_nav": float(daily["daily_add_short_weight"].quantile(0.95)),
                "max_daily_add_short_nav": float(daily["daily_add_short_weight"].max()),
                "mean_daily_add_short_usd_at_500k": float(
                    daily["daily_add_short_weight"].mean() * CAPITAL_USD
                ),
                "p95_daily_add_short_usd_at_500k": float(
                    daily["daily_add_short_weight"].quantile(0.95) * CAPITAL_USD
                ),
                "max_daily_add_short_usd_at_500k": float(
                    daily["daily_add_short_weight"].max() * CAPITAL_USD
                ),
                "mean_etb_gate_orders_per_day": float(daily["add_order_count"].mean()),
                "p95_etb_gate_orders_per_day": float(daily["add_order_count"].quantile(0.95)),
                "max_etb_gate_orders_per_day": int(daily["add_order_count"].max()),
                "max_single_add_short_usd_at_500k": float(
                    daily["max_single_add_short_weight"].max() * CAPITAL_USD
                ),
            }
        )
    action_audit = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.DataFrame(summaries), action_audit


def _schema_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    borrow_snapshot = pd.DataFrame(
        [
            ("asof_ts_utc", "timestamp", "UTC timestamp when Alpaca /assets snapshot was fetched."),
            ("trade_date", "date", "Trading date the snapshot is intended to gate."),
            ("symbol", "string", "Ticker symbol."),
            ("alpaca_asset_id", "string", "Alpaca asset id when available."),
            ("status", "string", "Alpaca asset status."),
            ("tradable", "bool", "Alpaca tradable flag."),
            ("shortable", "bool", "Alpaca shortable flag."),
            ("easy_to_borrow", "bool", "Alpaca ETB flag; required for new shorts."),
            ("source", "string", "Data source, e.g. alpaca_assets_api."),
            ("raw_payload_hash", "string", "Hash of raw API payload for auditability."),
        ],
        columns=["column", "dtype", "description"],
    )
    order_audit = pd.DataFrame(
        [
            ("decision_date", "date", "Signal/optimizer decision date."),
            ("submit_ts_utc", "timestamp", "Order submission timestamp."),
            ("symbol", "string", "Ticker symbol."),
            ("side", "string", "sell_short, buy_to_cover, sell, buy."),
            ("desired_delta_weight", "float", "Desired portfolio weight delta."),
            ("desired_notional_usd", "float", "Desired order notional."),
            ("shortable_at_submit", "bool", "Borrow gate shortable flag at submit time."),
            ("etb_at_submit", "bool", "Borrow gate ETB flag at submit time."),
            ("gate_decision", "string", "allow, reject_not_etb, reject_not_shortable, reject_unknown."),
            ("submitted_qty", "float", "Quantity submitted after gate."),
            ("broker_order_status", "string", "Alpaca order status."),
            ("reject_reason", "string", "Broker or internal reject reason."),
        ],
        columns=["column", "dtype", "description"],
    )
    transition_monitor = pd.DataFrame(
        [
            ("position_date", "date", "Date of open short position check."),
            ("symbol", "string", "Ticker symbol."),
            ("short_market_value_usd", "float", "Absolute short market value."),
            ("previous_easy_to_borrow", "bool", "Prior snapshot ETB flag."),
            ("current_easy_to_borrow", "bool", "Current snapshot ETB flag."),
            ("transition", "string", "unchanged_etb, etb_to_htb, htb_to_etb, unknown."),
            ("action", "string", "hold, block_add, reduce, close, manual_review."),
            ("notes", "string", "Operational notes."),
        ],
        columns=["column", "dtype", "description"],
    )
    return borrow_snapshot, order_audit, transition_monitor


def _plot(curve: pd.DataFrame, metrics: pd.DataFrame, action_summary: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    colors = {
        "etb_baseline_adv1m": "#0f766e",
        "etb_block_top20_recurring": "#c2410c",
        "etb_random_10pct_symbol_day": "#2563eb",
        "etb_random_20pct_symbol_day": "#7c3aed",
    }
    for portfolio, group in curve.groupby("portfolio", sort=False):
        axes[0].plot(
            pd.to_datetime(group["return_date"]),
            group["equity"],
            label=str(group["series"].iloc[0]),
            color=colors.get(portfolio),
        )
    axes[0].set_title("Phase6I Synthetic ETB-Gated Optimizer Replay")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].barh(
        metrics["series"],
        metrics["strict_annualized_return"] * 100,
        color=[colors.get(p, "#6b7280") for p in metrics["portfolio"]],
    )
    axes[1].set_title("Gross Annualized Return by ETB Gate Scenario")
    axes[1].set_xlabel("Annualized return %")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    if action_summary.empty:
        raise ValueError("No short action summary generated.")


def _write_memo(
    metrics: pd.DataFrame,
    action_summary: pd.DataFrame,
    borrow_snapshot: pd.DataFrame,
    order_audit: pd.DataFrame,
    transition_monitor: pd.DataFrame,
    outpath: Path,
) -> None:
    display_metrics = metrics[
        [
            "series",
            "strict_annualized_return",
            "strict_sharpe_no_rf",
            "strict_max_drawdown",
            "aggregate_turnover_mean",
            "target_turnover_mean",
            "note",
        ]
    ].copy()
    display_actions = action_summary[
        [
            "series",
            "mean_daily_add_short_nav",
            "p95_daily_add_short_nav",
            "max_daily_add_short_nav",
            "mean_etb_gate_orders_per_day",
            "p95_etb_gate_orders_per_day",
            "max_single_add_short_usd_at_500k",
        ]
    ].copy()
    lines = [
        "# Phase6I ETB-Gated Optimizer and Forward Borrow Schema",
        "",
        "Scope: validation-window only. Historical ETB availability is not available, so scenario results below are synthetic ETB-unavailability replays, not real historical HTB backtests.",
        "",
        "## Replay Results",
        "",
        "```text",
        display_metrics.to_string(index=False),
        "```",
        "",
        "## Borrow Gate Workload",
        "",
        "```text",
        display_actions.to_string(index=False),
        "```",
        "",
        "## Forward Borrow Snapshot Schema",
        "",
        "```text",
        borrow_snapshot.to_string(index=False),
        "```",
        "",
        "## Order Borrow Audit Schema",
        "",
        "```text",
        order_audit.to_string(index=False),
        "```",
        "",
        "## Borrow Transition Monitor Schema",
        "",
        "```text",
        transition_monitor.to_string(index=False),
        "```",
        "",
        "## Production Policy",
        "",
        "- New or increased short exposure must pass `shortable == true` and `easy_to_borrow == true` at order time.",
        "- If the gate fails, the optimizer should recompute without that symbol; do not submit and hope the broker accepts it.",
        "- Existing shorts that become non-ETB should not be increased. They should enter a monitor path: hold, reduce, close, or manual review depending on broker status and risk.",
        "- The research backtest cannot prove historical ETB availability. The paper/live system must collect point-in-time Alpaca borrow snapshots and order-level gate outcomes.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    strict_curve_path = STRICT_ROOT / "phase4z_strict_h10_daily_curve.csv"
    if POSITIONS_PATH.exists() and strict_curve_path.exists():
        positions = pd.read_csv(POSITIONS_PATH, parse_dates=["session_date"])
    else:
        positions, diagnostics, skipped = _build_positions()
        positions.to_csv(POSITIONS_PATH, index=False, compression="gzip")
        diagnostics.to_csv(OUTDIR / "phase6i_daily_validation.csv", index=False)
        skipped.to_csv(OUTDIR / "phase6i_skipped_validation.csv", index=False)
        build_phase4z_strict_horizon_daily_paths(
            positions_path=POSITIONS_PATH,
            output_root=STRICT_ROOT,
            phase3_rollup_path=DEFAULT_PHASE3_H10_ROLLUP,
            existing_curve_path=None,
            holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
            lead_portfolio="etb_baseline_adv1m",
        )

    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])
    metrics = _build_metrics(strict_curve, positions)
    curve = _build_curve_export(strict_curve)
    action_summary, action_audit = _build_short_action_audit(strict_curve, positions)
    borrow_snapshot, order_audit_schema, transition_monitor = _schema_tables()

    metrics.to_csv(OUTDIR / "phase6i_etb_gate_metrics.csv", index=False)
    curve.to_csv(OUTDIR / "phase6i_etb_gate_curve.csv", index=False)
    action_summary.to_csv(OUTDIR / "phase6i_short_action_summary.csv", index=False)
    action_audit.to_csv(OUTDIR / "phase6i_short_action_audit.csv", index=False)
    borrow_snapshot.to_csv(OUTDIR / "phase6i_borrow_snapshot_schema.csv", index=False)
    order_audit_schema.to_csv(OUTDIR / "phase6i_order_borrow_audit_schema.csv", index=False)
    transition_monitor.to_csv(OUTDIR / "phase6i_borrow_transition_monitor_schema.csv", index=False)
    _plot(curve, metrics, action_summary, OUTDIR / "phase6i_etb_gate_plot.png")
    _write_memo(
        metrics,
        action_summary,
        borrow_snapshot,
        order_audit_schema,
        transition_monitor,
        OUTDIR / "phase6i_etb_gate_memo.md",
    )

    print(f"Wrote Phase6I ETB-gated optimizer artifacts to {OUTDIR}")
    print(metrics.to_string(index=False))
    print(action_summary.to_string(index=False))


if __name__ == "__main__":
    main()
