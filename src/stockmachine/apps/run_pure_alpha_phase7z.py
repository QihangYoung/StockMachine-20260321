"""Phase7Z robustness packet for the frozen Phase7Y pure-alpha candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like
from stockmachine.research.p1_rigor import summarize_backtest_records
from stockmachine.research.robustness_analyzers import (
    build_tail_dependence_summary,
    build_time_stability_summary,
)
from stockmachine.research.robustness_frameworks import resolve_robustness_framework


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7z_frozen_candidate_robustness_20260518"
DEFAULT_PHASE7Y_ROOT = RESEARCH_ROOT / "phase7y_freeze_and_audit_20260518"
DEFAULT_PHASE7K_ROOT = RESEARCH_ROOT / "phase7k_locked_turnover_shared_capital_20260518_tb0p15"
DEFAULT_ASSET_QA_PATH = (
    RESEARCH_ROOT / "phase0_asset_class_qa_20260417" / "top1000_asset_class_qa_by_symbol.csv"
)

DEFAULT_PORTFOLIO = "shared_core_lambda_0p005_tb0p15"
DEFAULT_MODEL_NAME = "phase7y_frozen_candidate"
BASE_STOCK_COST_BPS = 4.0
BASE_GLD_COST_BPS = 1.0
TRADING_DAYS = 252.0

STOCK_INTERNAL_COST_LEVELS_BPS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 10.0)
STOCK_SCALING_COST_LEVELS_BPS: tuple[float, ...] = (0.0, 4.0, 8.0)
GLD_TOTAL_COST_LEVELS_BPS: tuple[float, ...] = (1.0, 2.0, 4.0)
BORROW_COST_LEVELS_BPS_ANNUAL: tuple[float, ...] = (0.0, 50.0, 200.0, 500.0)
MARGIN_COST_LEVELS_BPS_ANNUAL: tuple[float, ...] = (0.0, 300.0, 600.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase7Z robustness packet for the frozen candidate.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase7y-root", default=str(DEFAULT_PHASE7Y_ROOT))
    parser.add_argument("--phase7k-root", default=str(DEFAULT_PHASE7K_ROOT))
    parser.add_argument("--asset-qa-path", default=str(DEFAULT_ASSET_QA_PATH))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args(argv)

    rollup = build_phase7z_frozen_candidate_robustness(
        output_root=Path(args.output_root),
        phase7y_root=Path(args.phase7y_root),
        phase7k_root=Path(args.phase7k_root),
        asset_qa_path=Path(args.asset_qa_path),
        portfolio=str(args.portfolio),
        model_name=str(args.model_name),
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=False))
    return 0


def build_phase7z_frozen_candidate_robustness(
    *,
    output_root: Path,
    phase7y_root: Path,
    phase7k_root: Path,
    asset_qa_path: Path,
    portfolio: str,
    model_name: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    paths = _input_paths(phase7y_root=phase7y_root, phase7k_root=phase7k_root, asset_qa_path=asset_qa_path)
    frozen_config = _read_json(paths["phase7y_config"])
    frozen_curve = pd.read_csv(paths["phase7y_curve"], parse_dates=["date"])
    phase7k_curve = _load_phase7k_curve(paths["phase7k_curve"], portfolio=portfolio)
    diagnostics = _load_phase7k_diagnostics(paths["phase7k_diagnostics"], portfolio=portfolio)
    positions = _load_phase7k_positions(paths["phase7k_positions"], portfolio=portfolio)

    records = _build_records(frozen_curve=frozen_curve, phase7k_curve=phase7k_curve)
    records_path = output_root / "phase7z_frozen_candidate_records.csv"
    records.to_csv(records_path, index=False)

    scenario_summary = _build_named_scenario_summary(records)
    execution_stress = _build_execution_cost_stress(records)
    financing_stress = _build_financing_borrow_stress(records)
    scenario_path = output_root / "phase7z_scenario_summary.csv"
    execution_path = output_root / "phase7z_execution_cost_stress.csv"
    financing_path = output_root / "phase7z_financing_borrow_stress.csv"
    scenario_summary.to_csv(scenario_path, index=False)
    execution_stress.to_csv(execution_path, index=False)
    financing_stress.to_csv(financing_path, index=False)

    audit_records = _records_for_return_column(
        records,
        return_column="audit_net_return",
        model_name=f"{model_name}_scaling4",
    )
    framework = resolve_robustness_framework(strategy_project="us_equities_pure_alpha_h5", horizon=5)
    yearly = build_time_stability_summary(
        audit_records,
        model_name=f"{model_name}_scaling4",
        horizon=1,
        framework=framework,
        period="year",
    )
    quarterly = build_time_stability_summary(
        audit_records,
        model_name=f"{model_name}_scaling4",
        horizon=1,
        framework=framework,
        period="quarter",
    )
    tail = build_tail_dependence_summary(
        audit_records,
        model_name=f"{model_name}_scaling4",
        horizon=1,
        framework=framework,
    )
    market_state = _build_market_state_exposure(audit_records)
    rolling_beta = _build_rolling_beta(audit_records, windows=(60, 126, 252))
    rolling_beta_summary = _build_rolling_beta_summary(rolling_beta)
    concentration = _build_position_concentration_summary(
        positions=positions,
        phase7k_curve=phase7k_curve,
        records=records,
    )
    sector = _build_sector_exposure_summary(
        positions=positions,
        phase7k_curve=phase7k_curve,
        records=records,
    )
    leg_attribution = _build_phase7k_leg_proxy(diagnostics)
    borrow_proxy = _build_borrow_proxy_coverage(positions=positions, asset_qa_path=asset_qa_path)
    gate = _build_gate_check(
        audit_records=audit_records,
        yearly=yearly,
        tail=tail,
        scenario_summary=scenario_summary,
        execution_stress=execution_stress,
        financing_stress=financing_stress,
        market_state=market_state,
        rolling_beta_summary=rolling_beta_summary,
        concentration=concentration,
        sector=sector,
        leg_attribution=leg_attribution,
        borrow_proxy=borrow_proxy,
    )

    yearly_path = output_root / "phase7z_time_stability_yearly.csv"
    quarterly_path = output_root / "phase7z_time_stability_quarterly.csv"
    tail_path = output_root / "phase7z_tail_dependence.csv"
    market_path = output_root / "phase7z_market_state_exposure.csv"
    rolling_beta_path = output_root / "phase7z_rolling_realized_beta.csv"
    rolling_beta_summary_path = output_root / "phase7z_rolling_realized_beta_summary.csv"
    concentration_path = output_root / "phase7z_position_concentration.csv"
    sector_path = output_root / "phase7z_sector_exposure.csv"
    leg_path = output_root / "phase7z_leg_proxy_attribution.csv"
    borrow_path = output_root / "phase7z_borrow_proxy_coverage.csv"
    gate_path = output_root / "phase7z_gate_check.csv"
    plot_path = output_root / "phase7z_robustness_plot.png"
    memo_path = output_root / "phase7z_robustness_memo_zh.md"
    rollup_path = output_root / "phase7z_rollup.json"

    yearly.to_csv(yearly_path, index=False)
    quarterly.to_csv(quarterly_path, index=False)
    tail.to_csv(tail_path, index=False)
    market_state.to_csv(market_path, index=False)
    rolling_beta.to_csv(rolling_beta_path, index=False)
    rolling_beta_summary.to_csv(rolling_beta_summary_path, index=False)
    concentration.to_csv(concentration_path, index=False)
    sector.to_csv(sector_path, index=False)
    leg_attribution.to_csv(leg_path, index=False)
    borrow_proxy.to_csv(borrow_path, index=False)
    gate.to_csv(gate_path, index=False)
    _plot_robustness(records=records, rolling_beta=rolling_beta, output_path=plot_path)
    memo_path.write_text(
        _memo(
            frozen_config=frozen_config,
            scenario_summary=scenario_summary,
            execution_stress=execution_stress,
            financing_stress=financing_stress,
            yearly=yearly,
            tail=tail,
            market_state=market_state,
            rolling_beta_summary=rolling_beta_summary,
            concentration=concentration,
            sector=sector,
            leg_attribution=leg_attribution,
            borrow_proxy=borrow_proxy,
            gate=gate,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup: dict[str, Any] = {
        "generated_at": _utc_now(),
        "phase": "phase7z_frozen_candidate_robustness",
        "scope": "validation_only_no_test_read",
        "portfolio": portfolio,
        "model_name": model_name,
        "primary_audit_return": "audit_net_return = reported frozen return less stock-book scaling cost at 4 bps/traded notional",
        "base_cost_assumptions": {
            "stock_internal_cost_bps_per_traded_notional": BASE_STOCK_COST_BPS,
            "gld_overlay_cost_bps_per_traded_notional_already_in_phase7w": BASE_GLD_COST_BPS,
            "new_stock_scaling_cost_bps_per_traded_notional": 4.0,
            "stock_scaling_turnover_formula": "2 * abs(delta_insurance_overlay_weight)",
        },
        "output_root": str(output_root),
        "gate_status_counts": gate["status"].value_counts().to_dict(),
        "go_no_go": _go_no_go(gate),
        "paths": {
            "records": str(records_path),
            "scenario_summary": str(scenario_path),
            "execution_cost_stress": str(execution_path),
            "financing_borrow_stress": str(financing_path),
            "yearly": str(yearly_path),
            "quarterly": str(quarterly_path),
            "tail": str(tail_path),
            "market_state": str(market_path),
            "rolling_beta": str(rolling_beta_path),
            "rolling_beta_summary": str(rolling_beta_summary_path),
            "position_concentration": str(concentration_path),
            "sector_exposure": str(sector_path),
            "leg_proxy_attribution": str(leg_path),
            "borrow_proxy": str(borrow_path),
            "gate_check": str(gate_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
            "rollup": str(rollup_path),
        },
        "inputs": {key: str(path) for key, path in paths.items()},
        "limitations": [
            "Phase7Z does not open the future test lockbox.",
            "Stock-book scaling friction is a deterministic scenario haircut; it was not modeled inside Phase7W.",
            "Borrow and margin stresses are scenario haircuts, not point-in-time broker charges.",
            "Long/short leg attribution is only a diagnostic proxy because Phase7K did not persist separate leg PnL for the frozen shared-core path.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=False), encoding="utf-8")
    return rollup


def _input_paths(*, phase7y_root: Path, phase7k_root: Path, asset_qa_path: Path) -> dict[str, Path]:
    return {
        "phase7y_config": phase7y_root / "phase7y_frozen_candidate_config.json",
        "phase7y_curve": phase7y_root / "phase7y_frozen_candidate_daily_curve.csv",
        "phase7k_curve": phase7k_root / "phase7k_strict_daily_curve.csv",
        "phase7k_diagnostics": phase7k_root / "phase7k_daily_diagnostics.csv",
        "phase7k_positions": phase7k_root / "phase7k_locked_positions.csv.gz",
        "asset_qa": asset_qa_path,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_phase7k_curve(path: Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["signal_date", "return_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No Phase7K curve rows found for portfolio {portfolio!r}.")
    if "test_window_used" in frame and frame["test_window_used"].map(_is_true).any():
        raise ValueError("Phase7K curve contains test-window rows; refusing Phase7Z run.")
    return frame.sort_values("return_date").reset_index(drop=True)


def _load_phase7k_diagnostics(path: Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if "test_window_used" in frame and frame["test_window_used"].map(_is_true).any():
        raise ValueError("Phase7K diagnostics contain test-window rows; refusing Phase7Z run.")
    return frame.sort_values("session_date").reset_index(drop=True)


def _load_phase7k_positions(path: Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No Phase7K position rows found for portfolio {portfolio!r}.")
    if "test_window_used" in frame and frame["test_window_used"].map(_is_true).any():
        raise ValueError("Phase7K positions contain test-window rows; refusing Phase7Z run.")
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _build_records(*, frozen_curve: pd.DataFrame, phase7k_curve: pd.DataFrame) -> pd.DataFrame:
    required_frozen = {
        "date",
        "shared_core_net_return",
        "frozen_candidate_net_return",
        "risk_state",
        "risk_score",
        "insurance_overlay_weight",
    }
    required_k = {
        "signal_date",
        "return_date",
        "gross_return",
        "net_return",
        "cost_return",
        "cost_bps",
        "turnover",
        "positions",
        "gross_exposure",
        "net_exposure",
        "benchmark_oto_return",
    }
    missing_frozen = required_frozen.difference(frozen_curve.columns)
    missing_k = required_k.difference(phase7k_curve.columns)
    if missing_frozen:
        raise KeyError(f"Phase7Y curve missing columns: {sorted(missing_frozen)}")
    if missing_k:
        raise KeyError(f"Phase7K curve missing columns: {sorted(missing_k)}")

    frozen = frozen_curve.copy()
    frozen["date"] = pd.to_datetime(frozen["date"])
    merged = phase7k_curve.merge(frozen, left_on="return_date", right_on="date", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No common dates between Phase7K strict curve and Phase7Y frozen curve.")

    overlay_weight = merged["insurance_overlay_weight"].astype(float).fillna(0.0).clip(lower=0.0, upper=1.0)
    overlay_delta = overlay_weight.diff().abs().fillna(overlay_weight.abs())
    stock_scaling_turnover = 2.0 * overlay_delta
    overlay_turnover = overlay_delta
    stock_scaling_cost_4bps = stock_scaling_turnover * 4.0 / 10000.0

    reported = merged["frozen_candidate_net_return"].astype(float)
    audit = reported - stock_scaling_cost_4bps

    out = pd.DataFrame(
        {
            "entry_date": merged["return_date"],
            "exit_date": merged["return_date"],
            "signal_date": merged["signal_date"],
            "risk_state": merged["risk_state"].astype(str),
            "risk_score": merged["risk_score"].astype(float),
            "reported_net_return": reported,
            "audit_net_return": audit,
            "shared_core_net_return": merged["shared_core_net_return"].astype(float),
            "benchmark_return": merged["benchmark_oto_return"].astype(float),
            "stock_internal_gross_return": merged["gross_return"].astype(float),
            "stock_internal_net_return": merged["net_return"].astype(float),
            "stock_internal_turnover": merged["turnover"].astype(float).fillna(0.0),
            "stock_internal_cost_bps": merged["cost_bps"].astype(float).fillna(0.0),
            "stock_internal_cost_return": merged["cost_return"].astype(float).fillna(0.0),
            "insurance_overlay_weight": overlay_weight,
            "insurance_overlay_weight_delta": overlay_delta,
            "stock_scaling_turnover": stock_scaling_turnover,
            "overlay_turnover": overlay_turnover,
            "stock_scaling_cost_4bps_return": stock_scaling_cost_4bps,
            "gld_base_cost_bps": overlay_turnover * BASE_GLD_COST_BPS,
            "gross_exposure": merged["gross_exposure"].astype(float).fillna(0.0) * (1.0 - overlay_weight)
            + overlay_weight,
            "stock_gross_exposure_after_overlay": merged["gross_exposure"].astype(float).fillna(0.0)
            * (1.0 - overlay_weight),
            "long_gross": 1.0 - overlay_weight,
            "short_gross": 1.0 - overlay_weight,
            "gross_above_nav": (1.0 - overlay_weight).clip(lower=0.0),
            "net_exposure": merged["net_exposure"].astype(float).fillna(0.0) * (1.0 - overlay_weight)
            + overlay_weight,
            "positions": merged["positions"].fillna(0).astype(int),
            "test_window_used": False,
        }
    )
    out["gross_return"] = out["reported_net_return"]
    out["net_return"] = out["audit_net_return"]
    out["turnover"] = out["stock_internal_turnover"] + out["stock_scaling_turnover"] + out["overlay_turnover"]
    out["cost_bps"] = (
        out["stock_internal_cost_bps"]
        + out["stock_scaling_turnover"] * 4.0
        + out["overlay_turnover"] * BASE_GLD_COST_BPS
    )
    out["reported_equity"] = (1.0 + out["reported_net_return"]).cumprod()
    out["audit_equity"] = (1.0 + out["audit_net_return"]).cumprod()
    out["spy_equity"] = (1.0 + out["benchmark_return"]).cumprod()
    out["reported_drawdown"] = out["reported_equity"] / out["reported_equity"].cummax() - 1.0
    out["audit_drawdown"] = out["audit_equity"] / out["audit_equity"].cummax() - 1.0
    return out.sort_values("entry_date").reset_index(drop=True)


def _records_for_return_column(records: pd.DataFrame, *, return_column: str, model_name: str) -> pd.DataFrame:
    out = records.copy()
    out["net_return"] = out[return_column].astype(float)
    out["gross_return"] = out[return_column].astype(float)
    out["model"] = model_name
    return out


def _build_named_scenario_summary(records: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        ("reported_frozen_base", 4.0, 0.0, 1.0, 0.0, 0.0),
        ("audit_base_stock_scaling_4bps", 4.0, 4.0, 1.0, 0.0, 0.0),
        ("audit_gld_total_2bps", 4.0, 4.0, 2.0, 0.0, 0.0),
        ("harsh_execution_stock10_scaling8_gld4", 10.0, 8.0, 4.0, 0.0, 0.0),
        ("borrow200_scaling4", 4.0, 4.0, 1.0, 200.0, 0.0),
        ("borrow500_margin300_harsh_execution", 10.0, 8.0, 4.0, 500.0, 300.0),
    ]
    rows: list[dict[str, Any]] = []
    for name, stock_bps, scaling_bps, gld_bps, borrow_bps, margin_bps in scenarios:
        returns, cost_cols = _stress_returns(
            records,
            stock_internal_cost_bps=stock_bps,
            stock_scaling_cost_bps=scaling_bps,
            gld_total_cost_bps=gld_bps,
            borrow_cost_bps_annual=borrow_bps,
            margin_cost_bps_annual=margin_bps,
        )
        row = _daily_path_metrics(returns, benchmark=records["benchmark_return"])
        row.update(
            {
                "scenario": name,
                "stock_internal_cost_bps_per_traded_notional": stock_bps,
                "stock_scaling_cost_bps_per_traded_notional": scaling_bps,
                "gld_total_cost_bps_per_traded_notional": gld_bps,
                "borrow_cost_bps_annual_on_short_gross": borrow_bps,
                "margin_cost_bps_annual_on_gross_above_100pct_nav": margin_bps,
                **cost_cols,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_execution_cost_stress(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stock_bps in STOCK_INTERNAL_COST_LEVELS_BPS:
        for scaling_bps in STOCK_SCALING_COST_LEVELS_BPS:
            for gld_bps in GLD_TOTAL_COST_LEVELS_BPS:
                returns, cost_cols = _stress_returns(
                    records,
                    stock_internal_cost_bps=stock_bps,
                    stock_scaling_cost_bps=scaling_bps,
                    gld_total_cost_bps=gld_bps,
                    borrow_cost_bps_annual=0.0,
                    margin_cost_bps_annual=0.0,
                )
                metrics = _daily_path_metrics(returns, benchmark=records["benchmark_return"])
                rows.append(
                    {
                        "stock_internal_cost_bps_per_traded_notional": float(stock_bps),
                        "stock_scaling_cost_bps_per_traded_notional": float(scaling_bps),
                        "gld_total_cost_bps_per_traded_notional": float(gld_bps),
                        **cost_cols,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "stock_internal_cost_bps_per_traded_notional",
            "stock_scaling_cost_bps_per_traded_notional",
            "gld_total_cost_bps_per_traded_notional",
        ]
    )


def _build_financing_borrow_stress(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for borrow_bps in BORROW_COST_LEVELS_BPS_ANNUAL:
        for margin_bps in MARGIN_COST_LEVELS_BPS_ANNUAL:
            returns, cost_cols = _stress_returns(
                records,
                stock_internal_cost_bps=BASE_STOCK_COST_BPS,
                stock_scaling_cost_bps=4.0,
                gld_total_cost_bps=BASE_GLD_COST_BPS,
                borrow_cost_bps_annual=borrow_bps,
                margin_cost_bps_annual=margin_bps,
            )
            metrics = _daily_path_metrics(returns, benchmark=records["benchmark_return"])
            rows.append(
                {
                    "stock_internal_cost_bps_per_traded_notional": BASE_STOCK_COST_BPS,
                    "stock_scaling_cost_bps_per_traded_notional": 4.0,
                    "gld_total_cost_bps_per_traded_notional": BASE_GLD_COST_BPS,
                    "borrow_cost_bps_annual_on_short_gross": float(borrow_bps),
                    "margin_cost_bps_annual_on_gross_above_100pct_nav": float(margin_bps),
                    **cost_cols,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(
        [
            "borrow_cost_bps_annual_on_short_gross",
            "margin_cost_bps_annual_on_gross_above_100pct_nav",
        ]
    )


def _stress_returns(
    records: pd.DataFrame,
    *,
    stock_internal_cost_bps: float,
    stock_scaling_cost_bps: float,
    gld_total_cost_bps: float,
    borrow_cost_bps_annual: float,
    margin_cost_bps_annual: float,
) -> tuple[pd.Series, dict[str, float]]:
    reported = records["reported_net_return"].astype(float)
    stock_turnover = records["stock_internal_turnover"].astype(float).fillna(0.0)
    scale_turnover = records["stock_scaling_turnover"].astype(float).fillna(0.0)
    overlay_turnover = records["overlay_turnover"].astype(float).fillna(0.0)
    short_gross = records["short_gross"].astype(float).fillna(0.0)
    gross_above_nav = records["gross_above_nav"].astype(float).fillna(0.0)

    stock_internal_delta = stock_turnover * (float(stock_internal_cost_bps) - BASE_STOCK_COST_BPS) / 10000.0
    stock_scaling_cost = scale_turnover * float(stock_scaling_cost_bps) / 10000.0
    gld_cost_delta = overlay_turnover * (float(gld_total_cost_bps) - BASE_GLD_COST_BPS) / 10000.0
    borrow_cost = short_gross * float(borrow_cost_bps_annual) / 10000.0 / TRADING_DAYS
    margin_cost = gross_above_nav * float(margin_cost_bps_annual) / 10000.0 / TRADING_DAYS
    returns = reported - stock_internal_delta - stock_scaling_cost - gld_cost_delta - borrow_cost - margin_cost

    costs = {
        "mean_stock_internal_delta_cost_bps": float(stock_internal_delta.mean() * 10000.0),
        "mean_stock_scaling_cost_bps": float(stock_scaling_cost.mean() * 10000.0),
        "mean_gld_extra_cost_bps": float(gld_cost_delta.mean() * 10000.0),
        "mean_borrow_cost_bps": float(borrow_cost.mean() * 10000.0),
        "mean_margin_cost_bps": float(margin_cost.mean() * 10000.0),
        "mean_total_incremental_cost_bps": float(
            (stock_internal_delta + stock_scaling_cost + gld_cost_delta + borrow_cost + margin_cost).mean()
            * 10000.0
        ),
    }
    return returns, costs


def _build_market_state_exposure(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame["strategy_return"] = frame["net_return"].astype(float)
    frame["spy_return"] = frame["benchmark_return"].astype(float)
    bottom_decile = float(frame["spy_return"].quantile(0.10))
    top_decile = float(frame["spy_return"].quantile(0.90))
    abs_top_decile = float(frame["spy_return"].abs().quantile(0.90))
    states = [
        ("all", pd.Series(True, index=frame.index)),
        ("spy_up", frame["spy_return"] > 0),
        ("spy_down", frame["spy_return"] < 0),
        ("spy_top_decile", frame["spy_return"] >= top_decile),
        ("spy_bottom_decile", frame["spy_return"] <= bottom_decile),
        ("spy_abs_move_top_decile", frame["spy_return"].abs() >= abs_top_decile),
        ("insurance_calm", frame["risk_state"].eq("calm")),
        ("insurance_watch", frame["risk_state"].eq("watch")),
        ("insurance_stress", frame["risk_state"].eq("stress")),
    ]
    rows: list[dict[str, Any]] = []
    for state, mask in states:
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        metrics = _daily_path_metrics(sub["strategy_return"], benchmark=sub["spy_return"])
        rows.append(
            {
                "state": state,
                "days": int(len(sub)),
                "strategy_mean_bps": float(sub["strategy_return"].mean() * 10000.0),
                "strategy_hit_rate": float((sub["strategy_return"] > 0).mean()),
                "spy_mean_bps": float(sub["spy_return"].mean() * 10000.0),
                "mean_overlay_weight": float(sub["insurance_overlay_weight"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _build_rolling_beta(records: pd.DataFrame, *, windows: Sequence[int]) -> pd.DataFrame:
    frame = records[["entry_date", "net_return", "benchmark_return"]].copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["net_return"] = frame["net_return"].astype(float)
    frame["benchmark_return"] = frame["benchmark_return"].astype(float)
    rows: list[pd.DataFrame] = []
    for window in windows:
        cov = frame["net_return"].rolling(window).cov(frame["benchmark_return"])
        var = frame["benchmark_return"].rolling(window).var()
        corr = frame["net_return"].rolling(window).corr(frame["benchmark_return"])
        rows.append(
            pd.DataFrame(
                {
                    "entry_date": frame["entry_date"],
                    "window": int(window),
                    "rolling_beta": cov / var,
                    "rolling_corr": corr,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _build_rolling_beta_summary(rolling_beta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, group in rolling_beta.dropna(subset=["rolling_beta"]).groupby("window", sort=True):
        rows.append(
            {
                "window": int(window),
                "observations": int(len(group)),
                "mean_abs_rolling_beta": float(group["rolling_beta"].abs().mean()),
                "median_abs_rolling_beta": float(group["rolling_beta"].abs().median()),
                "p90_abs_rolling_beta": float(group["rolling_beta"].abs().quantile(0.90)),
                "max_abs_rolling_beta": float(group["rolling_beta"].abs().max()),
                "mean_abs_rolling_corr": float(group["rolling_corr"].abs().mean()),
                "p90_abs_rolling_corr": float(group["rolling_corr"].abs().quantile(0.90)),
                "max_abs_rolling_corr": float(group["rolling_corr"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _build_position_concentration_summary(
    *,
    positions: pd.DataFrame,
    phase7k_curve: pd.DataFrame,
    records: pd.DataFrame,
) -> pd.DataFrame:
    weights = _positions_with_overlay(positions=positions, phase7k_curve=phase7k_curve, records=records)
    rows: list[dict[str, Any]] = []
    for session_date, group in weights.groupby("return_date", sort=True):
        abs_weights = group["scaled_weight"].abs().sort_values(ascending=False)
        gross = float(abs_weights.sum())
        long_weights = group.loc[group["scaled_weight"] > 0, "scaled_weight"]
        short_weights = group.loc[group["scaled_weight"] < 0, "scaled_weight"]
        rows.append(
            {
                "return_date": session_date,
                "gross_exposure": gross,
                "long_gross": float(long_weights.sum()),
                "short_gross": float(-short_weights.sum()),
                "net_exposure": float(group["scaled_weight"].sum()),
                "positions": int((abs_weights > 1e-12).sum()),
                "long_positions": int((long_weights.abs() > 1e-12).sum()),
                "short_positions": int((short_weights.abs() > 1e-12).sum()),
                "max_abs_weight": float(abs_weights.max()) if not abs_weights.empty else np.nan,
                "top5_abs_share": _top_share(abs_weights, 5, gross),
                "top10_abs_share": _top_share(abs_weights, 10, gross),
                "hhi_abs_weight": float(((abs_weights / gross) ** 2).sum()) if gross > 0 else np.nan,
            }
        )
    daily = pd.DataFrame(rows)
    summary_rows = []
    for column in [
        "gross_exposure",
        "long_gross",
        "short_gross",
        "net_exposure",
        "positions",
        "long_positions",
        "short_positions",
        "max_abs_weight",
        "top5_abs_share",
        "top10_abs_share",
        "hhi_abs_weight",
    ]:
        series = daily[column].astype(float)
        summary_rows.append(
            {
                "scope": "summary",
                "metric": column,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p90": float(series.quantile(0.90)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(summary_rows)


def _build_sector_exposure_summary(
    *,
    positions: pd.DataFrame,
    phase7k_curve: pd.DataFrame,
    records: pd.DataFrame,
) -> pd.DataFrame:
    weights = _positions_with_overlay(positions=positions, phase7k_curve=phase7k_curve, records=records)
    sector = (
        weights.groupby(["return_date", "sic2_sector"], as_index=False)["scaled_weight"]
        .sum()
        .rename(columns={"scaled_weight": "net_sector_weight"})
    )
    rows: list[dict[str, Any]] = []
    for return_date, group in sector.groupby("return_date", sort=True):
        abs_exposure = group["net_sector_weight"].abs()
        idx = abs_exposure.idxmax()
        rows.append(
            {
                "return_date": return_date,
                "sic2_l1_exposure": float(abs_exposure.sum()),
                "sic2_max_abs_exposure": float(abs_exposure.max()),
                "sic2_worst_sector": str(group.loc[idx, "sic2_sector"]),
                "sic2_worst_sector_weight": float(group.loc[idx, "net_sector_weight"]),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    summary = {
        "return_date": "SUMMARY",
        "sic2_l1_exposure": float(daily["sic2_l1_exposure"].mean()),
        "sic2_max_abs_exposure": float(daily["sic2_max_abs_exposure"].mean()),
        "sic2_worst_sector": str(daily.sort_values("sic2_max_abs_exposure", ascending=False).iloc[0]["sic2_worst_sector"]),
        "sic2_worst_sector_weight": float(
            daily.sort_values("sic2_max_abs_exposure", ascending=False).iloc[0]["sic2_worst_sector_weight"]
        ),
    }
    return pd.concat([pd.DataFrame([summary]), daily], ignore_index=True)


def _positions_with_overlay(
    *,
    positions: pd.DataFrame,
    phase7k_curve: pd.DataFrame,
    records: pd.DataFrame,
) -> pd.DataFrame:
    date_map = phase7k_curve[["signal_date", "return_date"]].copy()
    overlay = records[["entry_date", "insurance_overlay_weight"]].rename(columns={"entry_date": "return_date"})
    merged = positions.merge(date_map, left_on="session_date", right_on="signal_date", how="inner")
    merged = merged.merge(overlay, on="return_date", how="left")
    merged["insurance_overlay_weight"] = merged["insurance_overlay_weight"].astype(float).fillna(0.0)
    merged["scaled_weight"] = merged["signed_weight"].astype(float) * (1.0 - merged["insurance_overlay_weight"])
    merged["sic2_sector"] = merged["sic2_sector"].fillna("UNKNOWN").astype(str)
    return merged


def _build_phase7k_leg_proxy(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ("gross_payoff_h5_bps", "net_payoff_h5_bps", "gross_payoff_h10_bps", "net_payoff_h10_bps", "gross_payoff_h20_bps", "net_payoff_h20_bps"):
        if column in diagnostics:
            series = pd.to_numeric(diagnostics[column], errors="coerce").dropna()
            rows.append(
                {
                    "source": "phase7k_signal_date_forward_payoff_proxy",
                    "leg": column,
                    "mean_bps": float(series.mean()) if not series.empty else np.nan,
                    "hit_rate": float((series > 0.0).mean()) if not series.empty else np.nan,
                    "note": "Spread-level forward payoff proxy; separate long/short realized PnL was not persisted.",
                }
            )
    for column in ("long_composite_score", "short_composite_score", "score_spread", "long_beta", "short_beta", "net_beta"):
        if column in diagnostics:
            series = pd.to_numeric(diagnostics[column], errors="coerce").dropna()
            rows.append(
                {
                    "source": "phase7k_construction_proxy",
                    "leg": column,
                    "mean_bps": float(series.mean()) if not series.empty else np.nan,
                    "hit_rate": float((series > 0.0).mean()) if not series.empty else np.nan,
                    "note": "Construction diagnostic, not a realized leg return.",
                }
            )
    return pd.DataFrame(rows)


def _build_borrow_proxy_coverage(*, positions: pd.DataFrame, asset_qa_path: Path) -> pd.DataFrame:
    short_positions = positions[positions["signed_weight"].astype(float) < -1e-12].copy()
    if not asset_qa_path.exists():
        return pd.DataFrame(
            [
                {
                    "scope": "target_short_rows",
                    "rows": int(len(short_positions)),
                    "unique_symbols": int(short_positions["symbol"].nunique()),
                    "current_shortable_coverage": np.nan,
                    "current_easy_to_borrow_coverage": np.nan,
                    "missing_asset_qa_symbols": int(short_positions["symbol"].nunique()),
                    "data_status": "asset_qa_missing",
                }
            ]
        )
    qa = pd.read_csv(asset_qa_path).drop_duplicates("symbol")
    merged = short_positions.merge(qa[["symbol", "shortable", "easy_to_borrow"]], on="symbol", how="left")
    symbol_coverage = merged.drop_duplicates("symbol")
    return pd.DataFrame(
        [
            {
                "scope": "target_short_rows",
                "rows": int(len(short_positions)),
                "unique_symbols": int(short_positions["symbol"].nunique()),
                "current_shortable_coverage": float(merged["shortable"].map(_is_true).mean()),
                "current_easy_to_borrow_coverage": float(merged["easy_to_borrow"].map(_is_true).mean()),
                "missing_asset_qa_symbols": int(merged["shortable"].isna().groupby(merged["symbol"]).max().sum()),
                "data_status": "current_metadata_proxy_not_point_in_time",
            },
            {
                "scope": "unique_short_symbols",
                "rows": int(len(symbol_coverage)),
                "unique_symbols": int(symbol_coverage["symbol"].nunique()),
                "current_shortable_coverage": float(symbol_coverage["shortable"].map(_is_true).mean()),
                "current_easy_to_borrow_coverage": float(symbol_coverage["easy_to_borrow"].map(_is_true).mean()),
                "missing_asset_qa_symbols": int(symbol_coverage["shortable"].isna().sum()),
                "data_status": "current_metadata_proxy_not_point_in_time",
            },
        ]
    )


def _build_gate_check(
    *,
    audit_records: pd.DataFrame,
    yearly: pd.DataFrame,
    tail: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    execution_stress: pd.DataFrame,
    financing_stress: pd.DataFrame,
    market_state: pd.DataFrame,
    rolling_beta_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    sector: pd.DataFrame,
    leg_attribution: pd.DataFrame,
    borrow_proxy: pd.DataFrame,
) -> pd.DataFrame:
    summary = summarize_backtest_records(audit_records, horizon=1)
    path_metrics = _daily_path_metrics(audit_records["net_return"], benchmark=audit_records["benchmark_return"])
    yearly_positive_rate = float((yearly["total_return"].astype(float) > 0.0).mean()) if not yearly.empty else np.nan
    top5 = tail[(tail["tail_side"].eq("top")) & (tail["trim_count"].eq(5))]
    top5_share = float(top5["removed_return_share_of_abs_sum"].iloc[0]) if not top5.empty else np.nan
    harsh = scenario_summary[scenario_summary["scenario"].eq("harsh_execution_stock10_scaling8_gld4")]
    harsh_ann = float(harsh["annualized_return"].iloc[0]) if not harsh.empty else np.nan
    borrow200 = financing_stress[
        financing_stress["borrow_cost_bps_annual_on_short_gross"].astype(float).eq(200.0)
        & financing_stress["margin_cost_bps_annual_on_gross_above_100pct_nav"].astype(float).eq(0.0)
    ]
    borrow200_ann = float(borrow200["annualized_return"].iloc[0]) if not borrow200.empty else np.nan
    full_state = market_state[market_state["state"].eq("all")]
    realized_beta = float(full_state["realized_beta_to_spy"].iloc[0]) if not full_state.empty else path_metrics["realized_beta_to_spy"]
    realized_corr = float(full_state["corr_to_spy"].iloc[0]) if not full_state.empty else path_metrics["corr_to_spy"]
    rb60 = rolling_beta_summary[rolling_beta_summary["window"].eq(60)]
    rb60_p90 = float(rb60["p90_abs_rolling_beta"].iloc[0]) if not rb60.empty else np.nan
    conc_top5 = _summary_metric(concentration, "top5_abs_share", "p90")
    sector_l1 = _summary_sector_metric(sector, "sic2_l1_exposure")
    spread_h10 = leg_attribution[leg_attribution["leg"].eq("net_payoff_h10_bps")]
    spread_h10_mean = float(spread_h10["mean_bps"].iloc[0]) if not spread_h10.empty else np.nan
    borrow_rows = borrow_proxy[borrow_proxy["scope"].eq("target_short_rows")]
    etb = float(borrow_rows["current_easy_to_borrow_coverage"].iloc[0]) if not borrow_rows.empty else np.nan

    rows = [
        _gate("validation_annualized_return_after_scaling_friction", float(summary["annualized_return"]), 0.08, ">=", "fail"),
        _gate("validation_sharpe_after_scaling_friction", float(summary["sharpe"]), 0.80, ">=", "fail"),
        _gate("validation_max_drawdown_after_scaling_friction", float(summary["max_drawdown"]), -0.15, ">=", "fail"),
        _gate("absolute_realized_beta_full_sample", abs(realized_beta), 0.05, "<=", "warn"),
        _gate("absolute_market_correlation_full_sample", abs(realized_corr), 0.10, "<=", "warn"),
        _gate("rolling_60_positive_rate", _rolling_positive_rate(audit_records["net_return"], 60), 0.50, ">=", "fail"),
        _gate("positive_year_ratio", yearly_positive_rate, 0.50, ">=", "fail"),
        _gate("top5_positive_day_abs_share", top5_share, 0.50, "<=", "fail"),
        _gate("harsh_execution_annualized_return", harsh_ann, 0.05, ">=", "fail"),
        _gate("borrow200_annualized_return", borrow200_ann, 0.05, ">=", "warn"),
        _gate("p90_abs_rolling_60_beta", rb60_p90, 0.20, "<=", "warn"),
        _gate("p90_top5_abs_position_share", conc_top5, 0.25, "<=", "warn"),
        _gate("mean_sic2_l1_sector_exposure", sector_l1, 0.50, "<=", "warn"),
        _gate("phase7k_net_payoff_h10_proxy_mean_bps", spread_h10_mean, 0.0, ">=", "warn"),
        _gate("current_etb_proxy_short_row_coverage", etb, 1.0, ">=", "warn"),
        {
            "gate": "long_short_realized_leg_pnl",
            "value": "not_persisted",
            "threshold": "required_before_production_freeze",
            "operator": "exists",
            "status": "WARN",
            "severity": "blocker_for_production_not_for_validation_research",
        },
        {
            "gate": "point_in_time_borrow_history",
            "value": "missing",
            "threshold": "required_before_product_freeze",
            "operator": "exists",
            "status": "WARN",
            "severity": "blocker_for_production_not_for_validation_research",
        },
        {
            "gate": "phase7z_test_lockbox_read",
            "value": "not_read",
            "threshold": "not_read",
            "operator": "==",
            "status": "PASS",
            "severity": "fail",
        },
    ]
    return pd.DataFrame(rows)


def _daily_path_metrics(returns: pd.Series, *, benchmark: pd.Series | None = None) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe_no_rf": np.nan,
            "sharpe_ann_over_vol": np.nan,
            "max_drawdown": np.nan,
            "final_equity": np.nan,
            "corr_to_spy": np.nan,
            "realized_beta_to_spy": np.nan,
        }
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    corr = np.nan
    beta = np.nan
    if benchmark is not None:
        bench = pd.to_numeric(benchmark, errors="coerce").reindex(returns.index)
        aligned = pd.concat([returns.rename("strategy"), bench.rename("benchmark")], axis=1).dropna()
        if len(aligned) > 2 and aligned["benchmark"].var(ddof=1) > 0:
            corr = float(aligned["strategy"].corr(aligned["benchmark"]))
            beta = float(aligned["strategy"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1))
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(equity.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1.0),
        "annualized_vol": vol,
        "sharpe_no_rf": sharpe,
        "sharpe_ann_over_vol": (
            float((equity.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1.0) / vol)
            if pd.notna(vol) and vol > 0.0
            else np.nan
        ),
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(equity.iloc[-1]),
        "corr_to_spy": corr,
        "realized_beta_to_spy": beta,
    }


def _rolling_positive_rate(returns: pd.Series, window: int) -> float:
    rolling = (1.0 + returns.astype(float)).rolling(window).apply(np.prod, raw=True) - 1.0
    valid = rolling.dropna()
    return float((valid > 0.0).mean()) if not valid.empty else np.nan


def _top_share(abs_weights: pd.Series, count: int, gross: float) -> float:
    if gross <= 0:
        return np.nan
    return float(abs_weights.head(count).sum() / gross)


def _summary_metric(frame: pd.DataFrame, metric: str, column: str) -> float:
    row = frame[frame["metric"].eq(metric)]
    return float(row[column].iloc[0]) if not row.empty else np.nan


def _summary_sector_metric(frame: pd.DataFrame, column: str) -> float:
    row = frame[frame["return_date"].astype(str).eq("SUMMARY")]
    return float(row[column].iloc[0]) if not row.empty else np.nan


def _gate(name: str, value: Any, threshold: Any, operator: str, severity_if_fail: str) -> dict[str, Any]:
    if operator == "==":
        status = "PASS" if str(value) == str(threshold) else severity_if_fail.upper()
    elif pd.isna(value):
        status = "WARN"
    elif operator == ">=":
        status = "PASS" if float(value) >= float(threshold) else severity_if_fail.upper()
    elif operator == "<=":
        status = "PASS" if float(value) <= float(threshold) else severity_if_fail.upper()
    else:
        raise ValueError(f"Unsupported gate operator: {operator}")
    return {
        "gate": name,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "status": status,
        "severity": severity_if_fail,
    }


def _plot_robustness(*, records: pd.DataFrame, rolling_beta: pd.DataFrame, output_path: Path) -> None:
    frame = records.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    harsh_returns, _ = _stress_returns(
        frame,
        stock_internal_cost_bps=10.0,
        stock_scaling_cost_bps=8.0,
        gld_total_cost_bps=4.0,
        borrow_cost_bps_annual=0.0,
        margin_cost_bps_annual=0.0,
    )
    frame["harsh_execution_equity"] = (1.0 + harsh_returns).cumprod()
    frame["audit_drawdown"] = frame["audit_equity"] / frame["audit_equity"].cummax() - 1.0
    frame["harsh_drawdown"] = frame["harsh_execution_equity"] / frame["harsh_execution_equity"].cummax() - 1.0
    beta60 = rolling_beta[rolling_beta["window"].eq(60)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(frame["entry_date"], frame["reported_equity"], label="reported frozen base", color="#0f766e", linewidth=2.0)
    axes[0].plot(frame["entry_date"], frame["audit_equity"], label="plus stock scaling 4bps", color="#2563eb", linewidth=1.8)
    axes[0].plot(frame["entry_date"], frame["harsh_execution_equity"], label="harsh execution", color="#c2410c", linewidth=1.4)
    axes[0].plot(frame["entry_date"], frame["spy_equity"], label="SPY", color="#6b7280", alpha=0.75)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left")

    axes[1].plot(frame["entry_date"], frame["audit_drawdown"] * 100.0, label="audit drawdown", color="#2563eb")
    axes[1].plot(frame["entry_date"], frame["harsh_drawdown"] * 100.0, label="harsh drawdown", color="#c2410c", alpha=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Drawdown %")
    axes[1].legend(loc="lower left")

    axes[2].plot(beta60["entry_date"], beta60["rolling_beta"], color="#7c3aed", label="60d realized beta")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.05, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axhline(-0.05, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Rolling beta")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    fig.suptitle("Phase7Z Frozen Candidate Robustness Packet (Validation Only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    frozen_config: Mapping[str, Any],
    scenario_summary: pd.DataFrame,
    execution_stress: pd.DataFrame,
    financing_stress: pd.DataFrame,
    yearly: pd.DataFrame,
    tail: pd.DataFrame,
    market_state: pd.DataFrame,
    rolling_beta_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    sector: pd.DataFrame,
    leg_attribution: pd.DataFrame,
    borrow_proxy: pd.DataFrame,
    gate: pd.DataFrame,
    plot_path: Path,
) -> str:
    scenario = _format_metrics_table(
        scenario_summary[
            [
                "scenario",
                "annualized_return",
                "annualized_vol",
                "sharpe_no_rf",
                "sharpe_ann_over_vol",
                "max_drawdown",
                "mean_total_incremental_cost_bps",
            ]
        ].copy(),
        pct_cols=("annualized_return", "annualized_vol", "max_drawdown"),
        float_cols=("sharpe_no_rf", "sharpe_ann_over_vol", "mean_total_incremental_cost_bps"),
    )
    exec_sample = execution_stress[
        execution_stress["stock_internal_cost_bps_per_traded_notional"].isin([4.0, 8.0, 10.0])
        & execution_stress["stock_scaling_cost_bps_per_traded_notional"].isin([0.0, 4.0, 8.0])
        & execution_stress["gld_total_cost_bps_per_traded_notional"].isin([1.0, 4.0])
    ].copy()
    exec_sample = _format_metrics_table(
        exec_sample[
            [
                "stock_internal_cost_bps_per_traded_notional",
                "stock_scaling_cost_bps_per_traded_notional",
                "gld_total_cost_bps_per_traded_notional",
                "annualized_return",
                "sharpe_no_rf",
                "sharpe_ann_over_vol",
                "max_drawdown",
                "mean_total_incremental_cost_bps",
            ]
        ].head(18),
        pct_cols=("annualized_return", "max_drawdown"),
        float_cols=("sharpe_no_rf", "sharpe_ann_over_vol", "mean_total_incremental_cost_bps"),
    )
    finance_sample = financing_stress[
        financing_stress["borrow_cost_bps_annual_on_short_gross"].isin([0.0, 200.0, 500.0])
        & financing_stress["margin_cost_bps_annual_on_gross_above_100pct_nav"].isin([0.0, 300.0])
    ].copy()
    finance_sample = _format_metrics_table(
        finance_sample[
            [
                "borrow_cost_bps_annual_on_short_gross",
                "margin_cost_bps_annual_on_gross_above_100pct_nav",
                "annualized_return",
                "sharpe_no_rf",
                "sharpe_ann_over_vol",
                "max_drawdown",
                "mean_total_incremental_cost_bps",
            ]
        ],
        pct_cols=("annualized_return", "max_drawdown"),
        float_cols=("sharpe_no_rf", "sharpe_ann_over_vol", "mean_total_incremental_cost_bps"),
    )
    y = _format_metrics_table(
        yearly[["period_label", "sessions", "total_return", "annualized_return", "sharpe", "max_drawdown"]].copy(),
        pct_cols=("total_return", "annualized_return", "max_drawdown"),
        float_cols=("sharpe",),
    )
    tail5 = tail[tail["trim_count"].eq(5)].copy()
    tail5 = _format_metrics_table(
        tail5[
            [
                "tail_side",
                "removed_return_share_of_abs_sum",
                "baseline_annualized_return",
                "trimmed_annualized_return",
                "delta_annualized_return",
                "delta_sharpe",
            ]
        ],
        pct_cols=("removed_return_share_of_abs_sum", "baseline_annualized_return", "trimmed_annualized_return", "delta_annualized_return"),
        float_cols=("delta_sharpe",),
    )
    m = _format_metrics_table(
        market_state[["state", "days", "strategy_mean_bps", "strategy_hit_rate", "realized_beta_to_spy", "corr_to_spy", "mean_overlay_weight"]].copy(),
        pct_cols=("strategy_hit_rate", "mean_overlay_weight"),
        float_cols=("strategy_mean_bps", "realized_beta_to_spy", "corr_to_spy"),
    )
    rb = _format_metrics_table(rolling_beta_summary.copy(), float_cols=tuple(c for c in rolling_beta_summary.columns if c != "window"))
    conc = _format_metrics_table(concentration.copy(), float_cols=("mean", "median", "p90", "max"))
    sector_summary = sector[sector["return_date"].astype(str).eq("SUMMARY")].copy()
    sector_summary = _format_metrics_table(
        sector_summary,
        float_cols=("sic2_l1_exposure", "sic2_max_abs_exposure", "sic2_worst_sector_weight"),
    )
    leg = _format_metrics_table(leg_attribution.copy(), float_cols=("mean_bps", "hit_rate"))
    borrow = _format_metrics_table(
        borrow_proxy.copy(),
        pct_cols=("current_shortable_coverage", "current_easy_to_borrow_coverage"),
    )

    candidate_id = str(frozen_config.get("candidate_id", "phase7y frozen candidate"))
    gate_counts = gate["status"].value_counts().to_dict()
    lines = [
        "# Phase7Z Frozen Candidate Robustness Memo",
        "",
        f"- Candidate: `{candidate_id}`",
        "- Scope: validation only; no future test lockbox read.",
        "- Primary audit return: reported Phase7Y frozen return minus stock-book scaling friction at 4 bps per traded notional.",
        "- Scaling formula: `stock_scaling_turnover = 2 * abs(delta insurance overlay weight)`, because the funded GLD overlay shrinks or rebuilds both sides of the 2x long/short stock book.",
        "- `sharpe_no_rf` follows the Phase6 robustness convention (`mean/std * sqrt(252)`); `sharpe_ann_over_vol` matches the Phase7W/Y convention (`annualized return / annualized vol`).",
        f"- Gate status counts: `{gate_counts}`",
        "",
        "## Scenario Summary",
        "",
        _markdown_table(scenario),
        "",
        "## Execution Cost Stress",
        "",
        _markdown_table(exec_sample),
        "",
        "## Borrow And Margin Stress",
        "",
        _markdown_table(finance_sample),
        "",
        "## Yearly Stability",
        "",
        _markdown_table(y),
        "",
        "## Tail Dependence",
        "",
        _markdown_table(tail5),
        "",
        "## Market-State Exposure",
        "",
        _markdown_table(m),
        "",
        "## Rolling Beta",
        "",
        _markdown_table(rb),
        "",
        "## Position Concentration",
        "",
        _markdown_table(conc),
        "",
        "## Sector Exposure",
        "",
        _markdown_table(sector_summary),
        "",
        "## Leg Proxy And Borrow Coverage",
        "",
        _markdown_table(leg),
        "",
        _markdown_table(borrow),
        "",
        "## Gate Check",
        "",
        _markdown_table(gate.copy()),
        "",
        "## Plot",
        "",
        f"![Phase7Z robustness plot]({plot_path.as_posix()})",
        "",
        "## First Reading",
        "",
        "- Phase7Z is mostly an audit completion step: it prices the missing funded-insurance stock scaling friction and reruns the old Phase6-style stability checks.",
        "- If the harsh execution and borrow rows remain comfortably positive, the remaining blockers are mainly realized market beta/correlation sign-off, point-in-time borrow history, and persisted long/short leg PnL.",
        "- The packet should be treated as freeze evidence, not as permission to tune after seeing these diagnostics.",
    ]
    return "\n".join(lines) + "\n"


def _format_metrics_table(
    frame: pd.DataFrame,
    *,
    pct_cols: Sequence[str] = (),
    float_cols: Sequence[str] = (),
) -> pd.DataFrame:
    out = frame.copy()
    for column in pct_cols:
        if column in out:
            out[column] = out[column].map(_fmt_pct_like)
    for column in float_cols:
        if column in out:
            out[column] = out[column].map(_fmt_float_like)
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _go_no_go(gate: pd.DataFrame) -> str:
    fail_count = int(gate["status"].eq("FAIL").sum())
    warn_count = int(gate["status"].eq("WARN").sum())
    if fail_count:
        return "no_go_until_failures_resolved"
    if warn_count:
        return "conditional_go_after_watch_signoff"
    return "go_for_lockbox_protocol"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
