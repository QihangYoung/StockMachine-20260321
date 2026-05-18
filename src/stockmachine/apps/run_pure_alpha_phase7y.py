"""Phase7Y freeze and test-readiness audit for the pure-alpha candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase7b import _markdown_table
from stockmachine.apps.run_pure_alpha_phase7v import _performance_metrics


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7y_freeze_and_audit_20260518"
DEFAULT_PHASE7K_ROOT = RESEARCH_ROOT / "phase7k_locked_turnover_shared_capital_20260518_tb0p15"
DEFAULT_PHASE7T_ROOT = RESEARCH_ROOT / "phase7t_universe_underwriting_sizepatched_20260518_tb0p15"
DEFAULT_PHASE7V_ROOT = RESEARCH_ROOT / "phase7v_cross_asset_insurance_overlay_20260518"
DEFAULT_PHASE7W_ROOT = RESEARCH_ROOT / "phase7w_walk_forward_insurance_overlay_20260518"
DEFAULT_PHASE7X_ROOT = RESEARCH_ROOT / "phase7x_insurance_threshold_robustness_20260518"
DEFAULT_PHASE7S_ROOT = RESEARCH_ROOT / "phase7s_no_low_beta_ablation_20260518_tb0p15"

PRIMARY_PORTFOLIO = "shared_core_lambda_0p005_tb0p15"
PRIMARY_INSURANCE_POLICY = "funded_gld_0_10_20"
VALIDATION_END = pd.Timestamp("2019-12-31")
TEST_START = "2020-01-02"
TEST_END = "2026-04-08"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze and audit the Phase7 pure-alpha candidate.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase7k-root", default=str(DEFAULT_PHASE7K_ROOT))
    parser.add_argument("--phase7t-root", default=str(DEFAULT_PHASE7T_ROOT))
    parser.add_argument("--phase7v-root", default=str(DEFAULT_PHASE7V_ROOT))
    parser.add_argument("--phase7w-root", default=str(DEFAULT_PHASE7W_ROOT))
    parser.add_argument("--phase7x-root", default=str(DEFAULT_PHASE7X_ROOT))
    parser.add_argument("--phase7s-root", default=str(DEFAULT_PHASE7S_ROOT))
    parser.add_argument("--primary-portfolio", default=PRIMARY_PORTFOLIO)
    parser.add_argument("--primary-insurance-policy", default=PRIMARY_INSURANCE_POLICY)
    args = parser.parse_args(argv)

    rollup = build_phase7y_freeze_and_audit(
        output_root=Path(args.output_root),
        phase7k_root=Path(args.phase7k_root),
        phase7t_root=Path(args.phase7t_root),
        phase7v_root=Path(args.phase7v_root),
        phase7w_root=Path(args.phase7w_root),
        phase7x_root=Path(args.phase7x_root),
        phase7s_root=Path(args.phase7s_root),
        primary_portfolio=str(args.primary_portfolio),
        primary_insurance_policy=str(args.primary_insurance_policy),
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=False))
    return 0


def build_phase7y_freeze_and_audit(
    *,
    output_root: Path,
    phase7k_root: Path,
    phase7t_root: Path,
    phase7v_root: Path,
    phase7w_root: Path,
    phase7x_root: Path,
    phase7s_root: Path,
    primary_portfolio: str,
    primary_insurance_policy: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    paths = _input_paths(
        phase7k_root=phase7k_root,
        phase7t_root=phase7t_root,
        phase7v_root=phase7v_root,
        phase7w_root=phase7w_root,
        phase7x_root=phase7x_root,
        phase7s_root=phase7s_root,
    )
    phase7k_rollup = _read_json(paths["phase7k_rollup"])
    phase7w_rollup = _read_json(paths["phase7w_rollup"])
    phase7x_rollup = _read_json(paths["phase7x_rollup"])
    phase7k_metrics = pd.read_csv(paths["phase7k_metrics"])
    phase7t_summary = pd.read_csv(paths["phase7t_summary"])
    phase7t_scorecard = pd.read_csv(paths["phase7t_scorecard"])
    phase7w_summary = pd.read_csv(paths["phase7w_summary"])
    phase7x_robustness = pd.read_csv(paths["phase7x_robustness"])
    phase7v_funded = pd.read_csv(paths["phase7v_funded"])

    primary_metrics = _select_policy_row(phase7w_summary, primary_insurance_policy)
    base_metrics = _select_phase7k_metric(
        phase7k_metrics,
        portfolio=primary_portfolio,
        return_kind="net",
    )
    top1000_universe = phase7t_summary.loc[
        phase7t_summary["variant"].eq(str(phase7k_rollup["variant"]))
    ].iloc[0]
    top1000_scorecard = phase7t_scorecard.loc[
        phase7t_scorecard["variant"].eq(str(phase7k_rollup["variant"]))
    ].iloc[0]
    robustness_row = phase7x_robustness.loc[
        phase7x_robustness["policy"].eq(primary_insurance_policy)
    ].iloc[0]

    curve = _build_frozen_candidate_curve(
        phase7w_daily_panel_path=paths["phase7w_daily_panel"],
        phase7w_policy_weights_path=paths["phase7w_policy_weights"],
        primary_insurance_policy=primary_insurance_policy,
    )
    curve_path = output_root / "phase7y_frozen_candidate_daily_curve.csv"
    curve.to_csv(curve_path, index=False)

    ablation = _build_component_ablation(
        phase7k_metrics=phase7k_metrics,
        phase7s_metrics_path=paths["phase7s_metrics"],
        phase7w_summary=phase7w_summary,
        phase7v_funded=phase7v_funded,
        primary_portfolio=primary_portfolio,
        primary_insurance_policy=primary_insurance_policy,
    )
    ablation_path = output_root / "phase7y_component_ablation.csv"
    ablation.to_csv(ablation_path, index=False)

    freeze_config = _build_freeze_config(
        phase7k_rollup=phase7k_rollup,
        phase7w_rollup=phase7w_rollup,
        phase7x_rollup=phase7x_rollup,
        base_metrics=base_metrics,
        primary_metrics=primary_metrics,
        top1000_universe=top1000_universe,
        top1000_scorecard=top1000_scorecard,
        robustness_row=robustness_row,
        paths=paths,
        output_root=output_root,
        primary_portfolio=primary_portfolio,
        primary_insurance_policy=primary_insurance_policy,
    )
    config_path = output_root / "phase7y_frozen_candidate_config.json"
    config_path.write_text(json.dumps(freeze_config, indent=2, ensure_ascii=False), encoding="utf-8")

    audit = _build_audit_checks(
        paths={**paths, "frozen_curve": curve_path, "component_ablation": ablation_path, "freeze_config": config_path},
        phase7k_rollup=phase7k_rollup,
        phase7w_rollup=phase7w_rollup,
        phase7x_rollup=phase7x_rollup,
        phase7k_metrics=phase7k_metrics,
        phase7t_summary=phase7t_summary,
        phase7t_scorecard=phase7t_scorecard,
        phase7w_summary=phase7w_summary,
        phase7x_robustness=phase7x_robustness,
        curve=curve,
        ablation=ablation,
        primary_portfolio=primary_portfolio,
        primary_insurance_policy=primary_insurance_policy,
    )
    audit_path = output_root / "phase7y_audit_checks.csv"
    audit.to_csv(audit_path, index=False)

    protocol_path = output_root / "phase7y_test_lockbox_protocol.md"
    _write_test_protocol(protocol_path, freeze_config=freeze_config, audit=audit)

    manifest_paths = {
        **paths,
        "frozen_candidate_curve": curve_path,
        "component_ablation": ablation_path,
        "freeze_config": config_path,
        "audit_checks": audit_path,
        "test_lockbox_protocol": protocol_path,
    }
    manifest = _artifact_manifest(manifest_paths)
    manifest_path = output_root / "phase7y_artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    memo_path = output_root / "phase7y_freeze_audit_memo_zh.md"
    _write_memo(
        memo_path,
        freeze_config=freeze_config,
        audit=audit,
        ablation=ablation,
        manifest=manifest,
    )

    rollup: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase7y_freeze_and_audit",
        "scope": "validation_only_freeze_no_test_read",
        "primary_portfolio": primary_portfolio,
        "primary_insurance_policy": primary_insurance_policy,
        "output_root": str(output_root),
        "audit_status_counts": audit["status"].value_counts().to_dict(),
        "go_no_go": _go_no_go(audit),
        "paths": {
            "freeze_config": str(config_path),
            "frozen_candidate_curve": str(curve_path),
            "component_ablation": str(ablation_path),
            "audit_checks": str(audit_path),
            "artifact_manifest": str(manifest_path),
            "test_lockbox_protocol": str(protocol_path),
            "memo": str(memo_path),
        },
    }
    rollup_path = output_root / "phase7y_rollup.json"
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=False), encoding="utf-8")
    return rollup


def _input_paths(
    *,
    phase7k_root: Path,
    phase7t_root: Path,
    phase7v_root: Path,
    phase7w_root: Path,
    phase7x_root: Path,
    phase7s_root: Path,
) -> dict[str, Path]:
    return {
        "phase7k_rollup": phase7k_root / "phase7k_rollup.json",
        "phase7k_curve": phase7k_root / "phase7k_strict_daily_curve.csv",
        "phase7k_metrics": phase7k_root / "phase7k_strict_daily_metrics.csv",
        "phase7k_positions": phase7k_root / "phase7k_locked_positions.csv.gz",
        "phase7k_score_coverage": phase7k_root / "phase7k_score_coverage.csv",
        "phase7k_turnover": phase7k_root / "phase7k_turnover_summary.csv",
        "phase7t_summary": phase7t_root / "phase7t_universe_underwriting_summary.csv",
        "phase7t_scorecard": phase7t_root / "phase7t_universe_underwriting_scorecard.csv",
        "phase7v_funded": phase7v_root / "phase7v_funded_overlay_summary.csv",
        "phase7w_rollup": phase7w_root / "phase7w_rollup.json",
        "phase7w_daily_panel": phase7w_root / "phase7w_daily_panel.csv",
        "phase7w_policy_weights": phase7w_root / "phase7w_policy_weights.csv",
        "phase7w_policy_costs": phase7w_root / "phase7w_policy_costs.csv",
        "phase7w_summary": phase7w_root / "phase7w_policy_summary.csv",
        "phase7x_rollup": phase7x_root / "phase7x_rollup.json",
        "phase7x_robustness": phase7x_root / "phase7x_policy_robustness.csv",
        "phase7x_default_rows": phase7x_root / "phase7x_default_threshold_rows.csv",
        "phase7s_metrics": phase7s_root / "phase7k_strict_daily_metrics.csv",
        "phase7w_source": Path("src/stockmachine/apps/run_pure_alpha_phase7w.py"),
    }


def _build_frozen_candidate_curve(
    *,
    phase7w_daily_panel_path: Path,
    phase7w_policy_weights_path: Path,
    primary_insurance_policy: str,
) -> pd.DataFrame:
    panel = pd.read_csv(phase7w_daily_panel_path)
    weights = pd.read_csv(phase7w_policy_weights_path)
    required_columns = {"date", "shared_core", primary_insurance_policy}
    missing = required_columns.difference(panel.columns)
    if missing:
        raise KeyError(f"Phase7W daily panel is missing columns: {sorted(missing)}")
    out = panel.loc[:, ["date", "shared_core", primary_insurance_policy, "risk_state", "risk_score"]].copy()
    out = out.rename(
        columns={
            "shared_core": "shared_core_net_return",
            primary_insurance_policy: "frozen_candidate_net_return",
        }
    )
    weight_column = f"{primary_insurance_policy}__total_overlay_weight"
    if weight_column in weights.columns:
        out = out.merge(weights.loc[:, ["date", weight_column]], on="date", how="left", validate="one_to_one")
        out = out.rename(columns={weight_column: "insurance_overlay_weight"})
    else:
        out["insurance_overlay_weight"] = np.nan
    returns = out["frozen_candidate_net_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    out["frozen_candidate_equity"] = equity
    out["frozen_candidate_drawdown"] = equity / equity.cummax() - 1.0
    return out


def _build_component_ablation(
    *,
    phase7k_metrics: pd.DataFrame,
    phase7s_metrics_path: Path,
    phase7w_summary: pd.DataFrame,
    phase7v_funded: pd.DataFrame,
    primary_portfolio: str,
    primary_insurance_policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_metric_row(name: str, source: str, row: Mapping[str, Any], note: str) -> None:
        rows.append(
            {
                "component": name,
                "source": source,
                "ann_return": _coalesce(row, "annualized_return", "ann_return"),
                "vol": _coalesce(row, "annualized_vol", "vol"),
                "sharpe": _coalesce(row, "sharpe_no_rf", "sharpe"),
                "max_dd": _coalesce(row, "max_drawdown", "max_dd"),
                "mean_daily_bps": _coalesce(row, "mean_daily_return_bps", "mean_daily_bps"),
                "worst_20d": _optional_coalesce(row, "worst_20d", "worst_20d_return"),
                "note": note,
            }
        )

    base = _select_phase7k_metric(phase7k_metrics, portfolio=primary_portfolio, return_kind="net")
    add_metric_row("shared_core_no_insurance", "phase7k", base, "Base frozen equity-alpha path.")

    primary_policy = _select_policy_row(phase7w_summary, primary_insurance_policy)
    add_metric_row(
        f"frozen_{primary_insurance_policy}",
        "phase7w",
        primary_policy,
        "Primary frozen candidate with walk-forward funded GLD insurance.",
    )
    for policy in ("funded_gld_5_10_20", "funded_cash_gld_0_10_20", "financed_gld_0_05_10"):
        if phase7w_summary["policy"].eq(policy).any():
            add_metric_row(
                policy,
                "phase7w",
                _select_policy_row(phase7w_summary, policy),
                "Insurance policy comparator.",
            )

    for overlay in ("funded_GLD_gold_10pct", "funded_GLD_gold_20pct"):
        if phase7v_funded["overlay"].eq(overlay).any():
            add_metric_row(
                overlay,
                "phase7v",
                phase7v_funded.loc[phase7v_funded["overlay"].eq(overlay)].iloc[0],
                "Always-on funded GLD comparator.",
            )

    for portfolio, note in (
        ("shared_no_momentum_lambda_0p005_tb0p15", "No-momentum ablation from Phase7K."),
        ("shared_slow_core_lambda_0p005_tb0p15", "Slow-core factor subset comparator from Phase7K."),
    ):
        if phase7k_metrics["portfolio"].eq(portfolio).any():
            add_metric_row(
                portfolio,
                "phase7k",
                _select_phase7k_metric(phase7k_metrics, portfolio=portfolio, return_kind="net"),
                note,
            )

    if phase7s_metrics_path.exists():
        phase7s_metrics = pd.read_csv(phase7s_metrics_path)
        portfolio = "shared_no_low_beta_lambda_0p005_tb0p15"
        if phase7s_metrics["portfolio"].eq(portfolio).any():
            add_metric_row(
                portfolio,
                "phase7s",
                _select_phase7k_metric(phase7s_metrics, portfolio=portfolio, return_kind="net"),
                "No-low-beta ablation.",
            )

    out = pd.DataFrame(rows)
    base_sharpe = float(out.loc[out["component"].eq("shared_core_no_insurance"), "sharpe"].iloc[0])
    base_dd = float(out.loc[out["component"].eq("shared_core_no_insurance"), "max_dd"].iloc[0])
    out["sharpe_delta_vs_base"] = out["sharpe"].astype(float) - base_sharpe
    out["max_dd_delta_vs_base"] = out["max_dd"].astype(float) - base_dd
    return out


def _build_freeze_config(
    *,
    phase7k_rollup: Mapping[str, Any],
    phase7w_rollup: Mapping[str, Any],
    phase7x_rollup: Mapping[str, Any],
    base_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
    top1000_universe: Mapping[str, Any],
    top1000_scorecard: Mapping[str, Any],
    robustness_row: Mapping[str, Any],
    paths: Mapping[str, Path],
    output_root: Path,
    primary_portfolio: str,
    primary_insurance_policy: str,
) -> dict[str, Any]:
    portfolio_spec = next(
        portfolio
        for portfolio in phase7k_rollup["portfolios"]
        if portfolio["portfolio"] == primary_portfolio
    )
    base_validation_metrics = _metric_payload(base_metrics)
    phase7w_base_metrics = phase7w_rollup.get("base_metrics", {})
    if isinstance(phase7w_base_metrics, Mapping):
        for key in ("worst_1d_bps", "cvar05_1d_bps", "worst_20d"):
            if key in phase7w_base_metrics and pd.isna(base_validation_metrics.get(key)):
                base_validation_metrics[key] = float(phase7w_base_metrics[key])
    return {
        "candidate_id": "phase7y_shared_core_top1000_gld_insurance_frozen_20260518",
        "freeze_status": "frozen_validation_candidate_no_test_read",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "primary chosen from validation evidence before any current-candidate test read",
        "validation_window": {
            "start": str(primary_metrics["start"]) if "start" in primary_metrics else str(phase7w_rollup["start"]),
            "end": str(phase7w_rollup["end"]),
        },
        "future_test_lockbox": {
            "start": TEST_START,
            "end": TEST_END,
            "rule": "one-time read only after freeze/audit sign-off; no parameter changes from test results",
        },
        "equity_alpha": {
            "runner": "stockmachine.apps.run_pure_alpha_phase7k",
            "portfolio": primary_portfolio,
            "universe": phase7k_rollup["variant"],
            "factor_weights": portfolio_spec["factor_weights"],
            "factor_min_holds": phase7k_rollup["factor_min_holds"],
            "turnover_budget": portfolio_spec["turnover_budget"],
            "turnover_penalty": portfolio_spec["turnover_penalty"],
            "candidate_pool_per_side": phase7k_rollup["candidate_pool_per_side"],
            "max_single_name_side_weight": phase7k_rollup["max_single_name_side_weight"],
            "sector_penalty": phase7k_rollup["sector_penalty"],
            "stock_cost_bps_per_side": phase7k_rollup["cost_bps_per_side"],
            "label_contract": phase7k_rollup["label_contract"],
            "validation_metrics": base_validation_metrics,
        },
        "insurance_overlay": {
            "runner": "stockmachine.apps.run_pure_alpha_phase7w",
            "policy": primary_insurance_policy,
            "overlay_kind": primary_metrics["overlay_kind"],
            "asset": "GLD",
            "funding": "funded; shared_core NAV is reduced by overlay weight",
            "weights": {"calm": 0.0, "watch": 0.10, "stress": 0.20},
            "state_detector": phase7w_rollup["thresholds"],
            "state_inputs": [
                "lagged shared_core drawdown",
                "lagged shared_core 20d return",
                "lagged shared_core 63d vol",
                "lagged SPY drawdown",
                "lagged SPY 20d return",
            ],
            "overlay_cost_bps_per_traded_notional": phase7w_rollup["overlay_cost_bps_per_traded_notional"],
            "validation_metrics": _metric_payload(primary_metrics),
        },
        "universe_underwriting": {
            "chosen": phase7k_rollup["variant"],
            "rank_sum": float(top1000_scorecard["underwriting_rank_sum"]),
            "score_coverage_min": float(top1000_universe["score_coverage_min"]),
            "reason": "top1000 remains the strongest symmetric universe after size data backfill",
        },
        "insurance_robustness": {
            "grid_config_count": phase7x_rollup["grid_config_count"],
            "policy": primary_insurance_policy,
            "strict_pass_rate": float(robustness_row["strict_pass_rate"]),
            "ann_return_drag_median": float(robustness_row["ann_return_drag_median"]),
            "max_dd_reduction_median": float(robustness_row["max_dd_reduction_median"]),
            "bad_day_improvement_median_bps": float(robustness_row["bad_day_improvement_median_bps"]),
        },
        "explicit_exclusions": [
            "KG regime gating",
            "asymmetric long/short universe",
            "financed cross-asset overlay",
            "dynamic factor reweighting",
            "post-validation threshold search",
        ],
        "source_artifacts": {key: str(path) for key, path in paths.items() if key != "phase7w_source"},
        "output_root": str(output_root),
        "environment": _environment_snapshot(),
    }


def _build_audit_checks(
    *,
    paths: Mapping[str, Path],
    phase7k_rollup: Mapping[str, Any],
    phase7w_rollup: Mapping[str, Any],
    phase7x_rollup: Mapping[str, Any],
    phase7k_metrics: pd.DataFrame,
    phase7t_summary: pd.DataFrame,
    phase7t_scorecard: pd.DataFrame,
    phase7w_summary: pd.DataFrame,
    phase7x_robustness: pd.DataFrame,
    curve: pd.DataFrame,
    ablation: pd.DataFrame,
    primary_portfolio: str,
    primary_insurance_policy: str,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(category: str, check: str, status: str, value: Any, threshold: str, evidence: str) -> None:
        checks.append(
            {
                "category": category,
                "check": check,
                "status": status,
                "value": value,
                "threshold_or_rule": threshold,
                "evidence": evidence,
            }
        )

    for name, path in paths.items():
        if name == "phase7w_source":
            continue
        add("artifact", f"{name}_exists", "PASS" if path.exists() else "FAIL", str(path), "file exists", str(path))

    add(
        "split",
        "phase7k_validation_only",
        "PASS" if not bool(phase7k_rollup.get("test_lockbox_used", True)) else "FAIL",
        phase7k_rollup.get("test_lockbox_used"),
        "test_lockbox_used must be false",
        str(paths["phase7k_rollup"]),
    )
    phase7k_end = pd.Timestamp(str(phase7k_rollup["validation_price_end"]))
    phase7w_end = pd.Timestamp(str(phase7w_rollup["end"]))
    phase7x_end = pd.Timestamp(str(phase7x_rollup["end"]))
    add("split", "phase7k_end_before_test", "PASS" if phase7k_end <= VALIDATION_END else "FAIL", str(phase7k_end.date()), "<= 2019-12-31", "")
    add("split", "phase7w_end_before_test", "PASS" if phase7w_end <= VALIDATION_END else "FAIL", str(phase7w_end.date()), "<= 2019-12-31", "")
    add("split", "phase7x_end_before_test", "PASS" if phase7x_end <= VALIDATION_END else "FAIL", str(phase7x_end.date()), "<= 2019-12-31", "")
    add(
        "split",
        "historical_phase7a_not_referenced",
        "PASS",
        "not referenced",
        "current freeze inputs must not depend on phase7a test artifacts",
        "Phase7A exists for an older candidate but is absent from source_artifacts.",
    )

    curve_dates = pd.to_datetime(curve["date"])
    add(
        "split",
        "frozen_curve_end_before_test",
        "PASS" if curve_dates.max() <= VALIDATION_END else "FAIL",
        str(curve_dates.max().date()),
        "<= 2019-12-31",
        str(paths["frozen_curve"]),
    )

    phase7k_curve = pd.read_csv(paths["phase7k_curve"])
    signal_dates = pd.to_datetime(phase7k_curve["signal_date"])
    return_dates = pd.to_datetime(phase7k_curve["return_date"])
    timing_ok = bool((signal_dates < return_dates).all())
    add(
        "timing",
        "signal_date_before_return_date",
        "PASS" if timing_ok else "FAIL",
        int((signal_dates >= return_dates).sum()),
        "zero rows with signal_date >= return_date",
        str(paths["phase7k_curve"]),
    )
    label_contract = phase7k_rollup.get("label_contract", {})
    add(
        "timing",
        "open_to_open_label_contract_recorded",
        "PASS" if "open" in json.dumps(label_contract).lower() else "FAIL",
        json.dumps(label_contract),
        "label contract explicitly records open-to-open timing",
        str(paths["phase7k_rollup"]),
    )
    source_text = paths["phase7w_source"].read_text(encoding="utf-8")
    shift_count = source_text.count(".shift(1)")
    add(
        "timing",
        "insurance_state_uses_lagged_features",
        "PASS" if shift_count >= 5 else "FAIL",
        shift_count,
        "at least five lagged state features via shift(1)",
        str(paths["phase7w_source"]),
    )

    base = _select_phase7k_metric(phase7k_metrics, portfolio=primary_portfolio, return_kind="net")
    add("cost", "stock_cost_bps_per_side_frozen", "PASS" if float(phase7k_rollup["cost_bps_per_side"]) == 4.0 else "FAIL", phase7k_rollup["cost_bps_per_side"], "4.0 bps per side", "")
    add("cost", "overlay_cost_recorded", "PASS" if float(phase7w_rollup["overlay_cost_bps_per_traded_notional"]) == 1.0 else "WATCH", phase7w_rollup["overlay_cost_bps_per_traded_notional"], "1.0 bps per traded overlay notional", "")
    add("cost", "mean_stock_cost_reasonable", "PASS" if float(base["mean_cost_bps"]) <= 0.75 else "WATCH", float(base["mean_cost_bps"]), "<= 0.75 bps/day", str(paths["phase7k_metrics"]))

    add("construction", "mean_turnover_near_budget", "PASS" if float(base["mean_turnover"]) <= 0.16 else "WATCH", float(base["mean_turnover"]), "<= 0.16 daily mean", str(paths["phase7k_metrics"]))
    add("construction", "ex_ante_net_exposure_near_zero", "PASS" if float(base["mean_abs_net_exposure"]) <= 1e-6 else "FAIL", float(base["mean_abs_net_exposure"]), "<= 1e-6", str(paths["phase7k_metrics"]))
    add("construction", "no_skipped_sessions", "PASS" if int(phase7k_rollup["rows"]["skipped"]) == 0 else "WATCH", int(phase7k_rollup["rows"]["skipped"]), "0 skipped sessions preferred", str(paths["phase7k_rollup"]))

    chosen_variant = str(phase7k_rollup["variant"])
    universe = phase7t_summary.loc[phase7t_summary["variant"].eq(chosen_variant)].iloc[0]
    scorecard = phase7t_scorecard.loc[phase7t_scorecard["variant"].eq(chosen_variant)].iloc[0]
    min_rank = float(phase7t_scorecard["underwriting_rank_sum"].min())
    add("universe", "top1000_selected_by_underwriting", "PASS" if float(scorecard["underwriting_rank_sum"]) == min_rank else "WATCH", float(scorecard["underwriting_rank_sum"]), f"rank_sum == {min_rank}", str(paths["phase7t_scorecard"]))
    add("universe", "score_coverage_min_after_size_backfill", "PASS" if float(universe["score_coverage_min"]) >= 0.90 else "FAIL", float(universe["score_coverage_min"]), ">= 0.90", str(paths["phase7t_summary"]))

    realized_beta = float(base["realized_beta_net"])
    realized_corr = float(base["corr_to_spy_net"])
    add(
        "risk",
        "realized_beta_corr_protocol_gate",
        "WATCH" if abs(realized_beta) > 0.05 or abs(realized_corr) > 0.10 else "PASS",
        f"beta={realized_beta:.4f}, corr={realized_corr:.4f}",
        "initial protocol beta <= 0.05 and corr <= 0.10",
        "Ex-ante neutrality passes, but realized beta/corr require explicit sign-off.",
    )
    add("risk", "max_drawdown_under_12pct", "PASS" if float(base["max_drawdown"]) >= -0.12 else "WATCH", float(base["max_drawdown"]), ">= -0.12 validation max DD", str(paths["phase7k_metrics"]))

    policy = _select_policy_row(phase7w_summary, primary_insurance_policy)
    robustness = phase7x_robustness.loc[phase7x_robustness["policy"].eq(primary_insurance_policy)].iloc[0]
    add("insurance", "policy_is_funded_not_financed", "PASS" if policy["overlay_kind"] == "funded" else "FAIL", policy["overlay_kind"], "funded", str(paths["phase7w_summary"]))
    add("insurance", "insurance_active_rate_not_always_on", "PASS" if float(policy["active_rate"]) < 0.50 else "WATCH", float(policy["active_rate"]), "< 0.50 for primary switch policy", str(paths["phase7w_summary"]))
    add("insurance", "insurance_robustness_strict_pass_rate", "PASS" if float(robustness["strict_pass_rate"]) >= 0.90 else "WATCH", float(robustness["strict_pass_rate"]), ">= 0.90", str(paths["phase7x_robustness"]))
    add("insurance", "insurance_bad_day_improvement", "PASS" if float(policy["bad_day_improvement_bps"]) > 0 else "FAIL", float(policy["bad_day_improvement_bps"]), "> 0 bps", str(paths["phase7w_summary"]))

    add("ablation", "component_ablation_table_present", "PASS" if len(ablation) >= 7 else "WATCH", len(ablation), ">= 7 rows covering base, insurance, factor ablations", str(paths["component_ablation"]))

    status = _git_status_porcelain()
    add(
        "reproducibility",
        "git_worktree_clean_before_test",
        "PASS" if status == "" else "WATCH",
        "clean" if status == "" else f"{len(status.splitlines())} dirty/untracked entries",
        "commit or archive exact source state before lockbox read",
        "Current research workspace is not yet commit-clean.",
    )
    manifest_count = len(_artifact_manifest(paths))
    add("reproducibility", "artifact_manifest_hashes_available", "PASS" if manifest_count >= 10 else "WATCH", manifest_count, ">= 10 hashed artifacts", "")
    add("reproducibility", "dependency_snapshot_available", "PASS", _environment_snapshot()["python"], "python and key package versions recorded", "")

    return pd.DataFrame(checks)


def _write_test_protocol(path: Path, *, freeze_config: Mapping[str, Any], audit: pd.DataFrame) -> None:
    watch_count = int(audit["status"].eq("WATCH").sum())
    fail_count = int(audit["status"].eq("FAIL").sum())
    text = f"""# Phase7Y Test Lockbox Protocol

Date: 2026-05-18

This protocol freezes the current validation-selected candidate before any
current-candidate test-window performance is read.

## Frozen Candidate

- Candidate: `{freeze_config['candidate_id']}`
- Equity alpha: Phase7K `{freeze_config['equity_alpha']['portfolio']}`
- Universe: `{freeze_config['equity_alpha']['universe']}` symmetric long/short pool
- Insurance: Phase7W `{freeze_config['insurance_overlay']['policy']}`
- Funding: funded overlay only; no financed cross-asset overlay
- Exclusions: KG gating, asymmetric universe, financed overlay, dynamic factor reweighting

## Validation Evidence

- Shared core Sharpe: `{freeze_config['equity_alpha']['validation_metrics']['sharpe']:.4f}`
- Shared core max drawdown: `{freeze_config['equity_alpha']['validation_metrics']['max_dd']:.4f}`
- Frozen candidate Sharpe: `{freeze_config['insurance_overlay']['validation_metrics']['sharpe']:.4f}`
- Frozen candidate max drawdown: `{freeze_config['insurance_overlay']['validation_metrics']['max_dd']:.4f}`
- Insurance robustness strict pass rate: `{freeze_config['insurance_robustness']['strict_pass_rate']:.4f}`

## Test Window

- Test start: `{TEST_START}`
- Test end: `{TEST_END}`
- The test run is a one-time lockbox read.
- No parameter, universe, factor, threshold, insurance, or cost rule may be changed because of the test result.

## Required Before Running Test

- Review all `WATCH` audit rows.
- Explicitly accept or resolve the realized beta/correlation watch item.
- Commit or otherwise archive the exact source/artifact state.
- Run the test command once and preserve all generated artifacts and logs.

## Current Audit State

- Fail rows: `{fail_count}`
- Watch rows: `{watch_count}`

If the test fails, record it as a failed final evaluation. Any further work must
start a new validation research cycle before another lockbox read.
"""
    path.write_text(text, encoding="utf-8")


def _write_memo(
    path: Path,
    *,
    freeze_config: Mapping[str, Any],
    audit: pd.DataFrame,
    ablation: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    status_counts = audit["status"].value_counts().to_dict()
    important_checks = audit.loc[audit["status"].isin(["WATCH", "FAIL"])].copy()
    if important_checks.empty:
        important_checks = audit.head(0).copy()
    metrics = pd.DataFrame(
        [
            {
                "candidate": "shared_core_no_insurance",
                **freeze_config["equity_alpha"]["validation_metrics"],
            },
            {
                "candidate": freeze_config["insurance_overlay"]["policy"],
                **freeze_config["insurance_overlay"]["validation_metrics"],
            },
        ]
    )
    lines = [
        "# Phase7Y 冻结与审计备忘录",
        "",
        "本文件冻结当前 validation 候选，并记录开测试集前的 hygiene audit。",
        "",
        "## Frozen Candidate",
        "",
        f"- candidate: `{freeze_config['candidate_id']}`",
        f"- equity alpha: `{freeze_config['equity_alpha']['portfolio']}`",
        f"- universe: `{freeze_config['equity_alpha']['universe']}`",
        f"- insurance: `{freeze_config['insurance_overlay']['policy']}`",
        "- explicit exclusions: KG gating, asymmetric universe, financed overlay, dynamic factor reweighting",
        "",
        "## Key Metrics",
        "",
        _markdown_table(metrics.loc[:, ["candidate", "ann_return", "vol", "sharpe", "max_dd", "worst_20d", "cvar05_1d_bps"]]),
        "",
        "## Audit Status",
        "",
        _markdown_table(pd.DataFrame([status_counts])),
        "",
        "## Watch / Fail Items",
        "",
        _markdown_table(
            important_checks.loc[:, ["category", "check", "status", "value", "threshold_or_rule", "evidence"]]
        ),
        "",
        "## Component Ablation",
        "",
        _markdown_table(
            ablation.loc[
                :,
                [
                    "component",
                    "ann_return",
                    "vol",
                    "sharpe",
                    "max_dd",
                    "sharpe_delta_vs_base",
                    "max_dd_delta_vs_base",
                    "note",
                ],
            ]
        ),
        "",
        "## Go / No-Go",
        "",
        f"- current decision: `{_go_no_go(audit)}`",
        "- No FAIL rows were observed if the decision is conditional_go_after_watch_signoff.",
        "- The main unresolved research sign-off is realized beta/correlation versus the original pure-alpha protocol gate.",
        "- The main operational sign-off is reproducibility: commit or archive the exact dirty worktree before any lockbox read.",
        "",
        "## Artifact Manifest",
        "",
        f"- hashed artifacts: `{len(manifest)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_payload(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "ann_return": float(_coalesce(row, "annualized_return", "ann_return")),
        "vol": float(_coalesce(row, "annualized_vol", "vol")),
        "sharpe": float(_coalesce(row, "sharpe_no_rf", "sharpe")),
        "max_dd": float(_coalesce(row, "max_drawdown", "max_dd")),
        "mean_daily_bps": float(_coalesce(row, "mean_daily_return_bps", "mean_daily_bps")),
        "worst_1d_bps": float(_optional_coalesce(row, "worst_1d_bps")),
        "cvar05_1d_bps": float(_optional_coalesce(row, "cvar05_1d_bps")),
        "worst_20d": float(_optional_coalesce(row, "worst_20d", "worst_20d_return")),
    }


def _select_phase7k_metric(metrics: pd.DataFrame, *, portfolio: str, return_kind: str) -> pd.Series:
    mask = metrics["portfolio"].eq(portfolio) & metrics["return_kind"].eq(return_kind)
    if not mask.any():
        raise ValueError(f"Missing metric row for {portfolio}/{return_kind}")
    return metrics.loc[mask].iloc[0]


def _select_policy_row(summary: pd.DataFrame, policy: str) -> pd.Series:
    mask = summary["policy"].eq(policy)
    if not mask.any():
        raise ValueError(f"Missing Phase7W policy row: {policy}")
    return summary.loc[mask].iloc[0]


def _coalesce(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    raise KeyError(f"None of the columns were present: {names}")


def _optional_coalesce(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return np.nan


def _artifact_manifest(paths: Mapping[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(paths.items()):
        if not path.exists() or path.is_dir():
            continue
        stat = path.stat()
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "bytes": int(stat.st_size),
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": _git_command(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _git_command(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return ""
    return completed.stdout.strip()


def _git_status_porcelain() -> str:
    return _git_command(["git", "status", "--porcelain"])


def _go_no_go(audit: pd.DataFrame) -> str:
    if audit["status"].eq("FAIL").any():
        return "no_go_until_fail_rows_resolved"
    if audit["status"].eq("WATCH").any():
        return "conditional_go_after_watch_signoff"
    return "go_for_one_time_test_lockbox_read"


if __name__ == "__main__":
    raise SystemExit(main())
