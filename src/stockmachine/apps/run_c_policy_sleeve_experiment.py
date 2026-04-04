from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from stockmachine.research.p1_rigor import summarize_backtest_records
from stockmachine.risk import (
    REGIME_LABELS,
    BenchmarkTrendDrawdownRegimeDetector,
    RegimeDetector,
    RegimeGatePolicy,
    apply_regime_gate,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blend Benchmark C policy returns with one or more stock-alpha sleeves."
    )
    parser.add_argument(
        "--core-records",
        default="artifacts/c_risk_tuning_regime_window_20260330/benchmark_c_policy_records_common404.csv",
    )
    parser.add_argument(
        "--sleeve",
        action="append",
        nargs=2,
        metavar=("MODEL_NAME", "RECORDS_PATH"),
        help="Add one sleeve model records path. Can be repeated.",
    )
    parser.add_argument(
        "--weights",
        default="0,0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated sleeve weights in decimal form.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/c_policy_sleeve_experiment",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--risk-match-to-core",
        action="store_true",
        help="Scale each blended sleeve series so annualized volatility matches the core-only portfolio.",
    )
    parser.add_argument(
        "--regime-gate",
        action="store_true",
        help="Apply a lagged white-box regime gate to the sleeve weight before blending.",
    )
    parser.add_argument("--regime-lookback-windows", type=int, default=12)
    parser.add_argument("--regime-drawdown-threshold", type=float, default=-0.10)
    parser.add_argument("--regime-warmup-multiplier", type=float, default=1.0)
    parser.add_argument("--regime-bull-multiplier", type=float, default=1.0)
    parser.add_argument("--regime-correction-multiplier", type=float, default=0.0)
    parser.add_argument("--regime-bear-multiplier", type=float, default=0.0)
    parser.add_argument("--regime-rebound-multiplier", type=float, default=0.5)
    return parser


def load_backtest_records(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["signal_date", "entry_date", "exit_date"],
    )
    required = {
        "signal_date",
        "entry_date",
        "exit_date",
        "gross_return",
        "net_return",
        "benchmark_return",
        "turnover",
        "cost_bps",
        "positions",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Records file '{path}' is missing required columns: {sorted(missing)}")
    return frame.sort_values("entry_date").reset_index(drop=True)


def align_core_and_sleeve_records(core_records: pd.DataFrame, sleeve_records: pd.DataFrame) -> pd.DataFrame:
    core = core_records.copy()
    sleeve = sleeve_records.copy()
    merged = core.merge(
        sleeve[
            [
                "entry_date",
                "signal_date",
                "exit_date",
                "gross_return",
                "net_return",
                "turnover",
                "cost_bps",
                "positions",
            ]
        ].rename(
            columns={
                "signal_date": "sleeve_signal_date",
                "exit_date": "sleeve_exit_date",
                "gross_return": "sleeve_gross_return",
                "net_return": "sleeve_net_return",
                "turnover": "sleeve_turnover",
                "cost_bps": "sleeve_cost_bps",
                "positions": "sleeve_positions",
            }
        ),
        on="entry_date",
        how="inner",
    )
    return merged.sort_values("entry_date").reset_index(drop=True)


def build_regime_detector_from_args(args: argparse.Namespace) -> BenchmarkTrendDrawdownRegimeDetector:
    return BenchmarkTrendDrawdownRegimeDetector(
        trend_lookback_windows=int(args.regime_lookback_windows),
        drawdown_threshold=float(args.regime_drawdown_threshold),
    )


def build_regime_gate_policy_from_args(args: argparse.Namespace) -> RegimeGatePolicy:
    return RegimeGatePolicy(
        multipliers={
            "warmup": float(args.regime_warmup_multiplier),
            "bull": float(args.regime_bull_multiplier),
            "correction": float(args.regime_correction_multiplier),
            "bear": float(args.regime_bear_multiplier),
            "rebound": float(args.regime_rebound_multiplier),
        }
    )


def _coerce_sleeve_weight_series(
    frame: pd.DataFrame,
    sleeve_weight: float | Sequence[float] | pd.Series | np.ndarray,
) -> pd.Series:
    if np.isscalar(sleeve_weight):
        value = float(sleeve_weight)
        if value < 0.0 or value > 1.0:
            raise ValueError("sleeve_weight must be between 0 and 1.")
        return pd.Series(np.full(len(frame), value, dtype=float), index=frame.index)

    weights = pd.Series(sleeve_weight, dtype=float)
    if len(weights) != len(frame):
        raise ValueError("Dynamic sleeve weight series must have the same length as aligned_records.")
    if ((weights < 0.0) | (weights > 1.0)).any():
        raise ValueError("Dynamic sleeve weights must stay between 0 and 1.")
    weights.index = frame.index
    return weights


def build_sleeve_blend_records(
    aligned_records: pd.DataFrame,
    *,
    sleeve_weight: float | Sequence[float] | pd.Series | np.ndarray,
) -> pd.DataFrame:
    weight_series = _coerce_sleeve_weight_series(aligned_records, sleeve_weight)
    core_weight_series = 1.0 - weight_series

    frame = aligned_records.copy()
    blended = pd.DataFrame(
        {
            "signal_date": frame["signal_date"],
            "entry_date": frame["entry_date"],
            "exit_date": frame["exit_date"],
            "gross_return": core_weight_series * frame["gross_return"].astype(float)
            + weight_series * frame["sleeve_gross_return"].astype(float),
            "net_return": core_weight_series * frame["net_return"].astype(float)
            + weight_series * frame["sleeve_net_return"].astype(float),
            "benchmark_return": frame["benchmark_return"].astype(float),
            "turnover": core_weight_series * frame["turnover"].astype(float)
            + weight_series * frame["sleeve_turnover"].astype(float),
            "cost_bps": core_weight_series * frame["cost_bps"].astype(float)
            + weight_series * frame["sleeve_cost_bps"].astype(float),
            "positions": core_weight_series * frame["positions"].astype(float)
            + weight_series * frame["sleeve_positions"].astype(float),
        }
    )
    return blended


def summarize_sleeve_blend(
    aligned_records: pd.DataFrame,
    *,
    sleeve_name: str,
    sleeve_weight: float,
    horizon: int,
) -> dict[str, object]:
    blended_records = build_sleeve_blend_records(aligned_records, sleeve_weight=sleeve_weight)
    summary = summarize_backtest_records(blended_records, horizon=horizon)
    summary["core_weight"] = 1.0 - float(sleeve_weight)
    summary["sleeve_weight"] = float(sleeve_weight)
    summary["sleeve_name"] = sleeve_name
    summary["correlation_to_core"] = float(
        aligned_records["net_return"].astype(float).corr(aligned_records["sleeve_net_return"].astype(float))
    )
    summary["sleeve_total_return"] = float((1.0 + aligned_records["sleeve_net_return"].astype(float)).prod() - 1.0)
    summary["excess_total_return_vs_core"] = (
        summary["total_return"] - float((1.0 + aligned_records["net_return"].astype(float)).prod() - 1.0)
    )
    return summary


def build_regime_gated_sleeve_blend_records(
    aligned_records: pd.DataFrame,
    *,
    sleeve_weight: float,
    detector: RegimeDetector,
    gate_policy: RegimeGatePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gated_alignment = apply_regime_gate(
        aligned_records,
        base_sleeve_weight=float(sleeve_weight),
        detector=detector,
        gate_policy=gate_policy,
        return_column="benchmark_return",
        date_column="entry_date",
    )
    blended_records = build_sleeve_blend_records(
        gated_alignment,
        sleeve_weight=gated_alignment["effective_sleeve_weight"],
    )
    return gated_alignment, blended_records


def summarize_regime_gated_sleeve_blend(
    aligned_records: pd.DataFrame,
    *,
    sleeve_name: str,
    sleeve_weight: float,
    horizon: int,
    detector: RegimeDetector,
    gate_policy: RegimeGatePolicy,
) -> tuple[dict[str, object], pd.DataFrame]:
    gated_alignment, blended_records = build_regime_gated_sleeve_blend_records(
        aligned_records,
        sleeve_weight=sleeve_weight,
        detector=detector,
        gate_policy=gate_policy,
    )
    summary = summarize_backtest_records(blended_records, horizon=horizon)
    summary["core_weight"] = 1.0 - float(sleeve_weight)
    summary["sleeve_weight"] = float(sleeve_weight)
    summary["sleeve_name"] = sleeve_name
    summary["correlation_to_core"] = float(
        aligned_records["net_return"].astype(float).corr(aligned_records["sleeve_net_return"].astype(float))
    )
    summary["sleeve_total_return"] = float((1.0 + aligned_records["sleeve_net_return"].astype(float)).prod() - 1.0)
    summary["excess_total_return_vs_core"] = (
        summary["total_return"] - float((1.0 + aligned_records["net_return"].astype(float)).prod() - 1.0)
    )
    summary["regime_gate_enabled"] = True
    summary["regime_lookback_windows"] = int(detector.trend_lookback_windows)
    summary["regime_drawdown_threshold"] = float(detector.drawdown_threshold)
    summary["effective_sleeve_weight_mean"] = float(gated_alignment["effective_sleeve_weight"].mean())
    summary["effective_sleeve_weight_min"] = float(gated_alignment["effective_sleeve_weight"].min())
    summary["effective_sleeve_weight_max"] = float(gated_alignment["effective_sleeve_weight"].max())
    regime_counts = gated_alignment["regime_label"].value_counts()
    for label in REGIME_LABELS:
        summary[f"regime_{label}_windows"] = int(regime_counts.get(label, 0))
        summary[f"regime_{label}_multiplier"] = gate_policy.multiplier_for(label)
    return summary, gated_alignment


def apply_risk_match(
    records: pd.DataFrame,
    *,
    target_annualized_volatility: float,
    horizon: int,
) -> tuple[pd.DataFrame, float, float]:
    summary = summarize_backtest_records(records, horizon=horizon)
    source_volatility = float(summary["annualized_volatility"])
    if not np.isfinite(source_volatility) or source_volatility <= 0.0:
        raise ValueError("Cannot risk-match a series with non-positive annualized volatility.")
    leverage_multiplier = float(target_annualized_volatility) / source_volatility

    scaled = records.copy()
    for column in ("gross_return", "net_return", "benchmark_return", "turnover", "cost_bps", "positions"):
        scaled[column] = scaled[column].astype(float) * leverage_multiplier
    return scaled, leverage_multiplier, source_volatility


def summarize_risk_matched_sleeve_blend(
    aligned_records: pd.DataFrame,
    *,
    sleeve_name: str,
    sleeve_weight: float,
    horizon: int,
    target_annualized_volatility: float,
) -> dict[str, object]:
    raw_summary = summarize_sleeve_blend(
        aligned_records,
        sleeve_name=sleeve_name,
        sleeve_weight=sleeve_weight,
        horizon=horizon,
    )
    raw_records = build_sleeve_blend_records(aligned_records, sleeve_weight=sleeve_weight)
    matched_records, leverage_multiplier, raw_volatility = apply_risk_match(
        raw_records,
        target_annualized_volatility=target_annualized_volatility,
        horizon=horizon,
    )
    matched_summary = summarize_backtest_records(matched_records, horizon=horizon)
    matched_summary.update(
        {
            "core_weight": 1.0 - float(sleeve_weight),
            "sleeve_weight": float(sleeve_weight),
            "sleeve_name": sleeve_name,
            "correlation_to_core": raw_summary["correlation_to_core"],
            "sleeve_total_return": raw_summary["sleeve_total_return"],
            "excess_total_return_vs_core": (
                matched_summary["total_return"] - float((1.0 + aligned_records["net_return"].astype(float)).prod() - 1.0)
            ),
            "risk_match_mode": "target_core_volatility",
            "target_annualized_volatility": float(target_annualized_volatility),
            "leverage_multiplier": leverage_multiplier,
            "raw_total_return": raw_summary["total_return"],
            "raw_annualized_return": raw_summary["annualized_return"],
            "raw_annualized_volatility": raw_volatility,
            "raw_sharpe": raw_summary["sharpe"],
            "raw_max_drawdown": raw_summary["max_drawdown"],
            "raw_mean_turnover": raw_summary["mean_turnover"],
            "raw_mean_cost_bps": raw_summary["mean_cost_bps"],
        }
    )
    return matched_summary


def parse_weights(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(item) for item in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    core_records = load_backtest_records(args.core_records)
    weights = parse_weights(args.weights)
    sleeve_specs = list(args.sleeve or [])
    if not sleeve_specs:
        raise ValueError("At least one --sleeve MODEL_NAME RECORDS_PATH pair is required.")
    target_core_summary = summarize_backtest_records(core_records, horizon=args.horizon)
    target_core_volatility = float(target_core_summary["annualized_volatility"])
    detector = build_regime_detector_from_args(args) if args.regime_gate else None
    gate_policy = build_regime_gate_policy_from_args(args) if args.regime_gate else None

    all_rows: list[dict[str, object]] = []
    for sleeve_name, sleeve_path in sleeve_specs:
        sleeve_records = load_backtest_records(sleeve_path)
        aligned = align_core_and_sleeve_records(core_records, sleeve_records)
        if aligned.empty:
            raise RuntimeError(f"No overlapping entry_date rows between core and sleeve '{sleeve_name}'.")

        sleeve_dir = output_root / sleeve_name
        sleeve_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, object]] = []
        for weight in weights:
            gated_alignment = None
            if args.regime_gate:
                assert detector is not None
                assert gate_policy is not None
                raw_summary, gated_alignment = summarize_regime_gated_sleeve_blend(
                    aligned,
                    sleeve_name=sleeve_name,
                    sleeve_weight=weight,
                    horizon=args.horizon,
                    detector=detector,
                    gate_policy=gate_policy,
                )
                if args.risk_match_to_core:
                    raw_records = build_sleeve_blend_records(
                        gated_alignment,
                        sleeve_weight=gated_alignment["effective_sleeve_weight"],
                    )
                    matched_records, leverage_multiplier, raw_volatility = apply_risk_match(
                        raw_records,
                        target_annualized_volatility=target_core_volatility,
                        horizon=args.horizon,
                    )
                    summary = summarize_backtest_records(matched_records, horizon=args.horizon)
                    summary.update(
                        {
                            "core_weight": raw_summary["core_weight"],
                            "sleeve_weight": raw_summary["sleeve_weight"],
                            "sleeve_name": raw_summary["sleeve_name"],
                            "correlation_to_core": raw_summary["correlation_to_core"],
                            "sleeve_total_return": raw_summary["sleeve_total_return"],
                            "regime_gate_enabled": raw_summary["regime_gate_enabled"],
                            "regime_lookback_windows": raw_summary["regime_lookback_windows"],
                            "regime_drawdown_threshold": raw_summary["regime_drawdown_threshold"],
                            "effective_sleeve_weight_mean": raw_summary["effective_sleeve_weight_mean"],
                            "effective_sleeve_weight_min": raw_summary["effective_sleeve_weight_min"],
                            "effective_sleeve_weight_max": raw_summary["effective_sleeve_weight_max"],
                            "risk_match_mode": "target_core_volatility",
                            "target_annualized_volatility": float(target_core_volatility),
                            "leverage_multiplier": leverage_multiplier,
                            "raw_total_return": raw_summary["total_return"],
                            "raw_annualized_return": raw_summary["annualized_return"],
                            "raw_annualized_volatility": raw_volatility,
                            "raw_sharpe": raw_summary["sharpe"],
                            "raw_max_drawdown": raw_summary["max_drawdown"],
                            "raw_mean_turnover": raw_summary["mean_turnover"],
                            "raw_mean_cost_bps": raw_summary["mean_cost_bps"],
                            "excess_total_return_vs_core": (
                                summary["total_return"]
                                - float((1.0 + aligned["net_return"].astype(float)).prod() - 1.0)
                            ),
                        }
                    )
                    for label in REGIME_LABELS:
                        summary[f"regime_{label}_windows"] = raw_summary[f"regime_{label}_windows"]
                        summary[f"regime_{label}_multiplier"] = raw_summary[f"regime_{label}_multiplier"]
                else:
                    summary = raw_summary
            else:
                if args.risk_match_to_core:
                    summary = summarize_risk_matched_sleeve_blend(
                        aligned,
                        sleeve_name=sleeve_name,
                        sleeve_weight=weight,
                        horizon=args.horizon,
                        target_annualized_volatility=target_core_volatility,
                    )
                else:
                    summary = summarize_sleeve_blend(
                        aligned,
                        sleeve_name=sleeve_name,
                        sleeve_weight=weight,
                        horizon=args.horizon,
                    )
            summary_rows.append(summary)
            if gated_alignment is not None:
                gated_alignment.to_csv(
                    sleeve_dir / f"aligned_records_weight_{str(weight).replace('.', 'p')}.csv",
                    index=False,
                )

        if not args.regime_gate:
            aligned.to_csv(sleeve_dir / "aligned_records.csv", index=False)

        sleeve_summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
        sleeve_summary.to_csv(sleeve_dir / "summary_metrics.csv", index=False)
        all_rows.extend(sleeve_summary.to_dict(orient="records"))

    combined = pd.DataFrame(all_rows).sort_values(
        ["sharpe", "annualized_return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    combined.to_csv(output_root / "summary_metrics.csv", index=False)

    payload = {
        "ok": True,
        "core_records": str(args.core_records),
        "output_root": str(output_root),
        "weights": weights,
        "risk_match_to_core": bool(args.risk_match_to_core),
        "regime_gate": bool(args.regime_gate),
        "target_core_annualized_volatility": target_core_volatility,
        "summary_path": str(output_root / "summary_metrics.csv"),
        "sleeves": [name for name, _ in sleeve_specs],
    }
    if args.regime_gate:
        payload["regime_detector"] = {
            "trend_lookback_windows": int(args.regime_lookback_windows),
            "drawdown_threshold": float(args.regime_drawdown_threshold),
        }
        payload["regime_gate_policy"] = {
            "warmup": float(args.regime_warmup_multiplier),
            "bull": float(args.regime_bull_multiplier),
            "correction": float(args.regime_correction_multiplier),
            "bear": float(args.regime_bear_multiplier),
            "rebound": float(args.regime_rebound_multiplier),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
