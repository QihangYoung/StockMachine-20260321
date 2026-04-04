from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from stockmachine.research.p1_rigor import summarize_backtest_records


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


def build_sleeve_blend_records(
    aligned_records: pd.DataFrame,
    *,
    sleeve_weight: float,
) -> pd.DataFrame:
    weight = float(sleeve_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("sleeve_weight must be between 0 and 1.")
    core_weight = 1.0 - weight

    frame = aligned_records.copy()
    blended = pd.DataFrame(
        {
            "signal_date": frame["signal_date"],
            "entry_date": frame["entry_date"],
            "exit_date": frame["exit_date"],
            "gross_return": core_weight * frame["gross_return"].astype(float)
            + weight * frame["sleeve_gross_return"].astype(float),
            "net_return": core_weight * frame["net_return"].astype(float)
            + weight * frame["sleeve_net_return"].astype(float),
            "benchmark_return": frame["benchmark_return"].astype(float),
            "turnover": core_weight * frame["turnover"].astype(float)
            + weight * frame["sleeve_turnover"].astype(float),
            "cost_bps": core_weight * frame["cost_bps"].astype(float)
            + weight * frame["sleeve_cost_bps"].astype(float),
            "positions": core_weight * frame["positions"].astype(float)
            + weight * frame["sleeve_positions"].astype(float),
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

    all_rows: list[dict[str, object]] = []
    for sleeve_name, sleeve_path in sleeve_specs:
        sleeve_records = load_backtest_records(sleeve_path)
        aligned = align_core_and_sleeve_records(core_records, sleeve_records)
        if aligned.empty:
            raise RuntimeError(f"No overlapping entry_date rows between core and sleeve '{sleeve_name}'.")

        sleeve_dir = output_root / sleeve_name
        sleeve_dir.mkdir(parents=True, exist_ok=True)
        aligned.to_csv(sleeve_dir / "aligned_records.csv", index=False)

        summary_rows: list[dict[str, object]] = []
        for weight in weights:
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
        "target_core_annualized_volatility": target_core_volatility,
        "summary_path": str(output_root / "summary_metrics.csv"),
        "sleeves": [name for name, _ in sleeve_specs],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
