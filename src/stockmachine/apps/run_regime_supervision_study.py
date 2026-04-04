from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from stockmachine.apps.run_c_policy_sleeve_experiment import load_backtest_records
from stockmachine.risk import (
    BenchmarkForwardRegimeLabeler,
    BenchmarkTrendDrawdownRegimeDetector,
    BenchmarkTrendDrawdownVolRegimeDetector,
    BenchmarkTrendDrawdownVolCrossAssetRegimeDetector,
    build_regime_confusion_matrix,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Study benchmark-centric regime supervision independent of any strategy consumer."
    )
    parser.add_argument(
        "--core-records",
        default="artifacts/c_risk_tuning_regime_window_20260330/benchmark_c_policy_records_common404.csv",
    )
    parser.add_argument(
        "--cross-asset-records",
        default="artifacts/c_risk_tuning_regime_window_20260330/proxy_asset_window_returns_common404.csv",
    )
    parser.add_argument("--output-root", default="artifacts/regime_supervision_study")
    parser.add_argument(
        "--detector-kind",
        choices=("trend_drawdown", "trend_drawdown_vol", "trend_drawdown_vol_cross_asset"),
        default="trend_drawdown",
    )
    parser.add_argument("--detector-lookback-windows", type=int, default=12)
    parser.add_argument("--detector-drawdown-threshold", type=float, default=-0.10)
    parser.add_argument("--detector-vol-lookback-windows", type=int, default=12)
    parser.add_argument("--detector-high-vol-annualized-threshold", type=float, default=0.18)
    parser.add_argument("--detector-cross-asset-lookback-windows", type=int, default=12)
    parser.add_argument("--detector-cross-asset-threshold", type=float, default=0.0)
    parser.add_argument("--horizon-sessions", type=int, default=5)
    parser.add_argument("--forward-windows", type=int, default=12)
    parser.add_argument("--forward-drawdown-threshold", type=float, default=-0.10)
    return parser


def _summarize_by_label(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    labeled = frame.loc[frame[label_column].notna()].copy()
    grouped = labeled.groupby(label_column, dropna=False)
    rows: list[dict[str, object]] = []
    for label, group in grouped:
        rows.append(
            {
                label_column: label,
                "windows": int(len(group)),
                "mean_forward_total_return": float(group["forward_regime_total_return"].mean()),
                "median_forward_total_return": float(group["forward_regime_total_return"].median()),
                "mean_forward_max_drawdown": float(group["forward_regime_max_drawdown"].mean()),
                "median_forward_max_drawdown": float(group["forward_regime_max_drawdown"].median()),
                "mean_forward_positive_rate": float(group["forward_regime_positive_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(label_column).reset_index(drop=True)


def _normalize_confusion_matrix(confusion: pd.DataFrame) -> pd.DataFrame:
    normalized = confusion.div(confusion.sum(axis=1).replace(0, pd.NA), axis=0)
    return normalized.fillna(0.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    core_records = load_backtest_records(args.core_records)
    detector_frame = core_records.copy()
    if args.detector_kind == "trend_drawdown_vol_cross_asset":
        cross_asset_records = pd.read_csv(
            args.cross_asset_records,
            parse_dates=["signal_date", "entry_date", "exit_date"],
        )
        keep_columns = [
            "entry_date",
            "spy_ret",
            "vxus_ret",
            "agg_ret",
            "cta_ret",
            "sgov_ret",
            "gldm_ret",
        ]
        missing = [column for column in keep_columns if column not in cross_asset_records.columns]
        if missing:
            raise KeyError(f"Cross-asset records are missing required columns: {missing}")
        detector_frame = detector_frame.merge(
            cross_asset_records[keep_columns],
            on="entry_date",
            how="left",
            validate="one_to_one",
        )
        detector = BenchmarkTrendDrawdownVolCrossAssetRegimeDetector(
            trend_lookback_windows=int(args.detector_lookback_windows),
            drawdown_threshold=float(args.detector_drawdown_threshold),
            vol_lookback_windows=int(args.detector_vol_lookback_windows),
            high_vol_annualized_threshold=float(args.detector_high_vol_annualized_threshold),
            horizon_sessions=int(args.horizon_sessions),
            cross_asset_lookback_windows=int(args.detector_cross_asset_lookback_windows),
            cross_asset_risk_on_threshold=float(args.detector_cross_asset_threshold),
        )
    elif args.detector_kind == "trend_drawdown_vol":
        detector = BenchmarkTrendDrawdownVolRegimeDetector(
            trend_lookback_windows=int(args.detector_lookback_windows),
            drawdown_threshold=float(args.detector_drawdown_threshold),
            vol_lookback_windows=int(args.detector_vol_lookback_windows),
            high_vol_annualized_threshold=float(args.detector_high_vol_annualized_threshold),
            horizon_sessions=int(args.horizon_sessions),
        )
    else:
        detector = BenchmarkTrendDrawdownRegimeDetector(
            trend_lookback_windows=int(args.detector_lookback_windows),
            drawdown_threshold=float(args.detector_drawdown_threshold),
        )
    labeler = BenchmarkForwardRegimeLabeler(
        forward_windows=int(args.forward_windows),
        drawdown_threshold=float(args.forward_drawdown_threshold),
    )

    detected = detector.label_frame(detector_frame, return_column="benchmark_return", date_column="entry_date")
    forward = labeler.label_frame(core_records, return_column="benchmark_return", date_column="entry_date")

    combined = detector_frame.copy()
    combined = combined.merge(detected, on="entry_date", how="left")
    combined = combined.merge(forward, on="entry_date", how="left")
    combined.to_csv(output_root / "labeled_records.csv", index=False)

    evaluable = combined.loc[
        (combined["regime_label"] != "warmup") & (combined["forward_regime_label"] != "unlabeled")
    ].reset_index(drop=True)
    evaluable.to_csv(output_root / "evaluable_records.csv", index=False)

    confusion = build_regime_confusion_matrix(
        evaluable,
        predicted_column="regime_label",
        target_column="forward_regime_label",
    )
    confusion.to_csv(output_root / "confusion_matrix.csv")
    _normalize_confusion_matrix(confusion).to_csv(output_root / "confusion_matrix_row_normalized.csv")

    detected_summary = _summarize_by_label(evaluable, "regime_label")
    detected_summary.to_csv(output_root / "future_outcomes_by_detected_regime.csv", index=False)
    forward_summary = _summarize_by_label(evaluable, "forward_regime_label")
    forward_summary.to_csv(output_root / "future_outcomes_by_forward_regime.csv", index=False)

    payload = {
        "ok": True,
        "core_records": str(args.core_records),
        "cross_asset_records": str(args.cross_asset_records),
        "output_root": str(output_root),
        "rows_total": int(len(combined)),
        "rows_evaluable": int(len(evaluable)),
        "detector": {
            "kind": str(args.detector_kind),
            "trend_lookback_windows": int(args.detector_lookback_windows),
            "drawdown_threshold": float(args.detector_drawdown_threshold),
        },
        "forward_labeler": {
            "forward_windows": int(args.forward_windows),
            "drawdown_threshold": float(args.forward_drawdown_threshold),
        },
        "artifacts": {
            "labeled_records": str(output_root / "labeled_records.csv"),
            "evaluable_records": str(output_root / "evaluable_records.csv"),
            "confusion_matrix": str(output_root / "confusion_matrix.csv"),
            "confusion_matrix_row_normalized": str(output_root / "confusion_matrix_row_normalized.csv"),
            "future_outcomes_by_detected_regime": str(output_root / "future_outcomes_by_detected_regime.csv"),
            "future_outcomes_by_forward_regime": str(output_root / "future_outcomes_by_forward_regime.csv"),
        },
    }
    if args.detector_kind == "trend_drawdown_vol":
        payload["detector"]["vol_lookback_windows"] = int(args.detector_vol_lookback_windows)
        payload["detector"]["high_vol_annualized_threshold"] = float(args.detector_high_vol_annualized_threshold)
        payload["detector"]["horizon_sessions"] = int(args.horizon_sessions)
    if args.detector_kind == "trend_drawdown_vol_cross_asset":
        payload["detector"]["vol_lookback_windows"] = int(args.detector_vol_lookback_windows)
        payload["detector"]["high_vol_annualized_threshold"] = float(args.detector_high_vol_annualized_threshold)
        payload["detector"]["cross_asset_lookback_windows"] = int(args.detector_cross_asset_lookback_windows)
        payload["detector"]["cross_asset_threshold"] = float(args.detector_cross_asset_threshold)
        payload["detector"]["horizon_sessions"] = int(args.horizon_sessions)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
