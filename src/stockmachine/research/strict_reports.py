from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def write_csv_artifact(path: str | Path, frame: pd.DataFrame) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artifact_path, index=False)
    return artifact_path


def write_json_artifact(path: str | Path, payload: Mapping[str, Any] | Sequence[Any] | Any) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return artifact_path


def write_summary_metrics_artifact(
    path: str | Path,
    rows_or_frame: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    ordered_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = rows_or_frame.copy() if isinstance(rows_or_frame, pd.DataFrame) else pd.DataFrame(list(rows_or_frame))
    if ordered_columns is not None:
        for column in ordered_columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[[*ordered_columns, *[column for column in frame.columns if column not in ordered_columns]]]
    write_csv_artifact(path, frame)
    return frame


def write_model_backtest_artifacts(
    output_dir: str | Path,
    *,
    model_name: str,
    summary_row: Mapping[str, Any],
    records: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    record_frame = records.copy()
    if record_frame.empty:
        record_frame = pd.DataFrame(
            columns=[
                "signal_date",
                "entry_date",
                "exit_date",
                "gross_return",
                "net_return",
                "benchmark_return",
                "turnover",
                "cost_bps",
                "positions",
            ]
        )
    summary_frame = pd.DataFrame([{**summary_row, "model": model_name}])
    records_path = write_csv_artifact(root / "backtest_records.csv", record_frame)
    summary_path = write_csv_artifact(root / "backtest_summary.csv", summary_frame)
    predictions_path = None
    if predictions is not None:
        predictions_path = write_csv_artifact(root / "predictions.csv", predictions.copy())
    return {
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path) if predictions_path is not None else "",
    }
