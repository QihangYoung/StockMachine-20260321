from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REQUIRED_BACKTEST_RECORD_COLUMNS: tuple[str, ...] = (
    "entry_date",
    "exit_date",
    "net_return",
    "benchmark_return",
    "turnover",
    "cost_bps",
    "positions",
)


@dataclass(slots=True, frozen=True)
class RobustnessArtifactBundle:
    """Standard artifact contract consumed by robustness analyzers."""

    strategy_project: str
    model_name: str
    records: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame | None = None
    protocol_meta: Mapping[str, Any] | None = None
    parameter_manifest: pd.DataFrame | None = None


def ensure_required_backtest_record_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_BACKTEST_RECORD_COLUMNS if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Backtest records are missing required columns: {joined}")
    normalized = frame.copy()
    normalized["entry_date"] = pd.to_datetime(normalized["entry_date"], errors="coerce")
    normalized["exit_date"] = pd.to_datetime(normalized["exit_date"], errors="coerce")
    return normalized


def build_robustness_artifact_bundle(
    *,
    strategy_project: str,
    model_name: str,
    records: pd.DataFrame,
    summary: pd.DataFrame | Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame | None = None,
    protocol_meta: Mapping[str, Any] | None = None,
    parameter_manifest: pd.DataFrame | None = None,
) -> RobustnessArtifactBundle:
    validated_records = ensure_required_backtest_record_columns(records)
    if isinstance(summary, pd.DataFrame):
        summary_frame = summary.copy()
    else:
        summary_frame = pd.DataFrame(list(summary))
    if summary_frame.empty:
        summary_frame = pd.DataFrame([{"model": model_name}])
    if "model" not in summary_frame.columns:
        summary_frame["model"] = model_name
    return RobustnessArtifactBundle(
        strategy_project=str(strategy_project),
        model_name=str(model_name),
        records=validated_records,
        summary=summary_frame,
        predictions=predictions.copy() if predictions is not None else None,
        protocol_meta=dict(protocol_meta) if protocol_meta is not None else None,
        parameter_manifest=parameter_manifest.copy() if parameter_manifest is not None else None,
    )


def load_backtest_records_artifact(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    return ensure_required_backtest_record_columns(frame)

