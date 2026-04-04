from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from stockmachine.research.strict_reports import write_csv_artifact, write_json_artifact


def write_robustness_overview_artifact(
    path: str | Path,
    rows_or_frame: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frame = rows_or_frame.copy() if isinstance(rows_or_frame, pd.DataFrame) else pd.DataFrame(list(rows_or_frame))
    write_csv_artifact(path, frame)
    return frame


def write_robustness_table_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    table = frame.copy()
    write_csv_artifact(path, table)
    return table


def write_time_stability_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_tail_dependence_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_parameter_stability_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_cost_execution_stress_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_turnover_concentration_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_universe_stability_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_selection_bias_artifact(
    path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return write_robustness_table_artifact(path, frame)


def write_robustness_gate_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    return write_json_artifact(path, payload)
