from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Mapping, Sequence

import pandas as pd


@dataclass(slots=True, frozen=True)
class ComparisonWindow:
    """A named date window used to align baseline comparisons."""

    name: str
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    include_start: bool = True
    include_end: bool = False


@dataclass(slots=True, frozen=True)
class ComparisonAlignment:
    """Shared metadata for a group of comparable frames."""

    frame_names: tuple[str, ...]
    shared_columns: tuple[str, ...]
    shared_dates: tuple[pd.Timestamp, ...]
    row_counts: tuple[tuple[str, int], ...]


def build_yearly_holdout_windows(
    *,
    validation_year: int,
    test_start_year: int | None = None,
    test_end_year: int | None = None,
) -> tuple[ComparisonWindow, ComparisonWindow]:
    """Build the default validation/test split used by the research baselines."""

    validation_window = ComparisonWindow(
        name="validation",
        start=f"{validation_year}-01-01",
        end=f"{validation_year + 1}-01-01",
    )
    test_window = ComparisonWindow(
        name="test",
        start=f"{test_start_year or (validation_year + 1)}-01-01",
        end=f"{test_end_year + 1}-01-01" if test_end_year is not None else None,
    )
    return validation_window, test_window


def slice_frame_by_windows(
    frame: pd.DataFrame,
    windows: Sequence[ComparisonWindow],
    *,
    date_column: str = "date",
) -> dict[str, pd.DataFrame]:
    """Slice a frame into named, date-bounded windows."""

    if date_column not in frame.columns:
        raise KeyError(f"Frame is missing required date column '{date_column}'.")

    dates = _normalize_datetime_series(frame[date_column])
    sliced_frames: dict[str, pd.DataFrame] = {}
    for window in windows:
        start = _normalize_timestamp(window.start)
        end = _normalize_timestamp(window.end)

        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= dates >= start if window.include_start else dates > start
        if end is not None:
            mask &= dates <= end if window.include_end else dates < end
        sliced_frames[window.name] = frame.loc[mask].copy()
    return sliced_frames


def validate_aligned_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    date_column: str = "date",
    required_columns: Sequence[str] = (),
) -> ComparisonAlignment:
    """Ensure multiple research frames can be compared under the same constraints."""

    if not frames:
        raise ValueError("At least one frame is required for alignment validation.")

    required = tuple(dict.fromkeys((date_column, *required_columns)))
    frame_names = tuple(frames.keys())
    reference_name = frame_names[0]
    reference_frame = frames[reference_name]
    reference_columns = tuple(reference_frame.columns)
    reference_dates = tuple(_normalized_unique_dates(reference_frame, date_column=date_column))

    row_counts: list[tuple[str, int]] = []
    for name, frame in frames.items():
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"Frame '{name}' is missing required columns: {missing}.")
        if tuple(frame.columns) != reference_columns:
            raise ValueError(
                f"Frame '{name}' does not match the reference schema for '{reference_name}'."
            )
        dates = tuple(_normalized_unique_dates(frame, date_column=date_column))
        if dates != reference_dates:
            raise ValueError(f"Frame '{name}' is not aligned with '{reference_name}' on {date_column}.")
        row_counts.append((name, int(len(frame))))

    return ComparisonAlignment(
        frame_names=frame_names,
        shared_columns=reference_columns,
        shared_dates=reference_dates,
        row_counts=tuple(row_counts),
    )


def comparison_summary_frame(evaluation: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Convert nested model/period results into a stable comparison table."""

    rows: list[dict[str, object]] = []
    for model_name, periods in evaluation.items():
        for period_name, metrics in periods.items():
            if is_dataclass(metrics):
                row = asdict(metrics)
            elif isinstance(metrics, Mapping):
                row = dict(metrics)
            else:
                raise TypeError(
                    "Comparison summary values must be dataclass instances or mappings."
                )
            row["model"] = model_name
            row["period"] = period_name
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["model", "period"])
    summary = pd.DataFrame(rows)
    columns = ["model", "period", *[column for column in summary.columns if column not in {"model", "period"}]]
    return summary[columns]


def _normalize_datetime_series(values: pd.Series) -> pd.Series:
    series = pd.to_datetime(values, utc=False)
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_convert(None)
    return series


def _normalize_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def _normalized_unique_dates(frame: pd.DataFrame, *, date_column: str) -> pd.Index:
    dates = _normalize_datetime_series(frame[date_column])
    return pd.Index(dates.drop_duplicates().sort_values())
