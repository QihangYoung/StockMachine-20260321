from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockmachine.research.protocols import get_default_research_protocol


_DEFAULT_PROTOCOL = get_default_research_protocol()
_TRADING_DAYS_PER_MONTH = 21


@dataclass(slots=True, frozen=True)
class WalkForwardSplitConfig:
    """Configuration for rolling train/validation/test research splits."""

    train_window_days: int = _DEFAULT_PROTOCOL.walk_forward.train_window_months * _TRADING_DAYS_PER_MONTH
    validation_window_days: int = _DEFAULT_PROTOCOL.walk_forward.validation_window_months * _TRADING_DAYS_PER_MONTH
    test_window_days: int = _DEFAULT_PROTOCOL.walk_forward.test_window_months * _TRADING_DAYS_PER_MONTH
    purge_window_days: int = _DEFAULT_PROTOCOL.walk_forward.purge_window_sessions
    embargo_window_days: int = _DEFAULT_PROTOCOL.walk_forward.embargo_window_sessions
    roll_frequency: str = _DEFAULT_PROTOCOL.walk_forward.roll_frequency
    date_column: str = "date"


@dataclass(slots=True, frozen=True)
class WalkForwardSplit:
    """One walk-forward fold with slice-ready research frames."""

    fold_index: int
    anchor_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_until: pd.Timestamp
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly metadata snapshot for orchestration/reporting."""

        return {
            "fold_index": self.fold_index,
            "anchor_date": self.anchor_date.isoformat(),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "embargo_until": self.embargo_until.isoformat(),
            "train_rows": len(self.train_frame),
            "validation_rows": len(self.validation_frame),
            "test_rows": len(self.test_frame),
        }


def build_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    config: WalkForwardSplitConfig | None = None,
) -> tuple[WalkForwardSplit, ...]:
    """Build rolling walk-forward splits with purge and embargo windows.

    The splitter works on trading dates rather than calendar days. It uses the
    first available trading date of each month as the fold anchor when
    ``roll_frequency`` is ``"monthly"``.
    """

    if config is None:
        config = WalkForwardSplitConfig()
    _validate_config(config)
    if frame.empty:
        return ()
    if config.date_column not in frame.columns:
        raise KeyError(f"Frame is missing required date column '{config.date_column}'.")

    prepared = _prepare_frame(frame, date_column=config.date_column)
    unique_dates = _unique_dates(prepared, date_column=config.date_column)
    anchor_positions = _monthly_anchor_positions(unique_dates, roll_frequency=config.roll_frequency)

    splits: list[WalkForwardSplit] = []
    for anchor_position in anchor_positions:
        split = _build_split(
            prepared,
            unique_dates=unique_dates,
            anchor_position=anchor_position,
            fold_index=len(splits),
            config=config,
        )
        if split is not None:
            splits.append(split)
    return tuple(splits)


def _build_split(
    frame: pd.DataFrame,
    *,
    unique_dates: pd.Index,
    anchor_position: int,
    fold_index: int,
    config: WalkForwardSplitConfig,
) -> WalkForwardSplit | None:
    test_start_idx = anchor_position
    validation_end_idx = test_start_idx - config.purge_window_days - 1
    validation_start_idx = validation_end_idx - config.validation_window_days + 1
    train_end_idx = validation_start_idx - config.purge_window_days - 1
    train_start_idx = train_end_idx - config.train_window_days + 1
    test_end_idx = test_start_idx + config.test_window_days - 1

    if (
        train_start_idx < 0
        or validation_start_idx < 0
        or validation_end_idx < 0
        or test_end_idx >= len(unique_dates)
        or train_end_idx < train_start_idx
        or validation_end_idx < validation_start_idx
        or test_end_idx < test_start_idx
    ):
        return None

    train_start = unique_dates[train_start_idx]
    train_end = unique_dates[train_end_idx]
    validation_start = unique_dates[validation_start_idx]
    validation_end = unique_dates[validation_end_idx]
    test_start = unique_dates[test_start_idx]
    test_end = unique_dates[test_end_idx]
    embargo_until_idx = min(test_end_idx + config.embargo_window_days, len(unique_dates) - 1)
    embargo_until = unique_dates[embargo_until_idx]

    train_frame = _slice_frame(frame, config.date_column, train_start, train_end)
    validation_frame = _slice_frame(frame, config.date_column, validation_start, validation_end)
    test_frame = _slice_frame(frame, config.date_column, test_start, test_end)
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        return None

    return WalkForwardSplit(
        fold_index=fold_index,
        anchor_date=test_start,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        embargo_until=embargo_until,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
    )


def _prepare_frame(frame: pd.DataFrame, *, date_column: str) -> pd.DataFrame:
    sort_columns = [date_column]
    if "symbol" in frame.columns:
        sort_columns.append("symbol")
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _unique_dates(frame: pd.DataFrame, *, date_column: str) -> pd.Index:
    dates = _normalize_datetime_series(frame[date_column])
    return pd.Index(dates.drop_duplicates().sort_values())


def _monthly_anchor_positions(unique_dates: pd.Index, *, roll_frequency: str) -> list[int]:
    if roll_frequency != "monthly":
        raise ValueError("walk-forward splitter currently supports roll_frequency='monthly' only.")
    periods = unique_dates.to_period("M")
    anchors: list[int] = []
    seen_periods: set[pd.Period] = set()
    for position, period in enumerate(periods):
        if period in seen_periods:
            continue
        seen_periods.add(period)
        anchors.append(position)
    return anchors


def _slice_frame(frame: pd.DataFrame, date_column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = _normalize_datetime_series(frame[date_column])
    mask = (dates >= start) & (dates <= end)
    return frame.loc[mask].copy()


def _normalize_datetime_series(values: pd.Series) -> pd.Series:
    series = pd.to_datetime(values, utc=False)
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_convert(None)
    return series


def _validate_config(config: WalkForwardSplitConfig) -> None:
    for field_name in ("train_window_days", "validation_window_days", "test_window_days"):
        value = getattr(config, field_name)
        if value <= 0:
            raise ValueError(f"{field_name} must be positive; got {value}.")
    for field_name in ("purge_window_days", "embargo_window_days"):
        value = getattr(config, field_name)
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative; got {value}.")
    if config.roll_frequency != "monthly":
        raise ValueError("walk-forward splitter currently supports roll_frequency='monthly' only.")
