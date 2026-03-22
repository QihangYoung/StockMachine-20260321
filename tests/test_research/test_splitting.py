from __future__ import annotations

import pandas as pd
import pytest

from stockmachine.research import DEFAULT_RESEARCH_PROTOCOL
from stockmachine.research.splitting import WalkForwardSplitConfig, build_walk_forward_splits


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", "2025-08-29", freq="B")
    symbols = ("AAPL", "MSFT", "NVDA")
    rows: list[dict[str, object]] = []

    for day_index, current_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "feature_a": float(day_index + symbol_index) / 10.0,
                    "feature_b": float(day_index * (symbol_index + 1)) / 25.0,
                    "target": float(day_index - symbol_index) / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_splitter_builds_monthly_folds_with_purge_and_embargo() -> None:
    frame = _synthetic_panel()
    config = WalkForwardSplitConfig(
        train_window_days=30,
        validation_window_days=10,
        test_window_days=5,
        purge_window_days=2,
        embargo_window_days=3,
    )

    splits = build_walk_forward_splits(frame, config=config)

    assert len(splits) >= 3
    assert [split.fold_index for split in splits] == list(range(len(splits)))

    unique_dates = pd.Index(pd.to_datetime(frame["date"]).drop_duplicates().sort_values())
    date_positions = {date: position for position, date in enumerate(unique_dates)}

    for split in splits:
        assert set(split.train_frame.columns) == set(frame.columns)
        assert set(split.validation_frame.columns) == set(frame.columns)
        assert set(split.test_frame.columns) == set(frame.columns)

        assert split.train_frame["date"].nunique() == config.train_window_days
        assert split.validation_frame["date"].nunique() == config.validation_window_days
        assert split.test_frame["date"].nunique() == config.test_window_days

        train_end_idx = date_positions[split.train_end]
        validation_start_idx = date_positions[split.validation_start]
        validation_end_idx = date_positions[split.validation_end]
        test_start_idx = date_positions[split.test_start]
        test_end_idx = date_positions[split.test_end]
        embargo_until_idx = date_positions[split.embargo_until]

        assert train_end_idx < validation_start_idx < validation_end_idx < test_start_idx < test_end_idx
        assert validation_start_idx == train_end_idx + config.purge_window_days + 1
        assert test_start_idx == validation_end_idx + config.purge_window_days + 1
        assert embargo_until_idx == min(test_end_idx + config.embargo_window_days, len(unique_dates) - 1)

        split_meta = split.as_dict()
        assert split_meta["fold_index"] == split.fold_index
        assert split_meta["train_rows"] == len(split.train_frame)
        assert split_meta["validation_rows"] == len(split.validation_frame)
        assert split_meta["test_rows"] == len(split.test_frame)


def test_walk_forward_splitter_rejects_non_monthly_roll() -> None:
    frame = _synthetic_panel()

    with pytest.raises(ValueError, match="monthly"):
        build_walk_forward_splits(
            frame,
            config=WalkForwardSplitConfig(roll_frequency="weekly"),
        )


@pytest.mark.parametrize(
    ("config_kwargs", "error_message"),
    [
        ({"train_window_days": 0}, "train_window_days"),
        ({"validation_window_days": -1}, "validation_window_days"),
        ({"test_window_days": 0}, "test_window_days"),
        ({"purge_window_days": -1}, "purge_window_days"),
        ({"embargo_window_days": -1}, "embargo_window_days"),
    ],
)
def test_walk_forward_splitter_rejects_invalid_window_sizes(
    config_kwargs: dict[str, int],
    error_message: str,
) -> None:
    frame = _synthetic_panel()

    with pytest.raises(ValueError, match=error_message):
        build_walk_forward_splits(frame, config=WalkForwardSplitConfig(**config_kwargs))


def test_walk_forward_splitter_rejects_missing_date_column() -> None:
    frame = _synthetic_panel().drop(columns=["date"])

    with pytest.raises(KeyError, match="date"):
        build_walk_forward_splits(frame)


def test_walk_forward_splitter_defaults_follow_shared_protocol() -> None:
    config = WalkForwardSplitConfig()

    assert config.train_window_days == DEFAULT_RESEARCH_PROTOCOL.walk_forward.train_window_months * 21
    assert config.validation_window_days == DEFAULT_RESEARCH_PROTOCOL.walk_forward.validation_window_months * 21
    assert config.test_window_days == DEFAULT_RESEARCH_PROTOCOL.walk_forward.test_window_months * 21
    assert config.purge_window_days == DEFAULT_RESEARCH_PROTOCOL.walk_forward.purge_window_sessions
    assert config.embargo_window_days == DEFAULT_RESEARCH_PROTOCOL.walk_forward.embargo_window_sessions
    assert config.roll_frequency == DEFAULT_RESEARCH_PROTOCOL.walk_forward.roll_frequency
