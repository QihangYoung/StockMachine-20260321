from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockmachine.domain import get_table_spec
from stockmachine.ingestion.storage import StorageLayout


def load_silver_table(
    table_name: str,
    *,
    layout: StorageLayout | None = None,
) -> pd.DataFrame:
    """Load every JSONL file for one silver table into a DataFrame."""

    storage = layout or StorageLayout()
    table_dir = storage.silver_table_dir(table_name)
    if not table_dir.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(table_dir.glob("*.jsonl")):
        frame = _read_jsonl(path)
        if not frame.empty:
            frames.append(frame.dropna(axis="columns", how="all"))

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return _dedupe_table_rows(table_name, combined)


def load_symbol_master_latest(*, layout: StorageLayout | None = None) -> pd.DataFrame:
    """Load the latest point-in-time symbol master snapshot."""

    frame = load_silver_table("symbol_master", layout=layout)
    if frame.empty:
        return frame

    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    latest_date = frame["as_of_date"].max()
    return frame.loc[frame["as_of_date"] == latest_date].copy().reset_index(drop=True)


def load_us_equities_dataset(*, layout: StorageLayout | None = None) -> dict[str, pd.DataFrame]:
    """Load the minimum silver tables required by the US equities backtest."""

    storage = layout or StorageLayout()
    universe_membership = load_silver_table("universe_membership", layout=storage)
    daily_bar = load_silver_table("daily_bar", layout=storage)
    adj_factor = load_silver_table("adj_factor", layout=storage)
    benchmark_index = load_silver_table("benchmark_index", layout=storage)
    industry_membership = load_silver_table("industry_membership", layout=storage)
    symbol_master = load_silver_table("symbol_master", layout=storage)

    for frame, date_column in (
        (universe_membership, "session_date"),
        (daily_bar, "session_date"),
        (adj_factor, "session_date"),
        (benchmark_index, "session_date"),
        (industry_membership, "as_of_date"),
        (symbol_master, "as_of_date"),
    ):
        if not frame.empty and date_column in frame.columns:
            frame[date_column] = pd.to_datetime(frame[date_column])

    return {
        "universe_membership": universe_membership,
        "daily_bar": daily_bar,
        "adj_factor": adj_factor,
        "benchmark_index": benchmark_index,
        "industry_membership": industry_membership,
        "symbol_master": symbol_master,
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _dedupe_table_rows(table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    try:
        primary_key = list(get_table_spec(table_name).primary_key)
    except KeyError:
        return frame.reset_index(drop=True)

    if not all(column in frame.columns for column in primary_key):
        return frame.reset_index(drop=True)

    deduped = frame.copy()
    sort_columns = []
    if "effective_time_utc" in deduped.columns:
        deduped["_effective_sort"] = pd.to_datetime(
            deduped["effective_time_utc"],
            errors="coerce",
            utc=True,
        )
        sort_columns.append("_effective_sort")
    if "load_time_utc" in deduped.columns:
        deduped["_load_sort"] = pd.to_datetime(
            deduped["load_time_utc"],
            errors="coerce",
            utc=True,
        )
        sort_columns.append("_load_sort")

    if sort_columns:
        deduped = deduped.sort_values(
            primary_key + sort_columns,
            kind="stable",
            na_position="first",
        )

    deduped = deduped.drop_duplicates(subset=primary_key, keep="last").reset_index(drop=True)
    return deduped.drop(columns=[column for column in ("_effective_sort", "_load_sort") if column in deduped.columns])
