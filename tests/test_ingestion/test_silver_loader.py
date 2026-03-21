import json

from stockmachine.data.loaders.silver import load_silver_table
from stockmachine.ingestion.storage import StorageLayout


def test_load_silver_table_dedupes_latest_primary_key(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    table_dir = layout.silver_table_dir("daily_bar")
    table_dir.mkdir(parents=True, exist_ok=True)

    older = {
        "session_date": "2025-01-02",
        "symbol": "AAPL",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000000.0,
        "vwap": 100.2,
        "dollar_volume": 100200000.0,
        "trade_count": 1000,
        "source_name": "source_a",
        "load_time_utc": "2025-01-03T00:00:00+00:00",
        "effective_time_utc": "2025-01-02T21:00:00+00:00",
        "source_version": None,
    }
    newer = dict(older)
    newer["open"] = 101.0
    newer["load_time_utc"] = "2025-01-03T01:00:00+00:00"

    with (table_dir / "older.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(older) + "\n")
    with (table_dir / "newer.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(newer) + "\n")

    loaded = load_silver_table("daily_bar", layout=layout)

    assert len(loaded) == 1
    assert loaded.iloc[0]["open"] == 101.0


def test_load_silver_table_prefers_present_effective_time_over_missing(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    table_dir = layout.silver_table_dir("daily_bar")
    table_dir.mkdir(parents=True, exist_ok=True)

    yahoo_like = {
        "session_date": "2025-01-02",
        "symbol": "AAPL",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000000.0,
        "vwap": 100.2,
        "dollar_volume": 100200000.0,
        "trade_count": None,
        "source_name": "yahoo_finance",
        "load_time_utc": "2026-03-21T14:40:23+00:00",
        "effective_time_utc": None,
        "source_version": None,
    }
    alpaca_like = dict(yahoo_like)
    alpaca_like["open"] = 101.0
    alpaca_like["source_name"] = "alpaca_market_data"
    alpaca_like["load_time_utc"] = "2026-03-21T15:28:31+00:00"
    alpaca_like["effective_time_utc"] = "2025-01-02T05:00:00+00:00"

    with (table_dir / "yahoo.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(yahoo_like) + "\n")
    with (table_dir / "alpaca.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(alpaca_like) + "\n")

    loaded = load_silver_table("daily_bar", layout=layout)

    assert len(loaded) == 1
    assert loaded.iloc[0]["source_name"] == "alpaca_market_data"
