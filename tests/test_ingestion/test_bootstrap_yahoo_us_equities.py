import json

from stockmachine.ingestion.jobs import backfill_static_metadata_history_from_silver
from stockmachine.ingestion.storage import StorageLayout


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_backfill_static_metadata_history_also_writes_universe_membership(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    _write_jsonl(
        layout.silver_table_dir("daily_bar") / "seed.jsonl",
        [
            {
                "session_date": "2025-01-02",
                "symbol": "AAPL",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "source_name": "test",
                "load_time_utc": "2025-01-02T00:00:00+00:00",
                "source_version": "v1",
            }
        ],
    )
    _write_jsonl(
        layout.silver_table_dir("symbol_master") / "seed.jsonl",
        [
            {
                "as_of_date": "2025-01-02",
                "symbol": "AAPL",
                "security_id": "AAPL",
                "company_name": "Apple",
                "exchange_mic": "XNAS",
                "currency": "USD",
                "security_type": "COMMON_STOCK",
                "asset_class": "EQUITY",
                "is_active": True,
                "list_date": None,
                "delist_date": None,
                "sector": "Technology",
                "industry": "Hardware",
                "country_of_listing": "US",
                "primary_share_class": True,
                "source_name": "test",
                "load_time_utc": "2025-01-02T00:00:00+00:00",
                "source_version": "v1",
            }
        ],
    )
    _write_jsonl(
        layout.silver_table_dir("industry_membership") / "seed.jsonl",
        [
            {
                "as_of_date": "2025-01-02",
                "symbol": "AAPL",
                "industry_system": "gics",
                "sector_name": "Technology",
                "industry_group_name": "Technology",
                "industry_name": "Hardware",
                "subindustry_name": "Computers",
                "source_name": "test",
                "load_time_utc": "2025-01-02T00:00:00+00:00",
                "source_version": "v1",
            }
        ],
    )

    result = backfill_static_metadata_history_from_silver(layout=layout)

    assert result["universe_membership_rows"] == 1
    output_path = layout.silver_table_dir("universe_membership") / "static_history_backfill.jsonl"
    assert output_path.exists()
