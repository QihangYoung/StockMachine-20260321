from __future__ import annotations

import pandas as pd

from stockmachine.ingestion.jobs import bootstrap_yahoo_us_equities as bootstrap_jobs
from stockmachine.ingestion.storage import StorageLayout


def test_backfill_static_metadata_history_from_silver_writes_session_scoped_snapshots(tmp_path, monkeypatch) -> None:
    daily_bar = pd.DataFrame(
        [
            {"session_date": "2025-01-02", "symbol": "AAPL"},
            {"session_date": "2025-01-02", "symbol": "MSFT"},
            {"session_date": "2025-01-03", "symbol": "AAPL"},
            {"session_date": "2025-01-03", "symbol": "MSFT"},
        ]
    )
    symbol_master = pd.DataFrame(
        [
            {
                "as_of_date": "2026-03-21",
                "symbol": "AAPL",
                "company_name": "Apple",
                "exchange_mic": "XNAS",
                "currency": "USD",
                "security_type": "COMMON_STOCK",
                "asset_class": "us_equity",
                "is_active": True,
                "source_name": "alpaca_trading",
                "load_time_utc": "2026-03-21T00:00:00+00:00",
                "source_version": None,
            },
            {
                "as_of_date": "2026-03-21",
                "symbol": "MSFT",
                "company_name": "Microsoft",
                "exchange_mic": "XNAS",
                "currency": "USD",
                "security_type": "COMMON_STOCK",
                "asset_class": "us_equity",
                "is_active": True,
                "source_name": "alpaca_trading",
                "load_time_utc": "2026-03-21T00:00:00+00:00",
                "source_version": None,
            },
        ]
    )
    industry_membership = pd.DataFrame(
        [
            {
                "as_of_date": "2025-12-30",
                "symbol": "AAPL",
                "industry_system": "yfinance_sector",
                "sector_name": "Technology",
                "industry_group_name": "Technology",
                "industry_name": "Consumer Electronics",
                "subindustry_name": "Consumer Electronics",
                "source_name": "yahoo_finance",
                "load_time_utc": "2026-03-21T00:00:00+00:00",
                "source_version": "bootstrap_v1",
            },
            {
                "as_of_date": "2025-12-30",
                "symbol": "MSFT",
                "industry_system": "yfinance_sector",
                "sector_name": "Technology",
                "industry_group_name": "Technology",
                "industry_name": "Software",
                "subindustry_name": "Software",
                "source_name": "yahoo_finance",
                "load_time_utc": "2026-03-21T00:00:00+00:00",
                "source_version": "bootstrap_v1",
            },
        ]
    )

    def _load_silver_table(table_name: str, *, layout: StorageLayout | None = None) -> pd.DataFrame:
        if table_name == "daily_bar":
            return daily_bar.copy()
        if table_name == "symbol_master":
            return symbol_master.copy()
        if table_name == "industry_membership":
            return industry_membership.copy()
        raise AssertionError(f"Unexpected table request: {table_name}")

    monkeypatch.setattr(bootstrap_jobs, "load_silver_table", _load_silver_table)
    layout = StorageLayout(root=tmp_path)

    result = bootstrap_jobs.backfill_static_metadata_history_from_silver(layout=layout)

    assert result == {
        "session_dates": 2,
        "symbols": 2,
        "symbol_master_rows": 4,
        "industry_membership_rows": 4,
    }

    symbol_master_backfill = (layout.silver_table_dir("symbol_master") / "static_history_backfill.jsonl").read_text(
        encoding="utf-8"
    )
    industry_backfill = (layout.silver_table_dir("industry_membership") / "static_history_backfill.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"as_of_date": "2025-01-02"' in symbol_master_backfill
    assert '"source_version": "static_history_backfill_v1"' in symbol_master_backfill
    assert '"as_of_date": "2025-01-03"' in industry_backfill
