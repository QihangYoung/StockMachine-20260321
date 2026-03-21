from datetime import datetime, timezone

from stockmachine.ingestion.collectors.base import RawRecord
from stockmachine.ingestion.normalizers import (
    AlpacaAdjFactorNormalizer,
    AlpacaAssetNormalizer,
    AlpacaBarNormalizer,
)


def test_alpaca_asset_normalizer_outputs_symbol_master_row() -> None:
    record = RawRecord(
        source_name="alpaca_trading",
        stream_name="assets",
        pulled_at_utc=datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc),
        payload={
            "id": "asset-1",
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "status": "active",
            "tradable": True,
            "class": "us_equity",
        },
    )

    batches = AlpacaAssetNormalizer().normalize([record])

    assert len(batches) == 1
    row = batches[0].rows[0]
    assert row["symbol"] == "AAPL"
    assert row["exchange_mic"] == "XNAS"
    assert row["security_type"] == "COMMON_STOCK"
    assert row["is_active"] is True


def test_alpaca_bar_normalizer_splits_benchmark_rows() -> None:
    records = [
        RawRecord(
            source_name="alpaca_market_data",
            stream_name="daily_bars",
            pulled_at_utc=datetime(2025, 1, 3, 1, 0, tzinfo=timezone.utc),
            payload={
                "symbol": "AAPL",
                "t": "2025-01-02T05:00:00Z",
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.5,
                "v": 1000000,
                "vw": 100.2,
                "n": 12000,
            },
        ),
        RawRecord(
            source_name="alpaca_market_data",
            stream_name="daily_bars",
            pulled_at_utc=datetime(2025, 1, 3, 1, 0, tzinfo=timezone.utc),
            payload={
                "symbol": "SPY",
                "t": "2025-01-02T05:00:00Z",
                "o": 500.0,
                "h": 503.0,
                "l": 499.0,
                "c": 501.0,
                "v": 2000000,
                "vw": 500.5,
                "n": 50000,
            },
        ),
    ]

    batches = AlpacaBarNormalizer().normalize(records)
    daily_batch = next(batch for batch in batches if batch.table_name == "daily_bar")
    benchmark_batch = next(batch for batch in batches if batch.table_name == "benchmark_index")

    assert len(daily_batch.rows) == 1
    assert daily_batch.rows[0]["symbol"] == "AAPL"
    assert daily_batch.rows[0]["dollar_volume"] == 100.2 * 1000000

    assert len(benchmark_batch.rows) == 1
    assert benchmark_batch.rows[0]["symbol"] == "SPY"
    assert benchmark_batch.rows[0]["return_1d"] is None


def test_alpaca_adj_factor_normalizer_builds_factor_and_dividend() -> None:
    raw_bar = RawRecord(
        source_name="alpaca_market_data",
        stream_name="daily_bars",
        pulled_at_utc=datetime(2025, 2, 8, 1, 0, tzinfo=timezone.utc),
        effective_time_utc=datetime(2025, 2, 7, 5, 0, tzinfo=timezone.utc),
        payload={
            "symbol": "AAPL",
            "t": "2025-02-07T05:00:00Z",
            "c": 227.63,
        },
    )
    adjusted_bar = RawRecord(
        source_name="alpaca_market_data",
        stream_name="daily_bars",
        pulled_at_utc=datetime(2025, 2, 8, 1, 5, tzinfo=timezone.utc),
        effective_time_utc=datetime(2025, 2, 7, 5, 0, tzinfo=timezone.utc),
        payload={
            "symbol": "AAPL",
            "t": "2025-02-07T05:00:00Z",
            "c": 226.41,
        },
    )
    corporate_action = RawRecord(
        source_name="alpaca_market_data",
        stream_name="corporate_actions",
        pulled_at_utc=datetime(2025, 2, 8, 1, 10, tzinfo=timezone.utc),
        effective_time_utc=datetime(2025, 2, 7, 0, 0, tzinfo=timezone.utc),
        payload={
            "symbol": "AAPL",
            "ex_date": "2025-02-07",
            "rate": 0.25,
            "action_type": "cash_dividends",
        },
    )

    batches = AlpacaAdjFactorNormalizer().normalize(
        raw_bar_records=[raw_bar],
        adjusted_bar_records=[adjusted_bar],
        corporate_action_records=[corporate_action],
    )

    assert len(batches) == 1
    row = batches[0].rows[0]
    assert row["session_date"] == "2025-02-07"
    assert row["symbol"] == "AAPL"
    assert round(row["price_adjust_factor"], 6) == round(226.41 / 227.63, 6)
    assert row["cash_dividend"] == 0.25
    assert row["split_factor"] == 1.0
