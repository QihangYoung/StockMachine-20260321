from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.data.vendors import AlpacaCredentials, AlpacaHttpClient
from stockmachine.ingestion.collectors import (
    AlpacaAssetCollector,
    AlpacaCorporateActionsCollector,
    AlpacaStockBarCollector,
    FetchWindow,
)
from stockmachine.ingestion.normalizers import (
    AlpacaAdjFactorNormalizer,
    AlpacaAssetNormalizer,
    AlpacaBarNormalizer,
)
from stockmachine.ingestion.storage import StorageLayout, write_jsonl
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, DEFAULT_UNIVERSE


def collect_symbol_master_snapshot(layout: StorageLayout | None = None) -> dict[str, int]:
    """Collect and persist one symbol_master snapshot from Alpaca assets."""

    storage = layout or StorageLayout()
    run_id = _run_id()
    client = AlpacaHttpClient(AlpacaCredentials.from_env())
    collector = AlpacaAssetCollector(client)
    normalizer = AlpacaAssetNormalizer()

    raw_records = collector.collect(FetchWindow(stream_name="assets"))
    normalized_batches = normalizer.normalize(raw_records)

    raw_path = storage.raw_stream_dir("alpaca_trading", "assets") / f"{run_id}.jsonl"
    write_jsonl(raw_path, [record.payload for record in raw_records])

    row_count = 0
    for batch in normalized_batches:
        output_path = storage.silver_table_dir(batch.table_name) / f"{run_id}.jsonl"
        write_jsonl(output_path, batch.rows)
        row_count += len(batch.rows)

    return {"raw_records": len(raw_records), "normalized_rows": row_count}


def collect_daily_bars(
    symbols: list[str],
    *,
    start_date: date,
    end_date: date,
    layout: StorageLayout | None = None,
    benchmark_symbols: tuple[str, ...] = ("SPY",),
    feed: str = "iex",
    adjustment: str = "raw",
) -> dict[str, int]:
    """Collect and persist one historical daily-bar batch from Alpaca."""

    storage = layout or StorageLayout()
    run_id = _run_id()
    client = AlpacaHttpClient(AlpacaCredentials.from_env())
    collector = AlpacaStockBarCollector(client, adjustment=adjustment, feed=feed)
    normalizer = AlpacaBarNormalizer(benchmark_symbols=benchmark_symbols)

    raw_records = collector.collect(
        FetchWindow(
            stream_name="daily_bars",
            start_date=start_date,
            end_date=end_date,
            symbols=tuple(symbols),
        )
    )
    normalized_batches = normalizer.normalize(raw_records)

    raw_path = storage.raw_stream_dir("alpaca_market_data", "daily_bars") / f"{run_id}.jsonl"
    write_jsonl(raw_path, [record.payload for record in raw_records])

    row_count = 0
    for batch in normalized_batches:
        output_path = storage.silver_table_dir(batch.table_name) / f"{run_id}.jsonl"
        write_jsonl(output_path, batch.rows)
        row_count += len(batch.rows)

    return {"raw_records": len(raw_records), "normalized_rows": row_count}


def collect_adj_factors(
    symbols: list[str],
    *,
    start_date: date,
    end_date: date,
    layout: StorageLayout | None = None,
    feed: str = "iex",
) -> dict[str, int]:
    """Collect adjusted bars and corporate actions, then build adj_factor rows."""

    storage = layout or StorageLayout()
    run_id = _run_id()
    client = AlpacaHttpClient(AlpacaCredentials.from_env())
    raw_bar_collector = AlpacaStockBarCollector(client, adjustment="raw", feed=feed)
    adjusted_bar_collector = AlpacaStockBarCollector(client, adjustment="all", feed=feed)
    corporate_actions_collector = AlpacaCorporateActionsCollector(client)
    normalizer = AlpacaAdjFactorNormalizer()

    fetch_window = FetchWindow(
        stream_name="daily_bars",
        start_date=start_date,
        end_date=end_date,
        symbols=tuple(symbols),
    )
    raw_bar_records = raw_bar_collector.collect(fetch_window)
    adjusted_bar_records = adjusted_bar_collector.collect(fetch_window)
    corporate_action_records = corporate_actions_collector.collect(
        FetchWindow(
            stream_name="corporate_actions",
            start_date=start_date,
            end_date=end_date,
            symbols=tuple(symbols),
        )
    )
    normalized_batches = normalizer.normalize(
        raw_bar_records=raw_bar_records,
        adjusted_bar_records=adjusted_bar_records,
        corporate_action_records=corporate_action_records,
    )

    write_jsonl(
        storage.raw_stream_dir("alpaca_market_data", "adj_factor_raw_bars") / f"{run_id}.jsonl",
        [record.payload for record in raw_bar_records],
    )
    write_jsonl(
        storage.raw_stream_dir("alpaca_market_data", "adj_factor_adjusted_bars") / f"{run_id}.jsonl",
        [record.payload for record in adjusted_bar_records],
    )
    write_jsonl(
        storage.raw_stream_dir("alpaca_market_data", "corporate_actions") / f"{run_id}.jsonl",
        [record.payload for record in corporate_action_records],
    )

    row_count = 0
    for batch in normalized_batches:
        output_path = storage.silver_table_dir(batch.table_name) / f"{run_id}.jsonl"
        write_jsonl(output_path, batch.rows)
        row_count += len(batch.rows)

    return {
        "raw_bar_records": len(raw_bar_records),
        "adjusted_bar_records": len(adjusted_bar_records),
        "corporate_action_records": len(corporate_action_records),
        "normalized_rows": row_count,
    }


def collect_research_seed(
    *,
    start_date: date,
    end_date: date,
    layout: StorageLayout | None = None,
    chunk_size: int = 25,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
    feed: str = "iex",
    adjustment: str = "raw",
    include_symbol_master: bool = True,
    include_adj_factor: bool = True,
) -> dict[str, int]:
    """Collect the default research universe into canonical silver tables."""

    storage = layout or StorageLayout()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    symbol_result = {"raw_records": 0, "normalized_rows": 0}
    if include_symbol_master:
        symbol_result = collect_symbol_master_snapshot(layout=storage)
    requested_symbols = list(dict.fromkeys([*DEFAULT_UNIVERSE, benchmark_symbol]))

    total_raw_records = symbol_result["raw_records"]
    total_normalized_rows = symbol_result["normalized_rows"]
    chunk_count = 0
    total_adjusted_bar_records = 0
    total_corporate_action_records = 0

    for chunk in _chunked(requested_symbols, chunk_size):
        result = collect_daily_bars(
            list(chunk),
            start_date=start_date,
            end_date=end_date,
            layout=storage,
            benchmark_symbols=(benchmark_symbol,),
            feed=feed,
            adjustment=adjustment,
        )
        total_raw_records += result["raw_records"]
        total_normalized_rows += result["normalized_rows"]
        if include_adj_factor:
            adj_factor_result = collect_adj_factors(
                list(chunk),
                start_date=start_date,
                end_date=end_date,
                layout=storage,
                feed=feed,
            )
            total_raw_records += (
                adj_factor_result["raw_bar_records"] + adj_factor_result["adjusted_bar_records"]
            )
            total_corporate_action_records += adj_factor_result["corporate_action_records"]
            total_normalized_rows += adj_factor_result["normalized_rows"]
            total_adjusted_bar_records += adj_factor_result["adjusted_bar_records"]
        chunk_count += 1

    return {
        "symbols": len(requested_symbols),
        "chunks": chunk_count,
        "included_symbol_master": int(include_symbol_master),
        "included_adj_factor": int(include_adj_factor),
        "raw_records": total_raw_records,
        "adjusted_bar_records": total_adjusted_bar_records,
        "corporate_action_records": total_corporate_action_records,
        "normalized_rows": total_normalized_rows,
    }


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [
        values[index:index + chunk_size]
        for index in range(0, len(values), chunk_size)
    ]
