from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from stockmachine.data.loaders.silver import load_silver_table
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
from stockmachine.research.universe import DEFAULT_RESEARCH_UNIVERSE_NAME
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, DEFAULT_UNIVERSE


def collect_symbol_master_snapshot(
    layout: StorageLayout | None = None,
    *,
    snapshot_date: date | None = None,
) -> dict[str, int]:
    """Collect and persist one symbol_master snapshot from Alpaca assets."""

    storage = layout or StorageLayout()
    run_id = _run_id()
    client = AlpacaHttpClient(AlpacaCredentials.from_env())
    collector = AlpacaAssetCollector(client)
    normalizer = AlpacaAssetNormalizer(as_of_date=snapshot_date)

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
    include_daily_bars: bool = True,
    include_adj_factor: bool = True,
    membership_start_date: date | None = None,
) -> dict[str, int]:
    """Collect the default research universe into canonical silver tables."""

    storage = layout or StorageLayout()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    symbol_result = {"raw_records": 0, "normalized_rows": 0}
    if include_symbol_master:
        symbol_result = collect_symbol_master_snapshot(layout=storage, snapshot_date=end_date)
    requested_symbols = list(dict.fromkeys([*DEFAULT_UNIVERSE, benchmark_symbol]))

    total_raw_records = symbol_result["raw_records"]
    total_normalized_rows = symbol_result["normalized_rows"]
    chunk_count = 0
    total_adjusted_bar_records = 0
    total_corporate_action_records = 0

    if include_daily_bars:
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

    membership_result = refresh_research_membership_history(
        start_date=membership_start_date or start_date,
        end_date=end_date,
        layout=storage,
    )
    total_normalized_rows += (
        membership_result["industry_membership_rows"] + membership_result["universe_membership_rows"]
    )

    return {
        "symbols": len(requested_symbols),
        "chunks": chunk_count,
        "included_symbol_master": int(include_symbol_master),
        "included_daily_bars": int(include_daily_bars),
        "included_adj_factor": int(include_adj_factor),
        "membership_refresh_start_date": (membership_start_date or start_date).isoformat(),
        "raw_records": total_raw_records,
        "adjusted_bar_records": total_adjusted_bar_records,
        "corporate_action_records": total_corporate_action_records,
        "normalized_rows": total_normalized_rows,
        "industry_membership_rows": membership_result["industry_membership_rows"],
        "universe_membership_rows": membership_result["universe_membership_rows"],
        "membership_session_dates": membership_result["session_dates"],
    }


def refresh_research_membership_history(
    *,
    start_date: date,
    end_date: date,
    layout: StorageLayout | None = None,
    universe_name: str = DEFAULT_RESEARCH_UNIVERSE_NAME,
) -> dict[str, int]:
    """Materialize industry and universe membership rows for every research session."""

    storage = layout or StorageLayout()
    session_dates = _load_research_session_dates(storage, start_date=start_date, end_date=end_date)
    if not session_dates:
        return {
            "session_dates": 0,
            "industry_membership_rows": 0,
            "universe_membership_rows": 0,
        }

    latest_symbol_master = _latest_symbol_master_rows(storage, symbols=DEFAULT_UNIVERSE)
    latest_industry = _latest_industry_rows(storage, symbols=DEFAULT_UNIVERSE)
    latest_universe_membership = _latest_universe_membership_rows(
        storage,
        universe_name=universe_name,
        symbols=DEFAULT_UNIVERSE,
    )

    if latest_industry.empty:
        latest_industry = _build_fallback_industry_rows(latest_symbol_master)
    if latest_universe_membership.empty:
        latest_universe_membership = _build_fallback_universe_membership_rows(
            symbols=DEFAULT_UNIVERSE,
            universe_name=universe_name,
        )

    load_time_utc = datetime.now(timezone.utc).isoformat()
    source_version = "research_seed_membership_refresh_v1"
    industry_rows = _expand_industry_membership_rows(
        latest_industry,
        session_dates=session_dates,
        load_time_utc=load_time_utc,
        source_version=source_version,
    )
    universe_rows = _expand_universe_membership_rows(
        latest_universe_membership,
        session_dates=session_dates,
        load_time_utc=load_time_utc,
        source_version=source_version,
    )

    run_id = _run_id()
    if industry_rows:
        write_jsonl(storage.silver_table_dir("industry_membership") / f"{run_id}.jsonl", industry_rows)
    if universe_rows:
        write_jsonl(storage.silver_table_dir("universe_membership") / f"{run_id}.jsonl", universe_rows)

    return {
        "session_dates": len(session_dates),
        "industry_membership_rows": len(industry_rows),
        "universe_membership_rows": len(universe_rows),
    }


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [
        values[index:index + chunk_size]
        for index in range(0, len(values), chunk_size)
    ]


def _load_research_session_dates(
    storage: StorageLayout,
    *,
    start_date: date,
    end_date: date,
) -> list[pd.Timestamp]:
    session_indexes: list[pd.Index] = []
    for table_name in ("daily_bar", "benchmark_index"):
        frame = load_silver_table(table_name, layout=storage)
        if frame.empty or "session_date" not in frame.columns:
            continue
        session_dates = pd.to_datetime(frame["session_date"], errors="coerce").dropna().dt.normalize()
        session_dates = session_dates.loc[
            (session_dates >= pd.Timestamp(start_date))
            & (session_dates <= pd.Timestamp(end_date))
        ]
        if not session_dates.empty:
            session_indexes.append(pd.Index(session_dates.drop_duplicates()))
    if not session_indexes:
        return []
    combined = session_indexes[0]
    for index in session_indexes[1:]:
        combined = combined.union(index)
    return list(combined.sort_values())


def _latest_symbol_master_rows(storage: StorageLayout, *, symbols: tuple[str, ...]) -> pd.DataFrame:
    frame = load_silver_table("symbol_master", layout=storage)
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")
    working = working.dropna(subset=["as_of_date", "symbol"]).copy()
    working = working.loc[working["symbol"].astype(str).isin(set(symbols))].copy()
    if working.empty:
        return pd.DataFrame()
    return _latest_rows_by_key(working, key_columns=("symbol",), sort_columns=("as_of_date", "load_time_utc"))


def _latest_industry_rows(storage: StorageLayout, *, symbols: tuple[str, ...]) -> pd.DataFrame:
    frame = load_silver_table("industry_membership", layout=storage)
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")
    working = working.dropna(subset=["as_of_date", "symbol"]).copy()
    working = working.loc[working["symbol"].astype(str).isin(set(symbols))].copy()
    if working.empty:
        return pd.DataFrame()
    return _latest_rows_by_key(
        working,
        key_columns=("symbol", "industry_system"),
        sort_columns=("as_of_date", "load_time_utc"),
    )


def _latest_universe_membership_rows(
    storage: StorageLayout,
    *,
    universe_name: str,
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    frame = load_silver_table("universe_membership", layout=storage)
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["session_date"] = pd.to_datetime(working["session_date"], errors="coerce")
    working = working.dropna(subset=["session_date", "symbol"]).copy()
    working = working.loc[working["universe_name"].astype(str) == universe_name].copy()
    working = working.loc[working["symbol"].astype(str).isin(set(symbols))].copy()
    if working.empty:
        return pd.DataFrame()
    if "is_member" in working.columns:
        working["is_member"] = working["is_member"].fillna(True).astype(bool)
        working = working.loc[working["is_member"]].copy()
    return _latest_rows_by_key(
        working,
        key_columns=("universe_name", "symbol"),
        sort_columns=("session_date", "load_time_utc"),
    )


def _latest_rows_by_key(
    frame: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    sort_columns: tuple[str, ...],
) -> pd.DataFrame:
    working = frame.copy()
    available_sort_columns = [column for column in sort_columns if column in working.columns]
    if available_sort_columns:
        working = working.sort_values(
            [*key_columns, *available_sort_columns],
            kind="stable",
            na_position="last",
        )
    return working.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)


def _build_fallback_industry_rows(symbol_master: pd.DataFrame) -> pd.DataFrame:
    if symbol_master.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "industry_system",
                "sector_name",
                "industry_group_name",
                "industry_name",
                "subindustry_name",
                "source_name",
                "load_time_utc",
                "source_version",
            ]
        )
    return pd.DataFrame(
        {
            "symbol": symbol_master["symbol"].astype(str),
            "industry_system": "symbol_master_fallback",
            "sector_name": symbol_master.get("sector", pd.Series(index=symbol_master.index, dtype="object")).fillna("Unknown"),
            "industry_group_name": symbol_master.get("sector", pd.Series(index=symbol_master.index, dtype="object")).fillna("Unknown"),
            "industry_name": symbol_master.get("industry", pd.Series(index=symbol_master.index, dtype="object")).fillna("Unknown"),
            "subindustry_name": symbol_master.get("industry", pd.Series(index=symbol_master.index, dtype="object")).fillna("Unknown"),
            "source_name": symbol_master.get("source_name", pd.Series(index=symbol_master.index, dtype="object")).fillna("symbol_master"),
            "load_time_utc": symbol_master.get("load_time_utc", pd.Series(index=symbol_master.index, dtype="object")),
            "source_version": symbol_master.get("source_version", pd.Series(index=symbol_master.index, dtype="object")).fillna("symbol_master_fallback_v1"),
        }
    ).reset_index(drop=True)


def _build_fallback_universe_membership_rows(
    *,
    symbols: tuple[str, ...],
    universe_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "universe_name": universe_name,
                "symbol": symbol,
                "is_member": True,
                "membership_source": "default_research_universe",
                "entry_date": None,
                "exit_date": None,
                "source_name": "research_seed_refresh",
                "load_time_utc": None,
                "source_version": "research_seed_membership_refresh_v1",
            }
            for symbol in symbols
        ]
    )


def _expand_industry_membership_rows(
    frame: pd.DataFrame,
    *,
    session_dates: list[pd.Timestamp],
    load_time_utc: str,
    source_version: str,
) -> list[dict[str, object]]:
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    for session_date in session_dates:
        as_of_date = str(pd.Timestamp(session_date).date())
        for row in frame.to_dict(orient="records"):
            payload = dict(row)
            payload["as_of_date"] = as_of_date
            payload["load_time_utc"] = load_time_utc
            payload["source_version"] = source_version
            rows.append(payload)
    return rows


def _expand_universe_membership_rows(
    frame: pd.DataFrame,
    *,
    session_dates: list[pd.Timestamp],
    load_time_utc: str,
    source_version: str,
) -> list[dict[str, object]]:
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    for session_date in session_dates:
        session_str = str(pd.Timestamp(session_date).date())
        for row in frame.to_dict(orient="records"):
            payload = dict(row)
            payload["session_date"] = session_str
            payload["load_time_utc"] = load_time_utc
            payload["source_version"] = source_version
            payload["is_member"] = bool(payload.get("is_member", True))
            payload["membership_source"] = payload.get("membership_source") or "static_history_backfill"
            payload["source_name"] = payload.get("source_name") or "research_seed_refresh"
            rows.append(payload)
    return rows
