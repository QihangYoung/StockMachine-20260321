from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from stockmachine.data.loaders.silver import load_silver_table
from stockmachine.ingestion.storage import StorageLayout, write_jsonl
from stockmachine.research.universe import DEFAULT_RESEARCH_UNIVERSE_NAME
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, DEFAULT_UNIVERSE

DEFAULT_MULTI_ASSET_ETF_UNIVERSE_NAME = "us_multi_asset_etfs_v1"
DEFAULT_MULTI_ASSET_ETF_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "VXUS",
    "AGG",
    "IEF",
    "LQD",
    "GLDM",
    "CTA",
    "SGOV",
)
DEFAULT_MULTI_ASSET_PROXY_UNIVERSE_NAME = "us_multi_asset_proxy_etfs_v1"
DEFAULT_MULTI_ASSET_PROXY_SYMBOLS: tuple[str, ...] = (
    "FMF",
    "DBMF",
    "BIL",
    "GLD",
)
DEFAULT_FMF_VALIDATION_UNIVERSE_NAME = "us_multi_asset_fmf_validation_v1"
DEFAULT_FMF_VALIDATION_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "VXUS",
    "IEF",
    "LQD",
    "GLD",
    "FMF",
    "BIL",
)


def bootstrap_us_equities_yahoo_to_silver(
    *,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    layout: StorageLayout | None = None,
    silver_file_stem: str = "yahoo_bootstrap",
) -> dict[str, int]:
    """Bootstrap canonical silver tables from Yahoo Finance for research."""

    return bootstrap_market_symbols_yahoo_to_silver(
        symbols=tuple(list(DEFAULT_UNIVERSE) + [BENCHMARK_SYMBOL]),
        metadata_symbols=tuple(DEFAULT_UNIVERSE),
        start=start,
        end=end,
        layout=layout,
        silver_file_stem=silver_file_stem,
        universe_name=DEFAULT_RESEARCH_UNIVERSE_NAME,
        membership_source="yahoo_bootstrap",
        benchmark_symbols=(BENCHMARK_SYMBOL,),
    )


def bootstrap_multi_asset_etfs_yahoo_to_silver(
    *,
    start: str = "2018-01-01",
    end: str = "2026-12-31",
    layout: StorageLayout | None = None,
    silver_file_stem: str = "multi_asset_etf_bootstrap",
) -> dict[str, int]:
    """Bootstrap the default multi-asset ETF universe into canonical silver tables."""

    return bootstrap_market_symbols_yahoo_to_silver(
        symbols=DEFAULT_MULTI_ASSET_ETF_SYMBOLS,
        metadata_symbols=DEFAULT_MULTI_ASSET_ETF_SYMBOLS,
        start=start,
        end=end,
        layout=layout,
        silver_file_stem=silver_file_stem,
        universe_name=DEFAULT_MULTI_ASSET_ETF_UNIVERSE_NAME,
        membership_source="multi_asset_etf_bootstrap",
        benchmark_symbols=(BENCHMARK_SYMBOL,),
    )


def bootstrap_multi_asset_proxy_etfs_yahoo_to_silver(
    *,
    start: str = "2018-01-01",
    end: str = "2026-12-31",
    layout: StorageLayout | None = None,
    silver_file_stem: str = "multi_asset_proxy_bootstrap",
) -> dict[str, int]:
    """Bootstrap the default proxy ETF set used in long-window multi-asset research."""

    return bootstrap_market_symbols_yahoo_to_silver(
        symbols=DEFAULT_MULTI_ASSET_PROXY_SYMBOLS,
        metadata_symbols=DEFAULT_MULTI_ASSET_PROXY_SYMBOLS,
        start=start,
        end=end,
        layout=layout,
        silver_file_stem=silver_file_stem,
        universe_name=DEFAULT_MULTI_ASSET_PROXY_UNIVERSE_NAME,
        membership_source="multi_asset_proxy_bootstrap",
        benchmark_symbols=(),
    )


def bootstrap_fmf_validation_etfs_yahoo_to_silver(
    *,
    start: str = "2013-08-01",
    end: str = "2026-12-31",
    layout: StorageLayout | None = None,
    silver_file_stem: str = "fmf_validation_etf_bootstrap",
) -> dict[str, int]:
    """Bootstrap the ETF universe used by the FMF-based validation rebuild."""

    return bootstrap_market_symbols_yahoo_to_silver(
        symbols=DEFAULT_FMF_VALIDATION_SYMBOLS,
        metadata_symbols=DEFAULT_FMF_VALIDATION_SYMBOLS,
        start=start,
        end=end,
        layout=layout,
        silver_file_stem=silver_file_stem,
        universe_name=DEFAULT_FMF_VALIDATION_UNIVERSE_NAME,
        membership_source="fmf_validation_etf_bootstrap",
        benchmark_symbols=(BENCHMARK_SYMBOL,),
    )


def bootstrap_market_symbols_yahoo_to_silver(
    *,
    symbols: tuple[str, ...] | list[str],
    metadata_symbols: tuple[str, ...] | list[str] | None = None,
    start: str,
    end: str,
    layout: StorageLayout | None = None,
    silver_file_stem: str = "yahoo_bootstrap",
    universe_name: str = DEFAULT_RESEARCH_UNIVERSE_NAME,
    membership_source: str = "yahoo_bootstrap",
    benchmark_symbols: tuple[str, ...] | list[str] = (BENCHMARK_SYMBOL,),
) -> dict[str, int]:
    """Bootstrap canonical silver tables from Yahoo Finance for an arbitrary symbol set."""

    storage = layout or StorageLayout()
    resolved_symbols = tuple(dict.fromkeys(str(symbol) for symbol in symbols))
    resolved_metadata_symbols = tuple(
        dict.fromkeys(str(symbol) for symbol in (metadata_symbols or resolved_symbols))
    )
    resolved_benchmark_symbols = tuple(dict.fromkeys(str(symbol) for symbol in benchmark_symbols))

    price_data = _download_history(tickers=list(resolved_symbols), start=start, end=end)
    metadata = _download_metadata(symbols=list(resolved_metadata_symbols))
    load_time_utc = datetime.now(timezone.utc).isoformat()
    snapshot_date = str(pd.to_datetime(price_data["date"]).max().date())

    raw_dir = storage.raw_stream_dir("yahoo_finance", "bootstrap")
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_dir / f"{silver_file_stem}_price_data.jsonl", price_data.to_dict(orient="records"))
    write_jsonl(raw_dir / f"{silver_file_stem}_metadata.jsonl", metadata.to_dict(orient="records"))

    symbol_master_rows = _build_symbol_master_rows(metadata, snapshot_date=snapshot_date, load_time_utc=load_time_utc)
    industry_rows = _build_industry_rows(metadata, snapshot_date=snapshot_date, load_time_utc=load_time_utc)
    universe_rows = _build_universe_membership_rows(
        metadata,
        session_date=snapshot_date,
        load_time_utc=load_time_utc,
        universe_name=universe_name,
        source_version="bootstrap_v1",
        membership_source=membership_source,
    )
    daily_rows, benchmark_rows, adj_factor_rows = _build_bar_rows(
        price_data,
        load_time_utc=load_time_utc,
        benchmark_symbols=resolved_benchmark_symbols,
    )

    target_filename = f"{silver_file_stem}.jsonl"
    write_jsonl(storage.silver_table_dir("symbol_master") / target_filename, symbol_master_rows)
    write_jsonl(storage.silver_table_dir("industry_membership") / target_filename, industry_rows)
    write_jsonl(storage.silver_table_dir("universe_membership") / target_filename, universe_rows)
    write_jsonl(storage.silver_table_dir("daily_bar") / target_filename, daily_rows)
    write_jsonl(storage.silver_table_dir("adj_factor") / target_filename, adj_factor_rows)
    write_jsonl(storage.silver_table_dir("benchmark_index") / target_filename, benchmark_rows)

    return {
        "silver_file_stem": silver_file_stem,
        "symbol_count": len(resolved_symbols),
        "metadata_symbol_count": len(resolved_metadata_symbols),
        "benchmark_symbol_count": len(resolved_benchmark_symbols),
        "symbol_master_rows": len(symbol_master_rows),
        "industry_membership_rows": len(industry_rows),
        "universe_membership_rows": len(universe_rows),
        "daily_bar_rows": len(daily_rows),
        "adj_factor_rows": len(adj_factor_rows),
        "benchmark_index_rows": len(benchmark_rows),
    }


def backfill_static_metadata_history_from_silver(
    *,
    layout: StorageLayout | None = None,
) -> dict[str, int]:
    """Expand the latest static research metadata across historical session dates.

    This is an explicit bootstrap helper for the current fixed US-equities
    research universe. It does not create true historical constituent history;
    instead, it materializes one static metadata snapshot per observed session so
    the point-in-time universe contract can operate without future-dated rows.
    """

    storage = layout or StorageLayout()
    daily_bar = load_silver_table("daily_bar", layout=storage)
    universe_membership = load_silver_table("universe_membership", layout=storage)
    symbol_master = load_silver_table("symbol_master", layout=storage)
    industry_membership = load_silver_table("industry_membership", layout=storage)

    if daily_bar.empty or symbol_master.empty:
        return {
            "session_dates": 0,
            "symbols": 0,
            "universe_membership_rows": 0,
            "symbol_master_rows": 0,
            "industry_membership_rows": 0,
        }

    session_dates = (
        pd.to_datetime(daily_bar["session_date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not session_dates:
        return {
            "session_dates": 0,
            "symbols": 0,
            "universe_membership_rows": 0,
            "symbol_master_rows": 0,
            "industry_membership_rows": 0,
        }

    active_symbols = set(daily_bar["symbol"].dropna().astype(str).unique().tolist())
    latest_universe_membership = _latest_rows_by_key(
        universe_membership,
        key_columns=("universe_name", "symbol"),
    )
    if not latest_universe_membership.empty and "symbol" in latest_universe_membership.columns:
        latest_universe_membership = latest_universe_membership.loc[
            latest_universe_membership["symbol"].astype(str).isin(active_symbols)
        ].copy()
    latest_symbol_master = _latest_rows_by_key(symbol_master, key_columns=("symbol",))
    latest_symbol_master = latest_symbol_master.loc[
        latest_symbol_master["symbol"].astype(str).isin(active_symbols)
    ].copy()
    latest_industry = _latest_rows_by_key(
        industry_membership,
        key_columns=("symbol", "industry_system"),
    )
    latest_industry = latest_industry.loc[
        latest_industry["symbol"].astype(str).isin(active_symbols)
    ].copy()

    load_time_utc = datetime.now(timezone.utc).isoformat()
    source_version = "static_history_backfill_v1"
    symbol_rows = _expand_static_snapshot_rows(
        latest_symbol_master,
        session_dates=session_dates,
        load_time_utc=load_time_utc,
        source_version=source_version,
    )
    industry_rows = _expand_static_snapshot_rows(
        latest_industry,
        session_dates=session_dates,
        load_time_utc=load_time_utc,
        source_version=source_version,
    )
    universe_rows = _expand_static_universe_membership_rows(
        latest_universe_membership,
        session_dates=session_dates,
        fallback_symbols=sorted(active_symbols),
        load_time_utc=load_time_utc,
        source_version=source_version,
    )

    write_jsonl(
        storage.silver_table_dir("universe_membership") / "static_history_backfill.jsonl",
        universe_rows,
    )
    write_jsonl(
        storage.silver_table_dir("symbol_master") / "static_history_backfill.jsonl",
        symbol_rows,
    )
    write_jsonl(
        storage.silver_table_dir("industry_membership") / "static_history_backfill.jsonl",
        industry_rows,
    )
    return {
        "session_dates": len(session_dates),
        "symbols": len(active_symbols),
        "universe_membership_rows": len(universe_rows),
        "symbol_master_rows": len(symbol_rows),
        "industry_membership_rows": len(industry_rows),
    }


def _download_history(*, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = _download_batch(tickers=tickers, start=start, end=end)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for symbol in tickers:
        ticker_frame = _extract_ticker_frame(raw, symbol=symbol)
        if ticker_frame.empty:
            missing.append(symbol)
            continue
        frames.append(ticker_frame)

    for symbol in missing.copy():
        try:
            retry_raw = _download_batch(tickers=[symbol], start=start, end=end)
            ticker_frame = _extract_ticker_frame(retry_raw, symbol=symbol)
        except RuntimeError:
            ticker_frame = pd.DataFrame()
        if ticker_frame.empty:
            ticker_frame = _download_single_history(symbol=symbol, start=start, end=end)
        if ticker_frame.empty:
            continue
        frames.append(ticker_frame)
        missing.remove(symbol)

    if missing:
        raise RuntimeError(f"Yahoo Finance download missing symbols after retry: {', '.join(missing)}")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=False)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def _download_batch(*, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("No market data returned from Yahoo Finance.")
    return raw


def _extract_ticker_frame(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        ticker_frame = raw[symbol].copy()
    else:
        ticker_frame = raw.copy()

    if ticker_frame.empty:
        return pd.DataFrame()

    ticker_frame = ticker_frame.reset_index()
    ticker_frame.columns = [str(column).lower().replace(" ", "_") for column in ticker_frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in ticker_frame.columns:
            return pd.DataFrame()
    ticker_frame = ticker_frame.dropna(subset=required).copy()
    if ticker_frame.empty:
        return pd.DataFrame()
    ticker_frame["symbol"] = symbol
    return ticker_frame


def _download_single_history(*, symbol: str, start: str, end: str) -> pd.DataFrame:
    ticker_frame = yf.Ticker(symbol).history(
        start=start,
        end=end,
        period="max",
        auto_adjust=False,
        actions=True,
    )
    if ticker_frame.empty:
        return pd.DataFrame()
    ticker_frame = ticker_frame.reset_index()
    ticker_frame.columns = [str(column).lower().replace(" ", "_") for column in ticker_frame.columns]
    if "date" not in ticker_frame.columns:
        ticker_frame = ticker_frame.rename(columns={"datetime": "date"})
    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in ticker_frame.columns:
            return pd.DataFrame()
    ticker_frame["date"] = pd.to_datetime(ticker_frame["date"], utc=False).dt.tz_localize(None)
    ticker_frame = ticker_frame[
        (ticker_frame["date"] >= pd.Timestamp(start))
        & (ticker_frame["date"] < pd.Timestamp(end))
    ].dropna(subset=required).copy()
    if ticker_frame.empty:
        return pd.DataFrame()
    ticker_frame["symbol"] = symbol
    return ticker_frame


def _download_metadata(*, symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        info = {}
        try:
            info = yf.Ticker(symbol).info
        except Exception:
            info = {}
        rows.append(
            {
                "symbol": symbol,
                "company_name": info.get("longName") or info.get("shortName") or symbol,
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "Unknown",
                "quote_type": info.get("quoteType") or "Unknown",
                "exchange": info.get("exchange") or "Unknown",
                "currency": info.get("currency") or "USD",
                "country": info.get("country") or "US",
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _build_symbol_master_rows(
    metadata: pd.DataFrame,
    *,
    snapshot_date: str,
    load_time_utc: str,
) -> list[dict[str, object]]:
    rows = []
    for row in metadata.itertuples(index=False):
        rows.append(
            {
                "as_of_date": snapshot_date,
                "symbol": row.symbol,
                "security_id": row.symbol,
                "company_name": row.company_name,
                "exchange_mic": _map_exchange(row.exchange),
                "currency": row.currency,
                "security_type": _resolve_security_type(row.quote_type),
                "asset_class": row.quote_type,
                "is_active": True,
                "list_date": None,
                "delist_date": None,
                "sector": row.sector,
                "industry": row.industry,
                "country_of_listing": row.country,
                "primary_share_class": True,
                "source_name": "yahoo_finance",
                "load_time_utc": load_time_utc,
                "source_version": "bootstrap_v1",
            }
        )
    return rows


def _build_industry_rows(
    metadata: pd.DataFrame,
    *,
    snapshot_date: str,
    load_time_utc: str,
) -> list[dict[str, object]]:
    rows = []
    for row in metadata.itertuples(index=False):
        rows.append(
            {
                "as_of_date": snapshot_date,
                "symbol": row.symbol,
                "industry_system": "yfinance_sector",
                "sector_name": row.sector,
                "industry_group_name": row.sector,
                "industry_name": row.industry,
                "subindustry_name": row.industry,
                "source_name": "yahoo_finance",
                "load_time_utc": load_time_utc,
                "source_version": "bootstrap_v1",
            }
        )
    return rows


def _build_universe_membership_rows(
    metadata: pd.DataFrame,
    *,
    session_date: str,
    load_time_utc: str,
    universe_name: str,
    source_version: str,
    membership_source: str,
) -> list[dict[str, object]]:
    rows = []
    for row in metadata.itertuples(index=False):
        rows.append(
            {
                "session_date": session_date,
                "universe_name": universe_name,
                "symbol": row.symbol,
                "is_member": True,
                "membership_source": membership_source,
                "entry_date": None,
                "exit_date": None,
                "source_name": "yahoo_finance",
                "load_time_utc": load_time_utc,
                "source_version": source_version,
            }
        )
    return rows


def _latest_rows_by_key(frame: pd.DataFrame, *, key_columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    if "as_of_date" in working.columns:
        working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")
    sort_columns = [column for column in ("as_of_date", "load_time_utc") if column in working.columns]
    if sort_columns:
        working = working.sort_values([*key_columns, *sort_columns], kind="stable", na_position="last")
    return working.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)


def _expand_static_snapshot_rows(
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


def _expand_static_universe_membership_rows(
    frame: pd.DataFrame,
    *,
    session_dates: list[pd.Timestamp],
    fallback_symbols: list[str],
    load_time_utc: str,
    source_version: str,
) -> list[dict[str, object]]:
    if frame.empty:
        frame = pd.DataFrame(
            [
                {
                    "universe_name": DEFAULT_RESEARCH_UNIVERSE_NAME,
                    "symbol": symbol,
                    "is_member": True,
                    "membership_source": "default_research_universe",
                    "entry_date": None,
                    "exit_date": None,
                    "source_name": "bootstrap",
                }
                for symbol in fallback_symbols
            ]
        )

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
            payload["source_name"] = payload.get("source_name") or "bootstrap"
            rows.append(payload)
    return rows


def _build_bar_rows(
    price_data: pd.DataFrame,
    *,
    load_time_utc: str,
    benchmark_symbols: tuple[str, ...] | list[str] = (BENCHMARK_SYMBOL,),
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    daily_rows = []
    benchmark_rows = []
    adj_factor_rows = []
    benchmark_symbol_set = {str(symbol) for symbol in benchmark_symbols}

    for row in price_data.itertuples(index=False):
        if any(pd.isna(value) for value in (row.open, row.high, row.low, row.close, row.volume)):
            continue
        close = float(row.close)
        adj_close = float(row.adj_close) if pd.notna(getattr(row, "adj_close", None)) else close
        price_adjust_factor = adj_close / close if close > 0 else 1.0
        split_factor = float(row.stock_splits) if pd.notna(getattr(row, "stock_splits", None)) else 0.0
        dividend = float(row.dividends) if pd.notna(getattr(row, "dividends", None)) else 0.0

        common = {
            "session_date": str(pd.Timestamp(row.date).date()),
            "symbol": row.symbol,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": close,
            "volume": float(row.volume),
            "source_name": "yahoo_finance",
            "load_time_utc": load_time_utc,
            "source_version": "bootstrap_v1",
        }
        adj_factor_rows.append(
            {
                "session_date": str(pd.Timestamp(row.date).date()),
                "symbol": row.symbol,
                "split_factor": split_factor if split_factor > 0 else 1.0,
                "cash_dividend": dividend,
                "price_adjust_factor": price_adjust_factor,
                "source_name": "yahoo_finance",
                "load_time_utc": load_time_utc,
                "effective_time_utc": None,
                "source_version": "bootstrap_v1",
            }
        )
        if row.symbol in benchmark_symbol_set:
            benchmark_rows.append(
                {
                    **common,
                    "return_1d": None,
                }
            )
        else:
            daily_rows.append(
                {
                    **common,
                    "vwap": None,
                    "dollar_volume": float(row.close) * float(row.volume),
                    "trade_count": None,
                    "effective_time_utc": None,
                }
            )
    return daily_rows, benchmark_rows, adj_factor_rows


def _map_exchange(raw_value: str) -> str:
    mapping = {
        "NMS": "XNAS",
        "NYQ": "XNYS",
        "ASE": "XASE",
        "NAS": "XNAS",
    }
    return mapping.get(raw_value, raw_value)


def _resolve_security_type(raw_quote_type: str) -> str:
    normalized = str(raw_quote_type or "").strip().lower()
    if "etf" in normalized:
        return "ETF"
    if "fund" in normalized:
        return "FUND"
    if "index" in normalized:
        return "INDEX"
    return "COMMON_STOCK"
