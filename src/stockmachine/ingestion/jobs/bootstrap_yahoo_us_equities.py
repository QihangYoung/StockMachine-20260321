from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from stockmachine.ingestion.storage import StorageLayout, write_jsonl
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, DEFAULT_UNIVERSE


def bootstrap_us_equities_yahoo_to_silver(
    *,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    layout: StorageLayout | None = None,
) -> dict[str, int]:
    """Bootstrap canonical silver tables from Yahoo Finance for research."""

    storage = layout or StorageLayout()
    tickers = list(DEFAULT_UNIVERSE) + [BENCHMARK_SYMBOL]
    price_data = _download_history(tickers=tickers, start=start, end=end)
    metadata = _download_metadata(symbols=list(DEFAULT_UNIVERSE))
    load_time_utc = datetime.now(timezone.utc).isoformat()
    snapshot_date = str(pd.to_datetime(price_data["date"]).max().date())

    raw_dir = storage.raw_stream_dir("yahoo_finance", "bootstrap")
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_dir / "price_data.jsonl", price_data.to_dict(orient="records"))
    write_jsonl(raw_dir / "metadata.jsonl", metadata.to_dict(orient="records"))

    symbol_master_rows = _build_symbol_master_rows(metadata, snapshot_date=snapshot_date, load_time_utc=load_time_utc)
    industry_rows = _build_industry_rows(metadata, snapshot_date=snapshot_date, load_time_utc=load_time_utc)
    daily_rows, benchmark_rows, adj_factor_rows = _build_bar_rows(price_data, load_time_utc=load_time_utc)

    write_jsonl(storage.silver_table_dir("symbol_master") / "yahoo_bootstrap.jsonl", symbol_master_rows)
    write_jsonl(storage.silver_table_dir("industry_membership") / "yahoo_bootstrap.jsonl", industry_rows)
    write_jsonl(storage.silver_table_dir("daily_bar") / "yahoo_bootstrap.jsonl", daily_rows)
    write_jsonl(storage.silver_table_dir("adj_factor") / "yahoo_bootstrap.jsonl", adj_factor_rows)
    write_jsonl(storage.silver_table_dir("benchmark_index") / "yahoo_bootstrap.jsonl", benchmark_rows)

    return {
        "symbol_master_rows": len(symbol_master_rows),
        "industry_membership_rows": len(industry_rows),
        "daily_bar_rows": len(daily_rows),
        "adj_factor_rows": len(adj_factor_rows),
        "benchmark_index_rows": len(benchmark_rows),
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
                "security_type": "COMMON_STOCK",
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


def _build_bar_rows(
    price_data: pd.DataFrame,
    *,
    load_time_utc: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    daily_rows = []
    benchmark_rows = []
    adj_factor_rows = []

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
        if row.symbol == BENCHMARK_SYMBOL:
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
