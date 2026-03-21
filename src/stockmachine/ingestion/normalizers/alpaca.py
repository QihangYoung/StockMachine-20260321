from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from stockmachine.ingestion.collectors.base import RawRecord

from .base import NormalizedBatch


class AlpacaAssetNormalizer:
    """Normalize Alpaca assets into symbol_master rows."""

    source_name = "alpaca_trading"
    target_tables = ("symbol_master",)

    def __init__(self, *, as_of_date: date | None = None) -> None:
        self._as_of_date = as_of_date

    def normalize(self, records: list[RawRecord]) -> list[NormalizedBatch]:
        rows = []
        for record in records:
            payload = record.payload
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "as_of_date": (
                        self._as_of_date or record.pulled_at_utc.astimezone(timezone.utc).date()
                    ).isoformat(),
                    "symbol": payload.get("symbol"),
                    "security_id": payload.get("id"),
                    "company_name": payload.get("name"),
                    "exchange_mic": _map_exchange(payload.get("exchange")),
                    "currency": "USD",
                    "security_type": _normalize_security_type(payload.get("class")),
                    "asset_class": payload.get("class"),
                    "is_active": payload.get("status") == "active" and payload.get("tradable", False),
                    "list_date": None,
                    "delist_date": None,
                    "sector": None,
                    "industry": None,
                    "country_of_listing": "US",
                    "primary_share_class": None,
                    "source_name": record.source_name,
                    "load_time_utc": record.pulled_at_utc.isoformat(),
                    "source_version": None,
                }
            )

        return [NormalizedBatch(table_name="symbol_master", rows=tuple(rows))]


class AlpacaBarNormalizer:
    """Normalize Alpaca stock bars into daily_bar and benchmark_index rows."""

    source_name = "alpaca_market_data"
    target_tables = ("daily_bar", "benchmark_index")

    def __init__(self, *, benchmark_symbols: tuple[str, ...] = ("SPY",)) -> None:
        self._benchmark_symbols = set(benchmark_symbols)

    def normalize(self, records: list[RawRecord]) -> list[NormalizedBatch]:
        daily_rows = []
        benchmark_rows = []

        for record in records:
            payload = record.payload
            if not isinstance(payload, dict):
                continue

            symbol = _to_str(payload.get("symbol"))
            session_date = _extract_session_date(payload.get("t"))
            if not symbol or not session_date:
                continue

            row = {
                "session_date": session_date,
                "symbol": symbol,
                "open": _to_float(payload.get("o")),
                "high": _to_float(payload.get("h")),
                "low": _to_float(payload.get("l")),
                "close": _to_float(payload.get("c")),
                "volume": _to_float(payload.get("v")),
                "vwap": _to_float(payload.get("vw")),
                "dollar_volume": _compute_dollar_volume(payload),
                "trade_count": _to_int(payload.get("n")),
                "source_name": record.source_name,
                "load_time_utc": record.pulled_at_utc.isoformat(),
                "effective_time_utc": (
                    record.effective_time_utc.isoformat()
                    if isinstance(record.effective_time_utc, datetime)
                    else None
                ),
                "source_version": None,
            }

            if symbol in self._benchmark_symbols:
                benchmark_rows.append(
                    {
                        "session_date": row["session_date"],
                        "symbol": row["symbol"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "return_1d": None,
                        "source_name": row["source_name"],
                        "load_time_utc": row["load_time_utc"],
                        "source_version": None,
                    }
                )
            else:
                daily_rows.append(row)

        batches = [NormalizedBatch(table_name="daily_bar", rows=tuple(daily_rows))]
        if benchmark_rows:
            batches.append(NormalizedBatch(table_name="benchmark_index", rows=tuple(benchmark_rows)))
        return batches


class AlpacaAdjFactorNormalizer:
    """Build adj_factor rows from raw bars, adjusted bars, and corporate actions."""

    source_name = "alpaca_market_data"
    target_tables = ("adj_factor",)

    def normalize(
        self,
        *,
        raw_bar_records: list[RawRecord],
        adjusted_bar_records: list[RawRecord],
        corporate_action_records: list[RawRecord],
    ) -> list[NormalizedBatch]:
        raw_index = _index_bar_records(raw_bar_records)
        adjusted_index = _index_bar_records(adjusted_bar_records)
        action_index = _index_corporate_actions(corporate_action_records)

        rows = []
        for session_symbol in sorted(set(raw_index) | set(adjusted_index)):
            raw_state = raw_index.get(session_symbol)
            adjusted_state = adjusted_index.get(session_symbol)
            if raw_state is None and adjusted_state is None:
                continue

            session_date, symbol = session_symbol
            raw_close = raw_state.get("close") if raw_state else None
            adjusted_close = adjusted_state.get("close") if adjusted_state else raw_close
            if raw_close is None or raw_close <= 0:
                price_adjust_factor = 1.0
            else:
                price_adjust_factor = (
                    adjusted_close / raw_close
                    if adjusted_close is not None
                    else 1.0
                )

            action_state = action_index.get(session_symbol, {})
            rows.append(
                {
                    "session_date": session_date,
                    "symbol": symbol,
                    "split_factor": float(action_state.get("split_factor", 1.0)),
                    "cash_dividend": float(action_state.get("cash_dividend", 0.0)),
                    "price_adjust_factor": float(price_adjust_factor),
                    "source_name": self.source_name,
                    "load_time_utc": _latest_load_time(
                        raw_state.get("load_time_utc") if raw_state else None,
                        adjusted_state.get("load_time_utc") if adjusted_state else None,
                        action_state.get("load_time_utc"),
                    ),
                    "effective_time_utc": _latest_load_time(
                        raw_state.get("effective_time_utc") if raw_state else None,
                        adjusted_state.get("effective_time_utc") if adjusted_state else None,
                        action_state.get("effective_time_utc"),
                    ),
                    "source_version": "bars_and_corporate_actions_v1",
                }
            )

        return [NormalizedBatch(table_name="adj_factor", rows=tuple(rows))]


def _extract_session_date(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str) or len(raw_value) < 10:
        return None
    return raw_value[:10]


def _normalize_security_type(raw_value: Any) -> str:
    if raw_value == "us_equity":
        return "COMMON_STOCK"
    return "UNKNOWN"


def _map_exchange(raw_value: Any) -> str | None:
    mapping = {
        "NASDAQ": "XNAS",
        "NYSE": "XNYS",
        "AMEX": "XASE",
        "ARCA": "ARCX",
        "BATS": "BATS",
    }
    if not isinstance(raw_value, str):
        return None
    return mapping.get(raw_value.upper(), raw_value.upper())


def _to_str(raw_value: Any) -> str | None:
    return raw_value if isinstance(raw_value, str) else None


def _to_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    return float(raw_value)


def _to_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    return int(raw_value)


def _compute_dollar_volume(payload: dict[str, Any]) -> float | None:
    volume = _to_float(payload.get("v"))
    vwap = _to_float(payload.get("vw"))
    close = _to_float(payload.get("c"))
    if volume is None:
        return None
    reference_price = vwap if vwap is not None else close
    if reference_price is None:
        return None
    return reference_price * volume


def _index_bar_records(records: list[RawRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        payload = record.payload
        if not isinstance(payload, dict):
            continue
        symbol = _to_str(payload.get("symbol"))
        session_date = _extract_session_date(payload.get("t"))
        close = _to_float(payload.get("c"))
        if not symbol or not session_date or close is None:
            continue
        indexed[(session_date, symbol)] = {
            "close": close,
            "load_time_utc": record.pulled_at_utc.isoformat(),
            "effective_time_utc": (
                record.effective_time_utc.isoformat()
                if isinstance(record.effective_time_utc, datetime)
                else None
            ),
        }
    return indexed


def _index_corporate_actions(records: list[RawRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        payload = record.payload
        if not isinstance(payload, dict):
            continue
        symbol = _to_str(payload.get("symbol"))
        session_date = _to_str(payload.get("ex_date"))
        action_type = _to_str(payload.get("action_type"))
        if not symbol or not session_date or not action_type:
            continue

        state = indexed.setdefault(
            (session_date, symbol),
            {
                "cash_dividend": 0.0,
                "split_factor": 1.0,
                "load_time_utc": record.pulled_at_utc.isoformat(),
                "effective_time_utc": (
                    record.effective_time_utc.isoformat()
                    if isinstance(record.effective_time_utc, datetime)
                    else None
                ),
            },
        )
        if action_type == "cash_dividends":
            state["cash_dividend"] += float(payload.get("rate") or 0.0)
        elif "split" in action_type:
            old_rate = float(payload.get("old_rate") or 1.0)
            new_rate = float(payload.get("new_rate") or 1.0)
            if old_rate > 0:
                state["split_factor"] *= new_rate / old_rate
    return indexed


def _latest_load_time(*values: Any) -> str | None:
    timestamps = [
        value
        for value in values
        if isinstance(value, str) and value
    ]
    if not timestamps:
        return None
    return max(timestamps)
