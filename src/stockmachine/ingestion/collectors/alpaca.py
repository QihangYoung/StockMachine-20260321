from __future__ import annotations

from datetime import datetime, timezone

from stockmachine.data.vendors import AlpacaHttpClient

from .base import FetchWindow, RawRecord


class AlpacaAssetCollector:
    """Collect US equity asset metadata from Alpaca."""

    source_name = "alpaca_trading"
    supported_streams = ("assets",)

    def __init__(self, client: AlpacaHttpClient) -> None:
        self._client = client

    def collect(self, window: FetchWindow) -> list[RawRecord]:
        if window.stream_name != "assets":
            raise ValueError(f"Unsupported stream for assets collector: {window.stream_name}")

        pulled_at_utc = datetime.now(timezone.utc)
        assets = self._client.list_assets(status="active", asset_class="us_equity")
        return [
            RawRecord(
                source_name=self.source_name,
                stream_name="assets",
                pulled_at_utc=pulled_at_utc,
                payload=asset,
                symbol=asset.get("symbol"),
                raw_key=asset.get("id") or asset.get("symbol"),
            )
            for asset in assets
            if isinstance(asset, dict)
        ]


class AlpacaStockBarCollector:
    """Collect US equity daily bars from Alpaca market data."""

    source_name = "alpaca_market_data"
    supported_streams = ("daily_bars",)

    def __init__(
        self,
        client: AlpacaHttpClient,
        *,
        adjustment: str = "raw",
        feed: str = "iex",
    ) -> None:
        self._client = client
        self._adjustment = adjustment
        self._feed = feed

    def collect(self, window: FetchWindow) -> list[RawRecord]:
        if window.stream_name != "daily_bars":
            raise ValueError(f"Unsupported stream for bar collector: {window.stream_name}")
        if not window.start_date or not window.end_date:
            raise ValueError("Bar collection requires both start_date and end_date.")
        if not window.symbols:
            raise ValueError("Bar collection requires at least one symbol.")

        pulled_at_utc = datetime.now(timezone.utc)
        bars = self._client.get_stock_bars(
            symbols=list(window.symbols),
            start=window.start_date.isoformat(),
            end=window.end_date.isoformat(),
            adjustment=self._adjustment,
            feed=self._feed,
        )
        return [
            RawRecord(
                source_name=self.source_name,
                stream_name="daily_bars",
                pulled_at_utc=pulled_at_utc,
                payload=bar,
                symbol=bar.get("symbol"),
                effective_time_utc=_parse_timestamp(bar.get("t")),
                raw_key=f"{bar.get('symbol')}:{bar.get('t')}",
                meta={"adjustment": self._adjustment, "feed": self._feed},
            )
            for bar in bars
            if isinstance(bar, dict)
        ]


class AlpacaCorporateActionsCollector:
    """Collect Alpaca corporate actions for a symbol set and date window."""

    source_name = "alpaca_market_data"
    supported_streams = ("corporate_actions",)

    def __init__(self, client: AlpacaHttpClient) -> None:
        self._client = client

    def collect(self, window: FetchWindow) -> list[RawRecord]:
        if window.stream_name != "corporate_actions":
            raise ValueError(f"Unsupported stream for corporate actions collector: {window.stream_name}")
        if not window.start_date or not window.end_date:
            raise ValueError("Corporate action collection requires both start_date and end_date.")
        if not window.symbols:
            raise ValueError("Corporate action collection requires at least one symbol.")

        pulled_at_utc = datetime.now(timezone.utc)
        actions = self._client.get_corporate_actions(
            symbols=list(window.symbols),
            start=window.start_date.isoformat(),
            end=window.end_date.isoformat(),
        )
        return [
            RawRecord(
                source_name=self.source_name,
                stream_name="corporate_actions",
                pulled_at_utc=pulled_at_utc,
                payload=action,
                symbol=action.get("symbol"),
                effective_time_utc=_parse_event_date(action.get("ex_date")),
                raw_key=str(action.get("id")) if action.get("id") else None,
            )
            for action in actions
            if isinstance(action, dict)
        ]


def _parse_timestamp(raw_value: object) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value:
        return None
    normalized = raw_value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _parse_event_date(raw_value: object) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value:
        return None
    return datetime.fromisoformat(f"{raw_value}T00:00:00+00:00")
