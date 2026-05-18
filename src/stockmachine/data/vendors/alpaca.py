from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class AlpacaRequestError(RuntimeError):
    """Raised when an Alpaca API request fails."""


@dataclass(slots=True, frozen=True)
class AlpacaCredentials:
    """Credentials and base URLs required for Alpaca requests."""

    api_key_id: str
    api_secret_key: str
    trading_base_url: str = "https://paper-api.alpaca.markets"
    data_base_url: str = "https://data.alpaca.markets"
    request_timeout_seconds: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        _load_local_env_file()
        api_key_id = os.getenv("ALPACA_API_KEY_ID")
        api_secret_key = os.getenv("ALPACA_API_SECRET_KEY")
        if not api_key_id or not api_secret_key:
            raise AlpacaRequestError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY before running collectors."
            )

        return cls(
            api_key_id=api_key_id,
            api_secret_key=api_secret_key,
            trading_base_url=os.getenv(
                "ALPACA_TRADING_BASE_URL",
                "https://paper-api.alpaca.markets",
            ).rstrip("/"),
            data_base_url=os.getenv(
                "ALPACA_DATA_BASE_URL",
                "https://data.alpaca.markets",
            ).rstrip("/"),
            request_timeout_seconds=float(os.getenv("ALPACA_HTTP_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("ALPACA_HTTP_MAX_RETRIES", "2")),
        )


class AlpacaHttpClient:
    """Small HTTP client for Alpaca trading and market data endpoints."""

    def __init__(self, credentials: AlpacaCredentials) -> None:
        self._credentials = credentials

    def list_assets(
        self,
        *,
        status: str = "active",
        asset_class: str = "us_equity",
    ) -> list[dict[str, Any]]:
        payload = self._get_trading(
            "/v2/assets",
            {"status": status, "asset_class": asset_class},
        )
        if not isinstance(payload, list):
            raise AlpacaRequestError("Unexpected Alpaca assets payload shape.")
        return payload

    def get_stock_bars(
        self,
        *,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
        adjustment: str = "raw",
        feed: str = "iex",
        limit: int = 10000,
        asof: str | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "adjustment": adjustment,
            "feed": feed,
            "limit": limit,
            "sort": "asc",
        }
        if asof:
            query["asof"] = asof

        bars: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while True:
            page_query = dict(query)
            if next_page_token:
                page_query["page_token"] = next_page_token
            payload = self._get_data("/v2/stocks/bars", page_query)
            bar_map = payload.get("bars", {})
            for symbol, symbol_bars in bar_map.items():
                if not isinstance(symbol_bars, list):
                    continue
                for bar in symbol_bars:
                    if isinstance(bar, dict):
                        bars.append({"symbol": symbol, **bar})

            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break

        return bars

    def get_corporate_actions(
        self,
        *,
        symbols: list[str],
        start: str,
        end: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "start": start,
            "end": end,
            "limit": limit,
        }

        actions: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while True:
            page_query = dict(query)
            if next_page_token:
                page_query["page_token"] = next_page_token
            payload = self._get_data("/v1/corporate-actions", page_query)
            action_groups = payload.get("corporate_actions", {})
            if not isinstance(action_groups, dict):
                raise AlpacaRequestError("Unexpected Alpaca corporate actions payload shape.")

            for action_type, items in action_groups.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        actions.append({"action_type": action_type, **item})

            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break

        return actions

    def _get_trading(self, path: str, query: dict[str, Any]) -> Any:
        url = f"{self._credentials.trading_base_url}{path}?{urlencode(query)}"
        return self._request(url)

    def _get_data(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._credentials.data_base_url}{path}?{urlencode(query)}"
        payload = self._request(url)
        if not isinstance(payload, dict):
            raise AlpacaRequestError("Unexpected Alpaca market data payload shape.")
        return payload

    def _request(self, url: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._credentials.max_retries + 1):
            request = Request(
                url=url,
                headers={
                    "accept": "application/json",
                    "APCA-API-KEY-ID": self._credentials.api_key_id,
                    "APCA-API-SECRET-KEY": self._credentials.api_secret_key,
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=self._credentials.request_timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                last_error = AlpacaRequestError(
                    f"Alpaca request failed with HTTP {exc.code}: {detail}"
                )
                if not retryable or attempt >= self._credentials.max_retries:
                    raise last_error from exc
            except (TimeoutError, URLError, IncompleteRead) as exc:
                reason = exc.reason if isinstance(exc, URLError) else "read timeout"
                if isinstance(exc, IncompleteRead):
                    reason = "incomplete read"
                last_error = AlpacaRequestError(f"Failed to reach Alpaca: {reason}")
                if attempt >= self._credentials.max_retries:
                    raise last_error from exc

            time.sleep(min(2 ** attempt, 5))

        if last_error is not None:
            raise last_error
        raise AlpacaRequestError("Alpaca request failed without a captured error.")


def _load_local_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
