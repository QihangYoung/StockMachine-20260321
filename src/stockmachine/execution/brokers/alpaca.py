from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from stockmachine.data.vendors.alpaca import AlpacaCredentials
from stockmachine.domain.datetime_utils import parse_iso_datetime_like


class AlpacaBrokerError(RuntimeError):
    """Raised when an Alpaca trading API request fails."""


@dataclass(slots=True, frozen=True)
class BrokerClock:
    """Normalized Alpaca clock payload."""

    timestamp: datetime
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BrokerAccount:
    """Normalized Alpaca account payload."""

    account_id: str
    status: str
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float
    long_market_value: float
    short_market_value: float
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BrokerPosition:
    """Normalized Alpaca position payload."""

    symbol: str
    quantity: float
    market_value: float
    avg_entry_price: float
    cost_basis: float
    unrealized_pl: float
    side: str | None = None
    current_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BrokerOrder:
    """Normalized Alpaca order payload."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    status: str
    quantity: float | None
    filled_quantity: float | None
    filled_avg_price: float | None
    limit_price: float | None
    submitted_at: datetime | None
    updated_at: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AlpacaTradingAdapter:
    """Domain-friendly Alpaca Trading API adapter."""

    credentials: AlpacaCredentials

    @classmethod
    def from_env(cls) -> "AlpacaTradingAdapter":
        return cls(credentials=AlpacaCredentials.from_env())

    def is_paper_trading_environment(self) -> bool:
        base_url = str(self.credentials.trading_base_url).lower()
        return "paper" in base_url

    def get_account(self) -> BrokerAccount:
        payload = self._request_json("GET", self._trading_url("/v2/account"))
        if not isinstance(payload, dict):
            raise AlpacaBrokerError("Unexpected Alpaca account payload shape.")
        return _parse_account(payload)

    def get_clock(self) -> BrokerClock:
        payload = self._request_json("GET", self._trading_url("/v2/clock"))
        if not isinstance(payload, dict):
            raise AlpacaBrokerError("Unexpected Alpaca clock payload shape.")
        return _parse_clock(payload)

    def list_positions(self) -> list[BrokerPosition]:
        payload = self._request_json("GET", self._trading_url("/v2/positions"))
        if not isinstance(payload, list):
            raise AlpacaBrokerError("Unexpected Alpaca positions payload shape.")
        return [_parse_position(item) for item in payload if isinstance(item, dict)]

    def list_orders(
        self,
        *,
        status: str | None = None,
        symbols: list[str] | None = None,
        limit: int | None = None,
        nested: bool | None = None,
        after: str | None = None,
        until: str | None = None,
        direction: str | None = None,
    ) -> list[BrokerOrder]:
        query: dict[str, Any] = {}
        if status is not None:
            query["status"] = status
        if symbols:
            query["symbols"] = ",".join(symbols)
        if limit is not None:
            query["limit"] = limit
        if nested is not None:
            query["nested"] = str(bool(nested)).lower()
        if after is not None:
            query["after"] = after
        if until is not None:
            query["until"] = until
        if direction is not None:
            query["direction"] = direction

        payload = self._request_json("GET", self._trading_url("/v2/orders", query))
        if not isinstance(payload, list):
            raise AlpacaBrokerError("Unexpected Alpaca orders payload shape.")
        return [_parse_order(item) for item in payload if isinstance(item, dict)]

    def get_order(self, order_id: str) -> BrokerOrder:
        payload = self._request_json("GET", self._trading_url(f"/v2/orders/{order_id}"))
        if not isinstance(payload, dict):
            raise AlpacaBrokerError("Unexpected Alpaca order payload shape.")
        return _parse_order(payload)

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float | None = None,
        notional: float | None = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        client_order_id: str | None = None,
        extended_hours: bool | None = None,
    ) -> BrokerOrder:
        if quantity is None and notional is None:
            raise AlpacaBrokerError("Either quantity or notional must be provided.")

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if quantity is not None:
            payload["qty"] = str(quantity)
        if notional is not None:
            payload["notional"] = str(notional)
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if client_order_id is not None:
            payload["client_order_id"] = client_order_id
        if extended_hours is not None:
            payload["extended_hours"] = extended_hours

        response = self._request_json(
            "POST",
            self._trading_url("/v2/orders"),
            payload=payload,
        )
        if not isinstance(response, dict):
            raise AlpacaBrokerError("Unexpected Alpaca submit order payload shape.")
        return _parse_order(response)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._request_json("DELETE", self._trading_url(f"/v2/orders/{order_id}"))
        return self.get_order(order_id)

    def _trading_url(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.credentials.trading_base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        return url

    def _request_json(self, method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.credentials.max_retries + 1):
            request = Request(
                url=url,
                data=None if payload is None else json.dumps(payload).encode("utf-8"),
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "APCA-API-KEY-ID": self.credentials.api_key_id,
                    "APCA-API-SECRET-KEY": self.credentials.api_secret_key,
                },
                method=method,
            )
            try:
                with urlopen(request, timeout=self.credentials.request_timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    if not body:
                        return {}
                    return json.loads(body)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                last_error = AlpacaBrokerError(
                    f"Alpaca trading request failed with HTTP {exc.code}: {detail}"
                )
                if not retryable or attempt >= self.credentials.max_retries:
                    raise last_error from exc
            except (TimeoutError, URLError, IncompleteRead) as exc:
                reason = exc.reason if isinstance(exc, URLError) else "read timeout"
                if isinstance(exc, IncompleteRead):
                    reason = "incomplete read"
                last_error = AlpacaBrokerError(f"Failed to reach Alpaca trading API: {reason}")
                if attempt >= self.credentials.max_retries:
                    raise last_error from exc

            time.sleep(min(2**attempt, 5))

        if last_error is not None:
            raise last_error
        raise AlpacaBrokerError("Alpaca trading request failed without a captured error.")


_BUY_RETRY_KEYWORDS = (
    "insufficient buying power",
    "buying power",
    "insufficient funds",
    "notional",
    "order notional",
)


def is_retryable_buy_rejection(error: Exception, *, side: str) -> bool:
    """Return True when a buy order rejection looks shrinkable and worth one retry."""

    if str(side).upper() != "BUY":
        return False
    return classify_buy_retry_reason(error) is not None


def classify_buy_retry_reason(error: Exception) -> str | None:
    """Map a broker rejection into a retry reason when shrinking might help."""

    message = str(error).lower()
    if "insufficient buying power" in message or "buying power" in message or "insufficient funds" in message:
        return "insufficient_buying_power"
    if "order notional" in message or "notional" in message:
        return "order_notional_exceeded"
    return None


def shrink_quantity_for_retry(quantity: int, *, shrink_ratio: float = 0.5) -> int:
    """Shrink a quantity for a single buy retry while keeping at least one share."""

    if quantity <= 1:
        return 0
    ratio = min(max(float(shrink_ratio), 0.0), 1.0)
    shrunk = int(quantity * ratio)
    if shrunk >= quantity:
        shrunk = quantity - 1
    return max(1, shrunk)


def _parse_account(payload: dict[str, Any]) -> BrokerAccount:
    return BrokerAccount(
        account_id=str(payload.get("id", "")),
        status=str(payload.get("status", "")),
        cash=_to_float(payload.get("cash")),
        equity=_to_float(payload.get("equity")),
        buying_power=_to_float(payload.get("buying_power")),
        portfolio_value=_to_float(payload.get("portfolio_value")),
        long_market_value=_to_float(payload.get("long_market_value")),
        short_market_value=_to_float(payload.get("short_market_value")),
        pattern_day_trader=_to_bool(payload.get("pattern_day_trader")),
        trading_blocked=_to_bool(payload.get("trading_blocked")),
        account_blocked=_to_bool(payload.get("account_blocked")),
        raw=payload,
    )


def _parse_clock(payload: dict[str, Any]) -> BrokerClock:
    return BrokerClock(
        timestamp=_to_datetime(payload.get("timestamp")) or datetime.utcnow(),
        is_open=_to_bool(payload.get("is_open")),
        next_open=_to_datetime(payload.get("next_open")),
        next_close=_to_datetime(payload.get("next_close")),
        raw=payload,
    )


def _parse_position(payload: dict[str, Any]) -> BrokerPosition:
    return BrokerPosition(
        symbol=str(payload.get("symbol", "")),
        quantity=_to_float(payload.get("qty")),
        market_value=_to_float(payload.get("market_value")),
        avg_entry_price=_to_float(payload.get("avg_entry_price")),
        cost_basis=_to_float(payload.get("cost_basis")),
        unrealized_pl=_to_float(payload.get("unrealized_pl")),
        side=str(payload.get("side")) if payload.get("side") is not None else None,
        current_price=_to_float(payload.get("current_price"))
        if payload.get("current_price") is not None
        else None,
        raw=payload,
    )


def _parse_order(payload: dict[str, Any]) -> BrokerOrder:
    return BrokerOrder(
        order_id=str(payload.get("id", "")),
        client_order_id=str(payload.get("client_order_id"))
        if payload.get("client_order_id") is not None
        else None,
        symbol=str(payload.get("symbol", "")),
        side=str(payload.get("side", "")),
        order_type=str(payload.get("type", "")),
        time_in_force=str(payload.get("time_in_force", "")),
        status=str(payload.get("status", "")),
        quantity=_to_float(payload.get("qty")) if payload.get("qty") is not None else None,
        filled_quantity=_to_float(payload.get("filled_qty"))
        if payload.get("filled_qty") is not None
        else None,
        filled_avg_price=_to_float(payload.get("filled_avg_price"))
        if payload.get("filled_avg_price") is not None
        else None,
        limit_price=_to_float(payload.get("limit_price"))
        if payload.get("limit_price") is not None
        else None,
        submitted_at=_to_datetime(payload.get("submitted_at")),
        updated_at=_to_datetime(payload.get("updated_at")),
        raw=payload,
    )


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _to_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return parse_iso_datetime_like(value)
