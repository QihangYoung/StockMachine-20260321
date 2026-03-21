from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import sleep
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from stockmachine.live.reconciler import BrokerOrderSnapshot

TERMINAL_ORDER_STATUSES = {"filled", "canceled", "cancelled", "rejected", "expired"}


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class OrderUpdateEvent:
    """Normalized order update event suitable for streaming or polling."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    change_type: str
    status: str
    previous_status: str | None
    quantity: int
    filled_quantity: int
    previous_filled_quantity: int
    avg_fill_price: float | None
    updated_at_utc: datetime
    observed_at_utc: datetime = field(default_factory=_default_now)
    source: str = "polling"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TradeUpdatePollResult:
    """Result from a bounded polling fallback loop."""

    events: tuple[OrderUpdateEvent, ...]
    polls: int
    stopped_reason: str
    open_order_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class TradeUpdateStreamResult:
    """Unified result for websocket-first streaming with polling fallback."""

    events: tuple[OrderUpdateEvent, ...]
    source_mode: str
    stopped_reason: str
    websocket_messages: int = 0
    polls: int = 0
    fallback_used: bool = False
    websocket_error: str | None = None
    open_order_ids: tuple[str, ...] = ()
    seed_order_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [_event_to_dict(event) for event in self.events],
            "source_mode": self.source_mode,
            "stopped_reason": self.stopped_reason,
            "websocket_messages": self.websocket_messages,
            "polls": self.polls,
            "fallback_used": self.fallback_used,
            "websocket_error": self.websocket_error,
            "open_order_ids": list(self.open_order_ids),
            "seed_order_ids": list(self.seed_order_ids),
        }


class TradeUpdateMessageSource(Protocol):
    """Transport seam that yields raw websocket messages."""

    def iter_messages(self) -> Iterator[object]:
        """Yield raw websocket frames or decoded payloads."""


def iter_trade_update_events(
    fetch_orders: Callable[[], Sequence[BrokerOrderSnapshot | Mapping[str, Any] | object]],
    *,
    stream_source: TradeUpdateMessageSource | None = None,
    max_polls: int | None = None,
    idle_polls_to_stop: int = 1,
    sleep_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = sleep,
    source: str = "polling",
) -> Iterator[OrderUpdateEvent]:
    """Yield trade update events from repeated broker order snapshots."""

    if stream_source is not None:
        result = stream_trade_updates(
            stream_source=stream_source,
            fetch_orders=fetch_orders,
            max_polls=max_polls,
            idle_polls_to_stop=idle_polls_to_stop,
            sleep_seconds=sleep_seconds,
            sleep_fn=sleep_fn,
            source=source,
        )
        for event in result.events:
            yield event
        return

    for event in poll_trade_updates(
        fetch_orders,
        max_polls=max_polls,
        idle_polls_to_stop=idle_polls_to_stop,
        sleep_seconds=sleep_seconds,
        sleep_fn=sleep_fn,
        source=source,
    ).events:
        yield event


def poll_trade_updates(
    fetch_orders: Callable[[], Sequence[BrokerOrderSnapshot | Mapping[str, Any] | object]],
    *,
    initial_snapshots: Sequence[BrokerOrderSnapshot | Mapping[str, Any] | object] | None = None,
    max_polls: int | None = None,
    idle_polls_to_stop: int = 1,
    sleep_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = sleep,
    source: str = "polling",
) -> TradeUpdatePollResult:
    """Poll broker order snapshots until the stream becomes idle or is capped."""

    if idle_polls_to_stop < 1:
        raise ValueError("idle_polls_to_stop must be at least 1.")
    if max_polls is not None and max_polls < 1:
        raise ValueError("max_polls must be at least 1 when provided.")

    previous_by_order_id = _snapshots_by_order_id(initial_snapshots or ())
    events: list[OrderUpdateEvent] = []
    idle_polls = 0
    polls = 0
    stopped_reason = "idle"

    while True:
        current_snapshots = tuple(
            snapshot if isinstance(snapshot, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(snapshot)
            for snapshot in fetch_orders()
        )
        current_by_order_id = {snapshot.order_id: snapshot for snapshot in current_snapshots}
        current_events = _diff_order_snapshots(previous_by_order_id, current_by_order_id, source=source)
        events.extend(current_events)
        polls += 1
        previous_by_order_id = current_by_order_id

        if current_events:
            idle_polls = 0
        else:
            idle_polls += 1

        if max_polls is not None and polls >= max_polls:
            stopped_reason = "max_polls"
            break
        if idle_polls >= idle_polls_to_stop:
            stopped_reason = "idle"
            break

        if sleep_seconds > 0:
            sleep_fn(sleep_seconds)
    else:
        stopped_reason = "idle"

    return TradeUpdatePollResult(
        events=tuple(events),
        polls=polls,
        stopped_reason=stopped_reason,
        open_order_ids=tuple(sorted(order_id for order_id, snapshot in previous_by_order_id.items() if snapshot.status.lower() not in TERMINAL_ORDER_STATUSES)),
    )


def stream_trade_updates(
    *,
    stream_source: TradeUpdateMessageSource | None = None,
    fetch_orders: Callable[[], Sequence[BrokerOrderSnapshot | Mapping[str, Any] | object]] | None = None,
    max_messages: int | None = None,
    max_polls: int | None = None,
    idle_messages_to_stop: int = 1,
    idle_polls_to_stop: int = 1,
    sleep_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = sleep,
    source: str = "websocket",
) -> TradeUpdateStreamResult:
    """Stream websocket updates first and fall back to polling when needed."""

    if stream_source is None:
        if fetch_orders is None:
            return TradeUpdateStreamResult(
                events=(),
                source_mode="none",
                stopped_reason="no_source_available",
            )
        polling_result = poll_trade_updates(
            fetch_orders,
            max_polls=max_polls,
            idle_polls_to_stop=idle_polls_to_stop,
            sleep_seconds=sleep_seconds,
            sleep_fn=sleep_fn,
            source=source,
        )
        return TradeUpdateStreamResult(
            events=polling_result.events,
            source_mode="polling",
            stopped_reason=polling_result.stopped_reason,
            polls=polling_result.polls,
            open_order_ids=polling_result.open_order_ids,
        )

    websocket_result = _consume_stream_updates(
        stream_source,
        max_messages=max_messages,
        idle_messages_to_stop=idle_messages_to_stop,
        source=source,
    )

    should_fallback = fetch_orders is not None and (
        websocket_result.websocket_error is not None
        or websocket_result.stopped_reason in {"idle", "exhausted"}
    )
    if not should_fallback:
        return TradeUpdateStreamResult(
            events=websocket_result.events,
            source_mode="websocket",
            stopped_reason=websocket_result.stopped_reason,
            websocket_messages=websocket_result.websocket_messages,
            open_order_ids=websocket_result.open_order_ids,
            websocket_error=websocket_result.websocket_error,
        )

    if fetch_orders is None:
        return TradeUpdateStreamResult(
            events=websocket_result.events,
            source_mode="websocket",
            stopped_reason=websocket_result.stopped_reason,
            websocket_messages=websocket_result.websocket_messages,
            open_order_ids=websocket_result.open_order_ids,
            websocket_error=websocket_result.websocket_error,
        )

    polling_result = poll_trade_updates(
        fetch_orders,
        initial_snapshots=websocket_result.current_snapshots,
        max_polls=max_polls,
        idle_polls_to_stop=idle_polls_to_stop,
        sleep_seconds=sleep_seconds,
        sleep_fn=sleep_fn,
        source="polling",
    )
    return TradeUpdateStreamResult(
        events=websocket_result.events + polling_result.events,
        source_mode="hybrid",
        stopped_reason=polling_result.stopped_reason,
        websocket_messages=websocket_result.websocket_messages,
        polls=polling_result.polls,
        fallback_used=True,
        websocket_error=websocket_result.websocket_error,
        open_order_ids=polling_result.open_order_ids,
        seed_order_ids=tuple(sorted(snapshot.order_id for snapshot in websocket_result.current_snapshots)),
    )


def _diff_order_snapshots(
    previous_by_order_id: Mapping[str, BrokerOrderSnapshot],
    current_by_order_id: Mapping[str, BrokerOrderSnapshot],
    *,
    source: str,
) -> list[OrderUpdateEvent]:
    events: list[OrderUpdateEvent] = []
    observed_at_utc = _default_now()

    for order_id, current in sorted(current_by_order_id.items(), key=lambda item: (item[1].symbol, item[0])):
        previous = previous_by_order_id.get(order_id)
        if previous is None:
            change_type = "new"
            previous_status = None
            previous_filled_quantity = 0
        else:
            previous_status = previous.status
            previous_filled_quantity = previous.filled_quantity
            if current.filled_quantity > previous.filled_quantity:
                change_type = "fill"
            elif current.status != previous.status:
                change_type = "status"
            elif current.avg_fill_price != previous.avg_fill_price:
                change_type = "price"
            else:
                continue

        events.append(
            OrderUpdateEvent(
                order_id=current.order_id,
                client_order_id=current.client_order_id,
                symbol=current.symbol,
                side=current.side,
                change_type=change_type,
                status=current.status,
                previous_status=previous_status,
                quantity=current.quantity,
                filled_quantity=current.filled_quantity,
                previous_filled_quantity=previous_filled_quantity,
                avg_fill_price=current.avg_fill_price,
                updated_at_utc=current.updated_at_utc,
                observed_at_utc=observed_at_utc,
                source=source,
                raw=current.raw,
            )
        )

    for order_id, previous in sorted(previous_by_order_id.items(), key=lambda item: (item[1].symbol, item[0])):
        if order_id in current_by_order_id:
            continue
        events.append(
            OrderUpdateEvent(
                order_id=previous.order_id,
                client_order_id=previous.client_order_id,
                symbol=previous.symbol,
                side=previous.side,
                change_type="missing",
                status="missing",
                previous_status=previous.status,
                quantity=previous.quantity,
                filled_quantity=0,
                previous_filled_quantity=previous.filled_quantity,
                avg_fill_price=previous.avg_fill_price,
                updated_at_utc=previous.updated_at_utc,
                observed_at_utc=observed_at_utc,
                source=source,
                raw=previous.raw,
            )
        )

    return events


@dataclass(slots=True, frozen=True)
class _StreamConsumption:
    events: tuple[OrderUpdateEvent, ...]
    current_snapshots: tuple[BrokerOrderSnapshot, ...]
    websocket_messages: int
    stopped_reason: str
    open_order_ids: tuple[str, ...]
    websocket_error: str | None = None


def _consume_stream_updates(
    stream_source: TradeUpdateMessageSource,
    *,
    max_messages: int | None,
    idle_messages_to_stop: int,
    source: str,
) -> _StreamConsumption:
    if idle_messages_to_stop < 1:
        raise ValueError("idle_messages_to_stop must be at least 1.")
    if max_messages is not None and max_messages < 1:
        raise ValueError("max_messages must be at least 1 when provided.")

    events: list[OrderUpdateEvent] = []
    previous_by_order_id: dict[str, BrokerOrderSnapshot] = {}
    websocket_messages = 0
    idle_messages = 0
    stopped_reason = "idle"
    websocket_error: str | None = None

    try:
        for raw_message in stream_source.iter_messages():
            snapshots = _normalize_trade_update_snapshots(raw_message)
            current_by_order_id = {snapshot.order_id: snapshot for snapshot in snapshots}
            current_events = _diff_order_snapshots(previous_by_order_id, current_by_order_id, source=source)
            events.extend(current_events)
            websocket_messages += 1
            previous_by_order_id = current_by_order_id

            if current_events:
                idle_messages = 0
            else:
                idle_messages += 1

            if max_messages is not None and websocket_messages >= max_messages:
                stopped_reason = "max_messages"
                break
            if idle_messages >= idle_messages_to_stop:
                stopped_reason = "idle"
                break
        else:
            stopped_reason = "exhausted"
    except Exception as exc:
        websocket_error = f"{type(exc).__name__}: {exc}"
        stopped_reason = "websocket_error"

    return _StreamConsumption(
        events=tuple(events),
        current_snapshots=tuple(previous_by_order_id.values()),
        websocket_messages=websocket_messages,
        stopped_reason=stopped_reason,
        open_order_ids=tuple(
            sorted(
                order_id
                for order_id, snapshot in previous_by_order_id.items()
                if snapshot.status.lower() not in TERMINAL_ORDER_STATUSES
            )
        ),
        websocket_error=websocket_error,
    )


def _normalize_trade_update_snapshots(
    payload: BrokerOrderSnapshot | Mapping[str, Any] | Sequence[object] | object,
) -> tuple[BrokerOrderSnapshot, ...]:
    if isinstance(payload, BrokerOrderSnapshot):
        return (payload,)
    if isinstance(payload, Mapping):
        nested = _extract_trade_update_snapshot_candidates(payload)
        if nested:
            return nested
        if _looks_like_order_snapshot(payload):
            return (BrokerOrderSnapshot.from_payload(payload),)
        raise ValueError("Unable to normalize trade update payload.")
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        snapshots: list[BrokerOrderSnapshot] = []
        for item in payload:
            snapshots.extend(_normalize_trade_update_snapshots(item))
        if snapshots:
            return tuple(snapshots)
        raise ValueError("Unable to normalize empty trade update payload sequence.")
    if isinstance(payload, (str, bytes, bytearray)):
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
        return _normalize_trade_update_snapshots(json.loads(text))
    raise TypeError(f"Unsupported trade update payload type: {type(payload)!r}")


def _extract_trade_update_snapshot_candidates(payload: Mapping[str, Any]) -> tuple[BrokerOrderSnapshot, ...]:
    candidates: list[BrokerOrderSnapshot] = []
    for key in ("order", "data", "payload", "event"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping) and _looks_like_order_snapshot(value):
            candidates.append(BrokerOrderSnapshot.from_payload(value))
            continue
        if isinstance(value, Mapping):
            candidates.extend(_normalize_trade_update_snapshots(value))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                candidates.extend(_normalize_trade_update_snapshots(item))
            continue
    if candidates:
        return tuple(candidates)
    orders_value = payload.get("orders")
    if isinstance(orders_value, Sequence) and not isinstance(orders_value, (str, bytes, bytearray)):
        for item in orders_value:
            candidates.extend(_normalize_trade_update_snapshots(item))
    return tuple(candidates)


def _looks_like_order_snapshot(payload: Mapping[str, Any]) -> bool:
    keys = set(payload.keys())
    return bool(
        {"id", "symbol", "side"} <= keys
        or {"order_id", "symbol", "side"} <= keys
        or {"id", "symbol"} <= keys
    )


def _snapshots_by_order_id(
    snapshots: Sequence[BrokerOrderSnapshot | Mapping[str, Any] | object],
) -> dict[str, BrokerOrderSnapshot]:
    by_order_id: dict[str, BrokerOrderSnapshot] = {}
    for snapshot in snapshots:
        normalized = snapshot if isinstance(snapshot, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(snapshot)
        by_order_id[normalized.order_id] = normalized
    return by_order_id


def _event_to_dict(event: OrderUpdateEvent) -> dict[str, Any]:
    return {
        "order_id": event.order_id,
        "client_order_id": event.client_order_id,
        "symbol": event.symbol,
        "side": event.side,
        "change_type": event.change_type,
        "status": event.status,
        "previous_status": event.previous_status,
        "quantity": event.quantity,
        "filled_quantity": event.filled_quantity,
        "previous_filled_quantity": event.previous_filled_quantity,
        "avg_fill_price": event.avg_fill_price,
        "updated_at_utc": event.updated_at_utc.isoformat(),
        "observed_at_utc": event.observed_at_utc.isoformat(),
        "source": event.source,
        "raw": dict(event.raw),
    }
