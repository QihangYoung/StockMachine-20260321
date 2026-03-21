from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class RunRecord:
    """Top-level run metadata stored in the local ledger."""

    run_id: str
    strategy_name: str
    market: str
    created_at_utc: datetime
    status: str = "open"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RunManifestRecord:
    """Persisted paper-run manifest used for audit and replay."""

    run_id: str
    session_date: date
    strategy_name: str
    model_name: str
    generated_at_utc: datetime
    client_order_id_prefix: str = "smk"
    dry_run: bool = False
    data_snapshot: dict[str, Any] = field(default_factory=dict)
    risk_policy: dict[str, Any] = field(default_factory=dict)
    execution_policy: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SignalRecord:
    """Persisted signal row."""

    run_id: str | None
    session_date: date
    symbol: str
    side: str
    score: float
    confidence: float
    horizon_bars: int
    timestamp_utc: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TargetRecord:
    """Persisted target-position row."""

    run_id: str | None
    session_date: date
    symbol: str
    target_weight: float
    max_weight: float
    reason: str
    timestamp_utc: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderDecisionRecord:
    """Persisted pre-trade decision row."""

    decision_id: str
    run_id: str | None
    session_date: date
    client_order_id: str
    symbol: str
    side: str
    decision_type: str
    decision_price: float | None
    estimated_notional: float
    approved: bool
    reason: str
    decision_at_utc: datetime
    slippage_bps: float | None = None
    fee_estimate: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderRecord:
    """Persisted broker order row."""

    order_id: str
    run_id: str | None
    session_date: date | None
    client_order_id: str | None
    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    status: str
    filled_quantity: int
    avg_fill_price: float | None
    submitted_at_utc: datetime
    updated_at_utc: datetime
    broker_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FillRecord:
    """Persisted fill row."""

    fill_id: str
    order_id: str
    run_id: str | None
    symbol: str
    side: str
    quantity: int
    price: float
    filled_at_utc: datetime
    expected_price: float | None = None
    slippage: float | None = None
    fee: float | None = None
    client_order_id: str | None = None
    broker_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FillAuditRecord:
    """Persisted fill-quality audit row."""

    audit_id: str
    order_id: str
    run_id: str | None
    session_date: date | None
    client_order_id: str | None
    symbol: str
    side: str
    quantity: int
    expected_price: float | None
    price: float
    slippage: float | None
    fee: float | None
    filled_at_utc: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EquitySnapshotRecord:
    """Persisted account snapshot row."""

    run_id: str | None
    session_date: date
    timestamp_utc: datetime
    cash: float
    equity: float
    gross_exposure: float
    payload: dict[str, Any] = field(default_factory=dict)
