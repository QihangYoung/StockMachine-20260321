from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Protocol, Sequence

from stockmachine.domain.models import OrderIntent, Signal, TargetPosition


@dataclass(slots=True, frozen=True)
class MarketBar:
    """Market data required by the backtest engine for one session."""

    session_date: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    adj_open: float | None = None


@dataclass(slots=True, frozen=True)
class PositionSnapshot:
    """Account-level position state for one symbol."""

    symbol: str
    quantity: int
    market_value: float
    weight: float


@dataclass(slots=True, frozen=True)
class AccountSnapshot:
    """Account state passed into portfolio and execution policies."""

    session_date: date
    cash: float
    equity: float
    gross_exposure: float
    positions: tuple[PositionSnapshot, ...] = ()


@dataclass(slots=True, frozen=True)
class BacktestResult:
    """Compact result summary for a completed run."""

    sessions: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    meta: Mapping[str, object] = field(default_factory=dict)


class SignalModel(Protocol):
    """Alpha contract used by the backtest runner."""

    def predict(self, session_date: date, universe: Sequence[str]) -> Sequence[Signal]:
        """Produce model signals for one session."""


class PortfolioPolicy(Protocol):
    """Portfolio and risk contract used inside the backtest runner."""

    def build_targets(
        self,
        session_date: date,
        signals: Sequence[Signal],
        account: AccountSnapshot,
    ) -> Sequence[TargetPosition]:
        """Transform signals into approved target positions."""


class ExecutionPolicy(Protocol):
    """Execution planning contract used by the backtest runner."""

    def generate_orders(
        self,
        session_date: date,
        targets: Sequence[TargetPosition],
        bars: Mapping[str, MarketBar],
        account: AccountSnapshot,
    ) -> Sequence[OrderIntent]:
        """Turn targets into executable order intents."""


class BacktestEngine(Protocol):
    """Backtest driver contract shared by prototype and future engine versions."""

    def run(self, start_date: date, end_date: date) -> BacktestResult:
        """Execute the full backtest between two dates."""
