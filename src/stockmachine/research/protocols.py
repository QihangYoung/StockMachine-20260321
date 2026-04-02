from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True, frozen=True)
class TimingProtocol:
    """Canonical timing assumptions for US-equities daily research."""

    feature_cutoff: str = "session_close_T"
    signal_timestamp: str = "after_session_close_T"
    entry_timestamp: str = "next_session_open_T_plus_1"
    exit_timestamp: str = "open_T_plus_6"
    holding_period_sessions: int = 5


@dataclass(slots=True, frozen=True)
class UniverseProtocol:
    """Point-in-time universe rules shared by research and reruns."""

    membership_resolution: str = "last_visible_snapshot_at_or_before_session_date"
    metadata_resolution: str = "last_visible_snapshot_at_or_before_session_date"
    membership_required_fields: tuple[str, ...] = ("symbol", "session_date")
    metadata_required_fields: tuple[str, ...] = (
        "symbol",
        "company_name",
        "sector",
        "industry",
        "quote_type",
        "exchange",
        "currency",
        "country",
    )
    missing_metadata_policy: str = "fill_unknown_fields_keep_symbol"
    missing_membership_policy: str = "exclude_symbol_for_that_session"


@dataclass(slots=True, frozen=True)
class WalkForwardProtocol:
    """Rolling split defaults for first-line model validation."""

    train_window_months: int = 36
    validation_window_months: int = 6
    test_window_months: int = 6
    roll_frequency: str = "monthly"
    purge_window_sessions: int = 6
    embargo_window_sessions: int = 1


@dataclass(slots=True, frozen=True)
class OutputContract:
    """Minimal required fields for audit-grade research outputs."""

    prediction_required_columns: tuple[str, ...] = (
        "date",
        "symbol",
        "score",
        "confidence",
        "target",
        "future_return",
        "benchmark_future_return",
        "model",
    )
    backtest_summary_required_columns: tuple[str, ...] = (
        "model",
        "sessions",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "benchmark_total_return",
        "mean_turnover",
        "mean_cost_bps",
    )
    replay_required_fields: tuple[str, ...] = (
        "launch_command",
        "runtime_parameters",
        "data_snapshot",
        "universe_snapshot",
        "repository_commit",
        "dependency_snapshot",
    )


@dataclass(slots=True, frozen=True)
class ResearchProtocol:
    """Shared research contract for P0 rigor hardening."""

    market: str = "US"
    frequency: str = "daily"
    benchmark_symbol: str = "SPY"
    timing: TimingProtocol = field(default_factory=TimingProtocol)
    universe: UniverseProtocol = field(default_factory=UniverseProtocol)
    walk_forward: WalkForwardProtocol = field(default_factory=WalkForwardProtocol)
    outputs: OutputContract = field(default_factory=OutputContract)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_RESEARCH_PROTOCOL = ResearchProtocol()
DEFAULT_H1_RESEARCH_PROTOCOL = ResearchProtocol(
    timing=TimingProtocol(
        exit_timestamp="open_T_plus_2",
        holding_period_sessions=1,
    ),
    walk_forward=WalkForwardProtocol(
        test_window_months=3,
        purge_window_sessions=2,
        embargo_window_sessions=1,
    ),
)


def get_default_research_protocol() -> ResearchProtocol:
    """Return the frozen P0 research protocol."""

    return DEFAULT_RESEARCH_PROTOCOL


def get_h1_research_protocol() -> ResearchProtocol:
    """Return the frozen first-line research protocol for the h1 strategy line."""

    return DEFAULT_H1_RESEARCH_PROTOCOL
