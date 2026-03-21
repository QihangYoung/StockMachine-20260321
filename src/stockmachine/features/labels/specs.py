from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LabelSpec:
    """A research label definition tied to execution timing."""

    name: str
    benchmark_symbol: str
    horizon_sessions: int
    feature_cutoff: str
    entry_timing: str
    exit_timing: str
    definition: str
    leakage_guardrail: str


US_EQUITIES_V1_LABEL = LabelSpec(
    name="next_5_session_excess_return_from_next_open",
    benchmark_symbol="SPY",
    horizon_sessions=5,
    feature_cutoff="session_close",
    entry_timing="next_session_open",
    exit_timing="open_after_5_sessions",
    definition=(
        "Return from the next session open to the open after five held sessions, "
        "minus SPY return over the same window."
    ),
    leakage_guardrail=(
        "Features may only use information available by the current session close "
        "and cannot assume same-close execution."
    ),
)
