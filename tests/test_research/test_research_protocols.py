from __future__ import annotations

from stockmachine.research import DEFAULT_RESEARCH_PROTOCOL, get_default_research_protocol


def test_default_research_protocol_matches_p0_contract() -> None:
    protocol = get_default_research_protocol()

    assert protocol.market == "US"
    assert protocol.frequency == "daily"
    assert protocol.benchmark_symbol == "SPY"
    assert protocol.timing.holding_period_sessions == 5
    assert protocol.walk_forward.train_window_months == 36
    assert protocol.walk_forward.validation_window_months == 6
    assert protocol.walk_forward.test_window_months == 6
    assert protocol.walk_forward.roll_frequency == "monthly"
    assert protocol.walk_forward.purge_window_sessions == 6
    assert protocol.walk_forward.embargo_window_sessions == 1
    assert protocol.outputs.prediction_required_columns[:4] == ("date", "symbol", "score", "confidence")


def test_default_research_protocol_is_exported_singleton() -> None:
    protocol = get_default_research_protocol()

    assert protocol is DEFAULT_RESEARCH_PROTOCOL
    payload = protocol.to_dict()
    assert payload["timing"]["entry_timestamp"] == "next_session_open_T_plus_1"
    assert payload["universe"]["missing_membership_policy"] == "exclude_symbol_for_that_session"
