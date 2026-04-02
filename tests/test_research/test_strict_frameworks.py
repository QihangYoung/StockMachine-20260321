from stockmachine.research.strict_frameworks import list_strict_framework_ids, resolve_strict_framework


def test_resolve_strict_framework_defaults_by_horizon() -> None:
    h5 = resolve_strict_framework(strategy_project=None, horizon=5)
    h1 = resolve_strict_framework(strategy_project=None, horizon=1)

    assert h5.strategy_project == "us_equities_h5"
    assert h5.protocol_family == "h5"
    assert h1.strategy_project == "us_equities_h1"
    assert h1.protocol_family == "h1"
    assert "us_equities_h1" in list_strict_framework_ids()
    assert "us_equities_h5" in list_strict_framework_ids()
