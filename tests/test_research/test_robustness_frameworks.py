from __future__ import annotations

from stockmachine.research.robustness_frameworks import (
    build_robustness_gate_config,
    list_robustness_framework_ids,
    resolve_robustness_framework,
    should_run_parameter_stability,
)


def test_resolve_robustness_framework_for_h1_and_h5() -> None:
    h1 = resolve_robustness_framework(strategy_project="us_equities_h1", horizon=1)
    h5 = resolve_robustness_framework(strategy_project="us_equities_h5", horizon=5)

    assert h1.attribution_date_column == "entry_date"
    assert h1.tail_trim_counts == (1, 5, 10)
    assert h5.cost_stress_levels == (10.0, 20.0, 40.0, 60.0)


def test_build_robustness_gate_config_applies_overrides() -> None:
    framework = resolve_robustness_framework(strategy_project="us_equities_h1", horizon=1)

    config = build_robustness_gate_config(
        framework,
        overrides={"max_top5_day_contribution_share": 0.4},
    )

    assert config["max_top5_day_contribution_share"] == 0.4
    assert "min_positive_year_ratio" in config


def test_parameter_stability_is_disabled_by_default_but_overridable() -> None:
    framework = resolve_robustness_framework(strategy_project="us_equities_h1", horizon=1)

    assert should_run_parameter_stability(framework) is False
    assert should_run_parameter_stability(framework, override=True) is True


def test_list_robustness_framework_ids_is_sorted() -> None:
    assert list_robustness_framework_ids() == ("us_equities_h1", "us_equities_h5")
