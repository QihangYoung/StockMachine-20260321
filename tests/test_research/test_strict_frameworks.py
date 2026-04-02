import pandas as pd

from stockmachine.research.strict_frameworks import (
    build_framework_overlay_config,
    build_framework_promotion_gate,
    list_strict_framework_ids,
    resolve_framework_cost_stress_levels,
    resolve_strict_framework,
)


def test_resolve_strict_framework_defaults_by_horizon() -> None:
    h5 = resolve_strict_framework(strategy_project=None, horizon=5)
    h1 = resolve_strict_framework(strategy_project=None, horizon=1)

    assert h5.strategy_project == "us_equities_h5"
    assert h5.protocol_family == "h5"
    assert h1.strategy_project == "us_equities_h1"
    assert h1.protocol_family == "h1"
    assert "us_equities_h1" in list_strict_framework_ids()
    assert "us_equities_h5" in list_strict_framework_ids()


def test_framework_helpers_expose_overlay_cost_and_promotion_gate_defaults() -> None:
    h1 = resolve_strict_framework(strategy_project="us_equities_h1", horizon=1)

    overlay = build_framework_overlay_config(h1)
    assert overlay.cost_bps_per_side == 10.0
    assert resolve_framework_cost_stress_levels(h1) == (10.0, 15.0, 20.0, 30.0)

    summary = pd.DataFrame(
        [
            {
                "model": "extra_trees",
                "annualized_return": 0.12,
                "sharpe": 0.8,
                "max_drawdown": -0.2,
                "mean_turnover": 0.4,
            }
        ]
    )
    yearly = pd.DataFrame([{"model": "extra_trees", "total_return": 0.05}])
    cost = pd.DataFrame([{"model": "extra_trees", "cost_bps_per_side": 20.0, "annualized_return": 0.03}])

    gate = build_framework_promotion_gate(
        h1,
        summary_frame=summary,
        yearly_summary=yearly,
        cost_summary=cost,
    )

    assert gate is not None
    assert gate["strategy_project"] == "us_equities_h1"
    assert gate["models"][0]["model"] == "extra_trees"
