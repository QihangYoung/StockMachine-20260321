from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class RobustnessFrameworkSpec:
    """Shared robustness-evaluation semantics for a strategy project."""

    framework_id: str
    strategy_project: str
    default_horizon: int
    attribution_date_column: str = "entry_date"
    tail_return_column: str = "net_return"
    tail_trim_counts: tuple[int, ...] = (1, 5, 10)
    cost_stress_levels: tuple[float, ...] = ()
    parameter_neighborhood_radius: int = 1
    run_parameter_stability_by_default: bool = False
    gate_defaults: Mapping[str, Any] = field(default_factory=dict)


_ROBUSTNESS_FRAMEWORKS: dict[str, RobustnessFrameworkSpec] = {
    "multi_asset_fmf_validation": RobustnessFrameworkSpec(
        framework_id="multi_asset_fmf_validation_robustness_v1",
        strategy_project="multi_asset_fmf_validation",
        default_horizon=1,
        attribution_date_column="entry_date",
        tail_return_column="net_return",
        tail_trim_counts=(1, 5, 10),
        cost_stress_levels=(5.0, 10.0, 20.0, 40.0),
        parameter_neighborhood_radius=1,
        run_parameter_stability_by_default=False,
        gate_defaults={
            "max_top5_day_contribution_share": 0.5,
            "min_positive_year_ratio": 0.5,
        },
    ),
    "us_equities_h5": RobustnessFrameworkSpec(
        framework_id="us_equities_h5_robustness_v1",
        strategy_project="us_equities_h5",
        default_horizon=5,
        attribution_date_column="entry_date",
        tail_return_column="net_return",
        tail_trim_counts=(1, 5, 10),
        cost_stress_levels=(10.0, 20.0, 40.0, 60.0),
        parameter_neighborhood_radius=1,
        run_parameter_stability_by_default=False,
        gate_defaults={
            "max_top5_day_contribution_share": 0.6,
            "min_positive_year_ratio": 0.5,
        },
    ),
    "us_equities_h1": RobustnessFrameworkSpec(
        framework_id="us_equities_h1_robustness_v1",
        strategy_project="us_equities_h1",
        default_horizon=1,
        attribution_date_column="entry_date",
        tail_return_column="net_return",
        tail_trim_counts=(1, 5, 10),
        cost_stress_levels=(10.0, 15.0, 20.0, 30.0),
        parameter_neighborhood_radius=1,
        run_parameter_stability_by_default=False,
        gate_defaults={
            "max_top5_day_contribution_share": 0.5,
            "min_positive_year_ratio": 0.5,
        },
    ),
    "us_equities_pure_alpha_h5": RobustnessFrameworkSpec(
        framework_id="us_equities_pure_alpha_h5_robustness_v1",
        strategy_project="us_equities_pure_alpha_h5",
        default_horizon=5,
        attribution_date_column="entry_date",
        tail_return_column="net_return",
        tail_trim_counts=(1, 5, 10),
        cost_stress_levels=(5.0, 10.0, 20.0, 40.0, 60.0),
        parameter_neighborhood_radius=1,
        run_parameter_stability_by_default=False,
        gate_defaults={
            "max_abs_ex_ante_net_beta": 0.05,
            "max_abs_realized_beta": 0.05,
            "max_abs_market_correlation": 0.10,
            "max_top5_day_contribution_share": 0.5,
            "min_positive_year_ratio": 0.5,
            "require_long_and_short_leg_contribution": True,
        },
    ),
}


def resolve_robustness_framework(
    *,
    strategy_project: str | None,
    horizon: int,
) -> RobustnessFrameworkSpec:
    if strategy_project not in (None, ""):
        try:
            framework = _ROBUSTNESS_FRAMEWORKS[str(strategy_project)]
        except KeyError as exc:
            available = ", ".join(sorted(_ROBUSTNESS_FRAMEWORKS))
            raise ValueError(
                f"Unsupported robustness framework strategy_project '{strategy_project}'. Available: {available}"
            ) from exc
        if int(horizon) != int(framework.default_horizon):
            raise ValueError(
                f"strategy_project '{strategy_project}' expects horizon={framework.default_horizon}, got {horizon}"
            )
        return framework

    if int(horizon) == 1:
        return _ROBUSTNESS_FRAMEWORKS["us_equities_h1"]
    return _ROBUSTNESS_FRAMEWORKS["us_equities_h5"]


def list_robustness_framework_ids() -> tuple[str, ...]:
    return tuple(sorted(_ROBUSTNESS_FRAMEWORKS))


def build_robustness_gate_config(
    framework: RobustnessFrameworkSpec,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(framework.gate_defaults)
    if overrides:
        if hasattr(overrides, "__dataclass_fields__"):
            config.update(asdict(overrides))
        else:
            config.update(dict(overrides))
    return config


def should_run_parameter_stability(
    framework: RobustnessFrameworkSpec,
    *,
    override: bool | None = None,
) -> bool:
    if override is not None:
        return bool(override)
    return bool(framework.run_parameter_stability_by_default)
