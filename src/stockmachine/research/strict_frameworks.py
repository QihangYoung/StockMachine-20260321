from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from stockmachine.research.universe import DEFAULT_RESEARCH_UNIVERSE_NAME


@dataclass(slots=True, frozen=True)
class StrictFrameworkSpec:
    """Project-scoped strict backtest framework contract."""

    framework_id: str
    strategy_project: str
    default_horizon: int
    universe_name: str
    protocol_family: str
    prediction_family: str
    backtest_family: str
    bundle_cache_version: int = 1
    prediction_cache_version: int = 1
    turnover_control_defaults: Mapping[str, Any] = field(default_factory=dict)


_STRICT_FRAMEWORKS: dict[str, StrictFrameworkSpec] = {
    "us_equities_h5": StrictFrameworkSpec(
        framework_id="us_equities_h5_strict_v1",
        strategy_project="us_equities_h5",
        default_horizon=5,
        universe_name=DEFAULT_RESEARCH_UNIVERSE_NAME,
        protocol_family="h5",
        prediction_family="h5",
        backtest_family="open_hold",
        bundle_cache_version=1,
        prediction_cache_version=1,
    ),
    "us_equities_h1": StrictFrameworkSpec(
        framework_id="us_equities_h1_strict_v1",
        strategy_project="us_equities_h1",
        default_horizon=1,
        universe_name=DEFAULT_RESEARCH_UNIVERSE_NAME,
        protocol_family="h1",
        prediction_family="h1",
        backtest_family="daily_rebalance",
        bundle_cache_version=1,
        prediction_cache_version=1,
        turnover_control_defaults={
            "no_trade_band": 0.05,
            "max_turnover": 1.0,
            "min_weight_change": 0.02,
            "hold_rank_buffer": 2,
            "entry_rank_buffer": 2,
            "max_new_names_per_rebalance": 2,
        },
    ),
}


def resolve_strict_framework(
    *,
    strategy_project: str | None,
    horizon: int,
) -> StrictFrameworkSpec:
    if strategy_project not in (None, ""):
        try:
            framework = _STRICT_FRAMEWORKS[str(strategy_project)]
        except KeyError as exc:
            available = ", ".join(sorted(_STRICT_FRAMEWORKS))
            raise ValueError(
                f"Unsupported strict framework strategy_project '{strategy_project}'. Available: {available}"
            ) from exc
        if int(horizon) != int(framework.default_horizon):
            raise ValueError(
                f"strategy_project '{strategy_project}' expects horizon={framework.default_horizon}, got {horizon}"
            )
        return framework

    if int(horizon) == 1:
        return _STRICT_FRAMEWORKS["us_equities_h1"]
    return _STRICT_FRAMEWORKS["us_equities_h5"]


def list_strict_framework_ids() -> tuple[str, ...]:
    return tuple(sorted(_STRICT_FRAMEWORKS))
