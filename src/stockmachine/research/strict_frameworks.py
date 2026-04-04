from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

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
    overlay_defaults: Mapping[str, Any] = field(default_factory=dict)
    prediction_defaults: Mapping[str, Any] = field(default_factory=dict)
    turnover_control_defaults: Mapping[str, Any] = field(default_factory=dict)
    cost_stress_levels: tuple[float, ...] = ()
    promotion_gate_defaults: Mapping[str, Any] = field(default_factory=dict)


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
        overlay_defaults={
            "min_close": 10.0,
            "min_median_dollar_volume_20": 50_000_000.0,
            "max_vol_20": 0.04,
            "max_positions_per_sector": 2,
            "cost_bps_per_side": 10.0,
            "sector_neutral": True,
        },
        cost_stress_levels=(10.0, 20.0, 40.0, 60.0),
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
        overlay_defaults={
            "min_close": 10.0,
            "min_median_dollar_volume_20": 50_000_000.0,
            "max_vol_20": 0.04,
            "max_positions_per_sector": 2,
            "cost_bps_per_side": 10.0,
            "sector_neutral": True,
        },
        prediction_defaults={
            "target_task": "bucket_classification",
            "bucket_count": 2,
            "positive_threshold_bps": 0.0,
        },
        turnover_control_defaults={
            "no_trade_band": 0.05,
            "max_turnover": 1.0,
            "min_weight_change": 0.02,
            "hold_rank_buffer": 2,
            "entry_rank_buffer": 2,
            "max_new_names_per_rebalance": 2,
        },
        cost_stress_levels=(10.0, 15.0, 20.0, 30.0),
        promotion_gate_defaults={
            "min_sharpe": 0.5,
            "min_annualized_return_at_20bps": 0.0,
            "max_drawdown_floor": -0.35,
            "max_mean_turnover": 1.25,
            "min_positive_year_ratio": 0.5,
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


def build_framework_overlay_config(
    framework: StrictFrameworkSpec,
    *,
    overrides: Mapping[str, Any] | None = None,
):
    from stockmachine.research.us_equities_baseline import OverlayConfig

    payload = dict(framework.overlay_defaults)
    if overrides:
        payload.update({key: value for key, value in overrides.items() if value is not None})
    return OverlayConfig(**payload)


def resolve_framework_cost_stress_levels(
    framework: StrictFrameworkSpec,
    *,
    override_levels: tuple[float, ...] | list[float] | None = None,
) -> tuple[float, ...]:
    if override_levels:
        return tuple(float(value) for value in override_levels)
    return tuple(float(value) for value in framework.cost_stress_levels)


def build_framework_promotion_gate(
    framework: StrictFrameworkSpec,
    *,
    summary_frame: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    gate_config: Any | None = None,
) -> dict[str, Any] | None:
    if not framework.promotion_gate_defaults:
        return None

    config = dict(framework.promotion_gate_defaults)
    if gate_config is not None:
        if hasattr(gate_config, "__dataclass_fields__"):
            config.update(asdict(gate_config))
        elif isinstance(gate_config, Mapping):
            config.update(dict(gate_config))

    decisions: list[dict[str, Any]] = []
    for row in summary_frame.itertuples(index=False):
        yearly = yearly_summary[yearly_summary["model"] == row.model].copy()
        cost_20 = cost_summary[
            (cost_summary["model"] == row.model)
            & (cost_summary["cost_bps_per_side"].astype(float) == 20.0)
        ].copy()
        positive_year_ratio = float((yearly["total_return"].astype(float) > 0).mean()) if not yearly.empty else float("nan")
        annualized_return_20bps = (
            float(cost_20["annualized_return"].iloc[0])
            if not cost_20.empty and pd.notna(cost_20["annualized_return"].iloc[0])
            else float("nan")
        )
        checks = {
            "sharpe": bool(pd.notna(row.sharpe) and float(row.sharpe) >= float(config["min_sharpe"])),
            "cost_20bps_annualized_return": bool(
                pd.notna(annualized_return_20bps)
                and float(annualized_return_20bps) >= float(config["min_annualized_return_at_20bps"])
            ),
            "max_drawdown": bool(
                pd.notna(row.max_drawdown) and float(row.max_drawdown) >= float(config["max_drawdown_floor"])
            ),
            "mean_turnover": bool(
                pd.notna(row.mean_turnover) and float(row.mean_turnover) <= float(config["max_mean_turnover"])
            ),
            "positive_year_ratio": bool(
                pd.notna(positive_year_ratio) and positive_year_ratio >= float(config["min_positive_year_ratio"])
            ),
        }
        decisions.append(
            {
                "model": row.model,
                "promote_to_paper_research": all(checks.values()),
                "checks": checks,
                "metrics": {
                    "annualized_return": row.annualized_return,
                    "sharpe": row.sharpe,
                    "max_drawdown": row.max_drawdown,
                    "mean_turnover": row.mean_turnover,
                    "positive_year_ratio": positive_year_ratio,
                    "annualized_return_at_20bps": annualized_return_20bps,
                },
            }
        )
    return {
        "strategy_project": framework.strategy_project,
        "framework_id": framework.framework_id,
        "gate_config": config,
        "models": decisions,
    }
