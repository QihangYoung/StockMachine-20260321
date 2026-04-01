from __future__ import annotations

from stockmachine.apps.paper_profiles import (
    list_builtin_strategy_profiles,
    load_strategy_profile,
    resolve_strategy_profile_path,
)
from stockmachine.domain.project_paths import build_strategy_project_paths


def test_list_builtin_strategy_profiles_uses_horizon_scoped_ids() -> None:
    profiles = list_builtin_strategy_profiles()

    assert "h5/us_extra_trees_daily" in profiles
    assert "us_extra_trees_daily" not in profiles


def test_load_strategy_profile_supports_legacy_alias_and_nested_id() -> None:
    legacy = load_strategy_profile("us_extra_trees_daily")
    nested = load_strategy_profile("h5/us_extra_trees_daily")
    workspace = build_strategy_project_paths("us_equities_h5")

    assert legacy == nested
    assert legacy.strategy_family == "us_equities"
    assert legacy.strategy_project == "us_equities_h5"
    assert legacy.strategy_horizon_bucket == "h5"
    assert legacy.strategy_track == "swing_rebalance"
    assert legacy.ledger_path == str(workspace.ledger_path)
    assert legacy.kill_switch_path == str(workspace.kill_switch_path)


def test_resolve_strategy_profile_path_maps_legacy_alias_into_h5_directory() -> None:
    resolved = resolve_strategy_profile_path("us_extra_trees_daily")

    assert resolved.as_posix().endswith("configs/strategies/h5/us_extra_trees_daily.json")
