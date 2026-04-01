from __future__ import annotations

from stockmachine.apps.strategy_projects import (
    list_builtin_strategy_projects,
    load_strategy_project,
    resolve_strategy_project_path,
)


def test_list_builtin_strategy_projects_includes_h1_and_h5() -> None:
    projects = list_builtin_strategy_projects()

    assert "us_equities_h1" in projects
    assert "us_equities_h5" in projects


def test_load_strategy_project_returns_expected_metadata() -> None:
    h1 = load_strategy_project("us_equities_h1")
    h5 = load_strategy_project("us_equities_h5")

    assert h1.status == "scaffold"
    assert h1.strategy_horizon_bucket == "h1"
    assert h1.entrypoints is not None
    assert h1.entrypoints["paper_profiles_dir"] == "configs/strategies/h1"
    assert h5.status == "active"
    assert h5.strategy_horizon_bucket == "h5"
    assert h5.entrypoints is not None
    assert h5.entrypoints["paper_profiles_dir"] == "configs/strategies/h5"


def test_resolve_strategy_project_path_supports_builtin_ids() -> None:
    resolved = resolve_strategy_project_path("us_equities_h1")

    assert resolved.as_posix().endswith("configs/strategy_projects/us_equities_h1.json")
