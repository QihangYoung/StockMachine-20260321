from __future__ import annotations

from stockmachine.domain.project_paths import build_strategy_project_paths


def test_build_strategy_project_paths_uses_project_scoped_convention() -> None:
    paths = build_strategy_project_paths("us_equities_h5")

    assert paths.root.as_posix().endswith("artifacts/strategy_projects/us_equities_h5")
    assert paths.research_root.as_posix().endswith("artifacts/strategy_projects/us_equities_h5/research")
    assert paths.paper_root.as_posix().endswith("artifacts/strategy_projects/us_equities_h5/paper")
    assert paths.reports_root.as_posix().endswith("artifacts/strategy_projects/us_equities_h5/paper/reports")
    assert paths.ledger_path.as_posix().endswith("artifacts/strategy_projects/us_equities_h5/paper/paper_ledger.sqlite3")
    assert paths.kill_switch_path.as_posix().endswith("artifacts/strategy_projects/us_equities_h5/paper/paper_daily.kill")
