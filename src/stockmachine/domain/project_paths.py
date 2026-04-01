from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class StrategyProjectPaths:
    """Canonical filesystem layout for one strategy project."""

    project_id: str
    root: Path
    research_root: Path
    paper_root: Path
    reports_root: Path
    ledger_path: Path
    kill_switch_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "root": str(self.root),
            "research_root": str(self.research_root),
            "paper_root": str(self.paper_root),
            "reports_root": str(self.reports_root),
            "ledger_path": str(self.ledger_path),
            "kill_switch_path": str(self.kill_switch_path),
        }


def build_strategy_project_paths(
    project_id: str,
    *,
    artifact_root: str | Path = "artifacts",
) -> StrategyProjectPaths:
    root = Path(artifact_root) / "strategy_projects" / project_id
    research_root = root / "research"
    paper_root = root / "paper"
    reports_root = paper_root / "reports"
    return StrategyProjectPaths(
        project_id=project_id,
        root=root,
        research_root=research_root,
        paper_root=paper_root,
        reports_root=reports_root,
        ledger_path=paper_root / "paper_ledger.sqlite3",
        kill_switch_path=paper_root / "paper_daily.kill",
    )
