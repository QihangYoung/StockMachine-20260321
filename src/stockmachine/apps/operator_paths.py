from __future__ import annotations

from pathlib import Path

from stockmachine.domain.project_paths import StrategyProjectPaths, build_strategy_project_paths


LEGACY_PAPER_DEMO_LEDGER_PATH = Path("artifacts/paper_demo/paper_ledger.sqlite3")


def resolve_operator_ledger_path(
    *,
    ledger_path: str | Path | None,
    strategy_project: str | None,
    artifact_root: str | Path = "artifacts",
) -> Path:
    if ledger_path not in (None, ""):
        return Path(ledger_path)
    workspace = resolve_strategy_workspace(
        strategy_project=strategy_project,
        artifact_root=artifact_root,
    )
    if workspace is not None:
        return workspace.ledger_path
    return LEGACY_PAPER_DEMO_LEDGER_PATH


def resolve_strategy_workspace(
    *,
    strategy_project: str | None,
    artifact_root: str | Path = "artifacts",
) -> StrategyProjectPaths | None:
    if strategy_project in (None, ""):
        return None
    return build_strategy_project_paths(str(strategy_project), artifact_root=artifact_root)
