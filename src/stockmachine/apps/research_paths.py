from __future__ import annotations

from pathlib import Path

from stockmachine.domain.project_paths import StrategyProjectPaths, build_strategy_project_paths


def infer_strategy_project(*, strategy_project: str | None, horizon: int) -> str:
    if strategy_project not in (None, ""):
        return str(strategy_project)
    return f"us_equities_h{int(horizon)}"


def resolve_research_workspace(
    *,
    strategy_project: str | None,
    horizon: int,
    artifact_root: str | Path = "artifacts",
) -> StrategyProjectPaths:
    project_id = infer_strategy_project(strategy_project=strategy_project, horizon=horizon)
    return build_strategy_project_paths(project_id, artifact_root=artifact_root)


def resolve_research_output_root(
    *,
    output_root: str | Path | None,
    default_dirname: str,
    strategy_project: str | None,
    horizon: int,
    artifact_root: str | Path = "artifacts",
) -> Path:
    if output_root not in (None, ""):
        return Path(output_root)
    workspace = resolve_research_workspace(
        strategy_project=strategy_project,
        horizon=horizon,
        artifact_root=artifact_root,
    )
    return workspace.research_root / default_dirname


def resolve_research_cache_dir(
    *,
    cache_dir: str | Path | None,
    default_dirname: str,
    strategy_project: str | None,
    horizon: int,
    artifact_root: str | Path = "artifacts",
) -> Path:
    if cache_dir not in (None, ""):
        return Path(cache_dir)
    workspace = resolve_research_workspace(
        strategy_project=strategy_project,
        horizon=horizon,
        artifact_root=artifact_root,
    )
    return workspace.research_root / "cache" / default_dirname
