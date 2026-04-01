from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_PROJECT_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILTIN_PROJECT_DIR = _PROJECT_REPO_ROOT / "configs" / "strategy_projects"


@dataclass(slots=True, frozen=True)
class StrategyProjectSpec:
    """Project-level metadata for one horizon-scoped strategy line."""

    project_id: str
    description: str
    strategy_family: str
    strategy_horizon_bucket: str
    status: str = "active"
    owner: str | None = None
    notes: str | None = None
    entrypoints: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategyProjectSpec":
        entrypoints = payload.get("entrypoints")
        normalized_entrypoints = dict(entrypoints) if isinstance(entrypoints, Mapping) else None
        return cls(
            project_id=str(payload["project_id"]),
            description=str(payload.get("description", "")),
            strategy_family=str(payload.get("strategy_family", "unknown")),
            strategy_horizon_bucket=str(payload.get("strategy_horizon_bucket", "unknown")),
            status=str(payload.get("status", "active")),
            owner=_optional_str(payload.get("owner")),
            notes=_optional_str(payload.get("notes")),
            entrypoints=normalized_entrypoints,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "project_id": self.project_id,
            "description": self.description,
            "strategy_family": self.strategy_family,
            "strategy_horizon_bucket": self.strategy_horizon_bucket,
            "status": self.status,
            "owner": self.owner,
            "notes": self.notes,
        }
        if self.entrypoints is not None:
            payload["entrypoints"] = dict(self.entrypoints)
        return payload


def list_builtin_strategy_projects() -> tuple[str, ...]:
    if not _BUILTIN_PROJECT_DIR.exists():
        return ()
    return tuple(sorted(path.stem for path in _BUILTIN_PROJECT_DIR.glob("*.json")))


def load_strategy_project(project_ref: str | Path) -> StrategyProjectSpec:
    project_path = resolve_strategy_project_path(project_ref)
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Strategy project must be a JSON object: {project_path}")
    return StrategyProjectSpec.from_mapping(payload)


def resolve_strategy_project_path(project_ref: str | Path) -> Path:
    candidate = Path(project_ref)
    if candidate.exists():
        return candidate.resolve()

    builtin_candidates = [candidate]
    if candidate.suffix != ".json":
        builtin_candidates.append(candidate.with_suffix(".json"))
    for relative_candidate in builtin_candidates:
        builtin_candidate = (_BUILTIN_PROJECT_DIR / relative_candidate).resolve()
        if builtin_candidate.exists():
            return builtin_candidate

    available = ", ".join(list_builtin_strategy_projects())
    raise FileNotFoundError(
        f"Strategy project '{project_ref}' was not found. Available built-ins: {available or '(none)'}."
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
