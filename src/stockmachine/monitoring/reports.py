from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from stockmachine.domain.project_paths import build_strategy_project_paths
from stockmachine.state.models import RunManifestRecord


@dataclass(slots=True, frozen=True)
class PaperRunFailure:
    """Structured failure note for a paper run stage."""

    stage: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(slots=True, frozen=True)
class PaperRunReport:
    """Compact JSON-ready summary for a paper trading run."""

    run_id: str
    session_date: date
    status: str
    dry_run: bool
    stage: str
    counts: Mapping[str, int] = field(default_factory=dict)
    failures: tuple[PaperRunFailure, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_date": self.session_date.isoformat(),
            "status": self.status,
            "dry_run": self.dry_run,
            "stage": self.stage,
            "counts": dict(self.counts),
            "failures": [failure.to_dict() for failure in self.failures],
            "meta": dict(self.meta),
        }


@dataclass(slots=True, frozen=True)
class PaperRunManifest:
    """JSON-ready manifest for one paper-trading run."""

    run_id: str
    session_date: date
    strategy_name: str
    model_name: str
    dry_run: bool
    client_order_id_prefix: str
    generated_at_utc: datetime
    data_snapshot: Mapping[str, Any] = field(default_factory=dict)
    risk_policy: Mapping[str, Any] = field(default_factory=dict)
    execution_policy: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_date": self.session_date.isoformat(),
            "strategy_name": self.strategy_name,
            "model_name": self.model_name,
            "dry_run": self.dry_run,
            "client_order_id_prefix": self.client_order_id_prefix,
            "generated_at_utc": self.generated_at_utc.astimezone(timezone.utc).isoformat(),
            "data_snapshot": dict(self.data_snapshot),
            "risk_policy": dict(self.risk_policy),
            "execution_policy": dict(self.execution_policy),
            "meta": dict(self.meta),
        }

    def to_record(self) -> RunManifestRecord:
        return RunManifestRecord(
            run_id=self.run_id,
            session_date=self.session_date,
            strategy_name=self.strategy_name,
            model_name=self.model_name,
            generated_at_utc=self.generated_at_utc.astimezone(timezone.utc),
            client_order_id_prefix=self.client_order_id_prefix,
            dry_run=self.dry_run,
            data_snapshot=dict(self.data_snapshot),
            risk_policy=dict(self.risk_policy),
            execution_policy=dict(self.execution_policy),
            meta=dict(self.meta),
        )


@dataclass(slots=True, frozen=True)
class PaperArtifactLink:
    """Resolved artifact directory for paper operator workflows."""

    artifact_dir: Path | None
    source: str
    exists: bool
    files: Mapping[str, str] = field(default_factory=dict)
    candidates: tuple[str, ...] = ()
    search_roots: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir is not None else None,
            "source": self.source,
            "exists": self.exists,
            "files": dict(self.files),
            "candidates": list(self.candidates),
            "search_roots": list(self.search_roots),
            "notes": list(self.notes),
        }


def build_paper_run_report(
    *,
    session_date: date,
    dry_run: bool,
    stage: str,
    counts: Mapping[str, int],
    failures: tuple[PaperRunFailure, ...] = (),
    meta: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    manifest: PaperRunManifest | Mapping[str, Any] | None = None,
) -> PaperRunReport:
    """Build a normalized report object for CLI output and later persistence."""

    resolved_run_id = run_id or uuid4().hex
    status = "failed" if failures else "success"
    report_meta = dict(meta or {})
    if manifest is not None:
        report_meta["run_manifest"] = manifest.to_dict() if isinstance(manifest, PaperRunManifest) else dict(manifest)
    return PaperRunReport(
        run_id=resolved_run_id,
        session_date=session_date,
        status=status,
        dry_run=dry_run,
        stage=stage,
        counts=dict(counts),
        failures=failures,
        meta=report_meta,
    )


def build_paper_run_manifest(
    *,
    run_id: str,
    session_date: date,
    strategy_name: str,
    model_name: str,
    dry_run: bool,
    client_order_id_prefix: str = "smk",
    generated_at_utc: datetime | None = None,
    data_snapshot: Mapping[str, Any] | None = None,
    risk_policy: Mapping[str, Any] | None = None,
    execution_policy: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> PaperRunManifest:
    """Build a normalized manifest for paper-trading audit trails."""

    return PaperRunManifest(
        run_id=run_id,
        session_date=session_date,
        strategy_name=strategy_name,
        model_name=model_name,
        dry_run=dry_run,
        client_order_id_prefix=client_order_id_prefix,
        generated_at_utc=generated_at_utc or datetime.now(timezone.utc),
        data_snapshot=dict(data_snapshot or {}),
        risk_policy=dict(risk_policy or {}),
        execution_policy=dict(execution_policy or {}),
        meta=dict(meta or {}),
    )


def build_paper_artifact_link(
    *,
    artifact_dir: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    manifest: PaperRunManifest | Mapping[str, Any] | None = None,
    run_name: str | None = None,
    session_date: date | None = None,
    model_name: str | None = None,
    strategy_project: str | None = None,
) -> PaperArtifactLink:
    """Resolve a likely artifact directory for paper operator workflows."""

    manifest_payload = _manifest_payload(manifest)
    resolved, source, candidates, search_roots, notes = _resolve_artifact_dir(
        explicit_artifact_dir=artifact_dir,
        artifact_root=artifact_root,
        manifest_payload=manifest_payload,
        run_name=run_name,
        session_date=session_date,
        model_name=model_name,
        strategy_project=strategy_project,
    )
    files = {}
    if resolved is not None:
        files = {
            "backtest_summary": str(resolved / "backtest_summary.csv"),
            "backtest_records": str(resolved / "backtest_records.csv"),
            "predictions": str(resolved / "predictions.csv"),
        }
    return PaperArtifactLink(
        artifact_dir=resolved,
        source=source,
        exists=resolved.exists() if resolved is not None else False,
        files=files,
        candidates=candidates,
        search_roots=search_roots,
        notes=notes,
    )


def _manifest_payload(manifest: PaperRunManifest | Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    if isinstance(manifest, PaperRunManifest):
        return manifest.to_dict()
    if is_dataclass(manifest):
        return dict(asdict(manifest))
    return dict(manifest)


def _resolve_artifact_dir(
    *,
    explicit_artifact_dir: str | Path | None,
    artifact_root: str | Path,
    manifest_payload: Mapping[str, Any],
    run_name: str | None,
    session_date: date | None,
    model_name: str | None,
    strategy_project: str | None,
) -> tuple[Path | None, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if explicit_artifact_dir is not None:
        resolved = Path(explicit_artifact_dir)
        return resolved, "explicit", (str(resolved),), (str(resolved.parent),), ()

    meta = dict(manifest_payload.get("meta") or {})
    manifest_candidate = (
        meta.get("artifact_dir")
        or meta.get("artifacts_dir")
        or meta.get("backtest_artifact_dir")
        or meta.get("research_artifact_dir")
    )
    if manifest_candidate is not None:
        resolved = Path(str(manifest_candidate))
        return resolved, "manifest_meta", (str(resolved),), (str(resolved.parent),), ()

    root = Path(artifact_root)
    search_roots = _artifact_search_roots(root, strategy_project=strategy_project)
    search_root_strings = tuple(str(candidate) for candidate in search_roots)
    if not any(candidate.exists() for candidate in search_roots):
        return None, "unresolved", (), search_root_strings, ("artifact_root_missing",)

    candidates = _discover_artifact_candidates(search_roots)
    candidate_strings = tuple(str(candidate) for candidate in candidates)
    if not candidates:
        return None, "unresolved", candidate_strings, search_root_strings, ("artifact_root_has_no_candidates",)

    requested_date = session_date.isoformat() if session_date is not None else None
    project_research_root = None
    if strategy_project not in (None, ""):
        project_research_root = build_strategy_project_paths(str(strategy_project), artifact_root=root).research_root
    scored_candidates: list[tuple[int, int, Path, tuple[str, ...]]] = []
    for candidate in candidates:
        candidate_text = str(candidate).lower()
        candidate_name = candidate.name.lower()
        notes: list[str] = []
        score = 0

        if project_research_root is not None and _path_is_within(candidate, project_research_root):
            score += 5
            notes.append("matched_strategy_project_root")

        if run_name:
            if run_name.lower() in candidate_text or run_name.lower() in candidate_name:
                score += 4
                notes.append("matched_run_name")
            else:
                notes.append("missing_run_name_match")
        if model_name:
            if model_name.lower() in candidate_text or model_name.lower() in candidate_name:
                score += 3
                notes.append("matched_model_name")
            else:
                notes.append("missing_model_name_match")
        if requested_date:
            if requested_date in candidate_text or requested_date in candidate_name:
                score += 2
                notes.append("matched_session_date")
            else:
                notes.append("missing_session_date_match")

        scored_candidates.append((score, int(_artifact_mtime(candidate)), candidate, tuple(notes)))

    scored_candidates.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    best_score, _, best_candidate, best_notes = scored_candidates[0]
    if best_score <= 0 and (run_name or model_name or requested_date):
        return None, "unresolved", candidate_strings, search_root_strings, ("no_direct_match_found",)

    return best_candidate, "discovered", candidate_strings, search_root_strings, best_notes


def _artifact_search_roots(root: Path, *, strategy_project: str | None) -> tuple[Path, ...]:
    roots: list[Path] = []
    if strategy_project not in (None, ""):
        roots.append(build_strategy_project_paths(str(strategy_project), artifact_root=root).research_root)
    roots.append(root)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return tuple(unique)


def _discover_artifact_candidates(roots: tuple[Path, ...] | list[Path]) -> list[Path]:
    required_files = ("backtest_summary.csv", "backtest_records.csv", "predictions.csv")
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for summary_file in root.rglob("backtest_summary.csv"):
            parent = summary_file.parent
            if all((parent / file_name).exists() for file_name in required_files):
                candidates.append(parent)
    return sorted(set(candidates), key=lambda path: (_artifact_mtime(path), str(path)), reverse=True)


def _path_is_within(candidate: Path, ancestor: Path) -> bool:
    try:
        candidate.resolve().relative_to(ancestor.resolve())
    except ValueError:
        return False
    return True


def _artifact_mtime(path: Path) -> float:
    mtimes = []
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        candidate = path / file_name
        if candidate.exists():
            mtimes.append(candidate.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0
