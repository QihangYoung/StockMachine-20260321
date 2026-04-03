from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stockmachine.research.research_governance import (
    ResearchSourceInputs,
    assess_universe_membership_coverage,
    build_research_data_coverage_assessment,
)
from stockmachine.research.strict_frameworks import resolve_strict_framework


StrictResearchSourceInputs = ResearchSourceInputs


@dataclass(slots=True, frozen=True)
class StrictResearchPreflightResult:
    """Strict-research data freshness and coverage preflight result."""

    ok: bool
    strategy_project: str
    framework_id: str
    reasons: tuple[str, ...]
    data_freshness_meta: dict[str, Any] = field(default_factory=dict)
    coverage_meta: dict[str, Any] = field(default_factory=dict)
    source_inputs: StrictResearchSourceInputs | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "strategy_project": self.strategy_project,
            "framework_id": self.framework_id,
            "reasons": list(self.reasons),
            "data_freshness_meta": dict(self.data_freshness_meta),
            "coverage_meta": dict(self.coverage_meta),
        }


def build_strict_research_preflight(
    *,
    horizon: int,
    strategy_project: str | None = None,
    layout=None,
    source_inputs: StrictResearchSourceInputs | None = None,
) -> StrictResearchPreflightResult:
    framework = resolve_strict_framework(strategy_project=strategy_project, horizon=horizon)
    assessment = build_research_data_coverage_assessment(
        layout=layout,
        source_inputs=source_inputs,
        universe_name=framework.universe_name,
    )
    return StrictResearchPreflightResult(
        ok=assessment.ok,
        strategy_project=framework.strategy_project,
        framework_id=framework.framework_id,
        reasons=assessment.reasons,
        data_freshness_meta={
            "strategy_project": framework.strategy_project,
            "framework_id": framework.framework_id,
            "universe_name": framework.universe_name,
            "research_session_count": assessment.research_session_count,
            "research_first_session_date": (
                assessment.research_first_session_date.isoformat()
                if assessment.research_first_session_date is not None
                else None
            ),
            "research_last_session_date": (
                assessment.research_last_session_date.isoformat()
                if assessment.research_last_session_date is not None
                else None
            ),
            "table_latest_dates": {
                key: value.isoformat() if value is not None else None
                for key, value in assessment.table_latest_dates.items()
            },
            "table_row_counts": dict(assessment.table_row_counts),
        },
        coverage_meta=dict(assessment.coverage_meta),
        source_inputs=assessment.source_inputs,
    )


def ensure_strict_research_preflight_ok(result: StrictResearchPreflightResult) -> None:
    if result.ok:
        return
    if "missing_universe_membership" in result.reasons:
        raise ValueError("Strict research bundle requires explicit universe_membership history; none was found.")
    if "missing_universe_membership_for_universe" in result.reasons:
        raise ValueError(
            "Strict research bundle requires explicit universe_membership history "
            f"for universe '{result.coverage_meta.get('universe_name')}'."
        )
    if "universe_membership_incomplete" in result.reasons:
        preview = result.coverage_meta.get("missing_sessions_preview", [])
        preview_text = ", ".join(str(item) for item in preview)
        raise ValueError(
            "Strict research bundle requires explicit universe_membership coverage for every research session; "
            f"missing {result.coverage_meta.get('missing_session_count', 0)} session(s), first missing: {preview_text}"
        )
    reasons = ", ".join(result.reasons) if result.reasons else "unknown"
    coverage = result.coverage_meta
    preview = coverage.get("missing_sessions_preview", [])
    if preview:
        raise ValueError(
            "Strict research preflight failed: "
            f"{reasons}. Missing universe coverage preview: {', '.join(str(item) for item in preview)}"
        )
    raise ValueError(f"Strict research preflight failed: {reasons}")
