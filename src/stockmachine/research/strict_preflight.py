from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.strict_frameworks import StrictFrameworkSpec, resolve_strict_framework
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, build_price_panel_from_silver


@dataclass(slots=True, frozen=True)
class StrictResearchSourceInputs:
    """Reusable source inputs for strict-research preflight and bundle building."""

    dataset: Mapping[str, pd.DataFrame]
    price_data: pd.DataFrame
    session_dates: pd.Index


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


def load_strict_research_source_inputs(
    *,
    layout: StorageLayout | None = None,
) -> StrictResearchSourceInputs:
    storage = layout or StorageLayout()
    dataset = load_us_equities_dataset(layout=storage)
    price_data = build_price_panel_from_silver(dataset)
    return StrictResearchSourceInputs(
        dataset=dataset,
        price_data=price_data,
        session_dates=_resolve_research_session_dates(price_data),
    )


def build_strict_research_preflight(
    *,
    horizon: int,
    strategy_project: str | None = None,
    layout: StorageLayout | None = None,
    source_inputs: StrictResearchSourceInputs | None = None,
) -> StrictResearchPreflightResult:
    framework = resolve_strict_framework(strategy_project=strategy_project, horizon=horizon)
    inputs = source_inputs or load_strict_research_source_inputs(layout=layout)

    dataset = inputs.dataset
    session_dates = inputs.session_dates
    reasons: list[str] = []
    research_last_session = session_dates.max().date() if not session_dates.empty else None
    research_first_session = session_dates.min().date() if not session_dates.empty else None

    table_dates = {
        "daily_bar": _latest_table_date(dataset.get("daily_bar"), "session_date"),
        "benchmark_index": _latest_table_date(dataset.get("benchmark_index"), "session_date"),
        "adj_factor": _latest_table_date(dataset.get("adj_factor"), "session_date"),
        "symbol_master": _latest_table_date(dataset.get("symbol_master"), "as_of_date"),
        "industry_membership": _latest_table_date(dataset.get("industry_membership"), "as_of_date"),
        "universe_membership": _latest_table_date(dataset.get("universe_membership"), "session_date"),
    }

    required_tables = {
        "daily_bar": dataset.get("daily_bar", pd.DataFrame()),
        "benchmark_index": dataset.get("benchmark_index", pd.DataFrame()),
        "adj_factor": dataset.get("adj_factor", pd.DataFrame()),
        "symbol_master": dataset.get("symbol_master", pd.DataFrame()),
        "industry_membership": dataset.get("industry_membership", pd.DataFrame()),
        "universe_membership": dataset.get("universe_membership", pd.DataFrame()),
    }
    blocking_tables = {
        "symbol_master": required_tables["symbol_master"],
        "industry_membership": required_tables["industry_membership"],
        "universe_membership": required_tables["universe_membership"],
    }
    for table_name, frame in blocking_tables.items():
        if frame is None or frame.empty:
            reasons.append(f"missing_{table_name}")

    if session_dates.empty:
        reasons.append("missing_research_sessions")

    if research_last_session is not None:
        if table_dates["symbol_master"] is not None and table_dates["symbol_master"] < research_last_session:
            reasons.append("stale_symbol_master")
        if table_dates["industry_membership"] is not None and table_dates["industry_membership"] < research_last_session:
            reasons.append("stale_industry_membership")

    coverage = assess_universe_membership_coverage(
        session_dates=session_dates,
        universe_membership_frame=dataset.get("universe_membership", pd.DataFrame()),
        universe_name=framework.universe_name,
    )
    if not coverage["ok"]:
        reasons.append(str(coverage["reason"]))

    data_freshness_meta = {
        "strategy_project": framework.strategy_project,
        "framework_id": framework.framework_id,
        "universe_name": framework.universe_name,
        "research_session_count": int(len(session_dates)),
        "research_first_session_date": research_first_session.isoformat() if research_first_session is not None else None,
        "research_last_session_date": research_last_session.isoformat() if research_last_session is not None else None,
        "table_latest_dates": {
            key: value.isoformat() if value is not None else None
            for key, value in table_dates.items()
        },
        "table_row_counts": {
            key: int(len(frame)) if frame is not None else 0
            for key, frame in required_tables.items()
        },
    }
    return StrictResearchPreflightResult(
        ok=not reasons,
        strategy_project=framework.strategy_project,
        framework_id=framework.framework_id,
        reasons=tuple(dict.fromkeys(reasons)),
        data_freshness_meta=data_freshness_meta,
        coverage_meta=dict(coverage),
        source_inputs=inputs,
    )


def assess_universe_membership_coverage(
    *,
    session_dates: pd.Index,
    universe_membership_frame: pd.DataFrame,
    universe_name: str,
) -> dict[str, Any]:
    if session_dates.empty:
        return {
            "ok": False,
            "reason": "missing_research_sessions",
            "universe_name": universe_name,
            "missing_session_count": 0,
            "missing_sessions_preview": [],
        }

    if universe_membership_frame.empty:
        return {
            "ok": False,
            "reason": "missing_universe_membership",
            "universe_name": universe_name,
            "missing_session_count": int(len(session_dates)),
            "missing_sessions_preview": [str(value.date()) for value in session_dates[:5]],
        }

    membership = universe_membership_frame.copy()
    membership["session_date"] = pd.to_datetime(membership["session_date"], errors="coerce").dt.normalize()
    membership = membership.dropna(subset=["session_date", "symbol"]).copy()
    membership = membership.loc[membership["universe_name"].astype(str) == universe_name].copy()
    if membership.empty:
        return {
            "ok": False,
            "reason": "missing_universe_membership_for_universe",
            "universe_name": universe_name,
            "missing_session_count": int(len(session_dates)),
            "missing_sessions_preview": [str(value.date()) for value in session_dates[:5]],
        }

    membership["is_member"] = membership.get("is_member", True)
    membership["is_member"] = membership["is_member"].fillna(True).astype(bool)
    if "entry_date" in membership.columns:
        membership["entry_date"] = pd.to_datetime(membership["entry_date"], errors="coerce").dt.normalize()
    if "exit_date" in membership.columns:
        membership["exit_date"] = pd.to_datetime(membership["exit_date"], errors="coerce").dt.normalize()

    active_mask = membership["is_member"]
    if "entry_date" in membership.columns:
        active_mask &= membership["entry_date"].isna() | (membership["entry_date"] <= membership["session_date"])
    if "exit_date" in membership.columns:
        active_mask &= membership["exit_date"].isna() | (membership["exit_date"] >= membership["session_date"])
    membership = membership.loc[active_mask].copy()

    available_dates = pd.Index(membership["session_date"].drop_duplicates().sort_values())
    missing_dates = pd.Index(session_dates).difference(available_dates)
    return {
        "ok": missing_dates.empty,
        "reason": "exact_coverage" if missing_dates.empty else "universe_membership_incomplete",
        "universe_name": universe_name,
        "available_session_count": int(len(available_dates)),
        "available_first_session_date": available_dates.min().date().isoformat() if not available_dates.empty else None,
        "available_last_session_date": available_dates.max().date().isoformat() if not available_dates.empty else None,
        "missing_session_count": int(len(missing_dates)),
        "missing_sessions_preview": [str(value.date()) for value in missing_dates[:5]],
    }


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


def _resolve_research_session_dates(price_data: pd.DataFrame) -> pd.Index:
    return pd.Index(
        pd.to_datetime(price_data.loc[price_data["symbol"] != BENCHMARK_SYMBOL, "date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )


def _latest_table_date(frame: pd.DataFrame | None, column: str) -> date | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return values.max().date()
