from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from stockmachine.state.ledger import LocalLedger

_TERMINAL_RUN_STATUSES = {"finished", "success", "completed"}


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "dry_run", "paper"}:
        return True
    if text in {"0", "false", "f", "no", "n", "execute", "live"}:
        return False
    return None


def _parse_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _normalize_mode_label(dry_run: bool) -> str:
    return "dry_run" if dry_run else "execute"


def _extract_mode_from_meta(meta: Mapping[str, Any]) -> str | None:
    if "dry_run" in meta:
        coerced = _coerce_bool(meta.get("dry_run"))
        if coerced is not None:
            return _normalize_mode_label(coerced)
    for key in ("mode", "execution_mode", "run_mode"):
        value = meta.get(key)
        if value is None:
            continue
        coerced = _coerce_bool(value)
        if coerced is not None:
            return _normalize_mode_label(coerced)
        text = str(value).strip().lower()
        if text in {"dry_run", "execute"}:
            return text
    return None


@dataclass(slots=True, frozen=True)
class SessionGuardRequest:
    """Inputs needed to decide whether a paper run may proceed."""

    ledger: LocalLedger
    strategy_name: str
    session_date: date
    dry_run: bool
    silver_session_dates: Sequence[date] = ()
    allow_previous_available_session: bool = True


@dataclass(slots=True, frozen=True)
class SessionGuardResult:
    """Normalized guard result for runner orchestration."""

    allowed: bool
    reasons: tuple[str, ...]
    effective_session_date: date | None
    data_freshness_meta: dict[str, Any] = field(default_factory=dict)
    duplicate_run_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "effective_session_date": self.effective_session_date.isoformat() if self.effective_session_date else None,
            "data_freshness_meta": dict(self.data_freshness_meta),
            "duplicate_run_ids": list(self.duplicate_run_ids),
        }


class SessionGuard:
    """Reusable session and idempotency guard for paper/demo runs."""

    def __init__(self, ledger: LocalLedger) -> None:
        self.ledger = ledger

    def evaluate(self, request: SessionGuardRequest) -> SessionGuardResult:
        available_dates = sorted({value for value in (_parse_date(d) for d in request.silver_session_dates) if value is not None})
        freshness_meta = {
            "requested_session_date": request.session_date.isoformat(),
            "silver_available_session_count": len(available_dates),
            "silver_first_session_date": available_dates[0].isoformat() if available_dates else None,
            "silver_last_session_date": available_dates[-1].isoformat() if available_dates else None,
            "dry_run": request.dry_run,
        }

        reasons: list[str] = []
        effective_session_date = None

        if not available_dates:
            reasons.append("missing_silver_data")
        else:
            exact_match = request.session_date in available_dates
            fallback = max((value for value in available_dates if value <= request.session_date), default=None)
            if exact_match:
                effective_session_date = request.session_date
                freshness_meta["resolution"] = "exact"
            elif fallback is not None and request.allow_previous_available_session:
                effective_session_date = fallback
                freshness_meta["resolution"] = "fallback_previous_available_session"
            elif fallback is None:
                reasons.append("no_silver_session_on_or_before_target")
                freshness_meta["resolution"] = "no_usable_session"
            else:
                reasons.append("previous_available_session_not_allowed")
                freshness_meta["resolution"] = "previous_available_session_not_allowed"

        if effective_session_date is not None:
            freshness_meta["effective_session_date"] = effective_session_date.isoformat()
            freshness_meta["session_lag_days"] = (request.session_date - effective_session_date).days
            freshness_meta["exact_session_match"] = effective_session_date == request.session_date

        duplicate_run_ids = self._find_completed_runs(
            strategy_name=request.strategy_name,
            session_date=request.session_date,
            dry_run=request.dry_run,
        )
        if duplicate_run_ids:
            reasons.append("duplicate_completed_run")
            freshness_meta["duplicate_run_count"] = len(duplicate_run_ids)
            freshness_meta["duplicate_run_ids"] = list(duplicate_run_ids)

        allowed = not reasons
        return SessionGuardResult(
            allowed=allowed,
            reasons=tuple(reasons),
            effective_session_date=effective_session_date,
            data_freshness_meta=freshness_meta,
            duplicate_run_ids=tuple(duplicate_run_ids),
        )

    def _find_completed_runs(self, *, strategy_name: str, session_date: date, dry_run: bool) -> list[str]:
        try:
            return self._find_completed_runs_from_manifests(strategy_name=strategy_name, session_date=session_date, dry_run=dry_run)
        except sqlite3.OperationalError:
            return self._find_completed_runs_from_runs(strategy_name=strategy_name, session_date=session_date, dry_run=dry_run)

    def _find_completed_runs_from_manifests(self, *, strategy_name: str, session_date: date, dry_run: bool) -> list[str]:
        session_date_text = session_date.isoformat()
        query = """
            SELECT rm.run_id, r.status
            FROM run_manifests rm
            LEFT JOIN runs r ON r.run_id = rm.run_id
            WHERE rm.strategy_name = ?
              AND rm.session_date = ?
              AND rm.dry_run = ?
        """
        rows = self._query_rows(query, (strategy_name, session_date_text, 1 if dry_run else 0))
        return [
            str(row["run_id"])
            for row in rows
            if str(row["status"] or "").lower() in _TERMINAL_RUN_STATUSES
        ]

    def _find_completed_runs_from_runs(self, *, strategy_name: str, session_date: date, dry_run: bool) -> list[str]:
        session_date_text = session_date.isoformat()
        query = "SELECT run_id, status, meta_json FROM runs WHERE strategy_name = ?"
        rows = self._query_rows(query, (strategy_name,))
        matched: list[str] = []
        for row in rows:
            if str(row["status"] or "").lower() not in _TERMINAL_RUN_STATUSES:
                continue
            meta = _parse_json(row["meta_json"])
            row_session_date = _parse_date(meta.get("session_date") or meta.get("sessionDate"))
            if row_session_date is None:
                row_session_date = _parse_date(meta.get("run_date"))
            if row_session_date is None or row_session_date.isoformat() != session_date_text:
                continue
            mode_label = _extract_mode_from_meta(meta)
            if mode_label is None:
                continue
            if mode_label != _normalize_mode_label(dry_run):
                continue
            matched.append(str(row["run_id"]))
        return matched

    def _query_rows(self, query: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        with sqlite3.connect(Path(self.ledger.path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return list(cursor.fetchall())


def evaluate_session_guard(request: SessionGuardRequest) -> SessionGuardResult:
    """Convenience helper mirroring SessionGuard.evaluate."""

    return SessionGuard(request.ledger).evaluate(request)
