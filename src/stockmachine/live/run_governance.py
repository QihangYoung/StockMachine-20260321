from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

import pandas as pd

from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.live.session_guard import SessionGuard, SessionGuardRequest
from stockmachine.monitoring.healthcheck import build_paper_daily_healthcheck
from stockmachine.state import LocalLedger

DEFAULT_KILL_SWITCH_PATH = Path("artifacts/paper_demo/paper_daily.kill")
_NON_BLOCKING_HEALTHCHECK_REASONS = {"missing_latest_run", "missing_latest_manifest"}


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_load(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_silver_session_dates(data_root: str | Path) -> tuple[tuple[date, ...], dict[str, Any]]:
    layout = StorageLayout(root=Path(data_root))
    dataset = load_us_equities_dataset(layout=layout)
    daily_bar = dataset.get("daily_bar")
    if daily_bar is None or daily_bar.empty:
        return (), {"resolution": "missing_silver_data", "data_root": str(Path(data_root))}

    session_dates = sorted(
        {
            item.date()
            for item in pd.to_datetime(daily_bar["session_date"], errors="coerce").dropna().unique()
        }
    )
    if not session_dates:
        return (), {"resolution": "missing_silver_data", "data_root": str(Path(data_root))}

    return (
        tuple(session_dates),
        {
            "resolution": "silver_loaded",
            "data_root": str(Path(data_root)),
            "silver_available_session_count": len(session_dates),
            "silver_first_session_date": session_dates[0].isoformat(),
            "silver_last_session_date": session_dates[-1].isoformat(),
        },
    )


def _read_kill_switch(path: str | Path | None) -> "DailyRunKillSwitchStatus":
    resolved = Path(path) if path is not None else DEFAULT_KILL_SWITCH_PATH
    if not resolved.exists():
        return DailyRunKillSwitchStatus(path=str(resolved), active=False, reason="absent", payload=None)

    try:
        content = resolved.read_text(encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - defensive
        return DailyRunKillSwitchStatus(
            path=str(resolved),
            active=True,
            reason=type(exc).__name__,
            payload={"error": str(exc)},
        )

    payload = _json_load(content)
    if isinstance(payload, dict):
        active_value = payload.get("active")
        active = True if active_value is None else bool(active_value)
        reason = str(payload.get("reason") or payload.get("message") or "file_present")
        return DailyRunKillSwitchStatus(path=str(resolved), active=active, reason=reason, payload=payload)

    normalized = content.lower()
    if normalized in {"0", "false", "off", "inactive", "clear", "none"}:
        return DailyRunKillSwitchStatus(path=str(resolved), active=False, reason="explicitly_cleared", payload=content)
    reason = content or "file_present"
    return DailyRunKillSwitchStatus(path=str(resolved), active=True, reason=reason, payload=content)


@dataclass(slots=True, frozen=True)
class DailyRunKillSwitchStatus:
    """Filesystem-based run gate for operators."""

    path: str
    active: bool
    reason: str
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "active": self.active,
            "reason": self.reason,
            "payload": self.payload,
        }


@dataclass(slots=True, frozen=True)
class DailyRunGovernanceRequest:
    """Inputs required to decide whether a daily paper run may proceed."""

    ledger_path: str | Path
    data_root: str | Path
    session_date: date
    strategy_name: str
    dry_run: bool
    universe: Sequence[str] = ()
    model_name: str = "hist_gbm"
    kill_switch_path: str | Path | None = None
    allow_unhealthy: bool = False
    allow_previous_available_session: bool = True


@dataclass(slots=True, frozen=True)
class DailyRunGovernanceResult:
    """Scheduler-friendly preflight result for daily paper operations."""

    policy_allowed: bool
    allowed: bool
    override_used: bool
    reasons: tuple[str, ...]
    healthcheck: Mapping[str, Any]
    session_guard: Mapping[str, Any] | None
    kill_switch: Mapping[str, Any]
    effective_session_date: date | None
    data_freshness_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_allowed": self.policy_allowed,
            "allowed": self.allowed,
            "override_used": self.override_used,
            "reasons": list(self.reasons),
            "healthcheck": dict(self.healthcheck),
            "session_guard": dict(self.session_guard) if self.session_guard is not None else None,
            "kill_switch": dict(self.kill_switch),
            "effective_session_date": self.effective_session_date.isoformat() if self.effective_session_date else None,
            "data_freshness_meta": dict(self.data_freshness_meta),
        }


def evaluate_daily_run_governance(request: DailyRunGovernanceRequest) -> DailyRunGovernanceResult:
    """Evaluate ledger health, session idempotency, and kill-switch state."""

    healthcheck = build_paper_daily_healthcheck(request.ledger_path)
    silver_dates, silver_meta = _load_silver_session_dates(request.data_root)
    kill_switch = _read_kill_switch(request.kill_switch_path)

    session_guard_result = None
    reasons: list[str] = []
    effective_session_date = request.session_date
    blocking_healthcheck_reasons = [
        reason for reason in healthcheck.reasons if reason not in _NON_BLOCKING_HEALTHCHECK_REASONS
    ]
    healthcheck_warnings = [
        reason for reason in healthcheck.reasons if reason in _NON_BLOCKING_HEALTHCHECK_REASONS
    ]

    if not healthcheck.ledger_readable:
        reasons.append("ledger_unreadable")
    if blocking_healthcheck_reasons:
        reasons.extend(f"healthcheck:{reason}" for reason in blocking_healthcheck_reasons)
    if kill_switch.active:
        reasons.append(f"kill_switch_active:{kill_switch.reason}")

    if silver_dates:
        with LocalLedger(request.ledger_path) as ledger:
            session_guard_result = SessionGuard(ledger).evaluate(
                SessionGuardRequest(
                    ledger=ledger,
                    strategy_name=request.strategy_name,
                    session_date=request.session_date,
                    dry_run=request.dry_run,
                    silver_session_dates=silver_dates,
                    allow_previous_available_session=request.allow_previous_available_session,
                )
            )
        if session_guard_result.effective_session_date is not None:
            effective_session_date = session_guard_result.effective_session_date
        if not session_guard_result.allowed:
            reasons.extend(f"session_guard:{reason}" for reason in session_guard_result.reasons)
    else:
        reasons.append("missing_silver_data")

    silver_meta.update(
        {
            "requested_session_date": request.session_date.isoformat(),
            "effective_session_date": effective_session_date.isoformat() if effective_session_date else None,
            "strategy_name": request.strategy_name,
            "model_name": request.model_name,
            "dry_run": request.dry_run,
            "universe_size": len(tuple(request.universe)),
            "kill_switch_active": kill_switch.active,
            "kill_switch_path": kill_switch.path,
        }
    )
    if healthcheck.latest_manifest is not None:
        silver_meta["latest_manifest_session_date"] = healthcheck.latest_manifest.get("session_date")
        silver_meta["latest_manifest_model_name"] = healthcheck.latest_manifest.get("model_name")
    if healthcheck_warnings:
        silver_meta["healthcheck_warnings"] = list(healthcheck_warnings)

    policy_allowed = healthcheck.ledger_readable and not blocking_healthcheck_reasons and kill_switch.active is False and (
        session_guard_result.allowed if session_guard_result is not None else False
    )
    if not silver_dates:
        policy_allowed = False

    override_used = bool(request.allow_unhealthy and not policy_allowed)
    allowed = policy_allowed or request.allow_unhealthy

    return DailyRunGovernanceResult(
        policy_allowed=policy_allowed,
        allowed=allowed,
        override_used=override_used,
        reasons=tuple(dict.fromkeys(reasons)),
        healthcheck=healthcheck.to_dict(),
        session_guard=session_guard_result.to_dict() if session_guard_result is not None else None,
        kill_switch=kill_switch.to_dict(),
        effective_session_date=effective_session_date,
        data_freshness_meta=silver_meta,
    )
