from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Any, Mapping

from stockmachine.domain.datetime_utils import parse_iso_datetime_like
from stockmachine.state import LocalLedger

_TERMINAL_RUN_STATUSES = {"finished", "success", "completed", "blocked"}


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _row_to_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _ensure_utc(parse_iso_datetime_like(value))


def _row_to_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


@dataclass(slots=True, frozen=True)
class PaperDailyHealthcheckResult:
    """Operator-friendly summary for the daily-run operating shell."""

    healthy: bool
    reasons: tuple[str, ...]
    ledger_readable: bool
    ledger_path: str
    latest_run: Mapping[str, Any] | None
    latest_manifest: Mapping[str, Any] | None
    open_orders_total: int
    open_orders_for_latest_run: int
    data_freshness_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "reasons": list(self.reasons),
            "ledger_readable": self.ledger_readable,
            "ledger_path": self.ledger_path,
            "latest_run": dict(self.latest_run) if self.latest_run is not None else None,
            "latest_manifest": dict(self.latest_manifest) if self.latest_manifest is not None else None,
            "open_orders_total": self.open_orders_total,
            "open_orders_for_latest_run": self.open_orders_for_latest_run,
            "data_freshness_meta": dict(self.data_freshness_meta),
        }


def build_paper_daily_healthcheck(ledger_path: str | Path) -> PaperDailyHealthcheckResult:
    """Inspect the ledger for daily-run readiness."""

    resolved_path = Path(ledger_path)
    reasons: list[str] = []
    latest_run: dict[str, Any] | None = None
    latest_manifest: dict[str, Any] | None = None
    data_freshness_meta: dict[str, Any] = {}
    open_orders_total = 0
    open_orders_for_latest_run = 0
    ledger_readable = False

    try:
        with LocalLedger(resolved_path) as ledger:
            ledger.initialize()
            ledger_readable = True

            latest_run = _fetch_latest_run(ledger.path)
            if latest_run is None:
                reasons.append("missing_latest_run")
            else:
                open_orders_for_latest_run = len(ledger.list_open_orders(run_id=str(latest_run["run_id"])))

            open_orders_total = len(ledger.list_open_orders())
            if open_orders_total > 0:
                reasons.append("lingering_open_orders")

            latest_manifest = _fetch_latest_manifest(ledger.path)
            if latest_manifest is None:
                reasons.append("missing_latest_manifest")
            else:
                data_freshness_meta = _manifest_freshness_summary(latest_manifest)
    except Exception as exc:
        reasons.append(type(exc).__name__)
        return PaperDailyHealthcheckResult(
            healthy=False,
            reasons=tuple(reasons),
            ledger_readable=False,
            ledger_path=str(resolved_path),
            latest_run=None,
            latest_manifest=None,
            open_orders_total=0,
            open_orders_for_latest_run=0,
            data_freshness_meta={"error": str(exc)},
        )

    healthy = ledger_readable and not reasons
    return PaperDailyHealthcheckResult(
        healthy=healthy,
        reasons=tuple(reasons),
        ledger_readable=ledger_readable,
        ledger_path=str(resolved_path),
        latest_run=latest_run,
        latest_manifest=latest_manifest,
        open_orders_total=open_orders_total,
        open_orders_for_latest_run=open_orders_for_latest_run,
        data_freshness_meta=data_freshness_meta,
    )


def _fetch_latest_run(ledger_path: Path) -> dict[str, Any] | None:
    query = """
        SELECT run_id, strategy_name, market, created_at_utc, status, meta_json
        FROM runs
        ORDER BY created_at_utc DESC, run_id DESC
        LIMIT 1
    """
    with sqlite3.connect(ledger_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "strategy_name": row["strategy_name"],
            "market": row["market"],
            "created_at_utc": _row_to_datetime(row["created_at_utc"]).isoformat() if row["created_at_utc"] else None,
            "status": row["status"],
            "meta": _json_load(row["meta_json"]),
        }


def _fetch_latest_manifest(ledger_path: Path) -> dict[str, Any] | None:
    query = """
        SELECT run_id, session_date, strategy_name, model_name, generated_at_utc,
               client_order_id_prefix, dry_run, data_snapshot_json, risk_policy_json,
               execution_policy_json, meta_json
        FROM run_manifests
        ORDER BY generated_at_utc DESC, run_id DESC
        LIMIT 1
    """
    with sqlite3.connect(ledger_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "session_date": _row_to_date(row["session_date"]).isoformat() if row["session_date"] else None,
            "strategy_name": row["strategy_name"],
            "model_name": row["model_name"],
            "generated_at_utc": _row_to_datetime(row["generated_at_utc"]).isoformat() if row["generated_at_utc"] else None,
            "client_order_id_prefix": row["client_order_id_prefix"],
            "dry_run": bool(row["dry_run"]),
            "data_snapshot": _json_load(row["data_snapshot_json"]),
            "risk_policy": _json_load(row["risk_policy_json"]),
            "execution_policy": _json_load(row["execution_policy_json"]),
            "meta": _json_load(row["meta_json"]),
        }


def _manifest_freshness_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(manifest.get("meta") or {})
    data_snapshot = manifest.get("data_snapshot", {})
    if not isinstance(data_snapshot, Mapping):
        data_snapshot = {}
    if isinstance(meta.get("session_guard"), Mapping):
        session_guard = dict(meta["session_guard"])
    elif isinstance(data_snapshot.get("session_guard"), Mapping):
        session_guard = dict(data_snapshot["session_guard"])
    else:
        session_guard = meta

    summary = {
        "run_id": manifest.get("run_id"),
        "session_date": manifest.get("session_date"),
        "strategy_name": manifest.get("strategy_name"),
        "model_name": manifest.get("model_name"),
        "dry_run": manifest.get("dry_run"),
        "client_order_id_prefix": manifest.get("client_order_id_prefix"),
        "resolution": session_guard.get("resolution"),
        "requested_session_date": session_guard.get("requested_session_date"),
        "effective_session_date": session_guard.get("effective_session_date"),
        "session_lag_days": session_guard.get("session_lag_days"),
        "silver_available_session_count": session_guard.get("silver_available_session_count"),
        "silver_first_session_date": session_guard.get("silver_first_session_date"),
        "silver_last_session_date": session_guard.get("silver_last_session_date"),
        "exact_session_match": session_guard.get("exact_session_match"),
    }
    return {key: value for key, value in summary.items() if value is not None}
