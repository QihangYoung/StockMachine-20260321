from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from stockmachine.monitoring.alerts import OperatorAlert
from stockmachine.monitoring.digest import build_daily_summary_payload, build_run_index_payload
from stockmachine.state.ledger import LocalLedger


@dataclass(slots=True, frozen=True)
class TrendPoint:
    """Tiny JSON-ready helper for one point in a recent trend."""

    key: str
    value: float
    count: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "count": self.count,
            "meta": dict(self.meta),
        }


def build_health_trend_payload(
    ledger: LocalLedger,
    *,
    limit_runs: int = 10,
    limit_sessions: int = 5,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """Summarize recent run health, anomalies, and trend direction."""

    run_index = build_run_index_payload(ledger, limit=limit_runs, reference_date=reference_date)
    entries = list(run_index["entries"])
    alerts = _merge_alerts(entry["alerts"] for entry in entries)
    anomaly_summary = build_anomaly_summary_payload(
        ledger,
        limit_runs=limit_runs,
        reference_date=reference_date,
    )
    trend_points = _build_trend_points(entries)
    session_breakdown = _build_session_breakdown(ledger, entries, limit_sessions=limit_sessions, reference_date=reference_date)

    return {
        "command": "health-trend",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {
            "run_limit": int(limit_runs),
            "session_limit": int(limit_sessions),
        },
        "run_index": run_index,
        "trend": {
            "status_counts": _status_counts(entry["status"] for entry in entries),
            "alert_counts": _alert_counts(entries),
            "open_order_run_count": sum(1 for entry in entries if int(entry["open_order_count"]) > 0),
            "rejected_run_count": sum(1 for entry in entries if int(entry["rejected_order_count"]) > 0),
            "stale_run_count": sum(1 for entry in entries if "data_stale" in entry["alert_codes"]),
            "average_open_orders": _average(int(entry["open_order_count"]) for entry in entries),
            "average_fill_count": _average(int(entry["fill_count"]) for entry in entries),
            "recent_trend_points": [point.to_dict() for point in trend_points],
        },
        "sessions": session_breakdown,
        "anomaly_summary": anomaly_summary,
        "alerts": [alert.to_dict() for alert in alerts],
    }


def build_anomaly_summary_payload(
    ledger: LocalLedger,
    *,
    limit_runs: int = 10,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """Summarize the most recent anomalies for operator triage."""

    run_index = build_run_index_payload(ledger, limit=limit_runs, reference_date=reference_date)
    entries = list(run_index["entries"])
    anomalies = _build_anomaly_summary(entries)
    alerts = _merge_alerts(entry["alerts"] for entry in entries)

    return {
        "command": "anomaly-summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"run_limit": int(limit_runs)},
        "count": len(anomalies),
        "anomalies": anomalies,
        "alerts": [alert.to_dict() for alert in alerts],
    }


def _build_anomaly_summary(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for entry in entries:
        alert_codes = list(entry.get("alert_codes", ()))
        if not alert_codes and int(entry.get("open_order_count", 0)) == 0 and int(entry.get("rejected_order_count", 0)) == 0:
            continue
        anomalies.append(
            {
                "run_id": entry.get("run_id"),
                "session_date": entry.get("session_date"),
                "strategy_name": entry.get("strategy_name"),
                "model_name": entry.get("model_name"),
                "status": entry.get("status"),
                "alert_codes": alert_codes,
                "open_order_count": int(entry.get("open_order_count", 0)),
                "rejected_order_count": int(entry.get("rejected_order_count", 0)),
                "fill_count": int(entry.get("fill_count", 0)),
                "order_count": int(entry.get("order_count", 0)),
            }
        )
    return anomalies


def _build_trend_points(entries: Sequence[Mapping[str, Any]]) -> list[TrendPoint]:
    points: list[TrendPoint] = []
    if not entries:
        return points

    latest = entries[0]
    points.append(
        TrendPoint(
            key="latest_run_open_orders",
            value=float(int(latest.get("open_order_count", 0))),
            meta={"run_id": latest.get("run_id"), "session_date": latest.get("session_date")},
        )
    )
    points.append(
        TrendPoint(
            key="latest_run_alert_count",
            value=float(len(latest.get("alerts", ()))),
            meta={"run_id": latest.get("run_id"), "session_date": latest.get("session_date")},
        )
    )

    status_counts = _status_counts(entry["status"] for entry in entries)
    for status, count in sorted(status_counts.items()):
        points.append(TrendPoint(key=f"status_{status}", value=float(count), count=count))

    return points


def _build_session_breakdown(
    ledger: LocalLedger,
    entries: Sequence[Mapping[str, Any]],
    *,
    limit_sessions: int,
    reference_date: date | None,
) -> list[dict[str, Any]]:
    session_dates = sorted({entry.get("session_date") for entry in entries if entry.get("session_date")}, reverse=True)
    selected_dates = session_dates[: max(0, int(limit_sessions))]
    breakdown: list[dict[str, Any]] = []
    for session_date in selected_dates:
        session_payload = build_daily_summary_payload(
            ledger,
            session_date=date.fromisoformat(str(session_date)),
            reference_date=reference_date,
        )
        breakdown.append(
            {
                "session_date": session_payload["session_date"],
                "run_count": session_payload["run_count"],
                "status_counts": session_payload["status_counts"],
                "alert_codes": [alert["code"] for alert in session_payload["alerts"]],
                "open_order_count": session_payload["open_order_count"],
                "rejected_order_count": session_payload["rejected_order_count"],
            }
        )
    return breakdown


def _merge_alerts(alert_groups: Sequence[Sequence[OperatorAlert | Mapping[str, Any]]]) -> tuple[OperatorAlert, ...]:
    merged: dict[str, OperatorAlert] = {}
    for group in alert_groups:
        for alert in group:
            candidate = alert if isinstance(alert, OperatorAlert) else OperatorAlert(**dict(alert))
            existing = merged.get(candidate.code)
            if existing is None:
                merged[candidate.code] = candidate
                continue
            merged[candidate.code] = OperatorAlert(
                code=existing.code,
                severity=_max_severity(existing.severity, candidate.severity),
                title=existing.title,
                message=existing.message,
                details={**existing.details, **candidate.details},
            )
    return tuple(merged.values())


def _status_counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _alert_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for code in entry.get("alert_codes", ()):
            counts[str(code)] = counts.get(str(code), 0) + 1
    return counts


def _average(values: Sequence[float] | Sequence[int]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return float(sum(collected)) / float(len(collected))


def _max_severity(left: str, right: str) -> str:
    order = {"info": 0, "warning": 1, "critical": 2}
    return left if order.get(left, 0) >= order.get(right, 0) else right
