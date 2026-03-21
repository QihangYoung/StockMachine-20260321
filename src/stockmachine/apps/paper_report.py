from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Any, Sequence

from stockmachine.monitoring.digest import (
    build_daily_summary_payload,
    build_operator_digest_payload,
    build_run_index_payload,
)
from stockmachine.monitoring.health_trend import build_anomaly_summary_payload, build_health_trend_payload
from stockmachine.state.ledger import LocalLedger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-run operator reports as JSON.")
    parser.add_argument(
        "--ledger-path",
        default="artifacts/paper_demo/paper_ledger.sqlite3",
        help="Path to the local paper-demo ledger SQLite file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_index = subparsers.add_parser("run-index", help="Summarize the latest run history.")
    run_index.add_argument("--limit", type=int, default=10)

    daily_summary = subparsers.add_parser(
        "daily-digest",
        aliases=["daily-summary"],
        help="Summarize runs for one session date.",
    )
    daily_summary.add_argument("--session-date", default=None)

    health_trend = subparsers.add_parser("health-trend", help="Summarize recent health and anomaly trends.")
    health_trend.add_argument("--limit-runs", type=int, default=10)
    health_trend.add_argument("--limit-sessions", type=int, default=5)
    health_trend.add_argument("--session-date", default=None)

    anomaly_summary = subparsers.add_parser("anomaly-summary", help="Summarize recent anomalies only.")
    anomaly_summary.add_argument("--limit-runs", type=int, default=10)
    anomaly_summary.add_argument("--session-date", default=None)

    operator_digest = subparsers.add_parser("operator-digest", help="Bundle run index and daily summary.")
    operator_digest.add_argument("--session-date", default=None)
    operator_digest.add_argument("--limit", type=int, default=10)

    return parser


def dispatch_command(args: argparse.Namespace) -> dict[str, Any]:
    with LocalLedger(args.ledger_path) as ledger:
        ledger.initialize()
        if args.command == "run-index":
            return build_run_index_payload(ledger, limit=args.limit)
        if args.command in {"daily-digest", "daily-summary"}:
            reference_date = _parse_optional_date(args.session_date)
            return build_daily_summary_payload(ledger, session_date=reference_date, reference_date=reference_date)
        if args.command == "health-trend":
            reference_date = _parse_optional_date(args.session_date)
            return build_health_trend_payload(
                ledger,
                limit_runs=args.limit_runs,
                limit_sessions=args.limit_sessions,
                reference_date=reference_date,
            )
        if args.command == "anomaly-summary":
            reference_date = _parse_optional_date(args.session_date)
            return build_anomaly_summary_payload(
                ledger,
                limit_runs=args.limit_runs,
                reference_date=reference_date,
            )
        if args.command == "operator-digest":
            reference_date = _parse_optional_date(args.session_date)
            return build_operator_digest_payload(
                ledger,
                session_date=reference_date,
                limit=args.limit,
                reference_date=reference_date,
            )
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = dispatch_command(args)
    print(json.dumps(payload, indent=2, default=_json_default, sort_keys=True))


def _parse_optional_date(value: str | None) -> date | None:
    if value in (None, ""):
        return None
    return date.fromisoformat(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


if __name__ == "__main__":
    main()
