from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from stockmachine.monitoring.reconciliation import (
    build_paper_reconciliation_summary,
    infer_expected_snapshot_from_manifest,
    load_expected_snapshot_from_artifact_dir,
)
from stockmachine.state import LocalLedger


def build_parser() -> argparse.ArgumentParser:
    artifact_parent = argparse.ArgumentParser(add_help=False)
    _add_artifact_arguments(artifact_parent)

    parser = argparse.ArgumentParser(description="Summarize paper trading reconciliation for one run.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/paper_demo/paper_ledger.sqlite3"),
        help="Path to the paper-trading SQLite ledger.",
    )
    parser.add_argument("--run-id", help="Reconcile a specific run id.")
    _add_artifact_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "latest-run",
        help="Reconcile the latest run in the ledger.",
        parents=[artifact_parent],
        add_help=False,
    )
    return parser


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--session-date",
        help="Optional target session date used when selecting the expected research snapshot.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Optional research/backtest artifact directory used to load expected snapshot data.",
    )
    parser.add_argument(
        "--artifact-model",
        help="Optional model name override when loading expected snapshot artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional top-k override for artifact-based expected snapshot reconstruction.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ledger = LocalLedger(args.ledger)
    ledger.initialize()
    try:
        run_id = args.run_id
        if args.command == "latest-run":
            run_id = None
        manifest = ledger.get_run_manifest(run_id) if run_id is not None else None
        if manifest is None:
            manifests = ledger.list_run_manifests()
            manifest = manifests[0] if manifests else None
        expected_snapshot = None
        if args.artifact_dir is not None:
            expected_snapshot = load_expected_snapshot_from_artifact_dir(
                args.artifact_dir,
                model_name=args.artifact_model,
                top_k=args.top_k,
                target_session_date=args.session_date or getattr(manifest, "session_date", None),
            )
        else:
            expected_snapshot = infer_expected_snapshot_from_manifest(
                manifest,
                model_name=args.artifact_model,
                top_k=args.top_k,
                target_session_date=args.session_date,
            )
        summary = build_paper_reconciliation_summary(ledger, run_id=run_id, expected_snapshot=expected_snapshot)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0
    finally:
        ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
