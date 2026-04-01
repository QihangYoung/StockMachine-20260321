from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from stockmachine.apps.operator_paths import resolve_operator_ledger_path, resolve_strategy_workspace
from stockmachine.monitoring.reconciliation import (
    build_paper_reconciliation_summary,
    infer_expected_snapshot_from_manifest,
    load_expected_snapshot_from_artifact_dir,
)
from stockmachine.monitoring.reports import build_paper_artifact_link
from stockmachine.state import LocalLedger


def build_parser() -> argparse.ArgumentParser:
    artifact_parent = argparse.ArgumentParser(add_help=False)
    _add_artifact_arguments(artifact_parent)

    parser = argparse.ArgumentParser(description="Summarize paper trading reconciliation for one run.")
    parser.add_argument(
        "--strategy-project",
        default=None,
        help="Optional strategy project id used to resolve default operator paths.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Artifact root used when resolving project-scoped default paths.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
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
    args.ledger = resolve_operator_ledger_path(
        ledger_path=args.ledger,
        strategy_project=getattr(args, "strategy_project", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
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
        resolved_artifact_dir = args.artifact_dir
        if resolved_artifact_dir is None and manifest is not None:
            manifest_meta = dict(manifest.meta or {})
            manifest_strategy_lineage = manifest_meta.get("strategy_lineage", {})
            manifest_strategy_project = None
            if isinstance(manifest_strategy_lineage, dict):
                manifest_strategy_project = manifest_strategy_lineage.get("strategy_project")
            artifact_link = build_paper_artifact_link(
                artifact_root=args.artifact_root,
                manifest=manifest,
                run_name=manifest.strategy_name,
                session_date=args.session_date or getattr(manifest, "session_date", None),
                model_name=args.artifact_model or manifest.model_name,
                strategy_project=getattr(args, "strategy_project", None) or manifest_strategy_project,
            )
            if artifact_link.artifact_dir is not None and artifact_link.exists:
                resolved_artifact_dir = artifact_link.artifact_dir
        if resolved_artifact_dir is not None:
            expected_snapshot = load_expected_snapshot_from_artifact_dir(
                resolved_artifact_dir,
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
        payload = summary.to_dict()
        workspace = resolve_strategy_workspace(
            strategy_project=getattr(args, "strategy_project", None),
            artifact_root=getattr(args, "artifact_root", "artifacts"),
        )
        if workspace is not None:
            payload["strategy_workspace"] = workspace.to_dict()
        if resolved_artifact_dir is not None:
            payload["resolved_artifact_dir"] = str(resolved_artifact_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
