# Paper Operator Playbook

This is the short runbook for the paper demo.

## Daily flow

1. Run the smoke harness first.
2. Review the JSON payload for `ok`, `preflight`, `artifact_link`, and `post_run`.
3. If the harness found an artifact directory, reuse it for reconciliation.
4. If the broker already filled an order but the ledger still shows it as open, run backfill first.
5. If there are still lingering open orders after backfill, run maintenance before launching a fresh daily run.

## Recommended commands

```powershell
python -m stockmachine.apps.paper_smoke --strategy-profile h5/us_extra_trees_daily --session-date 2026-03-22 --run-name paper-smoke --artifact-root artifacts
python -m stockmachine.apps.paper_backfill latest-run --strategy-project us_equities_h5 --artifact-root artifacts
python -m stockmachine.apps.paper_reconcile latest-run --strategy-project us_equities_h5 --artifact-root artifacts
python -m stockmachine.apps.paper_maintain latest-run --strategy-project us_equities_h5 --artifact-root artifacts
```

By default these commands now resolve the active h5 workspace under:

- `artifacts/strategy_projects/us_equities_h5/paper/paper_ledger.sqlite3`
- `artifacts/strategy_projects/us_equities_h5/paper/reports`
- `artifacts/strategy_projects/us_equities_h5/paper/paper_daily.kill`

Override with `--ledger-path`, `--artifact-dir`, or `--kill-switch-path` when a
run needs to point at a non-default workspace.

## Guardrails

- Keep `execute` off unless the preflight is healthy and the operator intends to submit paper orders.
- Prefer `artifact_link.exists == true` before relying on post-run reconciliation.
- If the manifest already carries an artifact directory, the reconcile command can infer the expected snapshot without extra parameters.
- If the smoke JSON shows stale open orders, backfill broker history before maintenance.
- The default paper runtime uses same-session `market/day` orders with a short post-submit polling window.

## What to inspect

- `paper_smoke.ok`
- `paper_smoke.preflight.allowed`
- `paper_smoke.artifact_link.source`
- `paper_smoke.post_run.reconciliation.notes`
- `paper_backfill.summary.open_orders_cleared`
- `paper_backfill.backfill.reconciliation.fill_events_created`
- `paper_reconcile.comparison.counts`
- `paper_maintain.plan.cancel_candidates`

## Minimal operator rule

If the smoke harness does not produce a clean `ok=true` payload, do not schedule the daily run.
