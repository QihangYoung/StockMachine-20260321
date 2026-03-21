# Paper Operator Playbook

This is the short runbook for the paper demo.

## Daily flow

1. Run the smoke harness first.
2. Review the JSON payload for `ok`, `preflight`, `artifact_link`, and `post_run`.
3. If the harness found an artifact directory, reuse it for reconciliation.
4. If there are lingering open orders, run maintenance before launching a fresh daily run.

## Recommended commands

```powershell
python -m stockmachine.apps.paper_smoke --session-date 2026-03-22 --run-name paper-smoke --artifact-root artifacts
python -m stockmachine.apps.paper_reconcile latest-run --ledger artifacts/paper_demo/paper_ledger.sqlite3
python -m stockmachine.apps.paper_maintain latest-run --ledger artifacts/paper_demo/paper_ledger.sqlite3
```

## Guardrails

- Keep `execute` off unless the preflight is healthy and the operator intends to submit paper orders.
- Prefer `artifact_link.exists == true` before relying on post-run reconciliation.
- If the manifest already carries an artifact directory, the reconcile command can infer the expected snapshot without extra parameters.
- If the smoke JSON shows stale open orders, clear them before another run.

## What to inspect

- `paper_smoke.ok`
- `paper_smoke.preflight.allowed`
- `paper_smoke.artifact_link.source`
- `paper_smoke.post_run.reconciliation.notes`
- `paper_reconcile.comparison.counts`
- `paper_maintain.plan.cancel_candidates`

## Minimal operator rule

If the smoke harness does not produce a clean `ok=true` payload, do not schedule the daily run.
