# Merge Guide For External Contributors

This document is a short integration map for teams merging new code into the
repository. It focuses on the current paper-trading stack, the research
pipeline, and the places where merge conflicts are most likely to become
behavioral regressions.

## What Is Already Stable

The following layers are now reasonably settled and should be treated as the
default integration surface:

- Shared domain objects: [src/stockmachine/domain/models.py](E:/CodeX/StockMachine-260321/src/stockmachine/domain/models.py)
- Backtest and runtime contracts: [src/stockmachine/backtest/protocols.py](E:/CodeX/StockMachine-260321/src/stockmachine/backtest/protocols.py)
- Silver data loading: [src/stockmachine/data/loaders/silver.py](E:/CodeX/StockMachine-260321/src/stockmachine/data/loaders/silver.py)
- Alpaca broker adapter: [src/stockmachine/execution/brokers/alpaca.py](E:/CodeX/StockMachine-260321/src/stockmachine/execution/brokers/alpaca.py)
- Alpaca account sync: [src/stockmachine/live/account_sync.py](E:/CodeX/StockMachine-260321/src/stockmachine/live/account_sync.py)
- Order reconciliation: [src/stockmachine/live/reconciler.py](E:/CodeX/StockMachine-260321/src/stockmachine/live/reconciler.py)
- Paper-run ledger: [src/stockmachine/state/ledger.py](E:/CodeX/StockMachine-260321/src/stockmachine/state/ledger.py)
- Paper runner: [src/stockmachine/apps/run_us_equities_paper.py](E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_us_equities_paper.py)
- Broker-aware risk gate: [src/stockmachine/risk/paper.py](E:/CodeX/StockMachine-260321/src/stockmachine/risk/paper.py)

## Merge Boundary Map

The repository is easiest to merge if contributors keep these boundaries
intact:

- `alpha` should emit signals only.
- `portfolio` and `risk` should turn signals into approved target positions.
- `execution` should turn approved targets into orders.
- `live` should handle broker sync, recovery, and reconciliation.
- `state` should own the local ledger and audit trail.
- `apps` should only orchestrate the end-to-end run.

If a change crosses more than one layer, it should usually be integrated in the
runner rather than hidden inside a lower-level helper.

## Highest-Risk Merge Areas

These files are the most likely to cause subtle regressions if changed without
coordinated review:

- [src/stockmachine/apps/run_us_equities_paper.py](E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_us_equities_paper.py)
- [src/stockmachine/risk/paper.py](E:/CodeX/StockMachine-260321/src/stockmachine/risk/paper.py)
- [src/stockmachine/state/ledger.py](E:/CodeX/StockMachine-260321/src/stockmachine/state/ledger.py)
- [src/stockmachine/live/reconciler.py](E:/CodeX/StockMachine-260321/src/stockmachine/live/reconciler.py)
- [src/stockmachine/research/us_equities_baseline.py](E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py)
- [src/stockmachine/data/loaders/silver.py](E:/CodeX/StockMachine-260321/src/stockmachine/data/loaders/silver.py)

The main reasons are:

- these modules carry shared object shapes
- they define timing and reconciliation behavior
- they encode ledger schema and report metadata
- they sit on the path from model output to broker submission

## Merge Order Recommendation

When combining work from another repository or another branch, the safest order
is:

1. Freeze or adapt shared contracts first.
2. Merge broker adapter and account-sync changes.
3. Merge ledger and reconciliation changes.
4. Merge runner orchestration and monitoring changes.
5. Merge research or evaluation changes last.

This order reduces the chance of having a runner that compiles but silently
changes runtime semantics.

## Known Design Constraints

The current codebase makes a few decisions that external changes should respect:

- `OrderIntent`, `Signal`, `TargetPosition`, and `AccountSnapshot` are the
  primary cross-layer objects.
- `PaperRunManifest` and the SQLite ledger are the audit trail for paper runs.
- The paper runner now depends on deterministic `client_order_id` generation.
- Broker-aware risk checks are expected to happen before submission, not after.
- Reconciliation is expected to update the ledger from broker snapshots.

If a proposed change changes one of these rules, it should be reviewed as an
architecture change rather than a small refactor.

## Current Paper Demo Flow

The current end-to-end paper path is:

```text
silver data
  -> universe selection
  -> signal model
  -> portfolio policy
  -> execution policy
  -> broker-aware risk gate
  -> Alpaca submit
  -> polling reconciliation
  -> local ledger and report
```

Relevant files:

- [src/stockmachine/apps/run_us_equities_paper.py](E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_us_equities_paper.py)
- [src/stockmachine/risk/paper.py](E:/CodeX/StockMachine-260321/src/stockmachine/risk/paper.py)
- [src/stockmachine/state/ledger.py](E:/CodeX/StockMachine-260321/src/stockmachine/state/ledger.py)
- [src/stockmachine/monitoring/reports.py](E:/CodeX/StockMachine-260321/src/stockmachine/monitoring/reports.py)

## Research Caveats To Preserve

The following limitations are still documented and should not be removed unless
they have been fully fixed in code and tests:

- current large-cap universe bias in the baseline reports
- latest-snapshot symbol master loading
- year-based holdout evaluation in the baseline experiment
- internal same-pool benchmark in the multi-expert report

Relevant references:

- [docs/us-equities-baseline-report.md](E:/CodeX/StockMachine-260321/docs/us-equities-baseline-report.md)
- [docs/evaluation-protocol.md](E:/CodeX/StockMachine-260321/docs/evaluation-protocol.md)
- [us_a_share_multi_expert_report.md](E:/CodeX/StockMachine-260321/us_a_share_multi_expert_report.md)

## Minimum Verification Checklist

After a merge, run at least these checks:

- paper smoke test for the runner
- risk policy unit tests
- ledger persistence tests
- reconciler tests
- silver loader tests
- the paper report CLI smoke

Suggested test entry points:

- `tests/test_apps/test_run_us_equities_paper.py`
- `tests/test_risk/test_paper.py`
- `tests/test_state/test_ledger.py`
- `tests/test_live/test_reconciler.py`
- `tests/test_ingestion/test_silver_loader.py`
- `tests/test_apps/test_paper_report.py`

## Open Merge Questions

Before merging larger changes, align on the following:

- whether the merge is research-only, paper-demo-only, or both
- whether the external branch owns any contracts already present here
- whether any current fallback behavior should be removed or preserved
- whether the target is to keep the paper runner as the single orchestrator

If those answers are unclear, start with the contract freeze instead of the
implementation merge.
