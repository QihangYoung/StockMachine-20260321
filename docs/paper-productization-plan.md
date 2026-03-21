# Paper Productization Plan

## Goal

Turn the current Alpaca paper-trading demo into a small daily-operable system.

The upgraded paper loop should be able to:

- refuse unsafe or duplicate runs before any order submission
- verify that the silver-layer dataset is fresh enough for the target session
- recover cleanly after a process restart
- keep order state synchronized even when the broker callback path is unavailable
- expose lightweight operator views for the latest run, open orders, and run details
- emit actionable alerts when the paper system blocks, fails, or drifts

## Scope

### In Scope

- market: US equities
- frequency: daily
- execution venue: Alpaca paper account
- orchestration: single-run CLI now, scheduler-friendly later
- state store: local SQLite ledger
- operator surface: CLI plus JSON summaries

### Out Of Scope

- live trading
- intraday execution logic
- production dashboard UI
- multi-broker support
- full strategy redesign

## Current Baseline

The repository already has:

- Alpaca market-data ingestion into silver tables
- baseline signal generation and formal silver-chain backtests
- a runnable Alpaca paper runner
- broker-aware order risk checks
- a local ledger with run, signal, target, order, fill, and audit tables
- real smoke-test evidence on an Alpaca paper account

The remaining work is mostly operational hardening rather than first-time feature creation.

## Productization Phases

### Phase 1: Single-Run Hardening

This phase makes one paper run safe and repeatable.

Required additions:

- session gate
  - verify whether the target session is valid for the selected run mode
  - block duplicate completed runs for the same strategy and session
- data freshness gate
  - verify silver coverage for the requested session date
  - expose the effective data session actually used by the runner
- hard execution caps
  - add global caps such as `max_total_orders` and `max_total_notional`
  - keep the existing per-order cap
- paper-environment guard
  - fail fast when the broker endpoint is not paper trading

Acceptance criteria:

- repeat execution for the same run key does not submit a second batch
- stale or missing data prevents execution
- the report clearly states which session date was actually used

### Phase 2: Order Lifecycle Hardening

This phase makes the paper loop restart-safe and easier to observe.

Required additions:

- recovery planner
  - compare ledger open orders with broker open orders
  - identify aligned orders, broker-only orders, and ledger-only orders
- restart recovery
  - perform one recovery sync before generating new orders
- polling fallback loop
  - normalize follow-up order snapshots into a reusable event stream
- future websocket seam
  - define the interface that a real Alpaca `trade_updates` listener will implement

Acceptance criteria:

- a restarted process can discover pre-existing open paper orders
- broker-only orders can be folded back into the ledger
- follow-up order polling can be reused by the runner without custom glue

### Phase 3: Operator Experience

This phase makes the system usable without opening SQLite manually.

Required additions:

- alert builder
  - summarize stale data, duplicate-run blocks, broker rejects, and lingering open orders
- paper ops CLI
  - `latest-run`
  - `open-orders`
  - `run-summary --run-id <id>`
- richer run summaries
  - expose alert payloads and the most relevant audit counters

Acceptance criteria:

- an operator can answer “did today run?”, “what is still open?”, and “what failed?”
- the latest run can be inspected from the CLI in one command

### Phase 4: Daily-Run Readiness

This phase prepares the demo to be scheduled and compared against research expectations.

Required additions:

- scheduler-friendly run contract
- health check command
- paper-versus-backtest reconciliation
- restart and drift runbook

This phase is intentionally after the first productization push.

## Workstream Split

The work is decomposed to keep write conflicts low.

### Workstream A: Run Governance

Owned paths:

- `src/stockmachine/live/session_guard.py`
- `src/stockmachine/live/__init__.py`
- `tests/test_live/test_session_guard.py`

Responsibilities:

- duplicate-run detection
- silver freshness checks
- effective session-date resolution
- reusable session guard result object

### Workstream B: Order Lifecycle

Owned paths:

- `src/stockmachine/live/recovery.py`
- `src/stockmachine/live/trade_updates.py`
- `src/stockmachine/live/__init__.py`
- `tests/test_live/test_recovery.py`
- `tests/test_live/test_trade_updates.py`

Responsibilities:

- restart recovery plan
- broker-versus-ledger open-order comparison
- polling update loop abstraction
- future websocket seam

### Workstream C: Operator Views

Owned paths:

- `src/stockmachine/monitoring/alerts.py`
- `src/stockmachine/monitoring/__init__.py`
- `src/stockmachine/apps/paper_ops.py`
- `tests/test_monitoring/test_alerts.py`
- `tests/test_apps/test_paper_ops.py`

Responsibilities:

- alert generation
- JSON operator CLI
- ledger-backed run inspection

### Main Thread

Responsibilities:

- integrate helpers into `run_us_equities_paper.py`
- extend risk placement and execution caps
- keep the end-to-end paper flow coherent
- run full validation and smoke tests

## Decoupling Analysis

### Safe To Build In Parallel

- session and freshness guards
- recovery planning
- alert generation
- ledger query CLI

These features mostly depend on stable ledger and report contracts.

### Better Integrated Centrally

- final runner gating order
- broker endpoint guard
- total-notional and total-order caps
- recovery invocation timing
- final report payload shape

These decisions affect the execution path and should be wired in one place.

## Delivery Sequence

1. freeze the productization plan
2. land helper modules from the three workstreams
3. integrate session guard and recovery into the runner
4. add total-order and total-notional hard caps
5. expose alerts and ops commands
6. run full tests
7. run one dry-run audit pass
8. run one minimal paper smoke and clean up the order

## First Completion Gate

This productization pass is successful when all of the following are true:

- `run_us_equities_paper.py` blocks duplicates and stale runs before order submission
- the runner can perform a restart recovery sync
- operator commands can inspect the latest run and open orders
- alerts are emitted for the most common operator-facing failures
- one dry-run and one paper smoke complete without manual database surgery
