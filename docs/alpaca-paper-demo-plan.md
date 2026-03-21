# Alpaca Paper Demo Plan

## Goal

Deliver a first end-to-end demo that can run a daily US-equities strategy
against an Alpaca paper trading account.

The demo should be able to:

- read the latest silver-layer market data
- produce model signals for one run
- turn signals into approved target positions
- convert approved targets into broker orders
- submit and reconcile orders against Alpaca paper trading
- persist enough state for audit and restart recovery

## Scope

### In Scope For The First Demo

- market: US equities
- frequency: daily
- strategy style: long-only
- alpha source: existing baseline model output
- execution mode: Alpaca paper account only
- orchestration: single-run CLI, safe to schedule later

### Out Of Scope For The First Demo

- live trading
- intraday strategy logic
- websocket-first execution flow
- advanced execution algorithms
- production-grade dashboarding
- multi-broker support

## Current Project Status

The repository already has a solid research and data foundation:

- Alpaca market-data ingestion into silver tables
- baseline model research and evaluation flow
- white-box portfolio and execution policy prototypes
- formal backtest contracts and a working prototype engine

The main gap is the trading-side runtime. The project does not yet have:

- a paper-trading broker adapter
- a persistent runtime ledger
- order-state reconciliation
- a paper runner entrypoint
- broker-aware operational risk checks

## Required Modules

### P0: Required For A Working Paper Demo

1. `execution/models.py`
   Define trading-side objects such as account snapshot, broker order, fill
   event, market clock, and execution report.

2. `execution/brokers/alpaca_trading.py`
   Implement the Alpaca paper-trading adapter with account, order, and
   position methods.

3. `live/account_sync.py`
   Map broker account and broker positions into project-friendly runtime
   objects.

4. `state/ledger.py`
   Persist signals, targets, orders, fills, and equity snapshots in a local
   recoverable store.

5. `live/reconciler.py`
   Poll broker order state and fold updates into the ledger.

6. `apps/run_us_equities_paper.py`
   Add a CLI that wires together data loading, signal generation, risk,
   execution planning, order submission, and reporting.

7. `monitoring/*`
   Emit one structured run summary and one failure summary for every paper run.

8. Broker-aware risk checks
   Extend portfolio and execution validation with buying power, duplicate-order,
   open-order, and market-open checks.

### P1: Important Soon After The First Demo

- Alpaca `trade_updates` websocket integration
- restart recovery from persisted open orders
- paper-versus-backtest reconciliation report
- stricter point-in-time universe handling
- richer cost and slippage attribution

## Workstream Split

The delivery plan is intentionally split into parallel workstreams with
minimized write conflicts.

### Workstream A: Broker Adapter

Owned paths:

- `src/stockmachine/execution/brokers/**`
- `src/stockmachine/live/account_sync.py`
- `src/stockmachine/live/__init__.py`
- `tests/test_execution/**`
- `tests/test_live/**`

Responsibilities:

- Alpaca trading adapter
- account snapshot mapping
- broker payload normalization
- broker adapter unit tests

### Workstream B: Ledger And Reconciliation

Owned paths:

- `src/stockmachine/state/**`
- `src/stockmachine/live/reconciler.py`
- `tests/test_state/**`
- `tests/test_live/test_reconciler.py`

Responsibilities:

- local runtime ledger
- order and fill persistence
- order-state reconciliation loop
- restart-safe state handling

### Workstream C: Runner And Monitoring

Owned paths:

- `src/stockmachine/apps/run_us_equities_paper.py`
- `src/stockmachine/monitoring/**`
- `tests/test_apps/**`
- `tests/test_monitoring/**`

Responsibilities:

- paper-run CLI
- run-phase orchestration
- dry-run support
- structured run report

### Main Thread

Responsibilities:

- freeze contracts between workstreams
- extend broker-aware risk validation
- integrate the workstreams into one runnable path
- verify tests and produce progress reports

## Decoupling Analysis

### Can Be Developed In Parallel

- trading adapter and account sync
- local ledger storage
- paper-run CLI shell
- monitoring and reporting helpers

These modules only need shared contracts, not shared implementations.

### Can Be Developed In Parallel After Contracts Freeze

- order reconciliation
- broker-aware execution validation
- paper runner wiring

These need stable object shapes for account snapshots, orders, and execution
reports.

### Should Be Integrated Centrally

- final order lifecycle from `OrderIntent` to broker order state
- final ledger schema used by both runner and reconciler
- final risk gate placement in the end-to-end paper flow

These pieces touch multiple layers and are better integrated in one place.

## Contract Freeze For Parallel Work

The following interfaces should stay small and stable:

1. broker adapter
   - `get_account()`
   - `get_clock()`
   - `list_positions()`
   - `list_orders(status)`
   - `get_order(order_id)`
   - `submit_order(order_request)`
   - `cancel_order(order_id)`

2. account sync
   - input: broker account payload and broker position payloads
   - output: project-side runtime account snapshot

3. ledger
   - append signal batch
   - append target batch
   - append broker order snapshots
   - append fill events
   - append equity snapshot
   - fetch open orders

4. runner
   - load data
   - get signals
   - build targets
   - validate targets
   - generate order intents
   - submit or dry-run
   - reconcile
   - emit report

## Delivery Sequence

1. freeze trading-side contracts
2. finish Alpaca trading adapter and account sync
3. finish local ledger and reconciler
4. land paper runner and monitoring shell
5. add broker-aware risk checks
6. integrate end to end and run a dry-run paper pass
7. run one real paper-account execution cycle

## Acceptance Criteria

The first paper demo is ready when all of the following are true:

- a single CLI command can execute one full paper run
- `--dry-run` produces signals, targets, orders, and a structured report
- live paper mode can submit orders to Alpaca paper trading
- order status can be polled and persisted after submission
- the run produces a recoverable local ledger entry
- the run fails safely when market is closed, buying power is insufficient, or
  there are conflicting open orders

## Reporting Expectations

Each implementation step should report:

- files changed
- tests added or updated
- current blockers
- integration assumptions

This keeps parallel work easy to merge and review.
