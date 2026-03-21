# Architecture

## Goal

Build a trading system that keeps alpha research flexible while making risk and
execution explicit, testable, and reviewable.

## Core Separation

The system is split into three main layers:

1. `alpha`
   - consumes market state and features
   - emits signals only
   - does not know about broker specifics
2. `portfolio` and `risk`
   - turn signals into target positions
   - apply white-box exposure, drawdown, liquidity, and compliance rules
   - may veto any signal
3. `execution`
   - turns approved target positions into executable orders
   - manages order style, slicing, retry, and slippage assumptions

This separation is the main guardrail of the repository.

## Data Flow

```text
market data
  -> features
  -> alpha signal
  -> portfolio target
  -> risk approval or veto
  -> execution intent
  -> order or fill events
  -> ledger, metrics, and monitoring
```

## Module Map

### `src/stockmachine/domain`

Shared contracts and typed objects that move across layers.

### `src/stockmachine/data`

Market data adapters, calendars, vendor wrappers, caching helpers, and
ingestion utilities.

### `src/stockmachine/ingestion`

Raw collection jobs, crawlers, parsing logic, source snapshots, and
normalization into internal canonical tables.

### `src/stockmachine/features`

Feature engineering, labeling, factor definitions, and transformation
pipelines.

### `src/stockmachine/alpha`

Model training, inference, signal normalization, and prediction orchestration.

### `src/stockmachine/portfolio`

Portfolio construction, rebalancing logic, and target weight generation.

### `src/stockmachine/risk`

White-box rules such as max position, drawdown brakes, turnover limits, and
exposure caps.

### `src/stockmachine/execution`

Order generation, slippage models, broker adapters, and fill handling.

### `src/stockmachine/backtest`

Simulation engine, accounting, transaction cost models, metrics, and reports.

### `src/stockmachine/monitoring`

Logs, alerts, audit trails, and post-trade diagnostics.

### `src/stockmachine/apps`

Entry points for training, backtesting, paper trading, and live runs.

## Runtime Modes

- `backtest`: replay historical data through the same contracts
- `paper`: run against current data while simulating order placement
- `live`: send approved orders to a broker

The same domain objects should move through all three modes.

## Contract Direction

The intended object flow is:

- `Signal`: output of prediction
- `TargetPosition`: output of portfolio and risk decision
- `OrderIntent`: output of execution planning

Prediction should never emit `OrderIntent` directly.

## Data Lifecycle

The intended data path is:

```text
external sources
  -> ingestion collectors and crawlers
  -> raw snapshots
  -> normalized internal tables
  -> feature pipeline
  -> model training and inference
```

Keeping raw snapshots separate from normalized tables helps later debugging when
source websites change or a parsing rule needs to be replayed.

## Non-Goals For The First Iteration

- end-to-end reinforcement learning with direct order outputs
- high-frequency trading infrastructure
- multi-asset support before a single equity workflow is stable
- large-scale distributed training before evaluation standards are clear
