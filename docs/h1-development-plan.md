# H1 Development Plan

## Goal

Stand up `us_equities_h1` as an independent 1-session US-equities strategy
line instead of treating `horizon=1` as a minor variation of the existing
`us_equities_h5` swing stack.

The purpose of this split is governance as much as modeling:

- separate feature research from the current 5-session feature set
- separate turnover and cost assumptions from the h5 baseline
- separate evaluation, paper, and operator defaults from the active h5 line

## Why H1 Is A Separate Project

The current production-like stack is optimized around a 5-session contract:

- labels are `T+1` open to `T+6` open
- backtests rebalance in non-overlapping 5-session cohorts
- execution and risk assumptions are tuned for lower-turnover swing behavior

An h1 strategy will likely differ in:

- feature design
- turnover profile
- execution sensitivity
- cost model and capacity limits
- evaluation criteria and promotion gates

Because of that, h1 should evolve as its own strategy subproject under
`us_equities_h1`.

## Current Status

- `us_equities_h1` exists as a scaffolded strategy project:
  [us_equities_h1.json](/E:/CodeX/StockMachine-260321/configs/strategy_projects/us_equities_h1.json)
- no runnable built-in h1 paper profiles exist yet
- no h1-specific feature builder, backtest engine, or research protocol exists yet

## Phase Plan

### Phase 1: Governance And Protocol Freeze

1. Freeze the h1 trading protocol.
2. Write a dedicated h1 research protocol document.
3. Record project-specific promotion gates for h1 before paper trading.

Deliverables:

- [h1-research-protocol.md](/E:/CodeX/StockMachine-260321/docs/h1-research-protocol.md)
- updated `us_equities_h1` project metadata and docs index

### Phase 2: H1 Feature MVP

1. Add a minimal h1 feature set built for 1-session holding periods.
2. Keep the first version daily-bar only.
3. Avoid minute data, market microstructure feeds, or complex intraday logic in v0.

Candidate first-wave features:

- overnight gap
- 1-day and 2-day reversal
- short-window momentum
- gap-to-range context
- recent realized volatility and range compression
- abnormal volume and volume acceleration
- benchmark-relative and sector-relative short-term strength

### Phase 3: H1 Backtest Contract

1. Add an h1-specific backtest engine.
2. Support daily rebalancing rather than 5-session non-overlapping cohorts.
3. Track daily turnover, cost drag, and optional turnover controls explicitly.

Expected additions:

- no-trade band support
- daily turnover cap
- minimum holding period option
- explicit cash-park handling for skipped trades

### Phase 4: Baseline Model Pass

1. Run a first strict h1 sweep on a small model set.
2. Start with simple tabular models before ensembles.

Recommended first models:

- `ridge` line as L2-regularized logistic classification
- `hist_gbm` classifier
- `extra_trees` classifier

### Phase 5: Cost And Stability Qualification

1. Run cost-stress tests at multiple per-side cost levels.
2. Review yearly stability, turnover concentration, and drawdown shape.
3. Reject any h1 line that only survives under unrealistically low costs.

### Phase 6: Paper Promotion Gate

Only consider paper shadow mode after h1 satisfies all of the following:

- net alpha remains positive under conservative transaction-cost assumptions
- turnover and order count fit realistic operational capacity
- yearly results are not dominated by one narrow regime
- operator workflow can explain the strategy without relying on implicit h5 assumptions

## Immediate Next Steps

The immediate next implementation steps are:

1. freeze the h1 trading and timing contract
2. publish the dedicated h1 research protocol
3. prepare the minimal h1 feature package

## Explicit Non-Goals For The First H1 Pass

- minute-bar alpha
- intraday execution optimization
- smart order routing
- short selling
- options overlays
- live trading promotion

## Reference Files

- [research-protocol.md](/E:/CodeX/StockMachine-260321/docs/research-protocol.md)
- [us-equities-v1.md](/E:/CodeX/StockMachine-260321/docs/us-equities-v1.md)
- [us_equities_h1.json](/E:/CodeX/StockMachine-260321/configs/strategy_projects/us_equities_h1.json)
- [h1 README](/E:/CodeX/StockMachine-260321/configs/strategies/h1/README.md)
