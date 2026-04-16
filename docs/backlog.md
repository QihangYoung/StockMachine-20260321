# Backlog

## Goal

This backlog turns the project roadmap into an execution-oriented task list.
The priority rule is simple:

- `P0`: required to establish a trustworthy research loop
- `P1`: required to produce a first tradable system prototype
- `P2`: improves robustness, breadth, and operational readiness

## Delivery Gates

The project should pass these gates in order:

1. problem definition is frozen enough for the first research loop
2. data ingestion is reproducible and point-in-time safe
3. backtest and evaluation reports are trustworthy
4. one baseline alpha model works end to end
5. white-box risk and execution simulation are integrated
6. paper trading is stable before any live execution

## P0: Research Foundation

### Scope Decisions

- choose first market: A-share or US equities
- choose bar frequency: daily by default
- choose holding horizon: 5 trading days by default
- choose strategy style: long-only by default unless shorting support is ready
- choose first universe: a stable benchmark universe such as CSI 300, CSI 500,
  or S&P 500

### Data Contracts

- define canonical schemas for `symbol_master`, `daily_bar`,
  `trading_calendar`, `adj_factor`, `benchmark_index`, `daily_basic`,
  `industry_membership`, `suspend_resume`, and `price_limit`
- define metadata fields for source, load time, effective time, and version
- define internal file and table naming conventions

### Ingestion Skeleton

- implement source catalog structure in `ingestion/sources`
- implement raw payload snapshot conventions in `ingestion/storage`
- implement collector interfaces for API pulls and crawlers
- implement checkpoint format for incremental sync jobs
- decide local storage format for the first version

### Evaluation Skeleton

- define walk-forward split policy
- define anti-leakage timing policy for feature cutoff and trade execution time
- define experiment report template
- define baseline metrics: Rank IC, spread return, turnover, Sharpe, drawdown
- define pass or fail rules for model promotion

### Deliverables

- frozen v1 problem statement
- canonical dataset schema document
- evaluation protocol document
- empty but stable ingestion and backtest interfaces

## P1: First End-to-End Prototype

### Data Ingestion

- implement stock or symbol master ingestion
- implement trading calendar ingestion
- implement daily OHLCV ingestion
- implement adjustment factor ingestion
- implement benchmark index ingestion
- implement daily indicator ingestion
- implement suspension and price-limit ingestion
- add normalization jobs from raw to silver tables
- add data quality checks for duplicates, missing dates, invalid prices, and
  symbol status mismatches

### Feature and Label Pipeline

- implement future 5-day excess return label generation
- implement baseline factor set:
  - momentum
  - short-term reversal
  - volatility
  - liquidity
  - benchmark-relative strength
  - sector-relative strength
- add winsorization, neutralization, and normalization hooks
- build reproducible training dataset generation

### Backtest Engine

- implement account ledger
- implement order and fill event model
- implement rebalance logic
- implement fee and slippage model
- implement market constraints for the chosen first market
- generate summary reports and factor diagnostics

### Baseline Alpha

- implement factor-score baseline
- implement one linear baseline model
- implement one tree-based model such as LightGBM or CatBoost
- add rolling training and rolling inference workflow
- store prediction outputs for later audit

### Risk and Portfolio

- implement target position generation from ranked signals
- implement max single-name cap
- implement gross exposure cap
- implement industry or sector exposure cap
- implement turnover cap
- implement drawdown brake

### Execution Simulation

- implement target-to-order conversion
- implement simple order sizing rules
- implement limit and market style simulation
- implement partial fill and cancel behavior assumptions
- track predicted price versus simulated fill price

### Deliverables

- reproducible research dataset
- one baseline backtest report
- one baseline signal model report
- one white-box risk rule set
- one simulated execution report

## P2: Robustness And Operations

### Data Expansion

- add announcement metadata ingestion
- add filing text or event extraction pipeline if needed
- add alternative sources for cross-checking critical fields
- add historical universe membership snapshots

### Modeling Upgrades

- add ensemble of linear and tree-based models
- add multi-horizon targets
- add confidence model or meta-labeling
- add market regime classifier
- add sector-neutral or style-neutral training variants

### Evaluation Upgrades

- add bootstrap-based uncertainty estimates
- add cost stress testing
- add capacity analysis
- add delayed execution sensitivity tests
- add parameter stability sweeps
- add daily marked-to-market risk panel for `h5` so contract-level `5-session`
  Sharpe and daily risk metrics are both available

### Operational Readiness

- implement paper trading app flow
- implement monitoring and alerting
- implement audit trail for predictions, targets, orders, and fills
- implement broker adapter abstraction
- implement daily health checks for data freshness and job failures

### Deliverables

- stable paper trading loop
- model comparison dashboard or report pack
- monitoring and audit baseline
- production-readiness checklist

### Parallel Research Thread: Multi-Asset Allocation

- define the first multi-asset line as a separate ETF-based strategy project
- freeze a small explicit cross-asset ETF universe and bucket map
- reuse white-box portfolio and risk contracts, but do not overload the
  current sector-based equity policy
- compare static baselines, white-box tactical allocation, and core-plus-sleeve
  variants before adding model-heavy allocation logic
- require the same strict walk-forward, cost-stress, and robustness discipline
  used by the current equity line

## Suggested Build Order

1. freeze v1 scope
2. define schemas and timing rules
3. ingest first daily-bar dataset
4. build label and feature pipeline
5. build trustworthy backtest
6. add baseline alpha
7. add white-box risk and portfolio logic
8. add execution simulation
9. expand robustness tests
10. run paper trading

## Current Recommendation

If we want the fastest path to a meaningful first result, the practical v1 is:

- market: choose one only
- frequency: daily
- horizon: 5 trading days
- style: long-only
- model path: factor baseline -> linear baseline -> tree model
- go-live path: backtest -> paper -> live
