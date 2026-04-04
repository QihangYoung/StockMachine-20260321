# H1 Research Protocol

## Status

This document freezes the first research contract for the scaffolded
`us_equities_h1` strategy line.

It is intentionally narrower than the current h5 production-like stack. The
goal is to define one auditable starting point for horizon-1 research before
new features, engines, or paper profiles are introduced.

## Goal

Define the first canonical h1 contract for:

- timing and label semantics
- daily-rebalance assumptions
- walk-forward validation defaults
- cost and turnover review expectations
- promotion gates before any paper runtime is attempted

## Frozen V0 Scope

- market: US equities
- frequency: daily bars
- style: long-only, cross-sectional ranking
- holding horizon: `1` session
- benchmark: `SPY`
- execution assumption: signal after session `T` close, enter next open
- promotion target: research only, not paper-ready

## Time Definitions

- `session_date`: the trading session being scored after the close
- `effective_session_date`: the latest data session actually used when the
  requested session is unavailable
- `prediction_date`: the session whose close-side features generated the signal

For h1 v0:

- features are built using information visible by session `T` close
- signals are generated after session `T` close
- entries occur at session `T+1` open
- exits occur at session `T+2` open
- one portfolio rebalance opportunity exists each session

This keeps h1 aligned with the same next-open execution family as h5 while
allowing daily rebalance behavior.

## Label Definition

The first h1 target family is:

- `bucket_classification`

V0 starts with a binary bucket over the same continuous economic target:

- continuous target = `next_1_session_excess_return_from_next_open`
- stock return = `adj_open(T+2) / adj_open(T+1) - 1`
- benchmark return = `SPY_open(T+2) / SPY_open(T+1) - 1`
- excess target = stock return minus benchmark return
- bucket rule = `target_bucket_2 = 1` when excess target `> 0 bps`, else `0`

Trading score semantics for the first classification pass:

- `score` = predicted probability of the positive bucket
- `confidence` = cross-sectional rank percentile of that score for the session

This keeps the backtest contract aligned with the existing ranking pipeline while
changing the supervised objective from point regression to bucket prediction.

## Universe And Point-in-Time Rules

The h1 line inherits the same point-in-time discipline as h5:

- membership is session-scoped
- metadata must resolve from the latest visible snapshot at or before the
  session date
- missing membership excludes the symbol for that session
- missing metadata should fall back conservatively rather than silently dropping
  symbols

Until a dedicated h1 universe is justified, the starting universe should remain
the current liquid common-stock US-equities universe.

## Walk-Forward Validation Defaults

The first h1 validation schedule should stay conservative and familiar:

- train window: `36` months
- validation window: `6` months
- test window: `3` months
- roll frequency: `monthly`

Rationale:

- h1 generates many more observations than h5
- shortening the test block reduces stale-model exposure without immediately
  moving to daily refits
- a monthly roll keeps the first h1 comparison surface manageable

## Purge And Embargo Defaults

Default first-line settings:

- purge window: `2` sessions
- embargo window: `1` session

Rationale:

- the label overlap problem is materially smaller than h5 because the holding
  horizon is one session
- the protocol still keeps a conservative gap around split boundaries

## Feature Policy For V0

The first h1 pass should not reuse the h5 feature set unchanged.

Allowed first-wave feature families:

- overnight gap and gap normalization
- 1-day to 3-day reversal and short momentum
- recent range, ATR proxy, and realized volatility
- abnormal volume and volume acceleration
- benchmark-relative and sector-relative short-horizon strength

Deferred for later phases:

- intraday bars
- book or quote features
- order-flow proxies
- minute-level execution context

## Backtest Contract

The h1 line should not reuse the existing non-overlapping 5-session engine as
its final evaluation path.

The first h1 engine must eventually support:

- daily rebalance opportunities
- explicit daily turnover measurement
- explicit daily transaction-cost deductions
- turnover controls such as no-trade bands or turnover caps

Until that engine exists, no h1 result should be treated as paper-eligible.

## Cost Review Requirements

Any h1 leaderboard must report at least:

- annualized return
- annualized volatility
- Sharpe
- max drawdown
- mean daily turnover
- annualized cost drag estimate
- yearly stability summary

Minimum cost review grid for research decisions:

- `10 bps/side`
- `15 bps/side`
- `20 bps/side`
- `30 bps/side`

## Initial Baseline Models

The first h1 model pass should start with simple single-model classifiers:

- `ridge`: L2-regularized logistic classification over the standardized h1 feature panel
- `hist_gbm`: histogram gradient boosting classifier
- `extra_trees`: ExtraTrees classifier

Ensembles are explicitly deferred until the first single-model h1 results show
positive net value after realistic cost assumptions.

## Promotion Gate

No h1 paper profile should be created until the strategy line has:

- a dedicated h1 feature builder
- a dedicated h1 backtest engine
- documented cost-stress results
- acceptable yearly stability
- a clear operator story for daily turnover and execution volume

## Relationship To Existing Docs

This document complements, not replaces:

- [research-protocol.md](/E:/CodeX/StockMachine-260321/docs/research-protocol.md)
- [us-equities-v1.md](/E:/CodeX/StockMachine-260321/docs/us-equities-v1.md)
- [h1-development-plan.md](/E:/CodeX/StockMachine-260321/docs/h1-development-plan.md)
