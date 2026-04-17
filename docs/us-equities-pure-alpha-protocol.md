# US Equities Pure Alpha Protocol

Date: 2026-04-17

## Purpose

This document freezes the first research contract for the `us_equities_pure_alpha_h5`
line.

The objective is a U.S. high-liquidity stock cross-sectional selection strategy
that is:

- long-short
- dollar neutral or near dollar neutral
- beta matched between the long and short books
- evaluated as pure stock-selection alpha, not as long equity beta

This line is separate from the existing long-only `us_equities_h5` paper stack.
It may reuse the same silver data and shared robustness infrastructure, but it
has its own universe, portfolio construction, and promotion gates.

## Calendar Contract

The global validation and test windows are aligned with the rebuilt Beta-thread
validation architecture:

- validation / research window: `2013-08-01` through `2019-12-31`
- final test lockbox: `2020-01-02` through `2026-04-08`

If the high-liquidity stock universe has usable point-in-time coverage only
after `2013-08-01`, the usable research start may move later. The final test
start must not move earlier.

The test lockbox is the last evaluation layer. It must not be used for:

- factor discovery
- feature selection
- model selection
- universe selection
- beta-matching parameter selection
- cost, borrow, or rebalance tuning
- deciding whether to keep or discard a candidate

Any test-window run must be explicitly labeled as a lockbox run and should only
happen after the validation-only research memo has frozen the candidate,
parameters, and expected failure modes.

## Universe Contract

The universe is independent from the Beta-thread ETF universe.

The intended universe is U.S. high-liquidity common stocks. The first point-in-
time implementation should prefer explicit eligibility rows over inferred
membership. A stock is eligible on a session only if the session-scoped universe
view says it is eligible.

Minimum v1 eligibility rules:

- U.S.-listed common equity or equivalent liquid primary share class
- price above a fixed minimum threshold, initially `10 USD`
- trailing median dollar volume above a fixed threshold, initially
  `50,000,000 USD`
- enough lagged return history to estimate beta and features
- no known trading halt, suspension, or non-tradable status for the session
- no stale price or missing adjusted open needed for entry / exit accounting

Short-side eligibility must be stricter than long-side eligibility when borrow
or locate data is unavailable. The first implementation may use a high-liquidity
proxy for shortability, but reports must label this as a borrow-data limitation.

## Signal Contract

Signals are cross-sectional. For a decision session `T`:

- features use only information available by `T` close
- portfolio formation happens after `T` close
- entry occurs at the next session open, `T+1`
- default exit occurs at the open after a `5` session holding period
- labels and realized returns use adjusted open-to-open returns

Every signal candidate should report:

- raw score
- rank or percentile within the eligible cross-section
- forward return target
- market-relative target
- beta-residual target when available
- sector and industry metadata used for diagnostics

## Beta Matching Contract

The portfolio must be constructed as a pure-alpha spread.

Default construction:

- long book: highest-ranked eligible names
- short book: lowest-ranked eligible names
- gross exposure: `1.0` long plus `1.0` short before optional scaling
- dollar neutrality: long capital and short capital should match
- beta neutrality: ex-ante long beta and short beta should match

Beta estimates must be lagged. The first default estimate is:

- benchmark: `SPY`
- lookback: `252` sessions
- minimum observations: `126`
- shrinkage target: `1.0`
- beta clip: `[0.0, 3.0]`

Initial beta gates:

- ex-ante absolute net beta target: `<= 0.05`
- validation realized market beta target: `abs(beta) <= 0.05`
- validation market correlation target: `abs(correlation) <= 0.10`

If beta matching fails on a session, the runner should either:

- reduce exposure until the constraint is met, or
- skip the session and record the skip reason

It should not silently run as a directional long-short book.

## Evaluation Metrics

Primary metrics:

- long-short net return
- annualized return
- annualized volatility
- Sharpe
- max drawdown
- realized regression beta versus `SPY`
- market correlation
- beta-adjusted alpha

Cross-sectional diagnostics:

- daily IC
- daily rank IC
- IC information ratio
- long-leg return
- short-leg return
- long-minus-short spread
- hit rate by leg

Implementation diagnostics:

- mean turnover
- cost sensitivity
- borrow / short-cost sensitivity when data exists
- long beta, short beta, and net beta by session
- gross and net exposure by session
- sector and industry exposure by side
- capacity proxies based on ADV participation

## Robustness Reuse

The line should reuse the shared robustness suite rather than building a
parallel evaluation layer.

Strategy project id:

- `us_equities_pure_alpha_h5`

Default robustness horizon:

- `5`

The long-short runner should emit the standard robustness artifact contract:

- `backtest_records.csv`
- `summary_metrics.csv`
- `predictions.csv`
- `protocol.json` or equivalent run metadata
- optional parameter and universe manifests

Required shared columns in `backtest_records.csv`:

- `entry_date`
- `exit_date`
- `net_return`
- `benchmark_return`
- `turnover`
- `cost_bps`
- `positions`

Pure-alpha extensions should also be emitted:

- `long_return`
- `short_return`
- `spread_return`
- `long_beta`
- `short_beta`
- `net_beta`
- `gross_exposure`
- `net_exposure`
- `long_count`
- `short_count`
- `beta_match_error`
- `borrow_cost_bps` when available

Shared robustness analyzers to reuse first:

- time stability
- tail dependence
- cost and execution stress
- turnover concentration
- selection-bias diagnostics from existing search manifests
- universe-stability diagnostics from existing universe manifests

Future pure-alpha-specific analyzers should add:

- realized beta stability
- leg balance stability
- sector and industry neutrality drift
- short-book crowding and borrow stress
- ADV capacity stress

## Promotion Discipline

A candidate may move from exploratory research to lockbox consideration only if
it passes validation-only review.

Minimum validation gates:

- positive long-short alpha after default costs
- realized beta and correlation within the beta-matching gates
- no single year explains the majority of total performance
- no top five days explain the majority of total performance
- performance remains positive after plausible cost stress
- both long and short legs contribute economically
- parameter choice is supported by a stable neighborhood, not one isolated peak
- universe and liquidity filters are frozen before any lockbox run

The lockbox test window should be opened only after these gates are documented.
If the test fails, the result should be recorded as a failed final evaluation
unless a new research cycle is explicitly declared.

