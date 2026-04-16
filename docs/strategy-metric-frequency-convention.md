# Strategy Metric Frequency Convention

## Goal

Freeze one consistent rule for:

- which Sharpe frequency is the research main metric for each strategy family
- which frequency should be used for real risk interpretation
- how to compare strategies across families without mixing incompatible numbers

## Core Rule

There is no single universal Sharpe frequency that is correct for every
strategy. The correct main metric depends on the strategy contract.

Use two layers:

- `research main metric`: follow the strategy's native return contract
- `risk main metric`: follow the highest realistic marked-to-market frequency
  of the portfolio path

This means:

- do not compare `5-session` Sharpe directly against `daily` Sharpe
- do not compare `21-session` Sharpe directly against `daily` Sharpe
- only compare Sharpe values directly when the sampling frequency, execution
  timing, cost assumptions, and sample window are all aligned

## Definitions

### Contract Return

The return naturally implied by the strategy's execution contract.

Examples:

- `h5`: next-open entry, hold for `5` sessions, exit at next open after the
  holding window
- `h1`: next-open entry, hold for `1` session

### Marked-to-Market Return

The return of the live portfolio path as it would be observed through time,
even when no rebalance occurs on that day.

Examples:

- an ETF multi-asset core that only rebalances every `21` sessions still has a
  daily marked-to-market return path
- a paper portfolio with persistent open positions should have a daily equity
  curve

## Family Rules

| Strategy family | Native contract | Research main metric | Risk main metric | Notes |
|---|---|---|---|---|
| `us_equities_h5` | `5-session open-hold` | `5-session Sharpe` | `daily` marked-to-market risk panel | Keep historical leaderboard continuity, but add daily risk diagnostics |
| `us_equities_h1` | `1-session open-hold / daily rebalance` | `daily Sharpe` | `daily Sharpe` and `daily vol` | Contract and MTM path are both daily |
| static multi-asset ETF core | continuous ETF portfolio | `daily Sharpe` | `daily Sharpe` and `daily vol` | Rebalance cadence does not change the fact that the portfolio exists daily |
| rolling multi-asset ETF core | continuous ETF portfolio with periodic rebalance | `daily Sharpe` | `daily Sharpe` and `daily vol` | `21` sessions is a rebalance interval, not a return contract |
| monthly NAV / illiquid future strategies | monthly valuation contract | `monthly Sharpe` | `monthly vol` | Only use monthly as main when the portfolio itself is only observed monthly |

## Decision Logic

### Use the contract-frequency Sharpe when

- comparing models within the same strategy family
- deciding whether a new feature, target, or overlay improved the native
  strategy contract
- maintaining continuity with an existing family leaderboard

### Use daily risk metrics when

- evaluating true holding-path risk
- comparing risk across strategy families
- deciding whether a strategy can sit inside a broader portfolio
- reviewing drawdown shape, realized volatility, and capital usage

## Current Repository Decisions

### `h5`

Keep:

- `5-session Sharpe` as the main research metric for model ranking and
  promotion inside the `h5` family

Add:

- daily marked-to-market risk panel for risk interpretation and cross-family
  comparison

Minimum daily risk panel:

- daily annualized volatility
- daily Sharpe
- max drawdown
- daily equity curve

### Multi-Asset Core

Use:

- `daily Sharpe` as the main research metric
- `daily annualized volatility` and `max drawdown` as the main risk metrics

Reason:

- the portfolio exists every session
- weights drift every day even between rebalance dates
- `21` sessions is a rebalance interval, not a hold-to-maturity contract

Optional compatibility views:

- `21-session` block returns
- monthly returns

These are secondary diagnostics, not the main headline metric.

## Interpretation Rule

If contract Sharpe and daily Sharpe disagree:

- trust contract Sharpe for same-family alpha ranking
- trust daily risk metrics for real risk interpretation

This prevents two common mistakes:

- overstating risk quality because a coarse sampling frequency hides path
  volatility
- rejecting a strong contract-level alpha signal just because it is being
  judged with a frequency that does not match the strategy definition

## Reporting Minimum

Every future report should say explicitly:

- `research main metric frequency`
- `risk main metric frequency`
- `rebalance interval`
- `execution timing assumption`

Example:

- `h5`: research main = `5-session`; risk main = `daily`; rebalance =
  `5-session`; execution = `T+1 open`
- `rolling multi-asset`: research main = `daily`; risk main = `daily`;
  rebalance = every `21` sessions; execution = next effective rebalance date

## Non-Goal

This convention does not claim that one frequency is universally superior.
It only freezes which frequency should be treated as primary for each strategy
family in this repository.
