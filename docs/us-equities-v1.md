# US Equities V1 Scope

## Status

This document freezes the first implementation target for the repository.
Anything outside this scope is intentionally deferred until the first research
loop is stable.

## Scope Summary

- market: US equities
- frequency: daily bars
- strategy style: long-only
- task: cross-sectional stock ranking
- label horizon: 5 trading sessions
- benchmark: `SPY`
- execution assumption: signal after session close, enter next session open

## Universe Definition

The first operational universe is a liquidity-filtered common-stock universe,
not a benchmark constituent universe.

Inclusion intent:

- listed common stocks on `XNAS`, `XNYS`, or `XASE`
- at least 120 historical daily bars
- close price at least `5 USD`
- trailing 60-session median dollar volume at least `10M USD`

Exclusions:

- ETFs
- ETNs
- ADRs
- preferred shares
- warrants
- rights
- units
- SPAC units or shell-style instruments
- OTC listings

The reason for this choice is practical: the universe can be reconstructed from
our own internal point-in-time data without needing paid constituent history on
day one.

## Label Definition

The first label is:

- `next_5_session_excess_return_from_next_open`

Definition:

- features are built using information available at session `T` close
- the strategy enters at `T+1` open
- the strategy exits at `T+6` open
- label is the stock return over that window minus `SPY` return over the same
  window

This keeps the research label aligned with the intended execution timing.

## Prediction Contract

The alpha layer outputs:

- symbol
- score
- confidence
- side
- horizon
- timestamp

It does not output:

- share count
- portfolio weight
- order type

## Evaluation Contract

The first validation setup is:

- rolling walk-forward
- train: 36 months
- validation: 6 months
- test: 6 months
- roll frequency: monthly
- purge window: 6 sessions around split boundaries

## What Is Explicitly Deferred

- intraday modeling
- short selling
- options or derivatives
- direct reinforcement-learning order policies
- alternative data beyond basic announcements
- live broker routing before paper stability
