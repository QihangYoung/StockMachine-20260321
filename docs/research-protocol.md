# Research Protocol

This document freezes the shared contract for the P0 research-rigor hardening pass.

It complements:

- [evaluation-protocol.md](./evaluation-protocol.md), which describes validation gates
- [experiment-manifest.md](./experiment-manifest.md), which describes replay and audit fields

## Goal

Define one canonical protocol for:

- timing assumptions
- point-in-time universe resolution
- walk-forward splitting
- purge and embargo behavior
- minimum output fields

This contract exists so every rerun, benchmark, and model comparison uses the
same assumptions.

## Time Definitions

- `session_date`: the trading session being predicted or evaluated
- `effective_session_date`: the latest available market-data session actually used when the requested date is not available
- `as_of_date`: the latest visible metadata snapshot allowed for that session

For the first US-equities daily workflow:

- features are built using information available by session `T` close
- signals are generated after session `T` close
- entries occur at the next session open, `T+1`
- exits occur at the open of `T+6`
- holding period is `5` sessions

## Point-in-Time Universe Contract

The research layer must resolve universe membership and metadata using the
latest visible snapshot at or before the target `session_date`.

### Membership rules

- membership is session-scoped
- a symbol missing from the point-in-time membership view is excluded for that session
- the default first implementation should prefer exclusion over silently filling forward membership

### Metadata rules

- symbol metadata and industry mappings are resolved using the latest visible snapshot at or before `session_date`
- missing metadata should not silently remove a symbol if membership is known
- missing metadata should be filled conservatively with `Unknown`-style placeholders

## Walk-Forward Contract

Default validation schedule:

- train window: `36` months
- validation window: `6` months
- test window: `6` months
- roll frequency: `monthly`

The train, validation, and test windows are defined in time order and must
never be shuffled.

## Purge And Embargo Contract

Default first-line settings:

- purge window: `6` sessions
- embargo window: `1` session

Meaning:

- purge removes training observations whose label horizon overlaps the validation or test boundary
- embargo adds a short buffer after each held-out block before training resumes

These defaults are intentionally conservative for the current 5-session label horizon.

## Required Output Fields

Minimum prediction fields:

- `date`
- `symbol`
- `score`
- `confidence`
- `target`
- `future_return`
- `benchmark_future_return`
- `model`

Minimum backtest summary fields:

- `model`
- `sessions`
- `total_return`
- `annualized_return`
- `annualized_volatility`
- `sharpe`
- `max_drawdown`
- `benchmark_total_return`
- `mean_turnover`
- `mean_cost_bps`

## Code Anchor

The code version of this contract lives in:

- [protocols.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/protocols.py)

Any splitter, universe helper, or rerun script added during P0 should align to that module.
