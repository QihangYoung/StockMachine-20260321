## Strict Framework Reuse Audit

Date: 2026-04-02

### Scope

This note records the first `h5/h1` strict-framework abstraction pass and the
remaining infrastructure reuse gaps discovered while wiring `us_equities_h1`
into the existing strict research suite.

### What Is Now Reused

- Shared strict bundle construction in
  `src/stockmachine/research/p1_rigor.py`
  now resolves a framework by `strategy_project + horizon` instead of being
  hard-coded to the `h5` baseline pipeline.
- Bundle cache and prediction cache signatures are now framework-aware via
  `src/stockmachine/research/strict_frameworks.py`.
- Research-frame construction now dispatches through the framework contract:
  - `h5` -> `build_research_frame(...)`
  - `h1` -> `build_h1_research_frame(...)`
- Prediction generation now dispatches through the framework contract:
  - `h5` -> `generate_walk_forward_predictions(...)`
  - `h1` -> `generate_h1_walk_forward_predictions(...)`
- Backtest-engine construction now dispatches through the framework contract:
  - `h5` -> `DailyOpenHoldBacktestEngine`
  - `h1` -> `DailyRebalanceOpenHoldBacktestEngine`
- `run_p1_rigor_suite` now passes `strategy_project` into strict bundle
  construction, so project-scoped output/cache paths can stay aligned with the
  actual protocol family being evaluated.

### Key Design Decision

`h1` and `h5` are now treated as different strict frameworks, but they still
share the same current silver research universe:

- `us_equities_research_v1`

This is intentional. The strategy project boundary is now independent from the
underlying silver universe name. That keeps strict evaluation aligned with the
actual shared silver inputs we have today.

### Real-Data Smoke Result

The new abstraction passes unit tests, but a real strict `h1` smoke run on the
current shared silver snapshot still fails before training starts.

Observed failure:

- strict bundle coverage requires explicit `universe_membership`
  for every research session date
- current shared silver has price data through `2026-03-31`
- `universe_membership` still stops earlier, leaving missing sessions on the
  latest tail

This is not a framework dispatch bug. It is a shared data-freshness / coverage
governance gap.

### Infrastructure That Still Needs Reuse

#### 1. Research freshness and coverage preflight

Paper runtime already has preflight logic that:

- refreshes silver incrementally
- checks freshness before continuing
- fails loudly when required tables lag behind

Strict research still lacks an equivalent reusable preflight layer.

Current consequence:

- strict runs fail late inside bundle construction instead of surfacing a clean
  "refresh / coverage gap" operator message up front

Recommended next step:

- extract a shared `research preflight` helper that validates
  `daily_bar`, `benchmark_index`, `adj_factor`, `symbol_master`, and
  `universe_membership` coverage before either paper or strict research starts

#### 2. H1 baseline cache reuse

`p1_rigor` now owns reusable bundle/prediction caching for strict research, but
`src/stockmachine/research/h1_us_equities.py` still rebuilds:

- dataset load
- price panel
- point-in-time metadata
- research frame
- walk-forward predictions

on every baseline run.

Recommended next step:

- make `run_us_equities_h1_baseline` consume the strict bundle/prediction cache
  path instead of maintaining a parallel one-off baseline build path

#### 3. Shared strict summary/report contract

`p1_rigor` and `h1` baseline both emit:

- predictions
- research frame
- backtest records
- summary metrics
- yearly stability / cost stress style tables

but the report-writing code paths are still duplicated.

Recommended next step:

- extract a shared strict report writer for
  summary, yearly stability, cost stress, and protocol metadata

#### 4. Shared protocol-to-engine config bridge

The new framework layer chooses the engine family correctly, but
framework-specific knobs are still split across:

- `strict_frameworks.py`
- `h1_us_equities.py`
- `us_equities_baseline.py`

Recommended next step:

- move reusable engine/policy defaults into one shared framework-config object
  so baseline apps and strict apps do not drift

#### 5. Shared benchmark / promotion-gate logic

`h1` baseline has its own:

- benchmark summary
- promotion gate

while strict research keeps evaluation concerns in `p1_rigor`.

Recommended next step:

- decide whether promotion gates are strategy-project infrastructure
  or app-local analysis
- if they are infrastructure, move them under the strict framework layer

### Suggested Order

1. Add shared research freshness / coverage preflight
2. Repoint `h1` baseline to strict bundle/prediction cache reuse
3. Extract shared strict report writers
4. Consolidate framework defaults and gates

### Status

- Framework dispatch abstraction: done
- Unit coverage for h1/h5 strict dispatch: done
- Real-data strict h1 smoke: blocked by shared silver coverage freshness gap
