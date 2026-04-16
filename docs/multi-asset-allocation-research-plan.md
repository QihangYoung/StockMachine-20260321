# Multi-Asset Allocation Research Plan

## Status

This document defines the recommended first research path for multi-asset
allocation inside the current repository.

It is intentionally scoped as a separate research thread, not as a forced
change to the current `us_equities_h5` mainline.

## Why This Is The Right First Shape

The repository is still single-equity-first by design, but three things are
already true:

- the `portfolio`, `risk`, `backtest`, and `paper` boundaries are reusable
- regime research already introduced cross-asset proxy handling
- the current US-equity data and broker path can support US-listed ETFs with
  much lower integration cost than futures, options, or direct macro series

Because of that, the best first definition of "multi-asset" here is:

- liquid US-listed ETFs
- daily bars
- long-only
- cash-allowed
- unlevered

This keeps the first allocation research line compatible with the current
execution and operational scaffolding.

## Recommended First Scope

- market: US-listed multi-asset ETF basket
- execution wrapper: same US-equity broker and paper path
- frequency: daily bars
- first rebalance cadence: every `5` sessions
- robustness cadence: rerun at `21` sessions after first baseline is stable
- style: long-only, unlevered, cash-allowed
- first objective: improve drawdown-adjusted performance versus simple static
  benchmarks without relying on fragile optimizer tuning

The first project should be treated as a separate strategy line, for example:

- `us_multi_asset_h5` for the first weekly-style tactical line

If a slower monthly line later proves materially different in turnover,
features, or evaluation, it should become a second project rather than a minor
parameter variant:

- `us_multi_asset_h21`

## First Universe Recommendation

The initial universe should stay intentionally small and liquid.

Suggested buckets:

- US equity beta: `SPY`, `QQQ`, `IWM`
- international equity beta: `VXUS`, `EEM`
- rates and cash: `IEF`, `TLT`, `SGOV`
- credit: `LQD`, `HYG`
- real assets: `GLD`, `DBC`

Design rules:

- prefer instruments with long history and high liquidity
- prefer one clear ETF per role before adding close substitutes
- keep the first basket small enough that every allocation change remains easy
  to explain
- freeze the universe explicitly instead of using dynamic discovery in v1

This first line should avoid:

- leveraged ETFs
- inverse ETFs
- futures
- options
- volatility ETPs
- thin commodity or country funds

## Research Questions

The first thread should answer four questions in order:

1. Can a small white-box multi-asset allocator beat simple static baselines on
   risk-adjusted terms?
2. Does the edge come from trend, relative strength, volatility scaling, or
   simply from diversification?
3. Do tactical allocation gains survive transaction-cost stress and rebalance
   cadence changes?
4. Does the existing regime infrastructure improve a tactical sleeve only after
   the detector inputs become materially stronger?

## Baseline Ladder

Research should move from simple to complex.

### 1. Static baselines

Start with:

- `SPY`
- `60/40` proxy such as `60% SPY + 40% IEF`
- equal-weight all-asset basket
- inverse-volatility or equal-risk basket with simple realized-vol scaling

These baselines are necessary so later tactical results are not mistaken for
basic diversification.

### 2. White-box tactical baselines

Add only transparent rules first:

- absolute momentum filter
- relative-strength top-N allocation
- dual-momentum style switch
- volatility-targeted allocation
- defensive or cash fallback when no risk assets pass filters

This stage should be the main research priority.

### 3. Core plus sleeve variants

Only after the tactical baselines are trustworthy:

- static core plus tactical sleeve
- tactical sleeve with vol matching
- tactical sleeve with optional regime gate

This path aligns well with the existing `run_core_sleeve_paper` design.

### 4. Model-based extensions

Model-heavy allocation should be deferred until the white-box baselines are
stable.

Examples of later-stage extensions:

- ETF return ranking models
- sleeve selection models
- forecast-based overlay on top of static or tactical core

## Data Contract

The first multi-asset line should reuse the existing data philosophy:

- raw source snapshots remain preserved
- modeling reads normalized internal tables only
- all joins remain point-in-time safe

The first required data set is small:

- ETF symbol master
- daily OHLCV bars
- adjustment factors
- trading calendar
- explicit bucket mapping for each ETF

Important constraints:

- v1 should prefer a frozen explicit ETF universe over dynamic asset discovery
- macro, credit, and rates series should not become required inputs until their
  point-in-time provenance is clean
- cross-asset proxy files created during regime studies may inform research,
  but they should not silently become production dependencies

## Research Protocol Recommendation

The first pass should maximize reuse of the existing strict framework.

That means:

- keep the same `session_date`, `effective_session_date`, and `as_of_date`
  semantics
- keep walk-forward evaluation with no shuffle
- keep purge and embargo behavior explicit
- keep cost stress and robustness reruns as promotion gates

Recommended starting point:

- reuse the current shared protocol defaults first
- freeze a line-specific protocol only if the multi-asset horizon materially
  diverges from `h5`

Minimum evaluation outputs should include:

- total return
- annualized return
- annualized volatility
- Sharpe
- max drawdown
- benchmark-relative return
- turnover
- mean cost bps

Recommended additional allocation metrics:

- downside capture versus `SPY`
- worst rolling `3`-month drawdown
- percent of time invested in cash or defensive assets
- concentration by bucket

## Portfolio Construction Direction

The current equity policy is sector-aware and top-k-oriented. That should not
be stretched into a multi-asset allocator by overloading sector fields.

The cleaner direction is:

- keep the current equity policy unchanged for `us_equities_*`
- add a separate allocation policy for multi-asset work
- express constraints by asset bucket rather than by sector

The first multi-asset policy should support:

- per-asset max weight
- per-bucket max weight
- minimum defensive allocation when all offensive assets fail filters
- optional cash allocation
- turnover cap
- realized-vol scaling as an optional overlay

This keeps the new line compatible with the repository's white-box design.

## Implementation Path

Recommended order:

1. Freeze the ETF universe and bucket map in config.
2. Add a separate strategy project id for the first multi-asset line.
3. Build a point-in-time-safe ETF dataset loader and research frame.
4. Implement static and white-box tactical baselines.
5. Add strict comparison reports versus `SPY`, `60/40`, and equal-weight
   basket baselines.
6. Add cadence sweep, cost stress, and top-N or breadth sensitivity.
7. Reuse `core + sleeve` paper plumbing only if research shows it is the best
   product shape.
8. Revisit regime gating only after detector inputs improve.

## Promotion Gates

The first multi-asset line should not advance just because it looks smoother
than equities.

Minimum promotion standard:

- beats at least one static baseline on risk-adjusted return in held-out data
- improves at least one downside metric versus `SPY`
- survives cost stress and rebalance-cadence sensitivity
- does not rely on one short regime window for most of its excess return
- remains simple enough to explain allocation changes in plain language

## Explicit Non-Goals

The first multi-asset thread should not try to do all of the following at once:

- mean-variance optimization as the mainline baseline
- direct macro forecasting
- futures portfolio construction
- leverage targeting that depends on borrowing assumptions
- short selling
- options overlays
- live deployment before paper-style operational checks exist

## Recommended Near-Term Deliverables

The first concrete output bundle should be:

- frozen multi-asset universe document
- first baseline backtest report pack
- cost-stress and cadence-sweep report
- decision memo on whether this line deserves a dedicated paper prototype

## Relationship To Existing Docs

This plan extends, but does not replace:

- [architecture.md](/E:/CodeX/StockMachine-260321/docs/architecture.md)
- [research-protocol.md](/E:/CodeX/StockMachine-260321/docs/research-protocol.md)
- [data-strategy.md](/E:/CodeX/StockMachine-260321/docs/data-strategy.md)
- [regime-research-status-20260404.md](/E:/CodeX/StockMachine-260321/docs/regime-research-status-20260404.md)

The main intent is to create a clean bridge from the current US-equities-first
system into a separate, auditable multi-asset research line.
