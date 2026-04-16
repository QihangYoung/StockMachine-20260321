# Multi-Asset Bucket And Covariance Spec

Date: 2026-04-08

## Scope

This document freezes the default bucket definition and covariance-estimation
recipe for the first `C 2.0` multi-asset core rebuild.

It applies to:

- `ERC` core research
- constrained risk-budget core research
- threshold-aware rebalance research

It does not cover:

- stock alpha sleeves
- single-name selection
- broker execution logic

## Design Goal

Build a bucket-level risk model that is:

- simple enough to audit
- stable enough for strategic allocation
- responsive enough to adapt when correlations change

The intended use is not to forecast returns.

The intended use is to estimate:

- bucket volatility
- bucket correlation
- bucket covariance
- bucket risk contribution

## Default Bucket Set

The default first-line bucket set is:

- `equity_us`
- `equity_ex_us`
- `duration`
- `credit`
- `inflation_hedge`
- `trend`
- `cash`

## Bucket Roles

- `equity_us`: primary US growth beta
- `equity_ex_us`: non-US equity diversification
- `duration`: recession and rate-cut hedge
- `credit`: spread carry and moderate-risk income sleeve
- `inflation_hedge`: inflation and fiat-risk hedge
- `trend`: crisis-diversifier and macro-trend sleeve
- `cash`: liquidity reserve and rebalance anchor

## Default ETF Mapping

### Primary representatives

- `equity_us` -> `SPY`
- `equity_ex_us` -> `VXUS`
- `duration` -> `IEF`
- `credit` -> `LQD`
- `inflation_hedge` -> `GLDM`
- `trend` -> `CTA`
- `cash` -> `SGOV`

### Secondary representatives for robustness tests

- `equity_us` -> `QQQ`, `IWM`
- `equity_ex_us` -> `VEA`, `VWO`
- `duration` -> `TLT`
- `credit` -> `HYG`
- `inflation_hedge` -> `GLD`, `DBC`
- `trend` -> `DBMF`, `KMLM`
- `cash` -> `BIL`

## Mapping Rules

The first research pass should follow these rules:

- use one primary ETF per bucket in the mainline run
- use secondary ETFs only for replacement checks or robustness checks
- do not mix multiple ETFs inside one bucket in the mainline covariance estimate
  until the single-representative version is understood
- if a representative has limited live history, document the proxy chain
  explicitly instead of silently filling history

## Return Construction

The default bucket return series should be built from:

- adjusted close-to-close daily total returns

Reason:

- this is the simplest stable return series for strategic covariance work
- it avoids mixing intraday execution assumptions into the bucket-risk model

### Bucket return definitions

- `equity_us_ret_t = SPY adjusted daily return`
- `equity_ex_us_ret_t = VXUS adjusted daily return`
- `duration_ret_t = IEF adjusted daily return`
- `credit_ret_t = LQD adjusted daily return`
- `inflation_hedge_ret_t = GLDM adjusted daily return`
- `trend_ret_t = CTA adjusted daily return`
- `cash_ret_t = SGOV adjusted daily return`

## History Policy

History should be split into two layers:

- `actual history`
- `proxy-chain history`

Main rule:

- always know which part of a bucket series is actual ETF data and which part is
  reconstructed

The research reports should show both:

- metrics including the proxy-extended history
- metrics using only the fully live ETF overlap window

## Calendar Alignment

Covariance estimation should use a common session calendar.

Alignment rules:

- use the intersection of available session dates for the chosen bucket set
- do not silently forward-fill missing bucket returns
- if a bucket is unavailable for a session, drop that session from the aligned
  matrix

This is intentionally conservative.

## Covariance Estimation Recipe

The default covariance estimate should be a blend of:

- a slower long-horizon covariance matrix
- a faster short-horizon covariance matrix

This avoids two common failures:

- a pure long window that reacts too slowly
- a pure short window that is too noisy

### Long-horizon covariance

Default:

- rolling sample covariance using the last `252` daily observations

Role:

- provide stability
- reflect medium-term structural relationships

### Short-horizon covariance

Default:

- `EWMA` covariance using the last `63` daily observations
- daily decay factor `lambda = 0.97`

Role:

- react to recent changes in volatility and correlation

### Blended covariance

Default:

- `Sigma_blend = 0.70 * Sigma_long + 0.30 * Sigma_short`

Role:

- anchor on stable long-run structure
- still respond when market relationships change

## Why These Default Parameters

### `252`-day long window

- about one trading year
- long enough to reduce noise
- short enough to avoid becoming purely stale for strategic allocation work

### `63`-day short window

- about one quarter
- responsive enough for changing market structure
- still more stable than very short windows such as `21` days

### `lambda = 0.97`

- gives recent observations more weight
- still retains enough memory for medium-frequency multi-asset work
- more stable than a very fast decay such as `0.94` in this use case

### `70 / 30` blend

- keeps the long matrix dominant
- allows recent vol and correlation shifts to matter
- is simple and easy to explain

## Parameter Grid For Robustness Tests

The first implementation should support this sensitivity grid:

### Long window

- `126`
- `252`
- `504`

### Short window

- `21`
- `63`
- `126`

### EWMA decay

- `0.94`
- `0.97`
- `0.99`

### Blend weights

- `80/20`
- `70/30`
- `60/40`

## Matrix Hygiene

The covariance matrix should be cleaned before use.

Required steps:

- ensure symmetry
- ensure positive semidefinite behavior or apply a nearest-PSD repair if needed
- report realized volatilities and correlations separately for debugging

Optional future upgrade:

- apply shrinkage toward a structured long-run target matrix

This should be deferred until the basic blended matrix is understood.

## Estimation Frequency

The covariance matrix should be recomputed:

- at each rebalance decision date

That means:

- for `5`-session strategies, update the matrix on each `5`-session rebalance
- for `21`-session strategies, update the matrix on each monthly-style rebalance

Do not recompute weights daily unless the strategy itself is daily-reactive.

## Risk Outputs To Produce

For every rebalance date, the risk model should produce:

- bucket volatility estimate
- bucket correlation matrix
- bucket covariance matrix
- portfolio volatility estimate
- marginal risk contribution by bucket
- total risk contribution by bucket
- risk-share percentage by bucket

## Default Usage By Strategy Family

### `ERC` core

Use:

- the blended covariance matrix
- equal target risk shares across all non-cash strategic buckets by default

Default first pass:

- equalize risk across:
  - `equity_us`
  - `equity_ex_us`
  - `duration`
  - `credit`
  - `inflation_hedge`
  - `trend`
- treat `cash` as optional reserve capital, not a sleeve that must carry equal
  risk

### Constrained risk-budget core

Use:

- the same blended covariance matrix
- target risk-share ranges rather than exact equality

Example first-pass ranges:

- `equity_us + equity_ex_us`: `30%` to `40%`
- `duration + cash`: `20%` to `30%`
- `credit`: `5%` to `15%`
- `inflation_hedge + trend`: `30%` to `40%`

## Stress Checks

The covariance process should be explicitly tested under:

- equity crash windows
- inflation shock windows
- bond-equity positive-correlation windows
- trend-diversifier weak periods

The purpose is not to predict future covariance perfectly.

The purpose is to verify that the estimation recipe does not become obviously
misleading in the regimes that matter most.

## Recommended First Implementation

For the first production-quality research pass, use:

- buckets: `equity_us`, `equity_ex_us`, `duration`, `credit`,
  `inflation_hedge`, `trend`, `cash`
- primary ETFs: `SPY`, `VXUS`, `IEF`, `LQD`, `GLDM`, `CTA`, `SGOV`
- return frequency: daily adjusted total return
- long covariance window: `252`
- short covariance window: `63`
- EWMA lambda: `0.97`
- blend: `70%` long, `30%` short

This should be the default configuration used to build the first `ERC` core.

## Recommended Next Step

After the first `ERC` run is working, the next sequence should be:

1. test bond-bucket split robustness
2. test proxy replacement sensitivity
3. move from `ERC` to constrained risk-budget targets
4. add threshold-aware rebalance logic
