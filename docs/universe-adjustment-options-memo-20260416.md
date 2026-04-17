# Universe Adjustment Options Memo

Date: 2026-04-16

This memo records why the rebuilt `FMF` validation universe looks weak, which replacement directions are plausible, and why we are temporarily shelving the universe-change branch.

All evidence referenced here is validation-only. The lockbox test window remains closed.

## Current Rebuilt Universe

The rebuilt validation universe is:

`SPY / VXUS / IEF / LQD / GLD / FMF / BIL`

Bucket interpretation:

- `SPY`: U.S. equity growth
- `VXUS`: ex-U.S. equity
- `IEF`: duration
- `LQD`: investment-grade credit
- `GLD`: gold / inflation hedge
- `FMF`: managed futures / trend
- `BIL`: cash

## Why The Universe Looks Weak

Validation-window single-ETF evidence shows that the universe is not evenly strong.

The strongest component is `LQD`. `SPY` is a usable broad equity benchmark, but it is not the strongest growth engine. `VXUS`, `GLD`, and especially `FMF` are weak in this validation sample.

Summary from `artifacts/single_etf_validation_performance_20260415.csv`:

| ETF | Role | Annual Return | Annual Vol | Sharpe | Working Read |
|---|---|---:|---:|---:|---|
| `BIL` | cash | `0.67%` | `0.26%` | `2.610` | cash sleeve, not a return engine |
| `LQD` | credit | `5.37%` | `4.85%` | `1.108` | strong defensive-return asset |
| `SPY` | broad U.S. equity | `12.65%` | `12.91%` | `0.980` | acceptable benchmark, not the best growth engine |
| `IEF` | duration | `3.27%` | `5.22%` | `0.627` | moderate defensive asset |
| `VXUS` | ex-U.S. equity | `5.23%` | `13.97%` | `0.374` | weak growth engine in validation |
| `GLD` | gold | `1.94%` | `13.51%` | `0.144` | hedge, not a stable return engine |
| `FMF` | trend | `-1.51%` | `13.58%` | `-0.111` | main drag in this universe |

The key issue is not merely parameter tuning. The universe itself gives the optimizer too few high-quality return sources.

## Replacement Directions

### Growth Engine

`SPY` should remain a benchmark, but not the only growth engine.

Candidate replacements or complements:

- `VGT`
- `XLK`
- `QQQ`
- `IGV`
- `SOXX`
- `SMH`

Validation evidence suggests that `VGT`, `XLK`, `SOXX`, `SMH`, and `IGV` are all materially stronger growth engines than `SPY` in this sample. The trade-off is style concentration, especially technology and semiconductor exposure.

### Credit / Cash

`LQD` and `BIL` remain useful.

- `LQD` is one of the strongest assets in the current validation sample.
- `BIL` remains the correct residual cash sleeve.

There is no immediate need to replace them.

### Ex-U.S. Equity

`VXUS` is weak in this validation sample.

Near-term options:

- remove from the core universe
- keep only as a benchmark / diversification check
- reintroduce later with a separate thesis

It should not be treated as a required growth engine in the next branch.

### Gold

`GLD` is weak across the full validation window, mostly because `2013 ~ 2015` was a poor gold environment.

However, `GLD` and `GLDM` are nearly identical in their overlapping validation window. The issue is market regime, not wrapper quality.

Working role:

- keep gold as a hedge candidate
- do not treat it as a primary return engine
- cap its role if included

### Trend

`FMF` is the most problematic replacement in the rebuilt universe.

It solves the long-history requirement but performs poorly in validation. It should not remain a required core sleeve unless a separate trend-sleeve validation supports it.

Candidate trend alternatives:

- `WTMF`: longer history, but has a 2021 strategy break
- `DBMF`: stronger modern candidate, but shorter live history
- `KMLM`: rules-based, but shorter live history
- `CTA`: legacy current-C representative, but too young for the rebuilt validation window

Working choice:

- remove `trend` from the main core for now
- treat trend as a separate optional sleeve research problem

## Plausible New Universes

### High-Sharpe Growth-Credit Core

Candidate universe:

`VGT / LQD / BIL`

Alternatives:

- `XLK / LQD / BIL`
- `SOXX / LQD / BIL`
- `IGV / LQD / BIL`

This is the cleanest direction for a high-Sharpe strategy with possible capped leverage.

### Growth-Credit-Hedge Core

Candidate universe:

`VGT / LQD / IEF / GLD / BIL`

This preserves some multi-asset structure but removes the weakest required sleeves, `VXUS` and `FMF`.

### Benchmark Universe

Candidate universe:

`SPY / LQD / BIL`

This should remain a benchmark, not the main candidate if the goal is stronger growth.

## Why We Are Shelving This Branch For Now

The universe-change argument is strong, but changing the universe now would open a large new search space:

- growth-engine choice
- hedge inclusion
- trend inclusion
- leverage policy
- dynamic sleeve rules

That is too much to optimize at once.

For now, we will keep the universe-change branch as a recorded hypothesis and temporarily shelve it. The next step is to apply the dynamic-sleeve idea to the traditional 7-bucket universe first, so we can isolate whether the dynamic sleeve mechanism itself helps before fully changing the universe.

## Working Decision

Shelve universe replacement as a separate branch.

Continue with:

- existing rebuilt 7-bucket validation universe
- validation-only experiments
- dynamic sleeve applied to the traditional multi-asset bucket set

Do not open the lockbox test window.

