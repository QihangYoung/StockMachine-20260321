# FMF Validation Rebuild Memo

Date: 2026-04-09

## Purpose

This memo freezes the current decision to rebuild the multi-asset validation and
evaluation architecture around an ETF-only universe that can support a longer
clean live-only window than the current `CTA`-based setup.

The immediate goal is not to preserve legacy comparability. The goal is to
maximize the reliability of validation and test evidence by expanding the common
live-only sample and by giving both validation and test windows meaningful
regime coverage.

## Why Rebuild

The current multi-asset research stack was built around:

- `current C`: `SPY / VXUS / AGG / CTA / GLDM`
- `C2 research`: `SPY / VXUS / IEF / LQD / GLDM / CTA / SGOV`

This produced a useful first research cycle, but it is not a strong final
validation architecture because:

- `CTA` is the clean live-only bottleneck. Its official inception is
  `2022-03-07`, which forces a very short common live-only sample.
- The longer-window workaround relied on proxy chains. That is acceptable for
  structural research, but not ideal for product-level validation.
- Our own proxy diagnostics showed that the `CTA` proxy chain was noisy, so we
  do not want the rebuilt validation system to depend on it.
- With the current universe, it is difficult to carve out both a meaningful
  validation window and a meaningful untouched test window while still covering
  enough macro regimes.

Therefore the right next step is to rebuild the dataset and validation protocol
 around a longer-history ETF representative for the `trend` bucket.

## Rebuild Universe

The rebuilt ETF universe is:

- `equity_us` -> `SPY`
- `equity_ex_us` -> `VXUS`
- `duration` -> `IEF`
- `credit` -> `LQD`
- `inflation_hedge` -> `GLD`
- `trend` -> `FMF`
- `cash` -> `BIL`

This is a deliberate break from the current `GLDM / CTA / SGOV` wrappers.
The point is to maximize common live-only history for validation and testing.

## Official Earliest Live Dates

The key issuer-reported first dates for the rebuilt universe are:

| ETF | Role | Earliest official live date |
|---|---|---:|
| `SPY` | equity_us | `1993-01-22` |
| `VXUS` | equity_ex_us | `2011-01-26` |
| `IEF` | duration | `2002-07-22` |
| `LQD` | credit | `2002-07-22` |
| `GLD` | inflation_hedge | `2004-11-18` |
| `BIL` | cash | `2007-05-25` |
| `FMF` | trend | `2013-08-01` |

Under this universe, the theoretical maximum common live-only window is:

- `2013-08-01` to `2026-04-08`

Important local note:

- the current local silver coverage still starts `SPY` at `2014-01-02`, so the
  rebuild requires extending local price history and adding the new symbols
  before rerunning the full experiment stack.

## Split Design Goals

The rebuilt split should satisfy four goals at the same time:

- maintain strict time order
- leave a large enough validation sample for disciplined retuning
- leave a large enough untouched test sample for final evaluation
- let the test window contain the most decision-critical modern regimes

For a month-frequency multi-asset process, rebalance opportunities matter at
least as much as calendar length. Using the full `2013-08-01 ~ 2026-04-08`
session calendar, a few candidate splits are:

| Split | Validation sessions | Validation rebalances (~21d) | Test sessions | Test rebalances (~21d) |
|---|---:|---:|---:|---:|
| strict half by sessions: `2019-11-29 / 2019-12-02` | `1595` | `75` | `1595` | `75` |
| regime-aligned: `2019-12-31 / 2020-01-02` | `1616` | `76` | `1574` | `74` |
| late split: `2020-02-18 / 2020-02-19` | `1648` | `78` | `1542` | `73` |
| post-crash split: `2020-03-31 / 2020-04-01` | `1678` | `79` | `1512` | `72` |

## Regime Coverage Assessment

### Validation: `2013-08-01 ~ 2019-12-31`

This validation window covers:

- long post-GFC bull market carry environment
- 2015-2016 growth scare / commodity weakness
- 2018 Q4 risk-off drawdown
- 2019 recovery and late-cycle environment

What it does not cover:

- the 2020 liquidity shock
- the 2022 inflation-driven bear market

That is acceptable because the purpose of validation is parameter selection, not
final proof of crisis robustness.

### Test: `2020-01-02 ~ 2026-04-08`

This test window covers:

- the full COVID crash and acute liquidity shock
- the 2020-2021 post-crash policy-driven bull market
- the 2022 inflation / tightening / stock-bond drawdown regime
- the 2023-2026 recovery, consolidation, and mixed macro environment

This is exactly the kind of modern, product-relevant out-of-sample window we
want the final candidate to survive.

## Recommended Split

Recommendation:

- validation: `2013-08-01 ~ 2019-12-31`
- test: `2020-01-02 ~ 2026-04-08`

Why this is better than a pure mechanical half split:

- it is still almost perfectly balanced in sample size
- it gives the test window a clean and interpretable start
- it keeps the full 2020 crash inside the test window rather than starting the
  test in the middle of the crisis
- it preserves both 2022 inflation stress and 2023-2026 post-stress behavior in
  the same untouched evaluation window

## Working Decision

The rebuild should proceed with:

- universe: `SPY / VXUS / IEF / LQD / GLD / FMF / BIL`
- target live-only common window: `2013-08-01 ~ 2026-04-08`
- validation window: `2013-08-01 ~ 2019-12-31`
- final test window: `2020-01-02 ~ 2026-04-08`

The next implementation step is to extend local history and add the new
representative ETF series, then rerun the multi-asset baseline and `C2`
candidate experiments on the rebuilt split.

