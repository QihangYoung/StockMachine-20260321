# Validation Rebuild Status Note

Date: 2026-04-09

## Current Position

The multi-asset validation architecture is being rebuilt around the `FMF`
universe proposal:

- `SPY`
- `VXUS`
- `IEF`
- `LQD`
- `GLD`
- `FMF`
- `BIL`

The intended maximum common live-only window is:

- `2013-08-01 ~ 2026-04-08`

The current recommended split is:

- validation: `2013-08-01 ~ 2019-12-31`
- test: `2020-01-02 ~ 2026-04-08`

## Why This Is Better Than The Previous Setup

This is not yet a perfect validation system, but it is clearly better than the
previous practical state, where the effective `2018 ~ 2026` sample had already
been consumed as a research-and-selection sample.

The new setup is an improvement for four reasons:

1. It restores a real untouched test window.

   The previous workflow relied too heavily on a single long sample that had
   already been used for hypothesis generation, candidate narrowing, and design
   updates. That made it hard to claim any clean final out-of-sample evidence.

2. It gives the test set materially better modern regime coverage.

   The new test window includes:

   - the full COVID crash and liquidity shock
   - the 2020-2021 post-crash risk-on recovery
   - the 2022 inflation / tightening / stock-bond drawdown regime
   - the 2023-2026 mixed post-inflation environment

   This is much closer to the kind of evidence we want for product-level
   evaluation.

3. It removes the current `CTA`-driven live-history bottleneck.

   Replacing the younger `CTA` sleeve with `FMF` is not a statement that `FMF`
   is universally superior. It is a validation-architecture decision intended
   to expand the common live-only window and reduce dependence on noisy proxy
   extensions.

4. It is still reasonably balanced.

   The proposed split is close to a half-and-half division in sample size while
   remaining cleaner from a regime-interpretation standpoint than a purely
   mechanical midpoint split.

## Known Imperfection

The validation and test windows are not distribution-matched.

- validation (`2013-2019`) is mostly a post-GFC expansion / growth-scare /
  `2018 Q4`-risk-off environment
- test (`2020-2026`) is a much more stressed and structurally different modern
  regime mix

That mismatch is a real risk. It means the best parameters on validation may
not be the best parameters on test.

However, this does not invalidate the split. It means we must use the split
correctly:

- validation should be used to find robust regions, not point-optimal answers
- test should be treated as the decisive modern out-of-sample window
- model complexity and parameter freedom should remain low

## Working Interpretation

The correct interpretation at this stage is:

- this rebuilt split is a meaningful step up in rigor
- it is not a perfect final architecture
- it is nevertheless preferable to continuing with a de facto all-validation
  `2018 ~ 2026` research process

## Immediate Rule

Until a stronger architecture is adopted, the project should follow these
rules:

- use the `FMF` rebuild universe as the new baseline validation universe
- use `2013-08-01 ~ 2019-12-31` for validation
- use `2020-01-02 ~ 2026-04-08` as the untouched final test window
- do not use test results to re-open parameter search without explicitly
  declaring a new research cycle

