# FMF Test Lockbox Protocol

Date: 2026-04-09

## Purpose

This note freezes a hard rule for the rebuilt `FMF` validation architecture:

- validation window: `2013-08-01 ~ 2019-12-31`
- final test lockbox: `2020-01-02 ~ 2026-04-08`

The purpose is to stop routine research work from repeatedly exposing the final
test window and thereby degrading the credibility of the eventual product-level
evaluation.

## Why The Lockbox Matters

Each time test-window results are inspected and then allowed to influence:

- which ETF universe we keep
- which risk-budget region we search next
- which rebalance rule we keep
- which candidate survives

the test window becomes less independent.

The damage is not necessarily catastrophic after a single exposure, but repeated
inspection steadily converts the test set into another research sample.

## Rule

From this point forward:

- routine baseline reruns must use only the validation window
- parameter search must use only the validation window
- candidate narrowing must use only the validation window
- test-window metrics must remain hidden unless we explicitly declare a formal
  evaluation round

## Runner Policy

The rebuilt `FMF` baseline runner should therefore behave as follows:

- default mode: validation only
- explicit unlock required: test window

In practice this means:

- normal runs should produce validation metrics and validation artifacts only
- the runner may still record the frozen test-window boundaries in metadata
- the runner must not emit test-window summary metrics unless a dedicated
  explicit switch is passed

## Allowed Uses Of The Test Window

The test window may be opened only for:

- a formal baseline checkpoint
- a formal final candidate comparison
- a formal promotion / keep / kill review

When that happens, the evaluation round must be explicitly named and recorded.

## What This Changes Operationally

The rebuilt `FMF` workflow now has two phases:

1. Validation research phase

   - run baselines on `2013-08-01 ~ 2019-12-31`
   - sweep parameter regions only on validation
   - shortlist candidates without reopening the lockbox

2. Formal evaluation phase

   - explicitly unlock `2020-01-02 ~ 2026-04-08`
   - run the frozen shortlist once
   - record the result as a named evaluation round

## Working Interpretation

This rule is intentionally strict.

The rebuilt split is only valuable if the test window stays substantially more
independent than the old `2018 ~ 2026` research process. A locked test window is
therefore not a convenience preference; it is the core mechanism that makes the
rebuild more credible.
