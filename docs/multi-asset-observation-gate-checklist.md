# Multi-Asset Observation Gate Checklist

## Purpose

This checklist is the default gate for deciding whether a multi-asset
allocation strategy should move from:

- promising research candidate

to:

- robust enough for controlled live observation

This checklist is only for multi-asset allocation strategies.

It excludes:

- stock alpha sleeves
- single-name selection models
- generic paper-ops or broker readiness items that do not change allocation
  quality

## Decision States

- `Research candidate`: the idea is interesting but not yet robust
- `Advance for observation`: robust enough to monitor in a tightly controlled
  live or paper observation phase
- `Reject or redesign`: too fragile to justify continued promotion in its
  current form

## Hard Red Flags

Any one of the following should block promotion until resolved:

- the strategy relies on one narrow regime or one short return burst for most
  of its excess return
- performance collapses under modest cost stress
- small parameter or weight changes materially break the result
- the risk contribution structure is dominated by one unintended bucket
- key conclusions depend on proxy history in a way that is not clearly
  understood

## Gate 1: Economic Logic

Pass only if all items are true:

- [ ] Every bucket has a clear economic role such as growth beta, duration
      hedge, credit carry, inflation hedge, trend, or cash reserve.
- [ ] No sleeve is included only because it improved historical Sharpe.
- [ ] The strategy thesis can be explained without referencing one specific
      backtest period.
- [ ] The strategy has a clear reason it should work across multiple macro
      environments rather than only one environment.
- [ ] The final bucket map and ETF mapping are explicitly frozen and documented.

Required evidence:

- bucket definitions
- ETF mapping
- one-paragraph rationale for each bucket

## Gate 2: Cross-Regime And Holdout Robustness

Pass only if all items are true:

- [ ] The strategy was evaluated across materially different historical
      environments, including at least one equity drawdown, one recovery
      phase, and one inflation or rate-shock phase.
- [ ] A frozen version of the strategy was checked on a holdout period that was
      not used for weight or rule selection.
- [ ] No single regime or calendar year appears to dominate the total excess
      return story.
- [ ] Weak periods are understandable and do not invalidate the economic logic.
- [ ] The strategy remains directionally acceptable when the sample is broken
      into subperiods.

Default review trigger:

- investigate carefully if more than `50%` of cumulative excess return comes
  from one regime block or one calendar year

Required evidence:

- subperiod summary table
- yearly summary
- holdout summary

## Gate 3: Parameter And Structure Robustness

Pass only if all items are true:

- [ ] Small weight changes around the chosen allocation do not cause a sharp
      collapse in Sharpe or drawdown behavior.
- [ ] Nearby rebalance cadences remain directionally sensible.
- [ ] Reasonable lookback or threshold changes do not destroy the strategy.
- [ ] The selected version is not just one isolated best point in a fragile
      local grid.
- [ ] Simpler neighboring variants remain competitive enough that the chosen
      version is believable.

Default review trigger:

- investigate carefully if the chosen configuration materially outperforms most
  neighboring variants by a margin that looks too sharp to be economically
  credible

Required evidence:

- local weight grid
- cadence sweep
- sensitivity summary

## Gate 4: Cost And Implementation Robustness

Pass only if all items are true:

- [ ] The strategy remains acceptable under at least modestly harsher cost
      assumptions than the base case.
- [ ] Turnover is consistent with the intended implementation style.
- [ ] All key sleeves are represented by liquid, realistically tradable
      instruments.
- [ ] Proxy replacement checks do not overturn the main conclusion.
- [ ] The strategy does not require unrealistic timing, stale prices, or
      unreasonably fast reallocations.

Default review trigger:

- investigate carefully if doubling the baseline cost assumption erases most of
  the strategy's risk-adjusted edge

Required evidence:

- cost-stress table
- turnover summary
- proxy replacement or proxy dependence note

## Gate 5: Risk Structure Robustness

Pass only if all items are true:

- [ ] Bucket risk contributions broadly align with the intended design.
- [ ] No unintended bucket dominates total portfolio risk.
- [ ] Crisis-diversifier sleeves actually help in the stress periods they are
      supposed to help.
- [ ] The strategy still makes sense when correlations move against the base
      case.
- [ ] Drawdown, downside capture, and concentration metrics are acceptable for
      the intended use.

Default review trigger:

- investigate carefully if one bucket contributes more than `50%` of total
  portfolio risk without that concentration being an explicit design choice

Required evidence:

- bucket risk contribution table
- drawdown decomposition
- stress-period sleeve contribution review

## Promotion Rule

A strategy is eligible to advance for controlled observation only if:

- all `5` gates pass
- no hard red flag is open
- the research owner can explain the strategy in plain language
- the current version is frozen before observation begins

If only `4` of `5` gates pass:

- do not promote yet
- keep the strategy in research
- resolve the failed gate first

If `3` or fewer gates pass:

- redesign or reject the current form

## Observation Means

Promotion through this checklist does **not** mean:

- production trust
- final approval
- unlimited capital confidence

It means only:

- strong enough to deserve tightly controlled observation with explicit
  monitoring

## Review Record Template

Use the following template when evaluating a candidate:

```text
Strategy:
Version:
Reviewer:
Date:

Gate 1 Economic Logic: PASS / FAIL
Gate 2 Cross-Regime And Holdout Robustness: PASS / FAIL
Gate 3 Parameter And Structure Robustness: PASS / FAIL
Gate 4 Cost And Implementation Robustness: PASS / FAIL
Gate 5 Risk Structure Robustness: PASS / FAIL

Hard Red Flags:
- none / list

Promotion Decision:
- Research candidate
- Advance for observation
- Reject or redesign

Most Important Reason:

Next Required Work:
- item 1
- item 2
- item 3
```

## Recommended Usage

This checklist should be used together with:

- the strategy memo
- the robustness-window memo
- the current backtest artifact pack

The main point is not to eliminate judgment.

The main point is to prevent promotion decisions from being made only because a
single backtest line looks attractive.
