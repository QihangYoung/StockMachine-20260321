# Post-Hoc Hypothesis Discipline Memo

## Goal

Freeze one explicit rule for what we may and may not do after seeing
backtest results.

This memo exists to prevent the research loop from quietly turning:

- sample evidence -> hypothesis generation

into:

- sample evidence -> immediate strategy conclusion

## Core Principle

After observing a backtest result, we may update:

- which questions are worth testing next
- which parameter regions deserve attention
- which failure modes seem most likely

But we may not immediately update:

- the official strategy definition
- the final preferred parameter point
- the promotion conclusion

without a new validation step that was not already used to create the idea.

## What Counts As Legitimate Post-Hoc Use

The following are allowed:

- using a weak result to diagnose where the current design is likely too
  defensive or too concentrated
- using a strong result to identify which sleeve may deserve a stress test
- changing experiment priority after observing where the current candidate
  underperformed
- defining the next sweep region after a baseline run

Examples:

- `C2 v0` appears too duration-heavy relative to realized outcomes, so the next
  sweep should test lower duration ranges
- proxy-era CTA behavior looks noisy, so the next pass should split proxy and
  live eras
- a candidate only wins because of one narrow year, so the next pass should
  emphasize year-by-year stability

These are all forms of:

- post-hoc hypothesis generation

They are acceptable.

## What Counts As Illegitimate Post-Hoc Use

The following are not allowed:

- promoting a new candidate because the same sample suggested and validated it
- freezing a new default weight vector directly from the best in-sample point
- declaring a parameter direction to be true rather than merely worth testing
- narrowing the search repeatedly on the same sample and then treating the
  final point as robust evidence

Examples:

- `duration down / CTA up / GLDM up` looked better in the current sample, so we
  now declare that this is the correct strategic budget
- `C2` underperformed the current `C`, so we directly move the official
  `C2` policy toward the current `C` shape without a fresh validation stage

These are all forms of:

- post-hoc strategy selection

They are not acceptable.

## Current Repository Application

### What We Observed

In the current long-window proxy-chain experiment:

- `current_c_policy_core` outperformed both rolling candidates on Sharpe
- `rolling_c2_v0_core` delivered lower volatility and shallower drawdown
- the rolling candidates appear structurally more duration-heavy than the
  current `C`

### What We Are Allowed To Conclude

We are allowed to say:

- `C2 v0` is a valid baseline but currently too defensive
- the next experiment should prioritize testing lower duration exposure and
  stronger crisis-diversifier budgets
- current `C` should remain as the benchmark to beat

### What We Are Not Allowed To Conclude

We are not allowed to say:

- the correct `C2` design is now known
- higher `CTA/GLDM` budget is confirmed
- duration should definitely be reduced in the final strategy

Those would require a new validation stage.

## Required Discipline For The Next C2 Sweep

The next sweep must obey all of the following:

1. Treat `duration down / crisis diversifier up` as a hypothesis region, not a
   conclusion.
2. Evaluate parameter regions, not only one best point.
3. Compare against an apples-to-apples baseline:
   `current C` with explicit periodic rebalance and `current C` with buy-and-drift.
4. Separate proxy-era and live-era interpretation where feasible.
5. Do not promote any candidate solely from the same sample used to define the
   region.

## Language Discipline

When discussing a result that comes from the current sample, use language such
as:

- "suggests"
- "points to"
- "is worth testing"
- "raises the hypothesis that"

Do not use language such as:

- "proves"
- "shows that the correct policy is"
- "confirms the final allocation should be"

unless a fresh validation step has been completed.

## Practical Rule

If a statement begins with:

- "because this sample showed..."

then the safe default is:

- it can justify the next experiment
- it cannot justify the next official strategy

This is the main working rule we should use going forward.
