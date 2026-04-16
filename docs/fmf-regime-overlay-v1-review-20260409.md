# FMF Regime Overlay V1 Review

Date: 2026-04-09

## Scope

This note reviews the first conservative regime-aware overlay prototype on the
rebuilt `FMF` validation line.

Important constraint:

- the `2020-01-02 ~ 2026-04-08` lockbox test window remained closed
- this review is validation-only

## Overlay Design

The overlay was intentionally narrow:

- base policy: `rolling_fmf_c2_e42_c10_d22_i20_t06`
- only two top-level levers move:
  - total equity risk budget
  - duration risk budget
- all other sleeves stay fixed:
  - credit
  - inflation hedge
  - trend
  - reserve cash

Three coarse states were used:

- `risk_on`
  - move `+4%` risk budget from duration to total equity
- `neutral`
  - keep the static lead policy
- `defensive`
  - move `-4%` risk budget from total equity to duration

State detection used lagged benchmark:

- trailing return
- drawdown
- realized volatility

The prototype is therefore a white-box overlay, not a meta-optimizer.

## Validation Results

Validation-common-window summary:

| Strategy | Annualized return | Annualized vol | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| `rolling_fmf_c2_e42_c10_d22_i20_t06` | `4.995%` | `4.201%` | `1.181` | `-5.245%` |
| `rolling_fmf_c2_regime_overlay_v1` | `5.055%` | `4.266%` | `1.177` | `-5.446%` |
| `rolling_c2_v0_seed_core` | `4.473%` | `4.176%` | `1.069` | `-5.274%` |

Interpretation:

- the overlay slightly increased return
- but it also increased volatility and drawdown
- net result: Sharpe is marginally worse than the static lead

So the first conservative overlay did **not** produce a clear validation win.

## State Mix

The detected validation-state counts were:

- `risk_on`: `993` days
- `neutral`: `524` days
- `defensive`: `97` days

This means the overlay spent most of validation in the `risk_on` state.

That is directionally consistent with the `2014-2019` validation environment,
but it also means:

- the overlay mostly acted like a mild risk-on lever
- it did not spend enough time in defensive mode to demonstrate a strong
  drawdown-control advantage

## What This Means

### 1. The conservative idea is viable, but not yet useful

The prototype proves that a very narrow, auditable regime overlay can be
implemented without touching the lockbox or turning the system into a
high-dimensional search problem.

That is a useful engineering result.

### 2. The current state definition is not yet adding enough value

The first state map did not improve the risk-adjusted result enough to justify
promotion over the static lead candidate.

This suggests one of two things:

- either the static lead is already close to the best validation compromise
- or the current state variables and thresholds are too blunt to improve it

### 3. This result supports caution

The first overlay did not fail badly, but it also did not clearly beat the
static lead.

That supports the earlier judgment:

- regime-aware adaptation is worth exploring
- but it should not be assumed to help automatically
- and it should not justify a rapid expansion into a larger dynamic search space

## Current Recommendation

Keep the status hierarchy unchanged:

- primary static lead:
  - `rolling_fmf_c2_e42_c10_d22_i20_t06`
- near-neighbor:
  - `rolling_fmf_c2_e42_c10_d22_i18_t08`
- conservative anchor:
  - `rolling_c2_v0_seed_core`

Treat `rolling_fmf_c2_regime_overlay_v1` as an exploratory side branch, not as
the new main candidate.

## Next Question

If we continue the regime-aware line, the next question should not be:

- how do we add more dynamic knobs?

It should be:

- can we define a better small set of state variables that creates a more
  informative `defensive` regime without exploding model complexity?
