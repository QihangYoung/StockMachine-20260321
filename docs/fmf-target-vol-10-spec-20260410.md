# FMF 10% Volatility Target Strategy Spec (2026-04-10)

## Objective

Design a long-only multi-asset strategy with:

- `0` leverage
- annualized volatility anchored around `10%`
- Sharpe as high as possible

This is the most practical translation of the user objective:

- high risk-adjusted return
- no gross leverage
- explicit volatility discipline

## Important Clarification

Under a strict `0`-leverage constraint, `10%` volatility is not always exactly reachable.

If the best long-only risky mix naturally runs below `10%` annualized volatility:

- we cannot lever it up
- so the strategy should treat `10%` as a **target-cap**, not a hard equality target

Therefore the first formal definition should be:

> maximize risk-adjusted return subject to long-only weights, cash allowed, and portfolio volatility `<= 10%`

Operationally this is:

- `vol target` as intent
- `vol cap` as constraint

## Why This Strategy Exists

The current rebuilt `FMF` line showed two things:

1. static long-only risk-budget portfolios can already reach strong validation Sharpe
2. the regime overlay branch is not yet strong enough to justify replacing the best static lead

That makes a `10%`-target strategy attractive because it is:

- industry-standard
- easy to explain
- directly aligned with product design

It also gives us a cleaner benchmark than continuing to optimize unconstrained bucket weights without a volatility budget.

## Industry Analogue

This spec is closest to:

- `target volatility`
- `risk control`
- `long-only maximum-Sharpe with vol cap`

It is **not** the same as:

- minimum-volatility only
- full unconstrained mean-variance optimization
- leveraged risk parity

## Research Universe

The first implementation should stay inside the rebuilt `FMF` validation universe:

- `equity_us` -> `SPY`
- `equity_ex_us` -> `VXUS`
- `duration` -> `IEF`
- `credit` -> `LQD`
- `inflation_hedge` -> `GLD`
- `trend` -> `FMF`
- `cash` -> `BIL`

This keeps the validation architecture consistent and avoids reopening the universe-selection problem before the strategy objective is tested.

## Portfolio Definition

### Risky Sleeves

Use the six non-cash buckets as the risky portfolio:

- `equity_us`
- `equity_ex_us`
- `duration`
- `credit`
- `inflation_hedge`
- `trend`

### Cash Sleeve

Use `BIL` as the cash / residual sleeve.

The strategy may leave capital in cash when:

- the optimal long-only risky mix would exceed the `10%` vol cap
- or when the chosen risky mix already has lower expected efficiency than holding some cash

## First-Pass Optimization Families

We should not start with only one optimizer. The right first pass is a small family comparison.

### Family A: Risk-Budget Core + Cash Scaling

Process:

1. solve a long-only strategic risky mix among the six non-cash buckets
2. estimate its realized volatility
3. scale down into cash until total portfolio volatility is `<= 10%`

Why test it:

- white-box
- robust
- less sensitive to return estimates

### Family B: Maximum Diversification + Cash Scaling

Process:

1. solve a long-only diversified risky mix using covariance structure only
2. scale risky exposure with cash to stay inside the `10%` cap

Why test it:

- often more stable than max-Sharpe
- still naturally aligned with multi-asset design

### Family C: Long-Only Max-Sharpe With Vol Cap

Process:

1. estimate expected returns with a simple and frozen rule
2. solve:
   - maximize estimated Sharpe
   - `w >= 0`
   - `sum(w) <= 1`
   - `portfolio vol <= 10%`
3. residual goes to cash

Why test it:

- closest to the raw user objective
- but should be treated as the highest-overfit-risk branch

## Recommended Expected-Return Discipline

If Family C is tested, expected return estimation must stay simple.

Allowed first-pass options:

- trailing medium-horizon momentum
- blended carry / momentum proxy
- simple equal expected return assumption by bucket group

Not allowed in first pass:

- complex predictive meta-model
- frequent retuning of return model parameters
- using lockbox data to select the return estimator

## Covariance / Volatility Model

Use the current rebuilt multi-asset default:

- long window sample covariance: `252` sessions
- short window EWMA covariance: `63` sessions
- blend: `70% long / 30% short`
- `ewma_lambda = 0.97`

This keeps the strategy comparable with the rest of the `FMF` line.

## Rebalance Frequency

Default:

- every `21` trading days
- `1` trading day effective lag

Reason:

- consistent with the existing rebuilt multi-asset process
- avoids mixing “new objective” with “new frequency” in the first pass

## Volatility Targeting Rule

The total portfolio should be built as:

- risky portfolio weights among non-cash buckets
- residual cash weight in `BIL`

If the risky portfolio estimated volatility is:

- `> 10%`: reduce risky sleeves proportionally until total vol is `10%`
- `< 10%`: do **not** lever up; keep remaining capital in cash only if the optimizer naturally implies it

This preserves the hard `0`-leverage constraint.

## Evaluation Metrics

Primary:

- Sharpe
- annualized return
- annualized volatility
- max drawdown

Secondary:

- monthly win rate
- downside deviation / Sortino
- turnover
- realized average cash weight
- realized risk share by bucket

## Baselines

The first experiment should compare against:

- `static_equal_weight_non_cash`
- `rolling_erc_core`
- `rolling_c2_v0_seed_core`
- `rolling_fmf_c2_e42_c10_d22_i20_t06`

These are all validation-only and already part of the rebuilt evidence stack.

## Validation Discipline

This strategy family must follow the current rebuilt protocol:

- all design and tuning stay on validation only
- lockbox test remains closed

That means:

- no optimizer family selection on the test set
- no expected-return model selection on the test set
- no vol-target parameter tuning on the test set

## First Implementation Order

### Step 1

Build Family A:

- risk-budget core
- cash scaling to `10%` cap

### Step 2

Build Family B:

- maximum-diversification core
- cash scaling to `10%` cap

### Step 3

Only then build Family C:

- long-only max-Sharpe with frozen return estimator

This order matters because it lets us answer:

- how much of the objective can be achieved with robust covariance-only structure
- before moving into more fragile return-estimation territory

## Working Recommendation

The best first productization path is:

- start with **risk-budget or diversification-first**
- add **cash-based 10% vol control**
- only later test explicit expected-return optimization

In short:

> first build a robust long-only 10%-vol core, then see whether a return model adds enough value to justify its extra fragility
