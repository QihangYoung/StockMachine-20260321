# Pure Alpha + Beta Product Memo

Date: 2026-05-07

## Purpose

This memo records the current product-level design constraint for combining the
U.S. equities pure-alpha sleeve with an optional `SPY` / S&P 500 beta sleeve.

This is a sizing and risk-budget memo, not a new backtest result. The final test
lockbox remains closed.

## Current Product Direction

The target product is no longer a low-volatility multi-asset ETF product. The
current design objective is:

- high growth with controlled ruin risk;
- pure-alpha stock selection as the primary return engine;
- optional S&P 500 beta as a simple, liquid growth engine;
- no forced deleveraging under ordinary stress assumptions;
- no use of the final test window for parameter tuning.

The beta sleeve may be implemented with `SPY`, `MES`, `ES`, or another liquid
S&P 500 instrument depending on account size and trading constraints. Instrument
choice is execution detail; the product-level constraint is the total stress
capital budget.

## Hard Safety-Cushion Rule

The product must reserve a final safety cushion of at least `25% NAV`.

Equivalently, theoretical stress capital occupancy should be anchored at no
more than `75% NAV`:

```text
alpha_margin_requirement
+ alpha_stress_loss
+ beta_margin_requirement
+ beta_stress_loss
<= 75% NAV
```

The remaining `25% NAV` is not a return-seeking risk budget. It is the final
liquidity reserve used to prevent forced liquidation, broker margin pressure,
and bad-timing deleveraging.

This rule should be treated as a hard product constraint, not a soft preference.
If a proposed exposure mix breaches the `75%` stress-occupancy budget, reduce
alpha scale, reduce beta notional, or reject the mix.

## What Counts As Safety Cushion

The safety cushion should be held in assets that remain liquid and operationally
usable during stress:

- cash;
- Treasury bills;
- Treasury-only money-market funds;
- very short Treasury ETF proxies such as `BIL` or `SGOV`, subject to broker
  collateral treatment.

The following should not be counted as the final safety cushion:

- `SPY` or other equity ETFs;
- `LQD`, `IEF`, or other duration / credit risk sleeves;
- `GLD` / `GLDM` or other gold exposure;
- option premium already spent;
- margin excess that depends on volatile risk assets keeping their value.

Gold, duration, and credit may still be useful risk sleeves, but they are not
the last-resort liquidity reserve.

## Current Alpha Sizing Implication

The current pure-alpha SOTA is approximately a `+100% / -100%` target gross
long-short strategy. In the validation artifact, actual average gross exposure
is lower because of holding overlap and netting, but the product should budget
against the target exposure unless broker margin data proves otherwise.

Working assumptions for first-pass product sizing:

- target alpha exposure: `+100% / -100%`, or `200%` gross;
- assumed broker margin requirement: `30%` of gross notional;
- implied alpha margin requirement: about `60% NAV`;
- observed validation max drawdown: about `8%`;
- first-pass alpha stress budget: roughly `60% + 8% = 68% NAV`.

This means full-size pure alpha already consumes most of the `75%` stress
budget. Adding beta on top is possible only if the beta sleeve is small enough
or the alpha sleeve is scaled down.

The lower actual average gross exposure observed in the backtest can be tracked
as a secondary reference, but the conservative product rule should use the
target-gross budget until real broker margin previews are available.

## Beta Sizing Implication

The beta sleeve should be sized from current notional and current margin
requirements at trade time:

```text
beta_margin_requirement = beta_notional * current_margin_rate
beta_stress_loss = beta_notional * assumed_spy_stress_drawdown
```

For small accounts, `MES` contract granularity can make one contract much larger
than the intended beta allocation. Therefore, beta sizing must satisfy both:

- target economic exposure;
- `75%` total stress-occupancy constraint after contract rounding.

If one `MES` plus full-size alpha breaches the stress budget, the product should
either reduce alpha scale or skip beta exposure until NAV is large enough.

## Opportunity Cost Of The 25% Cushion

The 25% cushion has no explicit option-premium cost, but it has an opportunity
cost because that capital is not deployed into higher-return sleeves.

The annual opportunity cost can be estimated as:

```text
25% NAV * (expected_return_of_displaced_risk_capital - safety_cushion_yield)
```

Illustrative scenarios:

| Displaced use of capital | Assumed return | Safety yield | Opportunity cost on total NAV |
|---|---:|---:|---:|
| S&P 500 beta | `8% ~ 12%` | `3.5% ~ 4.0%` | about `1.0% ~ 2.1%` per year |
| Pure alpha sleeve | about `15%` | `3.5% ~ 4.0%` | about `2.8%` per year |
| Aggressive alpha + beta mix | about `20%` | `3.5% ~ 4.0%` | about `4.0%` per year |

The practical central estimate is that the safety cushion costs roughly
`2.5% ~ 3.0% NAV` per year versus deploying the same capital into the pure-alpha
sleeve. The cost is the implicit insurance premium for avoiding forced
deleveraging, margin calls, and path-dependent ruin.

This cost should be reported explicitly in product comparisons. A strategy that
looks attractive before the safety cushion may be materially less attractive
after reserving the cushion.

## Operating Rules

- The 25% cushion is not available for opportunistic risk increases.
- If the projected cushion falls below `25%`, do not add new risk.
- If the projected cushion falls below `20%`, treat the portfolio as yellow
  zone and prepare a restoration plan.
- If the projected cushion falls below `15%`, treat the portfolio as red zone
  and prioritize liquidity restoration over return seeking.
- Broker margin previews should override static assumptions when they are more
  conservative.
- Any production sizing table must show both expected return and stress capital
  occupancy.

## Current Decision

Use the `25%` safety cushion as a hard design constraint. The working product
budget is therefore:

```text
maximum theoretical stress occupancy = 75% NAV
minimum final safety cushion = 25% NAV
```

Under the current assumptions, full-size pure alpha can stand on its own inside
this budget. Adding S&P 500 beta is still possible, but it must be funded by
available remaining stress budget or by scaling down the alpha sleeve.
