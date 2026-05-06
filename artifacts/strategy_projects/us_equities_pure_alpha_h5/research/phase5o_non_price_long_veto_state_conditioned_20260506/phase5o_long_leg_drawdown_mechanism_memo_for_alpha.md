# Phase5O Long-Leg Drawdown Mechanism Memo for Alpha

Generated: 2026-05-06 Asia/Shanghai

## Context

We tested several non-price long-side vetoes on top of the current SOTA:

- Phase5N filing/Form4 factors.
- Phase4 SEC companyfacts fundamental fragility.
- `filing OR fundamental` threshold sweep.
- state-conditioned variants such as `fof low mom60 veto t075`.

The best validation candidate so far is `np fof low mom60 veto t075`: annualized return 16.77%, Sharpe 2.164, max drawdown -8.19%. It improves full-sample Sharpe, but it does not materially repair the 2015 peak-to-trough drawdown.

This memo focuses on whether the long-leg drawdowns require non-price data, especially given prior alpha-thread attempts where pure-price approaches did not work well.

## Empirical Split

Current SOTA window decomposition:

| window | portfolio total | long leg sum | short leg sum | SPY | read |
|---|---:|---:|---:|---:|---|
| 2015 peak-to-trough | -7.66% | -11.75% | +3.83% | -4.38% | long reversal falling-knife problem |
| 2017 spring | -4.22% | +1.68% | -5.95% | +1.72% | short-leg squeeze/rally problem |
| 2018 Aug-Sep | -3.03% | -0.38% | -2.66% | +1.29% | mostly short-leg problem |
| 2019 May | -0.70% | -5.83% | +5.15% | -3.80% | long reversal falling-knife problem |
| 2019 Aug | -4.56% | -6.99% | +2.35% | -2.25% | long leg problem, but veto can delete rebound winners |

This means 2015 / 2019 May / 2019 Aug are relevant for long-side work. 2017 and 2018 should be handled by short-side risk controls, not long-side non-price gates.

## Main Mechanism

The common long-leg failure is not simply "bad company fundamentals" or "SEC red flags." It is:

**Short-horizon reversal is being applied during a trend-continuation / de-risking regime, so the long book buys losers before price has stabilized.**

This is why filing/companyfacts alone does not fix 2015. In the 2015 top80 long candidates, the `filing OR fund t075` trigger group was not worse than the full top80 set. The trigger group was slightly less bad, so vetoing it can only make a small dent.

For 2019 Aug, the mechanism is even trickier: many triggered names were rebound-capable winners. A hard non-price veto can delete the very stocks that recover after a volatility shock.

## Do We Need Non-Price?

Not in principle. The primary failure mode is visible in price/state data:

- sector trend continuation,
- market/sector breadth weakness,
- volatility expansion,
- cross-sectional correlation rising,
- no short-term stabilization after a sharp selloff,
- stock drawdown explained by sector/market rather than idiosyncratic oversold behavior.

However, the prior pure-price attempts performing poorly is an important signal. It suggests a naive price-only gate is probably too blunt. Common failure modes:

- It suppresses reversal exactly when the reversal premium is largest.
- It acts at market level when the problem is sector-level or stock-state-specific.
- It hard-vetoes instead of softly reshaping scores/gross.
- It does not distinguish "still falling" from "shock then stabilizing."
- It optimizes full-sample Sharpe while ignoring worst-window long-leg repair.

So the practical answer is:

**Non-price is not theoretically required, but it may be useful as a secondary prior once price has identified a falling-knife setup.**

It should not be the primary trigger.

## Recommended General Strategy

Avoid one-off window patches. Test one universal long-side state model:

1. Build a price-state `falling_knife_score`.
2. Use non-price only as a multiplier or tie-breaker.
3. Apply a soft score penalty or long gross throttle, not a hard veto.

Suggested shape:

```text
price_falling_knife =
    weak sector 20/60d momentum
  + weak stock sector-relative 20/60d momentum
  + high recent realized vol / vol expansion
  + high market/sector correlation regime
  + no 3-5d stabilization confirmation

non_price_fragility =
    filing distress
  + companyfacts fragility
  + insider sell pressure

adjusted_long_score =
    z(reversal_5d)
  - lambda * price_falling_knife * (0.5 + 0.5 * non_price_fragility)
```

Alternative implementation:

```text
long_gross_multiplier_t =
    clip(1 - k * aggregate_reversal_off_regime_score_t, floor, 1)
```

The first version is preferred because it is candidate-level and should avoid killing the whole reversal book.

## What Not To Do

- Do not keep sweeping filing/fundamental thresholds.
- Do not use filing/companyfacts as standalone hard vetoes.
- Do not build separate rules for 2015, 2019 May, and 2019 Aug.
- Do not evaluate only full-sample Sharpe; require window-level long-leg evidence.

## Proposed Alpha Experiment

Build one candidate-level hybrid overlay:

- `sector_momentum_60d_pct_low`
- `stock_sector_relative_momentum_20d/60d`
- `realized_vol_20d_pct` and vol expansion
- `short_stabilization_flag`: recent 3-5d return or close-above-short-MA confirmation
- existing `long_filing_distress_score`
- existing `long_fundamental_fragility_score`

Run four variants:

| variant | purpose |
|---|---|
| current SOTA | baseline |
| price falling-knife soft penalty | retest pure price, but softly and candidate-level |
| price falling-knife gross throttle | tests whether long gross needs to move |
| price falling-knife x non-price soft penalty | main hybrid candidate |

Promotion criteria should include:

- improves or does not materially worsen 2015 long-leg path,
- improves 2019 May,
- does not delete 2019 Aug rebound winners,
- keeps annualized return and Sharpe close to current SOTA or better,
- does not materially increase turnover.

## Current Read

The current non-price line is directionally useful for ranking/risk context, but it is not the missing state variable. The missing variable is a robust reversal-regime / falling-knife state. Given prior poor pure-price attempts, the next pass should not be naive price-only. It should be a candidate-level hybrid where price detects the state and non-price controls penalty severity.
