# US Equities Pure Alpha Characteristics Memo

Date: 2026-04-17

## Purpose

This memo explains what a U.S. equities long-short pure-alpha strategy is, why
its behavior differs from long-only stock selection, and what performance level
is usually realistic in institutional practice.

The target strategy line is:

- market: U.S. high-liquidity stocks
- style: cross-sectional stock selection
- portfolio: long-short
- objective: pure alpha, not equity beta
- construction rule: long and short books should be beta matched
- test discipline: final test set remains a lockbox

## Executive Summary

A long-short pure-alpha equity strategy is trying to answer one narrow question:

Can we reliably pick stocks that will outperform other stocks, after removing
the effect of the broad market going up or down?

In plain language, we do not want to make money because the whole U.S. market
went up. We want to make money because our long book was better than our short
book.

The ideal return source is:

`return = stock-selection alpha - costs - borrow/friction`

not:

`return = hidden S&P 500 exposure + stock-selection alpha`

This is why the research standard is much stricter than a normal long-only
strategy. A long-only strategy can survive with some market beta because the
market itself has a positive long-term drift. A pure-alpha long-short strategy
must prove that it is not secretly relying on that drift.

## The Six Core Characteristics

### 1. The Return Source Is Cross-Sectional Spread

The strategy does not mainly ask, "Will the market go up?"

It asks:

"Among the stocks we can trade today, which ones should outperform, and which
ones should underperform?"

The portfolio then buys the expected winners and shorts the expected losers.
The important result is the spread:

`long book return - short book return`

Simple example:

- market return: `+2.0%`
- long book return: `+3.0%`
- short book return: `+1.0%`
- long-short spread: `+2.0%`

In this case, the strategy made money because the long book beat the short book,
not merely because the market rose.

Another example:

- market return: `-2.0%`
- long book return: `-1.0%`
- short book return: `-3.0%`
- long-short spread: `+2.0%`

The market fell, but the strategy still made money because the long book fell
less than the short book.

This is the essence of pure alpha. We want relative selection skill to dominate
directional market exposure.

What to measure:

- long-leg return
- short-leg return
- long-minus-short spread
- daily IC and RankIC
- spread Sharpe
- hit rate by side

Red flag:

- the strategy only makes money when the market rallies
- the short book does not contribute
- the spread disappears after beta adjustment

### 2. Beta Matching Matters More Than Dollar Neutrality

Dollar neutrality means the long and short books have similar dollar notional.

For example:

- `100 USD` long
- `100 USD` short

This sounds neutral, but it may not be market neutral.

Why?

Because some stocks move more than the market, and some stocks move less. That
sensitivity is beta.

Example:

- long book beta: `1.3`
- short book beta: `0.8`
- long notional: `100`
- short notional: `100`

This book is dollar neutral, but it is still net long market beta:

`1.3 * 100 - 0.8 * 100 = +50 beta-dollar units`

If the market rises, the long book is expected to gain more than the short book
loses. If the market falls, the long book is expected to lose more. That is not
pure alpha. It is partly a disguised market bet.

Beta matching means we try to make the long side and short side equally exposed
to broad market moves.

The goal is:

`long beta exposure ~= short beta exposure`

so:

`net beta ~= 0`

In plain language, if the market moves sharply up or down, both sides should be
pushed by the market in roughly equal magnitude. The remaining profit or loss
should come mostly from whether our selected longs beat our selected shorts.

What to measure:

- ex-ante long beta
- ex-ante short beta
- ex-ante net beta
- realized regression beta versus `SPY`
- realized correlation versus `SPY`
- beta by market regime

Initial research gates:

- absolute ex-ante net beta: `<= 0.05`
- absolute realized beta: `<= 0.05`
- absolute market correlation: `<= 0.10`

Red flag:

- high Sharpe disappears when beta is neutralized
- the strategy is long high-beta growth and short low-beta defensives
- drawdowns line up with market selloffs despite "neutral" notional

### 3. Industry, Style, and Liquidity Exposures Must Be Controlled

Beta is only one kind of hidden exposure.

A strategy can also accidentally become:

- long technology, short utilities
- long mega-cap, short mid-cap
- long momentum, short value
- long high-quality, short low-quality
- long liquid names, short harder-to-trade names

Some of these tilts may be valid alpha ideas, but they should be intentional.
They should not sneak into the strategy and then be mistaken for stock-picking
skill.

Simple example:

Suppose the model buys mostly semiconductor stocks and shorts mostly consumer
staples. If semiconductors have a strong year, the strategy may look excellent.
But the real driver may be sector exposure, not stock selection inside the
sector.

For pure alpha, we prefer to ask:

"Within similar risk groups, did we pick the better stocks?"

This usually means controlling or reporting:

- sector
- industry
- size
- value
- momentum
- volatility
- liquidity
- short interest
- borrow cost when available

There are two acceptable paths:

- hard neutralization: force exposures close to zero
- transparent attribution: allow some exposures, but clearly report them and do
  not call them pure stock-selection alpha

For this project, the default bias should be conservative. If an exposure is not
intended, we should neutralize it or penalize it.

What to measure:

- sector exposure by long and short book
- industry concentration
- size and volatility spread between legs
- factor exposure regression
- contribution by sector and factor bucket

Red flag:

- one sector explains most of performance
- long and short books have very different liquidity profiles
- the strategy is effectively a style rotation strategy but is labeled pure
  alpha

### 4. Gross Exposure Is Usually Needed, But It Amplifies Friction

A long-only strategy has natural market exposure. If the market returns `10%`,
a long-only stock portfolio may benefit even if stock selection is only average.

A beta-neutral long-short strategy deliberately removes much of that market
tailwind.

That means the raw alpha spread is usually smaller than long-only total return.
To reach a useful volatility and return level, institutional long-short books
often use gross exposure.

Example:

- `100/100`: 100% long, 100% short, 200% gross
- `150/150`: 150% long, 150% short, 300% gross

Higher gross exposure can make a small spread economically meaningful.

But it also magnifies every implementation problem:

- transaction costs
- bid-ask spread
- market impact
- financing cost
- borrow cost
- locate failures
- short squeezes
- margin pressure
- turnover drag

So gross exposure is not free. It is a tool, not a source of alpha.

In plain language, if the signal has a small true edge, leverage can make it
visible. If the signal has no true edge, leverage only makes losses and costs
arrive faster.

What to measure:

- gross exposure
- net exposure
- return per unit gross exposure
- turnover per unit gross exposure
- cost drag
- borrow drag
- capacity under ADV limits

Red flag:

- backtest works only at unrealistic gross exposure
- performance collapses after modest cost stress
- turnover is too high for the target universe
- short costs are ignored

### 5. Capacity and Costs Are Central, Not Secondary

Pure-alpha long-short strategies often look better before costs than after
costs.

This is especially true when:

- signals rebalance frequently
- alpha decays quickly
- names are less liquid
- short book is expensive to borrow
- the strategy trades crowded names

Capacity means:

"How much capital can this strategy run before its own trading eats the edge?"

For high-liquidity U.S. stocks, capacity can be meaningfully better than small-
cap statistical arbitrage. But high-liquidity stocks are also heavily researched,
so the edge is usually more competitive and smaller.

The practical question is not:

"Does the signal work before costs?"

The practical question is:

"Does the signal still work after realistic costs, borrow assumptions, turnover,
and capacity constraints?"

Cost sources:

- commissions
- spread crossing
- market impact
- slippage
- financing
- borrow fees
- hard-to-borrow recalls
- failed locate constraints

What to measure:

- cost-stressed Sharpe
- annualized return under cost ladders
- turnover concentration
- ADV participation
- capacity at target impact
- borrow-cost sensitivity
- performance by liquidity bucket

Red flag:

- most alpha comes from the least liquid names
- returns vanish at `20 ~ 40 bps` cost stress
- short leg depends on names that would be expensive or impossible to borrow
- model turnover is too high relative to alpha half-life

### 6. Market Immunity Is Approximate, Never Perfect

No real strategy is perfectly immune to the market.

There are several reasons:

- beta is estimated from history, not known in advance
- beta changes during crises
- correlations rise in stress periods
- single-name events can become market-like during panic
- borrow and liquidity conditions worsen when markets fall
- long and short books may respond differently to volatility shocks

So "beta neutral" should never mean "cannot lose when the market moves."

It means:

"We have designed and measured the book so that broad market direction should
not be the main driver of returns."

The right standard is not theoretical perfection. The right standard is
continuous measurement and strict tolerance.

We should measure beta in multiple ways:

- ex-ante beta at portfolio formation
- realized full-sample beta
- rolling beta
- up-market beta
- down-market beta
- crisis-window beta
- correlation to `SPY`

In plain language, beta neutrality is like balancing a scale while the weights
keep changing. We can keep the scale close to balanced, but we must keep
checking it.

What to measure:

- rolling realized beta
- beta during large market up days
- beta during large market down days
- drawdown overlap with `SPY`
- residual return after market regression

Red flag:

- realized beta is much larger than ex-ante beta
- strategy loses heavily during broad market selloffs
- correlation rises sharply in stress periods
- "alpha" is mostly explained by market or sector regressions

## What Level Is Realistic?

Public market-neutral indices and institutional commentary suggest that true
long-short equity market-neutral alpha is valuable but difficult.

A useful rough scale:

| Level | Net Sharpe | Net Annual Return | Interpretation |
|---|---:|---:|---|
| Acceptable | `0.5 ~ 0.8` | `3% ~ 6%` | Worth studying if beta is clean, costs are realistic, and capacity is decent. |
| Strong | `0.8 ~ 1.3` | `5% ~ 10%` | Good institutional-quality market-neutral alpha. |
| Rare / excellent | `1.3 ~ 2.0+` | `8% ~ 15%+` | Hard to sustain live; usually requires strong data, execution, risk, and capacity control. |

These ranges are not guarantees. They depend on:

- gross exposure
- fees and financing
- borrow assumptions
- universe liquidity
- turnover
- signal decay
- portfolio constraints
- crowding

Backtests can easily overstate the achievable level. A validation Sharpe above
`2.0` should be treated with skepticism unless it survives cost stress,
subperiod stability, selection-bias checks, and capacity analysis.

For our first research phase, the correct priority is not maximum backtest
Sharpe. The correct priority is clean alpha:

- realized beta close to zero
- low market correlation
- both legs contributing
- performance stable across validation subperiods
- reasonable turnover and cost assumptions
- no dependence on the final test lockbox

## Working Standards For This Project

Initial validation targets:

- validation Sharpe: `> 0.7`
- annualized return after costs: `> 5%`
- absolute realized beta versus `SPY`: `<= 0.05`
- absolute market correlation: `<= 0.10`
- positive contribution from both long and short legs
- performance survives cost stress
- no top five days explain most of return

Initial disqualification rules:

- hidden market beta explains the result
- one sector or style explains the result
- short book is not realistically borrowable
- costs erase the edge
- test-window evidence is used for tuning

## Research Implication

The first implementation should not chase complex models immediately.

The right build order is:

1. Build a clean high-liquidity point-in-time universe.
2. Estimate lagged stock betas robustly.
3. Create a simple beta-matched long-short portfolio constructor.
4. Test simple transparent signals first.
5. Emit robustness-compatible artifacts.
6. Challenge the result with time stability, tail dependence, cost stress, and
   exposure diagnostics.
7. Only then consider more complex models.

If a simple signal cannot pass the beta-neutral and cost-aware gates, a more
complex model is likely to hide the same weakness rather than solve it.

## References

- HFR strategy classification: Equity Market Neutral definition.
- AQR, "Is Your Equity Hedge Fund Portfolio Resilient Enough for Uncertain Times?", 2024.
- Aon, "Using Equity Hedge and Market Neutral Strategies", 2014.
- Andrew Patton, "Are Market Neutral Hedge Funds Really Market Neutral?", Review of Financial Studies, 2009.

