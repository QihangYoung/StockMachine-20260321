# Pure Alpha Two-Track Roadmap: Style Regime and Non-Price Alpha

Date: 2026-05-11

Scope: research roadmap. Test lockbox remains closed.

## Current State

We decompose portfolio return as:

\[
R_{p,t}
=
(w_t^\top \beta_t) r^{mkt}_t
+
(w_t^\top B_t) f_t
+
w_t^\top \alpha_t
+
w_t^\top \epsilon_t
\]

where the four sources are market beta, style factors, alpha, and noise.

The current SOTA is best described as a beta-neutral price-style strategy, not yet a proven strict pure-alpha strategy.

Evidence:

- Market beta is explicitly controlled through beta matching.
- Size is now explicitly controlled through `size_hard_neutral`.
- Sector/industry exposure is controlled through SIC soft neutrality.
- The strategy still relies materially on price-style payoff: reversal, anti-momentum, overextension, and beta-residual momentum.
- In beta-residual label space, price ML can produce strong OOS payoff.
- In strict residual label space, after removing beta, size, liquidity, sector, and reversal/momentum style effects, price-only ML payoff is thin and unstable.
- Medium-capacity beta-residual ML only slightly exceeds current SOTA h10-equivalent raw gross performance, so SOTA already captures most obvious price-based payoff.

This means the next research should not simply be "larger price ML." We split into two tracks.

## Track 1: Style Regime Identification

### Goal

Recognize when the price-style payoff we already harvest is likely to be favorable, weak, or dangerous.

This is not the same as market timing. We are not trying to predict broad market direction. We are trying to monitor whether our own style engine, especially short-horizon reversal / anti-momentum payoff, is in a supportive or hostile regime.

### Why This Matters

If current SOTA earns mostly from price-style payoff, the key risk is style payoff regime change.

Examples:

- Reversal payoff may work in choppy/liquid mean-reverting markets.
- Momentum continuation may dominate during stress, squeezes, narrow leadership, or crowded deleveraging.
- Short-side loser identification may fail when former winners keep squeezing upward.
- Long-side dip-buying may fail when losers are structurally impaired rather than temporarily oversold.

The objective is not to perfectly forecast regimes. The objective is to build conservative, low-degree-of-freedom state variables that reduce exposure when the style engine is observably unhealthy.

### Candidate Signals

Primary diagnostics:

- Rolling realized payoff of reversal / anti-momentum spreads.
- Cross-sectional dispersion.
- Breadth of winners and losers.
- Market volatility and vol-of-vol.
- Index trend and crash/rebound state.
- Residual momentum continuation versus reversal balance.
- Short book squeeze pressure proxies.
- Long book falling-knife pressure proxies.

Preferred design:

- White-box first.
- Low parameter count.
- Use rolling out-of-sample payoff diagnostics.
- Require economic interpretation before accepting a gate.
- Avoid high-capacity regime classifiers unless we have many independent regimes, which we do not.

### Possible Actions

- Exposure scaling: reduce gross when style payoff is weak.
- Sleeve gating: turn off only the side that is failing, for example short-side former-winner shorts.
- Selector switching: use reversal selector in mean-reverting states, but require confirmation or use a weaker selector in continuation states.
- Risk-budget shifting: keep beta/sector/size neutral but reduce concentration when style health deteriorates.

### Acceptance Standard

A style-regime rule is useful only if:

- it improves drawdown and rolling return stability in validation;
- it does not rely on a small number of hand-picked failure windows;
- it has stable behavior across 2015, 2018, 2019-style stress windows;
- it preserves most of the gross return in normal regimes;
- it uses few parameters and has a clear economic story.

If a regime rule mainly improves one known bad window while harming others, treat it as overfit.

## Track 2: Non-Price Alpha

### Goal

Find return-predictive information that survives after controlling for market beta and known price styles.

This is the more direct route toward actual pure alpha.

### Why This Matters

The price-only experiments suggest that easy price payoff is already mostly captured by SOTA. After strict residualization, remaining price-only signal is thin.

To improve beyond current SOTA in a robust way, we likely need information that is not just another transformation of recent returns.

### Candidate Inputs

Fundamental / accounting:

- profitability quality;
- margin trend;
- revenue acceleration or deceleration;
- balance-sheet deterioration;
- leverage and refinancing risk;
- cash-flow quality;
- accruals;
- valuation versus growth/profitability context.

Insider / ownership / positioning:

- insider buying and selling pressure;
- institutional ownership changes;
- short interest and days-to-cover;
- borrow availability / hard-to-borrow status;
- crowded long and crowded short proxies.

Event / information flow:

- earnings surprise and post-earnings drift;
- guidance revisions;
- analyst revision direction;
- news sentiment and event intensity;
- filing changes;
- corporate actions.

Microstructure / flow:

- abnormal volume not explained by price move;
- liquidity deterioration;
- intraday reversal or continuation pressure;
- order-flow imbalance proxies, if available.

### Label Standard

Non-price features should be tested against multiple labels:

- beta residual;
- beta + size + liquidity + sector residual;
- strict style residual;
- side-specific long and short residual loser/winner labels.

A feature is interesting only if it retains signal after at least beta, size, liquidity, and sector controls. A feature is especially valuable if it survives strict style residualization.

### Acceptance Standard

A non-price factor or model is useful only if:

- it improves strict residual rank IC / top-bottom spread OOS;
- it is not merely a proxy for reversal/momentum/size/liquidity;
- it improves actual optimizer-level portfolio results, not just standalone label payoff;
- it has plausible economic mechanism;
- coverage is adequate for our high-liquidity universe;
- timestamping is point-in-time safe.

## What We Should Not Do Next

Do not keep increasing price-only model complexity as the main path. Phase6R shows that stronger models increase train fit much faster than OOS strict residual payoff.

Do not call beta-residual price payoff pure alpha. It is useful, but it likely includes style payoff.

Do not open the test lockbox for these theoretical improvements. We should continue validation-only research until we have a materially different hypothesis.

Do not build a high-dimensional style timing model before simpler white-box regime diagnostics are exhausted.

## Immediate Next Steps

Track 1:

1. Define a small style-health dashboard around rolling reversal / anti-momentum payoff, dispersion, breadth, and continuation pressure.
2. Re-express known bad windows as style-health failures rather than calendar windows.
3. Test low-degree-of-freedom gates or exposure scalers against validation only.

Track 2:

1. Send this memo to the non-price line as a factor request.
2. Ask for features explicitly designed to predict strict residual winner/loser behavior.
3. Require point-in-time availability, coverage, and neutralization diagnostics.
4. Evaluate new factors through the same label ladder used in Phase6Q.

## Working Position

Current SOTA remains the operational baseline.

The research objective is now:

- Track 1: make current price-style payoff safer and less regime-fragile.
- Track 2: find orthogonal non-price information that can become true alpha.

Only Track 2 can credibly move the product from "beta-neutral price-style strategy" toward "pure alpha." Track 1 is still valuable because it may make the current strategy safer while Track 2 matures.
