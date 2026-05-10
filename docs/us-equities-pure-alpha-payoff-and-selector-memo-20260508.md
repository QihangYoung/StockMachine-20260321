# Pure Alpha Payoff and Selector Scoring Memo

Date: 2026-05-08

Scope: validation-window research only. Test lockbox remains closed.

## 1. What `roll60 payoff` Means

`roll60 payoff` is a smoothed diagnostic of whether a transparent style factor is currently being rewarded.

For one signal date `t`, one factor, and one forward horizon `h`, first compute:

```text
daily factor payoff(t, h)
= mean(future residual return of long factor basket)
  - mean(future residual return of short factor basket)
```

Example: for `anti_momentum_20d`, the long factor basket is the bottom 20% by `momentum_20d_z`, and the short factor basket is the top 20% by `momentum_20d_z`.

Then:

```text
roll60 payoff(t, h)
= average(daily factor payoff over the latest 60 signal dates)
```

In Phase6M/6O, the main displayed unit is `bps / h10`. A value of `+20 bps` means that over the latest 60 signal dates, the diagnostic loser-minus-winner factor earned about 20 bps per h10 batch on average.

## 2. Important Look-Ahead Warning

The research diagnostic `roll60 payoff(t, h)` uses forward returns from signal date `t`. Therefore it is not directly tradable at date `t`.

For live or walk-forward gating, the usable version must be lagged by at least the forward horizon:

```text
usable roll60 payoff(t, h)
= average(payoff values whose full h-day forward return is already known)
```

For h10, this means the newest included signal date must be roughly `t - 10` sessions or earlier.

This distinction matters. The research series is useful to understand failure mechanics; the lagged series is the only acceptable input for an actual activation rule.

## 3. Current Selector Setup

Current size-neutral candidate:

```text
portfolio: size_hard_neutral
long variant: top1000_clean_core_beta_full
short variant: adv30m_clean_core_beta_full
long score: reversal_5d
short score: short_hybrid_soft_fw_overlay
holding: rolling h10 sleeves
constraints: dollar neutral, beta neutral, hard size neutral, SIC2 soft neutral
```

The selector score is not the final portfolio weight. It first ranks names into a candidate pool, then the optimizer chooses weights subject to constraints.

Daily candidate construction:

```text
long candidates  = top 80 by long score, plus incumbent long names
short candidates = top 80 by short score, plus incumbent short names
```

Weight optimizer objective:

```text
maximize selector score quality
minus turnover penalty
minus SIC2 imbalance penalty
subject to:
  long gross = 1
  short gross = 1
  long beta = short beta
  long size = short size
  max single-name side weight = 1 / 30
```

The score inside the optimizer is cross-sectionally z-scored inside the candidate set before entering the linear objective.

## 4. Long Selector: `reversal_5d`

The long-side selector is deliberately simple:

```text
return_5d = trailing 5-session adjusted close return
reversal_5d = -return_5d
```

Higher `reversal_5d` means the stock has fallen more over the recent 5-session window.

Interpretation:

```text
long selector wants recent short-term losers
```

This is a short-horizon mean-reversion signal. It assumes that recent underperformance is at least partly overreaction, liquidity pressure, or temporary positioning, and that the next h10 window has positive rebound odds.

Current caveat:

The signal has no explicit falling-knife confirmation in the current `size_hard_neutral` construction. Earlier research tested long overlays, but the current candidate still uses raw `reversal_5d`.

## 5. Short Selector: `short_hybrid_soft_fw_overlay`

The current short score is built in layers.

### 5.1 Base Features

For each date and universe, the panel builds:

```text
return_5d = -reversal_5d
return_5d_z = zscore(return_5d)
momentum_20d_z = zscore(momentum_20d)
momentum_60d_z = zscore(momentum_60d)
beta_residual_momentum_20d_z = zscore(beta_residual_momentum_20d)
vol_adjusted_momentum_20d_z = zscore(vol_adjusted_momentum_20d)
```

It also builds two overextension features:

```text
exhausted_winner_20_5
= momentum_20d_z + return_5d_z

residual_overextension_20_5
= beta_residual_momentum_20d_z + return_5d_z
```

These are high when a stock is already a medium-term winner and has also extended recently.

### 5.2 Baseline Short Score

The baseline short score is:

```text
short_core_plus_overextension
= mean_zscore(
    momentum_20d,
    beta_residual_momentum_20d_z,
    vol_adjusted_momentum_20d,
    exhausted_winner_20_5,
    residual_overextension_20_5
  )
```

Higher score means:

```text
strong recent winner
strong residual winner
strong vol-adjusted winner
possibly overextended in the latest 5 sessions
```

Naively, this is a short-the-overextended-winner signal.

### 5.3 Former-Winner / Recent-Weakness Layer

The strategy then estimates whether a strong winner has started to weaken.

```text
momentum_20_pos = max(momentum_20d_z, 0)
momentum_60_pos = max(momentum_60d_z, 0)
residual_20_pos = max(beta_residual_momentum_20d_z, 0)
vol_adjusted_pos = max(vol_adjusted_momentum_20d_z, 0)

former_winner_strength_mixed_20_60
= (momentum_20_pos + momentum_60_pos + residual_20_pos + vol_adjusted_pos) / 4

recent_weakness = max(-return_5d_z, 0)
recent_is_down = return_5d < 0
```

So the short overlay looks for:

```text
strong historical winner
plus recent price weakness confirmation
```

### 5.4 Hybrid Soft Overlay

The final short selector is:

```text
baseline = short_core_plus_overextension
strength = max(former_winner_strength_mixed_20_60, 0)

nonweak_050 =
  strength > 0.50 and recent_is_down is false

confirmed_weak_050 =
  strength > 0.50 and recent_is_down is true

soft_raw =
  baseline
  - 0.75 * strength if nonweak_050
  + 0.35 * recent_weakness if confirmed_weak_050

short_hybrid_soft_fw_overlay = zscore(soft_raw)
```

Interpretation:

```text
Do not short a strong winner merely because it is strong.
Prefer shorting a strong winner only after it has started to weaken.
Penalize strong winners that are still continuing upward.
Reward strong winners that have recent weakness confirmation.
```

This is why we often describe the short selector as:

```text
"强势背景 + 最近转弱确认，才允许空"
```

## 6. Why This Matters for Recent Diagnostics

The current strategy has large intentional exposure to:

```text
long recent losers
short recent winners / former winners
```

Therefore it depends on short-horizon cross-sectional mean reversion being rewarded.

Phase6M/6N/6O showed:

```text
average h10 mean-reversion payoff is positive,
but rolling payoff can turn negative for months,
and extending the horizon to h20/h40 usually does not fully rescue h10-negative episodes.
```

This suggests that future robustness work should focus on:

```text
1. reducing harmful medium-term anti-momentum exposure, especially anti_momentum_60d;
2. adding lagged rolling-payoff gates;
3. testing whether selector activation should depend on smoothed payoff state rather than fixed always-on reversal.
```

## 7. Practical Next Step

The clean next experiment is not to replace the whole selector. It is to add a white-box activation layer:

```text
if lagged roll60 payoff for a selector family is positive:
    keep normal selector weight
elif lagged roll60 payoff is near zero but slope is deteriorating:
    reduce selector score weight
else:
    disable or strongly penalize that selector family
```

This should be tested only inside the validation framework before any new test-lockbox use.
