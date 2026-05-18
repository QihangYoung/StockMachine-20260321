# Phase6T Dynamic Price-Style Allocation Memo

Date: 2026-05-11

Scope: validation-only. Test lockbox remains closed.

## Window Correction

The first Phase6T run used 2017-01-01 to 2019-12-31 because the initial walk-forward sketch reserved history through 2016 for payoff estimation. That made the reported window too narrow.

The current data chain has three relevant starts:

- raw Phase0 / Phase1 history reaches 2013, but current beta-full trading universe is not usable from that date;
- the h10 beta-full signal panel used by this strategy starts on 2014-08-05, after the 252-session beta lookback and related feature requirements;
- the dynamic posterior requires 60 matured h10 payoff observations, so the first tradable Phase6T date is 2014-11-11.

This memo therefore reports the corrected Phase6T validation window: 2014-11-11 to 2019-12-31.

## Objective

This experiment implements the first concrete version of the dynamic price-style allocation route. The purpose is not to claim pure alpha. The purpose is to test whether a simple posterior estimate of price-style payoff can improve a beta-neutral, size-neutral, long-short strategy.

The first version deliberately removes the current SOTA asymmetry:

- long universe: `adv30m_clean_core_beta_full`;
- short universe: `adv30m_clean_core_beta_full`;
- same score function on both sides;
- long buys high-score names;
- short sells low-score names.

The score is:

\[
s_{i,t} = \mathbf b_{i,t}^{\top}\tilde{\mathbf f}_t
\]

where \(\mathbf b_{i,t}\) is the stock's price-style exposure vector and \(\tilde{\mathbf f}_t\) is the posterior forecast of style payoff.

## First-Pass Style Set

The first pass uses \(K=4\) price styles selected by prior economic logic, not by validation-window optimization:

- `reversal_5d`;
- `anti_momentum_20d = -momentum_20d`;
- `anti_beta_residual_momentum_20d = -beta_residual_momentum_20d`;
- `anti_vol_adjusted_momentum_20d = -vol_adjusted_momentum_20d`.

All style exposures are cross-sectionally z-scored by date. Positive score means the stock has more exposure to the intended anti-momentum / reversal direction.

## Payoff And No-Lookahead Rule

For each style \(k\), the daily payoff is the top-minus-bottom h10 beta-residual return:

\[
y_{k,t}
=
\frac{1}{|Top_{k,t}|}
\sum_{i \in Top_{k,t}} r^{res}_{i,t:t+10}
-
\frac{1}{|Bottom_{k,t}|}
\sum_{i \in Bottom_{k,t}} r^{res}_{i,t:t+10}
\]

On trade date \(t\), the estimator can only use matured payoff observations:

\[
\mathcal F_t = \{y_\tau : \tau + 10 \le t\}
\]

This prevents h10 label leakage.

## Posterior Estimators

The experiment runs six posterior variants:

- `static_mean`: expanding mean payoff;
- `ewma_hl20`: EWMA half-life 20 sessions;
- `ewma_hl60`: EWMA half-life 60 sessions;
- `ewma_hl120`: EWMA half-life 120 sessions;
- `kalman_phi099_q0004`: univariate Kalman, \(\phi=0.99\), \(Q/R=0.0004\);
- `kalman_phi099_q0004_shrink025`: same Kalman with symmetric shrink-to-zero penalty.

The Kalman model is intentionally low degree-of-freedom in this first version. It is not tuned per style.

## Portfolio Construction

For each date and posterior variant:

- candidate pool uses the same adv30m universe on both sides;
- rank long candidates by \(s_{i,t}\);
- rank short candidates by \(-s_{i,t}\);
- remove cross-side symbol overlap;
- target long gross = 100% NAV and short gross = 100% NAV;
- hard match beta between long and short books;
- hard match true size exposure using `market_cap_log_z`;
- soft neutralize SIC2 sector exposure;
- max single-name side weight = \(1/30\);
- minimum nonzero names per side = 20;
- turnover penalty uses the existing default from the h10 rolling framework.

## Validation Results

Comparable curve metrics, 2014-11-11 to 2019-12-31:

| Portfolio | Final Equity | Ann. Return | Ann. Vol | Sharpe | Max DD | Rolling 60 Positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sota_size_hard_neutral` | 1.996 | 14.42% | 7.51% | 1.83 | -7.78% | 85.15% |
| `static_mean` | 1.485 | 8.03% | 8.74% | 0.93 | -21.67% | 61.50% |
| `ewma_hl120` | 1.471 | 7.83% | 8.71% | 0.91 | -28.35% | 61.89% |
| `ewma_hl60` | 1.246 | 4.39% | 8.38% | 0.55 | -27.12% | 54.38% |
| `kalman_phi099_q0004_shrink025` | 1.083 | 1.57% | 8.52% | 0.23 | -24.77% | 51.98% |
| `kalman_phi099_q0004` | 1.063 | 1.20% | 8.61% | 0.18 | -25.36% | 52.59% |
| `ewma_hl20` | 1.033 | 0.64% | 8.57% | 0.12 | -23.32% | 50.35% |

Key artifact:

- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6t_dynamic_price_style_allocation_20260511/phase6t_dynamic_style_allocation_plot.png`

## Interpretation

The corrected-window result is not strong.

The earlier 2017-only result made `ewma_hl120` look much more promising because it skipped the 2015-2016 drawdown regime. Once the available validation window starts at the proper post-warmup date, SOTA dominates every Phase6T variant on return, Sharpe, max drawdown, and rolling 60-session positive rate.

`static_mean` being competitive is important. It says the simple prior that these anti-momentum / reversal styles usually pay is already powerful. The dynamic model must beat a strong static prior, not just beat zero.

The short half-life EWMA and this first Kalman specification are weak. The likely failure is overreacting to noisy style payoff or shrinking away useful persistent style exposure. This is exactly the regime-timing danger we were worried about: if the state estimator is not very good, it can destroy a positive long-run style premium.

The experiment also confirms that this route is style-seeking. `ewma_hl120` has large intended exposure to anti-momentum / reversal styles. It should be treated as a beta-neutral price-style strategy, not as pure alpha.

## Decision

Do not replace current SOTA with Phase6T.

Keep Phase6T as a research branch because it gives us a clean, symmetric, validation-only harness for dynamic style allocation. However, the corrected-window result says this first dynamic style allocation design is not close to production-relevant.

## Next Experiments

1. Add capped style-seeking construction:

\[
|w_t^\top b_{k,t}| \le c_k
\]

This tests whether `ewma_hl120` can keep most of its return while reducing drawdown and rolling-window instability.

2. Add standalone posterior calibration:

- posterior sign hit rate;
- realized payoff by forecast-confidence bucket;
- payoff attribution by style and by calendar year.

3. Revisit Kalman only after calibration:

- use a small grid for \(\phi\) and \(Q/R\);
- avoid per-factor validation optimization;
- consider multivariate shrinkage only if univariate calibration is acceptable.

4. Keep the non-price alpha track separate.

If price-style allocation cannot improve risk quality without overfitting, the real path back toward pure alpha likely requires non-price signals rather than more aggressive price-regime timing.
