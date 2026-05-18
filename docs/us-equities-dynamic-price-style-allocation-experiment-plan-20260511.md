# Dynamic Price-Style Allocation Experiment Plan

Date: 2026-05-11

Scope: validation-only research plan. Test lockbox remains closed.

## Positioning

For price-based strategies, we should stop pretending that recent-return effects are pure alpha. The cleaner framing is:

\[
R_{p,t}
\approx
(w_t^\top B_t) f_t
+
w_t^\top \epsilon_t
\]

where:

- \(B_t\) is the matrix of stock exposures to selected price styles;
- \(f_t\) is the vector of realized price-style payoffs;
- \(w_t^\top B_t\) is intentional style exposure;
- \(w_t^\top \epsilon_t\) is residual noise.

This project is therefore not a pure-alpha experiment. It is a dynamic price-style allocation experiment:

\[
\text{estimate } p(f_t \mid \text{matured style payoff history})
\text{ and allocate style exposure accordingly.}
\]

The hypothesis is not that style timing is easy. The hypothesis is narrower:

> Good probabilistic estimation of noisy, partially persistent style payoff can produce a posterior that is more useful than a static prior.

## Core Question

Can a robust probability model of price-style payoff improve on the current static/hand-built SOTA by dynamically allocating exposure to styles whose expected payoff posterior is favorable?

Success means:

- better h10-equivalent gross return than current SOTA;
- better or comparable drawdown and rolling-return stability;
- lower dependence on hand-written selector patches;
- no validation-only overfit;
- clear attribution that gains come from intentional style exposure, not mislabeled pure alpha.

## First Evaluation Design

The first implementation should deliberately avoid the current SOTA's asymmetric design. We want to test the dynamic price-style allocation idea itself, not mix it with legacy long/short selector choices.

Initial portfolio design:

- Long universe: `adv30m_clean_core_beta_full`
- Short universe: `adv30m_clean_core_beta_full`
- Long and short use the same score:

\[
s_{i,t}
=
\mathbf b_{i,t}^{\top}\tilde{\mathbf f}_t
\]

- Long book buys high-score names.
- Short book shorts low-score names.
- No side-specific long selector in the first pass.
- No side-specific short selector in the first pass.
- No former-winner short overlay in the first pass.
- No falling-knife long overlay in the first pass.

Reason:

- symmetric universe and symmetric scoring make attribution cleaner;
- using the same score on both sides tests whether dynamic style payoff is strong enough by itself;
- asymmetric SOTA components can be added later only after the clean baseline is understood.

## Style Universe

First version should use a small, pre-defined set of price styles:

1. `reversal_5d`
2. `anti_momentum_20d = -momentum_20d`
3. `anti_beta_residual_momentum_20d = -beta_residual_momentum_20d`
4. `anti_vol_adjusted_momentum_20d = -vol_adjusted_momentum_20d`
5. `overextension_20_5 = momentum_20d - reversal_5d`, with sign chosen so positive payoff means the intended style works
6. Optional side-specific styles:
   - long dip-buying payoff;
   - short former-winner rollover payoff.

Do not select \(K\) by validation-window significance. The initial factor set must be chosen by prior economic logic.

## Payoff Definition

For each style \(k\), define a daily h10 realized payoff:

\[
y_{k,t}
=
\frac{1}{|Top_{k,t}|}
\sum_{i \in Top_{k,t}}
r^{res}_{i,t:t+10}
-
\frac{1}{|Bottom_{k,t}|}
\sum_{i \in Bottom_{k,t}}
r^{res}_{i,t:t+10}
\]

where:

- \(Top_{k,t}\) is the high-score bucket for style \(k\);
- \(Bottom_{k,t}\) is the low-score bucket for style \(k\);
- \(r^{res}_{i,t:t+10}\) is the chosen residual forward return;
- the sign of each style must be normalized so \(y_{k,t} > 0\) means the style worked.

Recommended payoff labels:

1. beta residual payoff;
2. beta + size + liquidity + SIC2 residual payoff;
3. strict style residual payoff as diagnostic only.

For this Track 1 strategy, beta residual is a legitimate objective because we explicitly accept price-style exposure. Strict residual is used to prove what is and is not pure alpha.

## No-Lookahead Rule

If the strategy trades on date \(t\) and payoff horizon is h10, then \(y_{k,t}\) is not known until \(t+10\).

The latest usable payoff observation on date \(t\) is:

\[
z_{k,t} = y_{k,t-10}
\]

All Kalman updates must use only matured payoff observations:

\[
\mathcal F_t = \{y_{\tau}: \tau + 10 \le t\}
\]

The trading-time posterior is:

\[
p(f_t \mid \mathcal F_t)
\]

not:

\[
p(f_t \mid y_{1:t})
\]

## Model Family

### Baseline 1: Static Prior

Use long-run expanding-window mean payoff:

\[
\hat f_{k,t}
=
\frac{1}{N_t}
\sum_{\tau \in \mathcal F_t}
y_{k,\tau}
\]

This is the static style allocation benchmark.

### Baseline 2: EWMA

\[
\hat f_{k,t}
=
(1-\lambda)y_{k,t-10}
+
\lambda \hat f_{k,t-1}
\]

Use a small fixed grid of half-lives:

- 20 sessions;
- 60 sessions;
- 120 sessions.

### Main Model: Kalman Filter

Univariate first:

\[
y_{k,t}
=
f_{k,t}
+
\epsilon_{k,t},
\quad
\epsilon_{k,t} \sim \mathcal N(0, R_k)
\]

\[
f_{k,t}
=
\phi f_{k,t-1}
+
\eta_{k,t},
\quad
\eta_{k,t} \sim \mathcal N(0, Q_k)
\]

Use shared, low-degree-of-freedom parameters:

- \(\phi \in \{0.97, 0.99, 1.00\}\);
- \(Q/R\) calibrated by a small half-life / signal-to-noise grid;
- no per-factor hyperparameter optimization in the first pass.

Output:

\[
\hat f_{k,t}
=
\mathbb E[f_{k,t} \mid \mathcal F_t]
\]

\[
P_{k,t}
=
\operatorname{Var}(f_{k,t} \mid \mathcal F_t)
\]

Confidence-adjusted payoff forecast:

\[
\tilde f_{k,t}
=
\hat f_{k,t}
-
\lambda_P \sqrt{P_{k,t}}
\]

where \(\lambda_P\) is fixed in advance, for example \(0\), \(0.25\), or \(0.5\).

### Second Pass: Multivariate Kalman

After univariate models establish value, test:

\[
\mathbf y_t
=
\mathbf f_t
+
\boldsymbol\epsilon_t,
\quad
\boldsymbol\epsilon_t \sim \mathcal N(0, R)
\]

\[
\mathbf f_t
=
\Phi \mathbf f_{t-1}
+
\boldsymbol\eta_t,
\quad
\boldsymbol\eta_t \sim \mathcal N(0, Q)
\]

Purpose:

- account for correlation among style payoffs;
- prevent overweighting duplicate styles;
- estimate posterior covariance of style returns.

Do not start here. Multivariate Kalman has more parameters and higher overfit risk.

## Stock Scoring

Each stock has style exposure vector:

\[
\mathbf b_{i,t}
=
(b_{i,1,t}, b_{i,2,t}, \ldots, b_{i,K,t})
\]

Use cross-sectional z-scored exposures.

Score:

\[
s_{i,t}
=
\mathbf b_{i,t}^\top \tilde{\mathbf f}_t
\]

Optional uncertainty-aware score:

\[
s_{i,t}
=
\mathbf b_{i,t}^\top \hat{\mathbf f}_t
-
\lambda
\sqrt{
\mathbf b_{i,t}^\top P_t \mathbf b_{i,t}
}
\]

First version should use the diagonal approximation:

\[
\sqrt{
\sum_k b_{i,k,t}^2 P_{k,t}
}
\]

## Portfolio Constructions

Run at least three versions.

### Version A: Style-Seeking

Allow intended price-style exposure.

First-pass implementation:

- use `adv30m_clean_core_beta_full` for both long and short candidate pools;
- rank both sides with the same dynamic score \(s_{i,t} = \mathbf b_{i,t}^{\top}\tilde{\mathbf f}_t\);
- construct long candidates from the high-score tail;
- construct short candidates from the low-score tail.

Constraints:

- dollar neutral;
- beta matched;
- size neutral or size controlled;
- sector/SIC soft neutral;
- ADV and concentration controls;
- no constraint that neutralizes the selected price styles.

Purpose:

- test whether dynamic style allocation improves the current price-based strategy.

### Version B: Capped Style-Seeking

Allow style exposure, but cap single-style concentration:

\[
|w_t^\top b_{k,t}| \le c_k
\]

Purpose:

- avoid the model putting the whole book into one style when posterior is noisy.

### Version C: Style-Neutral Control

Force:

\[
w_t^\top B_t \approx 0
\]

Purpose:

- diagnostic only.
- If Version A works and Version C collapses, profits are style returns.
- If Version C also works, there may be true alpha or nonlinear residual structure.

## Evaluation

Primary comparison:

- current SOTA `size_hard_neutral`;
- static style score;
- EWMA style score;
- Kalman style score;
- capped Kalman style score;
- style-neutral control.

Metrics:

- gross annualized return;
- gross volatility;
- Sharpe without risk-free;
- max drawdown;
- rolling 60-session return positive rate;
- h10-equivalent return positive rate;
- turnover;
- beta exposure;
- size exposure;
- sector exposure;
- style exposure attribution.

Payoff-level diagnostics:

- style payoff posterior \(\hat f_{k,t}\);
- posterior uncertainty \(P_{k,t}\);
- realized \(y_{k,t}\);
- hit rate of posterior sign;
- calibration by posterior confidence bucket.

## Walk-Forward Protocol

Validation-only.

The initial sketch reserved history through 2016 before evaluating 2017-2019:

- Train through 2016, evaluate 2017.
- Train through 2017, evaluate 2018.
- Train through 2018, evaluate 2019.

That is useful as a year-slice diagnostic, but it is too narrow for the headline validation readout. The full corrected validation protocol is:

- use the h10 beta-full signal panel from its first available date, 2014-08-05;
- require 60 matured h10 payoff observations before trading;
- first tradable dynamic allocation date is therefore 2014-11-11;
- evaluate the headline strategy from 2014-11-11 through 2019-12-31;
- keep 2017/2018/2019 slices as diagnostics, not as the primary reported window.

Within the evaluation window:

- update state daily using only matured h10 payoff;
- generate scores from current posterior;
- build portfolio using only same-day exposures and posterior forecasts.

No test lockbox usage.

## Acceptance Criteria

Dynamic style allocation is promising only if:

- it beats current SOTA on validation gross return by a meaningful margin;
- it does not materially worsen max drawdown;
- rolling 60-session positive rate is comparable or better;
- the improvement is not concentrated in one short calendar window;
- posterior sign has real calibration value;
- style exposure attribution matches the intended payoff forecasts;
- parameters are simple and stable.

If Kalman improves standalone payoff but not optimizer-level portfolio, it is not yet useful.

If Kalman improves return only by increasing exposure to one style, require capped-style version to still work.

If Phase6P-like complexity is needed to beat SOTA, reject for overfit risk.

## Expected Failure Modes

1. State estimate lags true regime shifts.
2. \(Q/R\) overreacts and chases payoff noise.
3. Selected styles are too correlated, causing disguised concentration.
4. h10 overlapping payoff observations overstate confidence.
5. Validation windows have too few independent regimes.
6. Portfolio constraints eat the standalone payoff.
7. Transaction costs erase the improvement.

## Immediate Implementation Plan

Phase DS1: payoff panel

- Build daily h10 payoff series for pre-defined price styles.
- Enforce maturity lag.
- Save beta-residual, size/sector residual, and strict residual payoff panels.

Phase DS2: posterior estimators

- Implement expanding mean, EWMA, and univariate Kalman.
- Output posterior mean, posterior variance, and confidence-adjusted forecasts.

Phase DS3: standalone score test

- Score stocks by \(\mathbf b_i^\top \tilde{\mathbf f}\).
- Evaluate top-bottom h10 payoff, rank IC, and drawdowns.

Phase DS4: portfolio integration

- Feed dynamic style score into current optimizer.
- Compare Version A/B/C against SOTA.

Phase DS5: attribution and decision

- Decompose realized portfolio return into selected style exposures and realized style payoff.
- Decide whether dynamic style allocation is worth keeping as a price-style strategy.

## Decision Rule

This line is successful if it produces a better beta-neutral price-style strategy.

It should not be marketed internally as pure alpha. It is a controlled, probabilistic, dynamic style allocation strategy. Pure alpha remains the goal of the separate non-price track.

## Phase6T First Implementation Status

Implemented in `stockmachine.apps.run_pure_alpha_phase6t`.

The first version follows the clean symmetric setup:

- long and short both use `adv30m_clean_core_beta_full`;
- both sides use the same score \(s_{i,t} = \mathbf b_{i,t}^{\top}\tilde{\mathbf f}_t\);
- h10 payoff updates use only matured observations;
- portfolio construction is dollar neutral, beta matched, true-size neutral, and SIC2 soft neutral;
- test lockbox is not used.

Result memo:

- `docs/us-equities-pure-alpha-phase6t-dynamic-style-allocation-20260511.md`

Current interpretation:

- the corrected headline window is 2014-11-11 to 2019-12-31, not 2017-2019;
- SOTA dominates the first Phase6T variants after this window correction;
- the earlier 2017-only result was too favorable because it skipped the 2015-2016 drawdown regime;
- the Kalman first pass is weak and should be revisited only after payoff calibration diagnostics;
- this branch remains a style-seeking price strategy, not pure alpha.

## Phase6T K=1 Reversal-Only Status

Implemented via `PHASE6T_STYLE_SET=k1_reversal`.

Result memo:

- `docs/us-equities-pure-alpha-phase6t-k1-reversal-20260512.md`

Current interpretation:

- K=1 `static_mean` is the best reversal-only variant, with 4.69% annualized return, 7.13% annualized volatility, Sharpe 0.68, and max drawdown -11.92%;
- the adaptive K=1 estimators do not beat the static reversal prior;
- K=1 reduces drawdown versus K=4 but gives up too much return;
- the result supports the view that this family is still mostly reversal-style exposure, not a robust regime-adaptive model.
