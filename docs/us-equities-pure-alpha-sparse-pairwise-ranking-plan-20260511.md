# US Equities Pure Alpha Sparse Pairwise Ranking Plan

Generated: 2026-05-11

## Purpose

This memo defines a validation-only experiment for replacing the current hand-built linear/transparent selectors with a nonlinear sparse pairwise ranking model.

The research question is:

> Can nonlinear price-state features rank future multi-factor residual winners and losers inside the same universe, after stripping out the public factor payoffs that we do not want to rely on?

This is not a test-lockbox experiment. The test set remains locked.

## Why This Experiment

The current SOTA still relies heavily on reversal and anti-momentum style exposure. That is explainable, but it creates a product risk: if the reversal or anti-momentum payoff changes sign, the strategy can suffer even when beta, sector, and size neutrality are well controlled.

If the score is only a linear combination of the same style factors we later neutralize, then the optimizer has little useful signal left:

$$
s_t = B_t \theta
$$

and if:

$$
w_t^\top B_t = 0
$$

then:

$$
w_t^\top s_t = w_t^\top B_t \theta = 0
$$

So a stricter pure-alpha path needs one of the following:

- information outside the neutralized factor space;
- nonlinear interactions among price-state variables;
- conditional relationships, such as "winner that recently cracked" or "loser that stabilized";
- non-price features.

This experiment focuses on the second and third paths: nonlinear price-state ranking.

## Core Idea

Instead of predicting the absolute future residual return of each stock, we predict pairwise preference:

$$
P(y_{i,t} > y_{j,t})
$$

where:

$$
y_{i,t}
$$

is stock \(i\)'s future residual return label over horizon \(h\), and \(i,j\) are two stocks from the same date and same candidate universe.

The raw label is a sparse pairwise preference matrix:

$$
Y_{ij,t}
=
\begin{cases}
1, & y_{i,t} - y_{j,t} > \tau_t \\
0, & y_{j,t} - y_{i,t} > \tau_t \\
\text{missing}, & |y_{i,t} - y_{j,t}| \le \tau_t \text{ or pair not sampled}
\end{cases}
$$

We intentionally keep this matrix sparse. Full pair enumeration is too large and too noisy.

For a universe of \(N\) names, full daily pair count is:

$$
\frac{N(N-1)}{2}
$$

For \(N=700\), that is roughly 245,000 pairs per day. Across roughly 1,350 validation sessions, full enumeration would produce hundreds of millions of pairs. Most of those pairs have tiny realized residual differences and mostly teach the model noise.

## Label Construction

The supervised target should be a strict realized residual label:

$$
y_{i,t}
=
r_{i,t \to t+h}
-
\widehat{r}^{risk/style}_{i,t \to t+h}
$$

The preferred MVP horizon is:

$$
h = 10
$$

because the current strategy is evaluated as rolling h10 sleeves, and prior horizon diagnostics showed h10 is near the strongest practical horizon for the current short-cycle reversal family.

The residual model should use only exposures known at signal date \(t\):

$$
X_{i,t}
=
[
\beta_{i,t},
\text{SIC2/SIC4 industry dummies},
\log(\text{market cap})_{i,t},
\log(\text{ADV})_{i,t},
\text{volatility}_{i,t},
\text{reversal/momentum exposures}_{i,t}
]
$$

Then estimate daily cross-sectional factor payoff after the fact:

$$
r_{t \to t+h}
=
X_t \lambda_{t \to t+h}
+
u_{t \to t+h}
$$

and use:

$$
y_{i,t} = u_{i,t \to t+h}
$$

as the pairwise label source.

This distinction matters: the future return is allowed in the label, but all exposures used to define the label must be known at time \(t\).

## Pair Sampling

We should not sample all pairs. The MVP should use sparse, balanced, high-information pairs.

Recommended daily sampling:

- Universe: start with the current SOTA universe pair, long candidate universe `top1000_clean_core_beta_full` and short candidate universe `adv30m_clean_core_beta_full`, but train one common ranking model per eligible universe row.
- Per date pair budget: 5,000 to 20,000 pairs.
- Margin filter: keep pairs only when:

$$
|y_{i,t} - y_{j,t}| > \tau_t
$$

where:

$$
\tau_t = 0.25 \times \text{IQR}(y_{\cdot,t})
$$

or a similar daily robust threshold.

- Balance: enforce approximately 50/50 labels after random orientation. If \(y_i > y_j\), randomly emit either \((i,j,1)\) or \((j,i,0)\) so the model cannot learn position/order artifacts.
- Hard negatives/positives: oversample pairs involving top/bottom residual deciles, because portfolio construction cares most about tails.
- Near-boundary sample: include a smaller share of medium-margin pairs so the model does not only learn obvious extremes.
- Date grouping: keep date identifiers for purged time split and diagnostic grouping.

The sparse matrix is therefore not "missing at random"; it is intentionally focused on economically useful ordering differences.

## Feature Design

For stock-level raw features \(z_{i,t}\), use only information available at \(t\):

- trailing returns: 1d, 3d, 5d, 10d, 20d, 40d, 60d;
- beta-residual trailing returns over similar windows;
- volatility: 10d, 20d, 60d;
- residual volatility;
- drawdown depth: 20d, 60d;
- rebound/recovery confirmation: 3d and 5d after recent drawdown;
- recent weakness after strength;
- volume shock and dollar-volume trend;
- size and ADV proxies;
- beta and realized-volatility proxies;
- sector/industry identifiers as categorical or one-hot if the model supports them safely.

For each pair, first MVP input should be anti-symmetric difference:

$$
\Delta z_{ij,t} = z_{i,t} - z_{j,t}
$$

This has an important consistency property:

$$
\Delta z_{ji,t} = -\Delta z_{ij,t}
$$

If the model is well behaved, then:

$$
P(i > j) + P(j > i) \approx 1
$$

Second-stage features may add nonlinear pair features:

$$
|\Delta z_{ij,t}|
$$

and selected interactions:

$$
z_{i,k,t} z_{i,l,t} - z_{j,k,t} z_{j,l,t}
$$

But MVP should start with differences only plus a tree model. Tree splits already create nonlinear interactions.

## Model

Recommended MVP:

- Model: LightGBM binary classifier or sklearn HistGradientBoostingClassifier if dependency simplicity is preferred.
- Objective: binary logloss on sparse pair labels.
- Input: pairwise feature differences.
- Output:

$$
\hat{p}_{ij,t} = P(y_{i,t} > y_{j,t})
$$

Training must use time splits, not random pair splits. A pair from 2018 must not be used to train a model evaluated on 2017, and pairs from the same date must never be split across train and validation folds.

Recommended validation split inside the existing validation window:

- Train: 2014-08-05 to 2017-12-31
- Model-selection validation: 2018-01-01 to 2018-12-31
- Final validation holdout inside validation: 2019-01-01 to 2019-12-31

This still does not touch the test lockbox.

## Rank Aggregation

The model produces pairwise probabilities, but the portfolio optimizer needs one score per stock per date.

For each date, sample or evaluate pairwise probabilities against a reference set and compute Borda-style score:

$$
s_{i,t}
=
\frac{1}{|\mathcal{J}_{i,t}|}
\sum_{j \in \mathcal{J}_{i,t}}
(\hat{p}_{ij,t} - 0.5)
$$

where:

$$
\mathcal{J}_{i,t}
$$

is the set of comparison names for stock \(i\) on date \(t\).

This is not a hard topological sort. A hard topological sort assumes pairwise preferences are acyclic, but financial labels will produce cycles:

$$
A > B,\quad B > C,\quad C > A
$$

So the correct target is maximum-consistency ranking, not perfect ordering. Borda aggregation is a robust MVP approximation.

Second-stage alternatives:

- Bradley-Terry score fit per date;
- spectral ranking/PageRank from pairwise win probabilities;
- Kemeny-style rank aggregation approximation.

## Portfolio Integration

The ranking score should feed the existing optimizer:

$$
\max_{w_t}
w_t^\top s_t
-
\lambda \cdot \text{turnover}_t
$$

subject to:

$$
w_t^\top \beta_t = 0
$$

$$
w_t^\top \text{size}_t = 0
$$

$$
\text{SIC2 exposure controlled}
$$

and, for the style-neutral experiment:

$$
w_t^\top B^{reversal/momentum}_t \approx 0
$$

The important test is whether the pairwise model still has value after neutralizing the linear public style exposures that originally powered the hand-built selectors.

## Evaluation

We need evaluate at three levels.

### 1. Pairwise Model Quality

Metrics:

- pairwise AUC;
- logloss;
- accuracy with margin buckets;
- calibration of \(\hat{p}_{ij,t}\);
- performance by year;
- performance by market state;
- performance by pair margin.

Useful baseline:

- random pair classifier;
- linear logistic pair model;
- current hand-built selector converted to pairwise preference.

### 2. Ranking Quality

Metrics by date:

- Spearman rank IC versus residual label;
- top-minus-bottom residual spread;
- NDCG@top/bottom tails;
- decile monotonicity;
- stability of selected names.

The key question is not whether the whole ranking is perfect. The optimizer mostly cares whether top and bottom tails are better than random.

### 3. Portfolio Quality

Compare validation-only portfolio results against:

- current SOTA `size_hard_neutral`;
- current SOTA plus style-neutral constraints;
- linear residualized-score baseline;
- pairwise sparse model without style-neutral constraints;
- pairwise sparse model with reversal/momentum style-neutral constraints.

Portfolio metrics:

- gross annualized return;
- annualized volatility;
- Sharpe;
- max drawdown;
- rolling 60-session return;
- yearly return;
- turnover;
- realized beta;
- factor exposure;
- long/short leg attribution;
- residual contribution attribution.

The highest-value result would be:

> Pairwise model retains meaningful return after reversal/momentum neutralization.

That would indicate we have found information beyond simply harvesting the reversal/momentum payoff.

## Anti-Overfitting Rules

- No test-set access.
- No random row split; split by time.
- No feature using future prices.
- No label exposure using future-built betas or future-known fundamentals.
- No hyperparameter search against full validation curve.
- Predefine pair sampling rules.
- Predefine the style-neutral constraints before looking at portfolio results.
- Report failure clearly if style-neutral performance collapses.

## Expected Failure Modes

1. The model learns the same reversal/momentum style in nonlinear clothing.

Mitigation: run style exposure diagnostics and style-neutral portfolio constraints.

2. Pair labels are too noisy.

Mitigation: margin filter, robust residual labels, top/bottom oversampling.

3. Pairwise accuracy improves but portfolio PnL does not.

Mitigation: evaluate tail ranking and portfolio objective alignment, not just AUC.

4. Rank aggregation creates unstable daily scores.

Mitigation: Borda score smoothing, comparison reference-set stability, turnover-aware optimizer.

5. The model overfits 2015/2019 specific stress windows.

Mitigation: year-by-year reporting and simple model complexity cap.

## MVP Deliverables

Phase6P should produce:

- residual label panel for validation only;
- sparse pair training file with date, symbol_i, symbol_j, label, margin, and pair features;
- trained pairwise classifier artifact;
- daily pairwise ranking score panel;
- model-quality diagnostics;
- ranking-quality diagnostics;
- portfolio comparison against current SOTA;
- style exposure report;
- memo with pass/fail interpretation.

## Go / No-Go Criteria

This path is worth continuing only if at least one of the following is true:

- pairwise model beats current hand-built selector in rank IC or tail residual spread;
- portfolio improves validation path quality without increasing style exposure;
- style-neutral pairwise portfolio preserves a meaningful fraction of SOTA return;
- drawdown windows become less synchronized with reversal/momentum payoff undercrosses.

If none hold, we should conclude that current price-only information is mostly public style payoff, and non-price features become the next necessary research input.

