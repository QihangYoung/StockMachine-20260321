# Phase7C Repaired Graph Reversal Payoff Memo

Date: 2026-05-17

Scope: validation-only. The test lockbox is not used.

## Purpose

Phase7B produced a useful negative result: the first hand-written graph-prior
score did not improve reversal payoff sorting. Before trying more complicated
models, Phase7C fixes two structural defects in the graph and adds a conservative
edge-reliability audit.

## Structural Repairs

Phase7B had two graph-shape problems:

1. `capital_concentration` was both a feature node and a mechanism node,
   creating a semantic self-loop in the exported graph.
2. `structural_loser_filter` was overloaded: insider support, fundamental
   fragility, and filing quality all pointed into the same mechanism node even
   though they are different economic channels.

Phase7C repairs this as:

```text
capital_concentration -> capital_concentration_regime -> ReversalPayoff
insider_support_spread -> insider_support -> ReversalPayoff
fundamental_fragility_spread -> fundamental_loser_risk -> ReversalPayoff
filing_red_flag_spread -> filing_quality_risk -> ReversalPayoff
```

Other mechanism nodes are also made more explicit:

```text
trend_dominance -> trend_continuation_risk -> ReversalPayoff
liquidity_fragility -> liquidity_stress -> ReversalPayoff
beta_fragility -> risk_off_fragility -> ReversalPayoff
breadth_weakness -> risk_off_fragility -> ReversalPayoff
sector_shock_concentration -> sector_shock_risk -> ReversalPayoff
```

## Edge Reliability Audit

Phase7C keeps the graph prior dominant. Data is allowed only to audit existing
edges, not invent new edges.

For each edge, the app computes:

```text
prior_contribution_e,t = sign_e * weight_e * z(feature_e,t)
```

Then it estimates edge reliability using only matured historical payoff:

```text
reliability_corr_e,t
= rolling_corr(prior_contribution_e,t-h, reversal_payoff_t-h)
```

The first audited score is prior-preserving:

```text
effective_contribution_e,t
= prior_contribution_e,t * (0.8 + 0.2 * reliability_corr_e,t)
```

This keeps structural direction dominant and only changes edge strength within
a bounded range.

The second audited score is a stricter diagnostic:

```text
positive_reliability_contribution_e,t
= prior_contribution_e,t if reliability_corr_e,t >= 0 else 0
```

This is not a production rule; it is an edge audit that asks whether bad edges
are hurting the graph.

## Validation Question

The test remains:

```text
Can the repaired graph rank future reversal payoff states?
```

Useful output is not a high standalone Sharpe. Useful output is monotonic or
economically sensible payoff sorting across score bins, plus improved payoff
when the graph says reversal should be trusted.

## Command

```text
python -m stockmachine.apps.run_pure_alpha_phase7c
```

Default output:

```text
artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7c_repaired_graph_reversal_payoff_20260517
```

## First Run Result

The repaired graph was first run on the validation window:

```text
2014-11-11 through 2019-12-31
```

Panel size:

- joined stock-date rows: 915,312;
- daily graph-state rows: 1,351;
- evaluation rows: 1,282;
- graph nodes: 20;
- graph edges: 20.

Key comparison:

| Rule | Mean payoff | Hit rate | Lift vs always-on |
|---|---:|---:|---:|
| `always_on` | 23.49 bps | 54.99% | 0.00 bps |
| `repaired_graph_prior_score_top_tercile` | 25.96 bps | 57.71% | +2.47 bps |
| `reliability_80_20_score_top_tercile` | 27.94 bps | 57.94% | +4.45 bps |
| `positive_reliability_score_top_tercile` | 29.78 bps | 57.01% | +6.29 bps |
| `reliable_core_score_top_tercile` | 41.22 bps | 60.05% | +17.73 bps |
| `reliable_core_score_bottom_tercile` | 8.08 bps | 49.30% | -15.41 bps |

Important correction: this first-run top/bottom tercile table used full-window
validation quantiles, so it is diagnostic only. It is not a valid live
activation rule.

## Walk-Forward Rerun

The implementation was updated so top/bottom tercile activation uses only prior
daily scores:

```text
threshold_t = quantile(score_{t-252} ... score_{t-1})
min history = 60 sessions
```

The rerun keeps the same validation window and lockbox status:

```text
2014-11-11 through 2019-12-31
test lockbox used: False
```

Walk-forward decision metrics:

| Rule | Sessions | Coverage | Mean payoff | Hit rate | Lift vs always-on |
|---|---:|---:|---:|---:|---:|
| `always_on` | 1282 | 100.00% | 23.49 bps | 54.99% | 0.00 bps |
| `repaired_graph_prior_score_walk_forward_top_tercile` | 440 | 34.32% | 27.11 bps | 57.27% | +3.62 bps |
| `reliability_80_20_score_walk_forward_top_tercile` | 441 | 34.40% | 26.91 bps | 56.92% | +3.42 bps |
| `positive_reliability_score_walk_forward_top_tercile` | 429 | 33.46% | 24.92 bps | 55.71% | +1.43 bps |
| `reliable_core_score_walk_forward_top_tercile` | 464 | 36.19% | 39.72 bps | 59.70% | +16.23 bps |
| `reliable_core_score_walk_forward_bottom_tercile` | 471 | 36.74% | 9.03 bps | 49.89% | -14.46 bps |

The useful improvement is the `reliable_core_score`, which keeps only the
structural edges that both make financial sense and showed positive reliability
in the rolling audit:

```text
fundamental_fragility_spread -> fundamental_loser_risk
reversal_dislocation -> mispricing_convergence
liquidity_fragility -> liquidity_stress
beta_fragility -> risk_off_fragility
```

The reliable-core score has stronger payoff sorting:

```text
Spearman(score, payoff) = 0.1020
```

Its top score bin produced:

```text
63.54 bps mean payoff
64.59% hit rate
```

Its bottom score bin produced:

```text
-2.96 bps mean payoff
47.47% hit rate
```

Interpretation:

The repaired full graph is only modestly better than Phase7B. The real gain
comes from pruning unreliable edges. That supports the graph approach, but with
a stricter principle: do not assume every plausible structural channel is useful
for this specific target. The graph should first pass an edge-reliability audit,
then only trusted channels should enter the activation score.

Edges that look useful in this target:

- fundamental fragility;
- raw reversal dislocation;
- liquidity fragility;
- beta fragility.

Edges that need redesign or exclusion before promotion:

- insider support;
- sector shock concentration;
- filing red flags;
- trend dominance;
- breadth weakness;
- capital concentration as currently measured.

Next step:

Promote `reliable_core_score` from diagnostic to a soft activation overlay
candidate. The first overlay should not disable reversal entirely. It should map
score quantiles to selector intensity, for example:

```text
bottom tercile: 0.25x reversal exposure
middle tercile: 0.75x reversal exposure
top tercile: 1.25x reversal exposure
```

Then compare the resulting portfolio path against always-on SOTA inside the
same validation-only strict h10 framework.
