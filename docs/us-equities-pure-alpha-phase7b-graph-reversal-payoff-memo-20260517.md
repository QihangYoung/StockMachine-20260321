# Phase7B Graph-Prior Reversal Payoff Memo

Date: 2026-05-17

Scope: validation-only. The test lockbox is not used.

## Question

We want to predict whether the current market state supports a positive
reversal payoff inside a fixed universe:

\[
E[R^{rev}_{t,t+h} \mid \mathcal{F}_t, G, K] > 0
\]

Here \(G\) is a knowledge graph / asset relation graph and \(K\) is structural
financial prior. The target is not single-name direction. The target is whether
the reversal mechanism is theoretically active now.

For the first experiment, define:

```text
reversal payoff(t)
= mean(forward beta-residual return of recent losers)
  - mean(forward beta-residual return of recent winners)
```

Recent losers are the top quantile by `reversal_5d`; recent winners are the
bottom quantile by `reversal_5d`.

## Hypothesis

Reversal payoff is positive when recent losers are more likely to be temporary
mispricing / liquidity overreaction candidates.

Reversal payoff is negative when recent losers are more likely to be structural
losers, or when winners are being reinforced by capital concentration, trend
dominance, or genuine improvement.

The graph-prior score therefore uses mechanism nodes, not just price nodes:

- `reversal_dislocation`: recent winner-loser return spread is large.
- `insider_support`: loser basket has better insider support than winner basket.
- `fundamental_fragility`: loser basket has worse balance-sheet/profit stress.
- `filing_red_flags`: loser basket has more SEC filing red-flag pressure.
- `trend_dominance`: winners are also stronger intermediate-term winners.
- `capital_concentration`: size-weighted recent return beats equal-weight return.
- `liquidity_fragility`: loser basket is less liquid than winner basket.
- `beta_fragility`: loser basket has higher beta than winner basket.
- `sector_shock_concentration`: losers are concentrated in a few SIC sectors.
- `breadth_weakness`: market breadth is weak.

## Graph Representation

The MVP graph has four node types:

- `asset`: stock-level universe members.
- `structural_feature`: observable state variables on baskets or the market.
- `mechanism`: latent economic interpretation.
- `target`: `ReversalPayoff`.

Edges are typed structural priors:

```text
structural_feature -> mechanism -> ReversalPayoff
```

Each edge has:

- `sign`: whether higher feature values support or hurt reversal payoff.
- `weight`: hand-set prior strength.
- `channel`: economic transmission channel.

This is intentionally not a learned graph. It is the explicit structural prior
we want to test.

## Experiment

The first runnable experiment is:

```text
python -m stockmachine.apps.run_pure_alpha_phase7b
```

Default universe:

```text
top1000_clean_core_beta_full
```

Default data:

- price/style panel: Phase3 h10 signal panel;
- size graph feature: Phase6J true-size panel;
- filing and insider graph features: non-price Phase1 two-factor panel;
- fundamental graph features: non-price Phase4 companyfacts panel;
- SIC sector graph: cached SEC submissions mapping.

## Validation

The experiment outputs:

- daily reversal payoff target;
- graph node states;
- graph edge priors;
- daily graph-prior scores;
- score-bin payoff tables;
- decision-rule comparison;
- contribution diagnostics.

The key validity checks are:

1. High graph-prior score should have higher realized reversal payoff than low
   graph-prior score.
2. `graph_prior_positive` should improve average payoff or hit rate versus
   `always_on`.
3. Graph score should beat a deterministic random-sign graph sanity check.
4. Useful output is monotonic payoff sorting, not necessarily high raw Sharpe.

## Guardrails

- This is validation-only.
- No test-window performance is read.
- The graph is a fallible structural prior, not truth.
- First pass is a mechanism-state diagnostic, not a promoted trading overlay.
- If useful, the next pass should convert graph score into a soft activation
  weight for the existing reversal selector.

## Interpretation Standard

Success does not mean "the graph predicts prices." Success means:

```text
The graph helps decide when reversal payoff should be trusted.
```

If the graph only restates price-only lagged payoff, it is not enough. If it
adds sorting power through structural non-price evidence, it becomes a credible
state variable for a later portfolio overlay.

## First Run Result

The first run was completed on the validation window:

```text
python -m stockmachine.apps.run_pure_alpha_phase7b
```

Artifacts:

```text
artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7b_graph_reversal_payoff_20260517
```

Summary:

- joined stock-date panel rows: 915,312;
- daily graph-state rows: 1,351;
- evaluation rows: 1,282;
- graph nodes: 17;
- graph edges: 20.

The initial hand-weighted graph prior did not validate:

- `always_on` reversal payoff: 23.49 bps mean, 54.99% hit rate;
- `graph_prior_positive`: 22.66 bps mean, 55.43% hit rate;
- `graph_prior_top_tercile`: 23.35 bps mean, 56.31% hit rate;
- `graph_prior_bottom_tercile`: 29.74 bps mean, 54.80% hit rate;
- graph score Spearman vs payoff: 0.0035;
- random-sign graph Spearman vs payoff: 0.0798.

Interpretation:

The current graph-prior construction is not yet useful as an activation signal.
The validation window already rewards reversal strongly on average, and the
hand-written structural score does not improve the payoff sorting. This does
not reject the graph approach; it rejects the first edge-weight/sign
specification.

Most useful diagnostics from the first run:

- lower loser fundamental fragility aligns with better reversal payoff;
- pure reversal dislocation has weak positive information;
- the insider-support edge appears mis-specified or too noisy in this form;
- trend/sector concentration edges may be suppressing good reversal states
  rather than only filtering bad ones.

Next experimental move:

Keep the graph structure, but add a prior-preserving edge audit:

```text
effective edge weight
= 80% structural prior weight
  + 20% expanding historical edge reliability
```

The key guardrail is that historical payoff may only calibrate existing edge
reliability; it should not invent a new black-box timing model.
