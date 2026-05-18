# Phase7D Signed Graph Reversal Payoff Memo

Date: 2026-05-17

Scope: validation-only. The test lockbox is not used.

## Purpose

Phase7C showed that the graph can identify stronger and weaker reversal states,
but it did not establish that weak reversal states should be traded in reverse.
Phase7D tests a signed action space:

```text
y: trade reversal      = long losers - short winners
 0: stay flat
-y: trade continuation = long winners - short losers
```

The target remains one variable:

```text
reversal_payoff = long recent losers - short recent winners
```

If reversal payoff is truly negative, a negative action should produce positive
payoff.

## Signed Graph Structure

Phase7D adds explicit continuation mechanisms to the graph:

```text
trend_dominance -> winner_return_continuation -> ReversalPayoff (-)
capital_concentration -> capital_concentration_regime -> ReversalPayoff (-)
breadth_weakness -> narrow_market_leadership -> ReversalPayoff (-)
winner_fundamental_resilience -> winner_fundamental_upgrade -> ReversalPayoff (-)
winner_profit_resilience -> winner_fundamental_upgrade -> ReversalPayoff (-)
winner_size_leadership -> mega_cap_leadership -> ReversalPayoff (-)
winner_liquidity_leadership -> institutional_winner_accumulation -> ReversalPayoff (-)
```

It keeps the reliable reversal core:

```text
reversal_dislocation -> mispricing_convergence -> ReversalPayoff (+)
fundamental_fragility_spread -> loser_falling_knife_risk -> ReversalPayoff (-)
liquidity_fragility -> loser_liquidity_stress -> ReversalPayoff (-)
beta_fragility -> risk_off_fragility -> ReversalPayoff (-)
```

All score thresholds are walk-forward:

```text
threshold_t = quantile(score_{t-252} ... score_{t-1})
min history = 252 sessions
```

## Command

```text
$env:PYTHONPATH='src'; python -m stockmachine.apps.run_pure_alpha_phase7d
```

Default output:

```text
artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7d_signed_graph_reversal_payoff_20260517
```

## Result

Validation window:

```text
2014-11-11 through 2019-12-31
```

Panel size:

- joined stock-date rows: 915,312;
- daily graph-state rows: 1,351;
- evaluation rows: 1,282;
- graph nodes: 26;
- graph edges: 26.

The 252-session walk-forward threshold is ready on 1,099 evaluation sessions.
On that ready subset, always-on reversal payoff is:

```text
32.01 bps
```

Key action comparison:

| Score | Rule | Calendar mean signed payoff | Active mean signed payoff | Reverse days | Reverse underlying reversal payoff |
|---|---:|---:|---:|---:|---:|
| `signed_core_score` | always-on ready | 32.01 bps | 32.01 bps | 0 | n/a |
| `signed_core_score` | stop bottom tercile | 23.79 bps | 36.31 bps | 0 | n/a |
| `signed_core_score` | reverse bottom tercile only | -8.22 bps | -23.84 bps | 379 | +23.84 bps |
| `signed_core_score` | signed top/bottom tercile | 6.55 bps | 9.49 bps | 379 | +23.84 bps |
| `signed_full_graph_score` | reverse bottom tercile only | -10.40 bps | -30.99 bps | 369 | +30.99 bps |
| `concentration_signed_score` | reverse bottom tercile only | -17.09 bps | -48.53 bps | 387 | +48.53 bps |

## Interpretation

Phase7D does not validate reverse trading. In the live-valid walk-forward
bottom regimes, reversal payoff remains positive, so continuation trades lose.

This is strongest for the concentration score:

```text
concentration_signed_score bottom tercile reversal payoff = +48.53 bps
```

That means the current concentration proxies are not detecting a durable
negative reversal regime. They may be detecting exactly the opposite: broad
stress or crowded concentration states where losers later rebound strongly.

The reliable-core score still works as a quality filter:

```text
top tercile reversal payoff = 42.73 bps
bottom tercile reversal payoff = 23.84 bps
```

But the bottom tercile is not negative. It supports lower exposure or maybe a
no-trade band; it does not support shorting the reversal basket.

## Conclusion

The signed action space should stay in the research design, but current evidence
says:

```text
Do not reverse the trade from this graph yet.
```

The correct next step is not to force signed exposure, but to add stronger
continuation-specific evidence:

- analyst EPS revision / revenue revision for winners;
- industry-chain evidence for real technology adoption;
- ETF/fund flow concentration;
- short interest and borrow stress;
- option skew / crash-risk pricing;
- explicit mega-cap winner fundamental upgrade vs valuation overextension.

Only if those features produce walk-forward bottom regimes with negative
`reversal_payoff` should continuation exposure be promoted.
