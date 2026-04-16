# Dynamic Sleeve Growth-Engine Observation

Date: 2026-04-16

This note summarizes a validation-only observation from the `growth ETF + LQD` sleeve experiments.

The current question is whether a monthly walk-forward dynamic sleeve can generalize better than a static sleeve. The answer appears to depend strongly on the equity growth engine.

## Scope

All results in this note use validation data only.

- Validation window: `2014-08-05 ~ 2019-12-31`
- Lockbox test window: not used
- Sleeve structure: `equity ETF + LQD`, with `BIL` as residual cash when volatility scaling applies
- Dynamic objective: maximize trailing Sharpe in the lookback window
- Dynamic execution shape:
  - strong damping
  - discrete execution rungs
  - monthly rebalance
  - no leverage in this experiment family

Relevant artifacts:

- `artifacts/spy_lqd_dynamic_sleeve_current_params_20260416/summary_with_k3.csv`
- `artifacts/dynamic_sleeve_equity_floor_check_20260416/floor_sweep_summary.csv`
- `artifacts/vgt_lqd_discrete_sleeve_vt15_20260416/summary.csv`
- `artifacts/soxx_lqd_dynamic_framework_vt15_20260416/summary.csv`

## Main Observation

The same dynamic sleeve framework behaves differently across growth engines.

For `SPY`, the dynamic sleeve can beat the static sleeve on Sharpe because it reduces equity exposure and improves risk efficiency.

For `VGT` and `SOXX`, the dynamic sleeve often trails the static sleeve because it can cut exposure to assets with strong long-run trend and right-tail payoff.

In short:

- `SPY`: dynamic sleeve acts like useful de-risking.
- `VGT / SOXX`: dynamic sleeve can become over-defensive and interrupt strong trend exposure.

## SPY Result

With the current dynamic framework, adding `K=3m` produced the strongest `SPY+LQD` dynamic result.

| Strategy | Annual Return | Annual Vol | Sharpe | Max Drawdown | Avg SPY |
|---|---:|---:|---:|---:|---:|
| `SPY+LQD discrete dynamic 3m` | `8.13%` | `6.79%` | `1.197` | `-12.35%` | `44.87%` |
| `SPY+LQD discrete dynamic 6m` | `7.90%` | `7.11%` | `1.111` | `-13.64%` | `46.57%` |
| `SPY+LQD static ref vt15` | `10.42%` | `10.00%` | `1.042` | `-15.00%` | `74.83%` |

Interpretation:

The dynamic version does not improve absolute return. It improves Sharpe by reducing SPY exposure, lowering realized volatility, and reducing drawdown.

This suggests that `SPY` is weak enough as a growth engine that de-risking can improve its risk-adjusted profile.

## VGT Result

For `VGT`, the unconstrained dynamic sleeve was initially weaker than the static sleeve. Adding an equity floor materially improved the dynamic version.

| Strategy | Annual Return | Annual Vol | Sharpe | Max Drawdown | Avg VGT |
|---|---:|---:|---:|---:|---:|
| `VGT discrete 3m floor30` | `12.77%` | `9.59%` | `1.332` | `-14.25%` | `52.30%` |
| `VGT discrete 3m floor00` | `11.28%` | `8.62%` | `1.308` | `-14.08%` | `44.20%` |
| `VGT+LQD static vt15` | `13.05%` | `9.86%` | `1.323` | `-13.69%` | approximately `55.88%` risky-sleeve VGT |

Interpretation:

A mild floor, especially `floor30`, helps because it prevents the dynamic controller from reducing exposure too much after short-term volatility or drawdown.

This is important: the dynamic layer can add value, but only if it is not allowed to under-own a high-quality growth engine for too long.

## SOXX Result

For `SOXX`, adding a floor tends to increase return but reduce Sharpe.

| Strategy | Annual Return | Annual Vol | Sharpe | Max Drawdown | Avg SOXX |
|---|---:|---:|---:|---:|---:|
| `SOXX discrete 6m floor00` | `11.75%` | `9.31%` | `1.261` | `-12.96%` | `35.84%` |
| `SOXX discrete 6m floor30` | `13.59%` | `11.42%` | `1.189` | `-15.07%` | `45.93%` |
| `SOXX discrete 3m floor40` | `14.61%` | `12.58%` | `1.161` | `-14.51%` | `53.24%` |
| `SOXX+LQD static vt15` | `13.22%` | `9.97%` | `1.326` | `-12.02%` | approximately `41.46%` risky-sleeve SOXX |

Interpretation:

`SOXX` is a more volatile and more concentrated growth engine. Forcing higher SOXX exposure raises return, but it also magnifies volatility and drawdown enough to hurt Sharpe.

For `SOXX`, the dynamic sleeve should be treated as a return-seeking variant only if the product objective explicitly accepts deeper drawdowns.

## Why The Behavior Differs

The dynamic sleeve's main strength is risk control. It is good at reducing exposure when recent Sharpe deteriorates.

That helps when the equity engine is broad and less powerful, as with `SPY`.

It can hurt when the equity engine has strong long-run trend and right-tail payoff, as with `VGT` or `SOXX`, because the controller may cut exposure after temporary turbulence and then miss the rebound or continuation.

Therefore:

- For `SPY`, dynamic Sharpe optimization improves the risk-adjusted profile by de-risking.
- For `VGT`, dynamic Sharpe optimization needs a mild equity floor to avoid excessive de-risking.
- For `SOXX`, higher floors are mainly return enhancers, not Sharpe enhancers.

## Working Conclusion

We should not treat dynamic sleeve design as universally good or bad.

The correct design depends on the growth engine:

- `SPY`: dynamic sleeve is useful as a defensive Sharpe improver.
- `VGT`: dynamic sleeve is promising with a modest equity floor.
- `SOXX`: dynamic sleeve should remain a high-return, higher-risk branch; static sleeve is still the cleaner high-Sharpe anchor.

Current most useful dynamic candidate from this set:

- `VGT+LQD discrete 3m floor30`

Current static high-Sharpe anchors:

- `VGT+LQD static vt15`
- `SOXX+LQD static vt15`

Next research should avoid broad routing and avoid opening the test lockbox. If we continue dynamic research, the most valuable direction is a narrow validation-only check around:

- `VGT`
- `K=3m`
- strong damping
- 5-rung execution
- equity floor around `30%`

