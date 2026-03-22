# Sequence Expert Migration

This note records the first migration pass for sequence-model experts inspired by
the peer multi-expert project.

## Scope

The first pass intentionally migrates only the model layer:

- `lstm_regressor`
- `transformer_regressor`

It does not migrate the peer project's CSV runtime, broker code, or state store.
The current implementation reuses our own:

- silver-table research input
- strict walk-forward protocol
- white-box portfolio and execution assumptions
- artifact and leaderboard workflow

## Shared Contract

The first pass keeps the feature contract conservative:

- reuse the existing 12 baseline tabular features
- form one rolling sequence window per symbol
- predict the same 5-session excess-return target used by the rest of the model zoo

Both sequence models now emit the same prediction schema as the existing
regressors and rankers:

- `date`
- `symbol`
- `score`
- `confidence`
- `target`
- `future_return`
- `benchmark_future_return`
- `model`

## Dependency Strategy

PyTorch is optional instead of a hard dependency for the whole repository.

- optional extra: `sequence`
- package: `torch`

This keeps the paper/runtime stack lightweight while still allowing sequence
research in environments that install the extra.

## Code Paths

- sequence builders: `src/stockmachine/research/builders/sequence_models.py`
- alpha registry: `src/stockmachine/alpha/registry.py`
- research pipeline integration: `src/stockmachine/research/us_equities_baseline.py`

## First Strict Backtest

Output root:

- `artifacts/p2_sequence_migration`

Summary:

| model | total_return | annualized_return | sharpe | max_drawdown | benchmark_total_return |
|---|---:|---:|---:|---:|---:|
| `lstm_regressor` | `0.83%` | `0.73%` | `0.12` | `-14.14%` | `17.46%` |
| `transformer_regressor` | `8.06%` | `7.09%` | `0.47` | `-15.35%` | `17.46%` |

Interpretation:

- the first-pass `transformer_regressor` is usable but not competitive with the
  current strict top ensemble group
- the first-pass `lstm_regressor` is too weak to promote beyond research
- both models now clear the engineering bar of "can train, predict, and backtest
  under the existing strict protocol"

## Next Steps

The next sequence-model upgrades should focus on:

1. richer sequence-specific features
2. better validation and early-stopping logic
3. longer lookback and hyperparameter sweeps
4. sequence-model-specific ensembles only after single-model quality improves
