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

## Second Pass: Richer Features and Early Stopping

The second pass keeps the same strict protocol but upgrades three things inside
our own sequence builder:

- richer sequence-specific features derived from the existing research frame
- validation-aware early stopping instead of fixed-epoch-only fitting
- a more peer-like Transformer head with learnable positional embeddings and
  pooled sequence output

Output root:

- `artifacts/p2_sequence_migration_v2`

Summary:

| model | total_return | annualized_return | sharpe | max_drawdown | benchmark_total_return |
|---|---:|---:|---:|---:|---:|
| `lstm_regressor` | `24.16%` | `21.09%` | `0.92` | `-15.96%` | `17.46%` |
| `transformer_regressor` | `6.78%` | `5.97%` | `0.37` | `-22.07%` | `17.46%` |

Interpretation:

- `lstm_regressor` improved materially and is now at least credible as a
  research-only candidate
- `transformer_regressor` still trails the current strict top models, even
  after the richer feature and training pass
- the main sequence-model bottleneck is no longer "can it run in our stack",
  but "can it add enough alpha to justify its extra complexity"

## V1 vs V2 Snapshot

| model | v1 total_return | v2 total_return | v1 sharpe | v2 sharpe |
|---|---:|---:|---:|---:|
| `lstm_regressor` | `0.83%` | `24.16%` | `0.12` | `0.92` |
| `transformer_regressor` | `8.06%` | `6.78%` | `0.47` | `0.37` |

## Next Steps

The next sequence-model upgrades should focus on:

1. longer lookback and hyperparameter sweeps for the now-credible `lstm_regressor`
2. another dedicated Transformer pass before promotion to the strict shortlist
3. sequence-model-specific ensembles only after single-model quality improves
4. optional ranking or classification sequence variants after the regression path stabilizes
