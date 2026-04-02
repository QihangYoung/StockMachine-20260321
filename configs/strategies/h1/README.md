## H1 Strategy Profiles

This folder is reserved for 1-session / daily-rebalance strategy lines.

The intent is to keep horizon-1 research, execution, and governance isolated from
the existing `h5` swing stack.

Current reference docs:

- [h1 development plan](/E:/CodeX/StockMachine-260321/docs/h1-development-plan.md)
- [h1 research protocol](/E:/CodeX/StockMachine-260321/docs/h1-research-protocol.md)

First research entrypoint:

```text
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_us_equities_h1_baseline --strategy-project us_equities_h1
```

The current `h1` baseline uses a turnover-control v2 shell by default:

- `rank buffer`: incumbents can stay slightly below the raw cutoff
- `entry buffer`: new names must clear a stricter rank threshold
- `max_new_names_per_rebalance`: cap on same-day replacements
- `min_weight_change`: small per-name weight changes are skipped
