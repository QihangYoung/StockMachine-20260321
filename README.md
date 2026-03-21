# StockMachine

StockMachine is a research-first trading system scaffold with a strict split
between prediction, risk, and execution.

The repository is intentionally starting from architecture and documentation
instead of implementation details. The goal is to make the core boundaries
clear before any model, backtest, or broker code grows around them.

## Design Principles

- Prediction produces signals, not orders.
- Risk and portfolio logic remain white-box and auditable.
- Execution is a separate concern from alpha generation.
- Backtest, paper trading, and live trading share the same contracts.
- Logs, metrics, and post-trade review are part of the system, not add-ons.

## Layers

1. `alpha`: find opportunities and emit ranked signals.
2. `portfolio` and `risk`: decide whether to act and how large to size.
3. `execution`: convert approved targets into broker-friendly orders.

## Repository Layout

```text
configs/                  Runtime and environment configuration
docs/                     Architecture and research notes
src/stockmachine/         Python package
tests/                    Test packages reserved for each subsystem
```

See [docs/architecture.md](docs/architecture.md) for the full module map,
[docs/us-equities-v1.md](docs/us-equities-v1.md) for the frozen first scope,
and [docs/model-design.md](docs/model-design.md) for predictive model framing.
For Alpaca setup and credential checks, see [docs/alpaca-setup.md](docs/alpaca-setup.md).

## Current Status

This repository currently contains:

- project skeleton
- architecture notes
- predictive model discussion notes
- baseline domain contracts
- US equities v1 scope and validation protocol
- Alpaca market-data ingestion and silver-table loading
- baseline US-equities research and backtest flow
- Alpaca paper-trading adapter, account sync, local ledger, and reconciliation
- first paper runner with dry-run support and structured reports

It does not yet contain:

- fully managed websocket-based broker callbacks
- production-grade live trading operations
- multi-broker execution support
- full online model serving lifecycle

## Data Strategy

The project is planned around a hybrid data strategy instead of relying on a
single source.

- official exchange and disclosure websites for authoritative raw information
- open-source data connectors for fast research iteration
- an internal ingestion module for crawling, normalization, and storage
- a canonical internal dataset used by backtest and modeling

This means we can start quickly with public data, while still preparing for
better source control and reproducibility later.

## First Runnable Collector

The first real source integration targets US equities via Alpaca.

Run either after `python -m pip install -e .` or with `PYTHONPATH=src`.

Required environment variables:

- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`

Optional overrides:

- `ALPACA_TRADING_BASE_URL`
- `ALPACA_DATA_BASE_URL`

Template:

- [`.env.example`](/E:/CodeX/StockMachine-260321/.env.example)

First validation command:

```text
python -m stockmachine.apps.check_alpaca_access --symbol AAPL --benchmark SPY --feed iex
```

Example commands:

```text
python -m stockmachine.apps.collect_us_equities_v1 symbol-master
python -m stockmachine.apps.collect_us_equities_v1 daily-bars --symbols AAPL MSFT SPY --start 2024-01-01 --end 2024-03-31
python -m stockmachine.apps.collect_us_equities_v1 adj-factors --symbols AAPL MSFT SPY --start 2024-01-01 --end 2024-03-31
python -m stockmachine.apps.collect_us_equities_v1 research-seed --start 2025-01-01 --end 2026-03-21 --chunk-size 25
```

Outputs are written to:

- `data/raw/...` for captured upstream payloads
- `data/silver/...` for normalized internal tables

When multiple silver batches exist for the same canonical primary key, the
loader now keeps the most recently loaded row. That lets newer Alpaca pulls
override older Yahoo bootstrap rows cleanly.

## Silver Bootstrap

When Alpaca credentials are not available yet, a temporary Yahoo bootstrap can
seed canonical silver tables for research and backtest wiring:

```text
python -m stockmachine.apps.bootstrap_us_equities_silver --start 2019-01-01 --end 2025-12-31
python -m stockmachine.apps.run_us_equities_silver_chain --model hist_gbm --predict-start 2025-01-01
```

This bridge is for research plumbing only and should later be replaced by our
official ingestion sources.

## Next Suggested Milestones

1. Continue paper-demo productization with scheduler-friendly daily runs.
2. Harden restart recovery, lingering-order maintenance, and operator automation.
3. Add websocket `trade_updates` handling for faster order-state convergence.
4. Extract the paper signal path out of the research module into a cleaner alpha runtime layer.
5. Harden point-in-time universe metadata and live-cost attribution.

## Paper Demo

The first paper runner can now wire together:

- latest silver data
- walk-forward baseline signals
- white-box portfolio construction
- broker-aware pre-trade checks
- session/data/idempotency guard rails
- restart recovery against broker open orders
- Alpaca paper account sync
- local ledger persistence
- polling-based order reconciliation
- operator CLI summaries via `paper_ops`

Dry-run smoke command:

```text
$env:PYTHONPATH='src'; python -m stockmachine.apps.run_us_equities_paper --session-date 2026-03-22 --model hist_gbm --run-name smoke-dryrun --execution-equity-cap 500 --max-order-notional 550 --max-total-notional 550 --max-total-orders 2
```

Execute mode is available, but it will place orders into the Alpaca paper account, so treat it as an explicit next step rather than a default smoke test.

Operator inspection commands:

```text
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_smoke --strategy-profile us_hist_gbm_random_forest_rank_daily --artifact-root artifacts
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_smoke --session-date 2026-03-22 --model hist_gbm --run-name stable-smoke --artifact-root artifacts
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_ops latest-run
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_ops open-orders
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_daily healthcheck
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_daily run --strategy-profile us_hist_gbm_random_forest_rank_daily --session-date 2026-03-22 --execution-equity-cap 500 --max-order-notional 550 --max-total-notional 550 --max-total-orders 2
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_daily run --session-date 2026-03-22 --model hist_gbm --run-name stable-daily-demo --execution-equity-cap 500 --max-order-notional 550 --max-total-notional 550 --max-total-orders 2 --artifact-dir artifacts/us_equities_silver_chain_2025_hist_gbm_alpaca_adj
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_reconcile latest-run --artifact-dir artifacts/us_equities_silver_chain_2025_hist_gbm_alpaca_adj
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_maintain latest-run --broker-orders-json path/to/open_orders.json --stale-after-minutes 60
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_report run-index --limit 10
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_report daily-summary --session-date 2026-03-22
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_report health-trend --session-date 2026-03-22 --limit-runs 10 --limit-sessions 5
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_report anomaly-summary --session-date 2026-03-22 --limit-runs 10
$env:PYTHONPATH='src'; python -m stockmachine.apps.paper_report operator-digest --session-date 2026-03-22 --limit 10
```

The daily runner also supports a filesystem kill switch at
`artifacts/paper_demo/paper_daily.kill` by default, or a custom path via
`--kill-switch-path`.

`paper_smoke` is the safest operator entrypoint for repeatable checks. It
auto-discovers a likely research/backtest artifact directory, runs the same
preflight path as `paper_daily`, and returns the recommended follow-up
commands for reconcile and maintenance.

Built-in paper strategy profiles now live under
[`configs/strategies`](/E:/CodeX/StockMachine-260321/configs/strategies). The
first low-risk migration set includes:

- `us_hist_gbm_daily`
- `us_ridge_daily`
- `us_hist_gbm_ridge_mean_daily`
- `us_hist_gbm_ridge_rank_daily`
- `us_hist_gbm_random_forest_rank_daily`

Profiles provide default model and risk parameters, while CLI flags still
override them when needed.

The current productization plan lives in [docs/paper-productization-plan.md](/E:/CodeX/StockMachine-260321/docs/paper-productization-plan.md).
