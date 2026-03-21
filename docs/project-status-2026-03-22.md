# Project Status 2026-03-22

## Summary

As of 2026-03-22, the project has moved from a research-only scaffold to a runnable US-equities paper-trading system prototype with a stable daily-ops shell.

The strongest part of the system is now the Alpaca paper runtime:

- structured market-data ingestion for US equities
- silver-table research and backtest flow
- white-box portfolio, risk, and execution policies
- Alpaca paper broker adapter and account sync
- local ledger, recovery, maintenance, and reconciliation
- scheduler-friendly daily runner, smoke harness, and operator reporting

The main gap is no longer "can the system run", but "how production-stable and research-rigorous is it over time".

## Module Progress

| Module | Status | Notes |
| --- | --- | --- |
| Data ingestion and silver tables | 75% | US equities `symbol_master`, `daily_bar`, `benchmark_index`, and `adj_factor` are wired; point-in-time rigor still needs work. |
| Research and baseline models | 65% | Baseline modeling and walk-forward style research are usable, but not yet fully hardened. |
| Backtest and evaluation | 72% | Artifact-based evaluation and paper-vs-artifact reconciliation exist; formal walk-forward and stricter sample handling remain. |
| Portfolio, risk, and execution policies | 82% | White-box controls are in place for research and paper trading. |
| Alpaca paper runtime | 92% | End-to-end paper workflow, smoke harness, recovery, maintenance, and reporting are all available. |
| Monitoring, audit, and operator views | 88% | Run index, digest, anomaly summary, health trend, reconciliation, and manifests are all present. |
| Stable operations | 70% | Daily-ops shell is in place; longer-running reliability still needs more validation. |
| A-share expansion | 15% | Architecture is market-aware, but A-share data and trading rules are not yet on the main path. |

## Current Milestones

- `M1 Research Prototype`: completed
- `M2 Alpaca Paper Demo`: completed
- `M3 Stable Daily-Run Shell`: completed
- `M4 Stable Operations Version`: in progress
- `M5 Research-Rigor Hardening`: not completed
- `M6 A-share Expansion`: not started

## What Already Works

### Data and Research

- Alpaca-backed US-equities ingestion and normalized silver tables
- research baseline on US equities with artifact outputs
- adjusted-price handling and benchmark-aware analysis
- paper-vs-artifact reconciliation based on recorded run manifests

### Paper Trading Runtime

- Alpaca paper account connectivity and account sync
- broker-aware pre-trade risk checks
- local SQLite ledger for runs, orders, fills, and manifests
- restart recovery against broker open orders
- stale/open-order maintenance planning
- dry-run and real paper smoke paths

### Operations

- `paper_daily` for scheduler-friendly daily execution
- `paper_smoke` for repeatable safe smoke checks
- `paper_reconcile` for run-level comparison against research artifacts
- `paper_report` for run index, daily digest, operator digest, health trend, and anomaly summary
- kill-switch support for daily runs

## Main Risks Still Open

- point-in-time universe and metadata rigor are not fully hardened
- research validation is not yet at the strictest walk-forward standard
- websocket-backed live order-update flow has a seam and runner integration, but still needs more real execute validation
- long-running operational validation across many consecutive sessions is still light
- one known non-blocking cleanup remains in the silver loader warning path

## Recommended Next Directions

### 1. Finish Stable Operations

This is the best immediate path because it is closest to a continuously running system.

- run a small real paper execute validation for websocket/fill handling
- run multi-day smoke validation across consecutive sessions
- harden lingering-order maintenance and restart recovery
- connect alerts to a real delivery channel

### 2. Harden Research Rigor

This is the path that most improves confidence in the strategy itself.

- point-in-time universe handling
- formal walk-forward splitting
- purge and embargo rules
- stricter cost and robustness analysis

### 3. Improve Strategy Quality

This should come after the first two tracks are more stable.

- richer daily basic features
- sector/style neutralization improvements
- confidence and meta-label modeling
- multi-horizon and ensemble models

## Recommended Priority

The recommended priority order is:

1. finish the stable-operations line
2. harden research rigor
3. improve the strategy layer
4. start the A-share expansion after the US path is stable

## Key Entry Points

- [paper_daily.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/paper_daily.py)
- [paper_smoke.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/paper_smoke.py)
- [paper_reconcile.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/paper_reconcile.py)
- [paper_report.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/paper_report.py)
- [run_us_equities_paper.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_us_equities_paper.py)
- [project-status-2026-03-22.md](/E:/CodeX/StockMachine-260321/docs/project-status-2026-03-22.md)
