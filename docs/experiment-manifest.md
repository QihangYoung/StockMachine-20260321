# Experiment Manifest

This document defines the minimum replay packet for any experiment that may be cited in a report.

## Goal

The experiment is only considered reproducible enough for audit when another person can:

1. restore the same code version
2. restore the same data snapshot
3. rerun the same command with the same parameters
4. obtain artifacts that can be compared against the preserved outputs

If any of these steps is impossible, the result should be treated as a research note, not an audit-grade experiment.

## Minimum fields to record

Record the following for every cited run:

- `experiment_name`
- `experiment_owner`
- `run_timestamp`
- `repository_commit`
- `branch_name`
- `working_directory`
- `python_version`
- `dependency_snapshot`
- `launch_command`
- `runtime_parameters`
- `data_snapshot`
- `universe_snapshot`
- `model_artifact`
- `prediction_artifact`
- `backtest_artifact`
- `report_artifact`
- `random_seed`
- `cost_assumptions`
- `risk_rules`
- `notes`

## Suggested manifest template

```yaml
experiment_name: us_zeroshot_suite_v2
experiment_owner: <name>
run_timestamp: 2026-03-22T09:30:00+08:00
repository_commit: <git-sha>
branch_name: <branch>
working_directory: E:\CodeX\StockMachine-260321
python_version: 3.10.x
dependency_snapshot: requirements.freeze.txt
launch_command: >
  <exact command line used to launch the run>
runtime_parameters:
  market: US
  universe: us_large_cap_30
  start_date: 2024-01-01
  end_date: 2025-12-23
  holding_period_sessions: 5
  cost_bps: 10
  random_seed: 42
data_snapshot:
  raw_input: <path-or-uri>
  normalized_features: <path-or-uri>
  labels: <path-or-uri>
  hashes:
    raw_input: <sha256>
    normalized_features: <sha256>
    labels: <sha256>
universe_snapshot:
  file: <path-or-uri>
  hash: <sha256>
model_artifact:
  file: <path-or-uri>
  hash: <sha256>
prediction_artifact:
  file: <path-or-uri>
  hash: <sha256>
backtest_artifact:
  file: <path-or-uri>
  hash: <sha256>
report_artifact:
  file: <path-or-uri>
  hash: <sha256>
cost_assumptions:
  slippage_bps: 10
  fees_included: true
risk_rules:
  - liquidity_filter
  - sector_neutralization
  - max_position_cap
notes: >
  Add any exception, fallback path, or manual intervention here.
```

## Minimum commands to archive

The exact command text used for the experiment should be preserved. A minimal replay packet should include:

1. the environment capture command
   - `python --version`
   - `pip freeze` or the project-specific dependency export command
2. the code snapshot command
   - `git rev-parse HEAD`
   - `git status --short`
3. the launch command
   - the exact experiment entrypoint plus all parameters
4. the artifact hash command
   - `sha256sum` or `Get-FileHash` for every cited artifact

If a notebook was used, also archive:

- notebook path
- executed cell range
- kernel version
- any manually edited input cells

## Required artifact set

At minimum, preserve the following outputs for any report-cited run:

- raw data files
- normalized feature files
- label files
- universe membership file
- model artifact
- prediction summary
- backtest summary
- baseline comparison
- report markdown

If the experiment includes paper trading, also preserve:

- signal log
- target-position log
- order submission log
- fill or rejection log
- account snapshot
- ledger snapshot

## Review gate

Before a result is promoted into a report, check whether the replay packet is complete.

Treat the run as non-auditable if any of these are missing:

- launch command
- parameter snapshot
- dependency snapshot
- data hash
- artifact hash
- code version

## Relation to the evaluation protocol

This manifest is the operational companion to `evaluation-protocol.md`.

- `evaluation-protocol.md` defines how to validate a result
- this document defines what must be preserved so the result can be replayed later

