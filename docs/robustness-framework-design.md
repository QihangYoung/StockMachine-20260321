# Robustness Framework Design

Date: 2026-04-04

## Goal

This note defines the first shared robustness-evaluation framework for
StockMachine research. The framework is intended to sit beside strict backtest
execution, not inside any one strategy implementation.

The design goal is:

- strategy code produces standard research artifacts
- robustness infrastructure consumes those artifacts
- robustness output is comparable across `h1`, `h5`, and future strategy lines

In other words, the robustness layer should challenge alpha claims rather than
help generate alpha.

## Why This Must Be Shared Infrastructure

Robustness analysis should not live inside:

- `src/stockmachine/research/h1_us_equities.py`
- `src/stockmachine/research/us_equities_baseline.py`
- `src/stockmachine/research/p1_rigor.py`

Those files own strategy semantics, prediction generation, and backtest
construction. Robustness should consume their outputs through a stable contract.

This keeps three concerns separate:

1. strategy design
2. backtest execution
3. research challenge and validation

## External References

This design is informed by several common model-governance and validation ideas:

- Federal Reserve SR 11-7: model risk management and independent challenge
- White's Reality Check: correct for repeated testing / data snooping
- Probability of Backtest Overfitting: track search-driven false discoveries
- Deflated Sharpe Ratio: adjust interpretation of selected Sharpe leaders

These references do not imply full implementation parity. They inform the
module boundaries and evaluation philosophy.

## Scope Of The First Version

Version 1 focuses on four robustness families:

1. time stability
2. tail dependence
3. parameter stability
4. cost and execution stress

Universe perturbation and selection-bias analyzers are part of the intended
framework but can be added in the second pass once the shared contract is in
place.

Parameter stability should be implemented as shared infrastructure in V1, but
it should not run by default in routine robustness evaluation. The reason is
that parameter-stability analysis usually depends on pre-existing sweep outputs
and can be materially more expensive if it triggers search generation.

Selection-bias diagnostics should follow the same philosophy. They should
consume existing search surfaces and metadata, not force new experimental runs.

## Core Principles

- Shared inputs, shared outputs
- Strategy-specific semantics only through framework config
- Analyzer modules are independent from strategy implementation files
- Reports are written with the same conventions as strict research artifacts
- Gate decisions are derived from analyzer outputs, not just raw return metrics

## Proposed Module Layout

### `src/stockmachine/research/robustness_contracts.py`

Owns the input contract.

Responsibilities:

- define the standard artifact bundle expected by robustness analyzers
- validate required columns in backtest records and summary tables
- normalize optional payloads such as predictions, protocol metadata, and
  parameter manifests

### `src/stockmachine/research/robustness_frameworks.py`

Owns small strategy-specific semantics.

Responsibilities:

- resolve robustness framework by `strategy_project + horizon`
- define date-assignment rules for yearly or quarterly attribution
- define default extreme-day thresholds
- define default parameter-neighborhood settings
- define whether parameter stability should run in routine evaluation by default
- define default cost stress levels and gating thresholds

### `src/stockmachine/research/robustness_reports.py`

Owns shared report-writing helpers.

Responsibilities:

- write robustness overview tables
- write analyzer-specific tables
- write robustness gate payloads
- keep naming and output layout consistent across strategy lines

### Future modules

These should be added after the contract and framework layers settle:

- `robustness_analyzers.py`
- `robustness_gates.py`
- `robustness_selection_bias.py`

The first implementation pass may keep selection-bias diagnostics inside
`robustness_analyzers.py` as a lightweight search-diagnostic summary before any
heavier PBO or DSR-specific module is introduced.

## Standard Inputs

The robustness layer should assume the following standard artifacts when
available:

- `backtest_records.csv`
- `summary_metrics.csv` or a one-model summary row
- `predictions.csv`
- `protocol.json` or equivalent run metadata
- optional parameter search manifest

Required backtest-record columns for V1:

- `entry_date`
- `exit_date`
- `net_return`
- `benchmark_return`
- `turnover`
- `cost_bps`
- `positions`

The strategy runner remains responsible for producing these artifacts.

## Standard Outputs

Version 1 should standardize these artifacts:

- `robustness_overview.csv`
- `time_stability_yearly.csv`
- `tail_dependence_summary.csv`
- `parameter_stability_summary.csv`
- `cost_stress_summary.csv`
- `robustness_gate.json`

Not every analyzer must emit content in every run. Empty but schema-valid
outputs are acceptable when a strategy does not provide the needed upstream
context yet.

## Analyzer Semantics

### Time stability

Purpose:

- determine whether performance survives temporal slicing

Minimum metrics:

- positive-year ratio
- worst year
- yearly return standard deviation
- rolling 1-year and 2-year summary hooks

### Tail dependence

Purpose:

- determine whether total performance depends on a few extreme days

Minimum metrics:

- top 1 / 5 / 10 positive day contribution share
- bottom 1 / 5 / 10 negative day contribution share
- leave-top-k-out and leave-bottom-k-out result deltas
- cross-year trade contribution diagnostics

### Parameter stability

Purpose:

- determine whether the selected parameter point is isolated or supported by a
  healthy neighborhood

Minimum metrics:

- leader vs neighborhood median
- local neighborhood sample count
- sensitivity to one-step parameter perturbations

Implementation note:

- the analyzer should consume an existing parameter-surface table
- it should not require a strategy runner to retrain models by default
- routine robustness suites should skip this analyzer unless explicitly enabled

### Cost and execution stress

Purpose:

- determine whether alpha survives plausible friction changes

Minimum metrics:

- annualized return under framework-specific cost ladders
- turnover concentration
- optional exposure and capacity proxies

### Selection-bias diagnostics

Purpose:

- measure how much confidence should be discounted after trying many
  model/feature/parameter variants

Minimum metrics:

- trial count
- leader percentile within tested configurations
- leader vs median and leader vs upper-quantile gap
- leader z-score relative to tested candidates
- optional family concentration diagnostics when a search-family column exists

Implementation note:

- the first version is intentionally lightweight
- it is a search-diagnostic layer, not a full statistical proof against
  overfitting
- PBO and DSR style methods can be layered on later

## Framework-Level Strategy Adapters

The robustness layer should not re-encode whole strategies. It only needs a
small amount of semantics. Initial framework fields should include:

- `strategy_project`
- `default_horizon`
- `attribution_date_column`
- `tail_return_column`
- `tail_trim_counts`
- `cost_stress_levels`
- `parameter_neighborhood_radius`
- `gate_defaults`

This mirrors the role already played by `strict_frameworks.py` for strict
research.

## Initial Gate Philosophy

The first gate should classify outcomes into:

- `pass`
- `warning`
- `fail`

It should not rely on a single score.

Examples of likely fail conditions:

- removing top 5 days turns annualized return negative
- 20 bps or 30 bps stress turns the strategy unviable
- best parameter point is isolated from neighbors

Examples of warning conditions:

- more than half of performance comes from a tiny number of days
- time stability is acceptable but concentrated in one regime

## Integration Plan

### Phase 1

- land design document
- land contract, framework, and report skeleton
- keep analyzers stubbed or thin

### Phase 2

- integrate time stability and tail dependence analyzers
- wire outputs into `h1` and `p1_rigor`

### Phase 3

- add parameter stability and cost-stress gate integration
- reuse in both `h1` and `h5`

### Phase 4

- add selection-bias tools such as PBO or DSR-style corrections

## Success Criteria

The first shared robustness infrastructure is considered successful when:

- `h1` and `h5` can both resolve a robustness framework
- analyzers consume standard contracts instead of strategy-internal state
- report writing is shared
- future robustness analyzers can be added without editing strategy runners
