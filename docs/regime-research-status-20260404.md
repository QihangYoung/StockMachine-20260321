## Regime Research Status (2026-04-04)

### Scope
This document summarizes the first round of regime research built on top of the current `C policy + stock-alpha sleeve` product direction.

The goal of this round was not to ship a production regime gate yet. The goal was to:

1. Add a reusable, low-coupling regime infrastructure layer.
2. Test a first white-box regime gate on `C + 6% lightgbm_ranker`.
3. Evaluate regime detectors independently of any single strategy, using benchmark-centric supervision.
4. Decide whether regime work should stay on the critical path right now.

### What Was Implemented

#### 1. Regime infrastructure
New reusable regime module:

- [src/stockmachine/risk/regime.py](/E:/CodeX/StockMachine-260321/src/stockmachine/risk/regime.py)
- [src/stockmachine/risk/__init__.py](/E:/CodeX/StockMachine-260321/src/stockmachine/risk/__init__.py)

Implemented components:

- `RegimeDetector` protocol
- `RegimeGatePolicy`
- `apply_regime_gate(...)`
- `BenchmarkTrendDrawdownRegimeDetector`
- `BenchmarkTrendDrawdownVolRegimeDetector`
- `BenchmarkTrendDrawdownVolCrossAssetRegimeDetector`
- `BenchmarkForwardRegimeLabeler`
- `build_regime_confusion_matrix(...)`

Design intent:

- detector and gate policy are separated
- detector is strategy-agnostic
- gate policy is consumer-specific
- all detector inputs are strictly lagged / point-in-time safe

#### 2. Sleeve experiment integration
The sleeve experiment app now supports optional regime gating:

- [src/stockmachine/apps/run_c_policy_sleeve_experiment.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_c_policy_sleeve_experiment.py)

This allows:

- ungated sleeve blends
- gated sleeve blends
- risk-matched gated sleeve blends

#### 3. Independent supervision study
New study app:

- [src/stockmachine/apps/run_regime_supervision_study.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_regime_supervision_study.py)

This app evaluates detectors without using strategy PnL as the training or scoring target.

It compares:

- detected regime from lagged current information
- forward regime labels defined from future benchmark path

### Strategy-Level Gate Prototype Result

First gate policy tested on `C + 6% lightgbm_ranker`:

- `warmup = 1.0`
- `bull = 1.0`
- `correction = 0.0`
- `bear = 0.0`
- `rebound = 0.5`

Artifacts:

- [artifacts/c_policy_lightgbm_regime_gate_20260404/summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/c_policy_lightgbm_regime_gate_20260404/summary_metrics.csv)
- [artifacts/c_policy_lightgbm_regime_gate_20260404/comparison_vs_ungated.csv](/E:/CodeX/StockMachine-260321/artifacts/c_policy_lightgbm_regime_gate_20260404/comparison_vs_ungated.csv)
- [artifacts/c_policy_lightgbm_regime_gate_risk_matched_20260404/summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/c_policy_lightgbm_regime_gate_risk_matched_20260404/summary_metrics.csv)
- [artifacts/c_policy_lightgbm_regime_gate_risk_matched_20260404/comparison_vs_ungated.csv](/E:/CodeX/StockMachine-260321/artifacts/c_policy_lightgbm_regime_gate_risk_matched_20260404/comparison_vs_ungated.csv)

Headline result:

- `C only`: Sharpe `1.562`
- `C + 6% lightgbm_ranker`: Sharpe `1.620`
- `C + 6% lightgbm_ranker (gated)`: Sharpe `1.621`

Interpretation:

- the first gate prototype is not harmful
- it slightly improves risk-adjusted performance
- but the improvement is small
- this is not enough evidence to declare the detector production-ready

### Independent Detector Evaluation

#### Forward regime supervision
Independent supervision was defined from future benchmark path, not from any strategy return.

Forward labels were created from future 12-window benchmark path:

- `bull`
- `correction`
- `bear`
- `rebound`

This labeler is implemented in:

- [src/stockmachine/risk/regime.py](/E:/CodeX/StockMachine-260321/src/stockmachine/risk/regime.py)

#### V1: trend + drawdown
Artifacts:

- [artifacts/regime_supervision_study_common404_20260404](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_study_common404_20260404)

Result:

- detector can describe current market state to some degree
- but detected regimes do not cleanly separate future market outcomes
- in particular, detected `bear` windows still had strong future benchmark returns on average

#### V2: trend + drawdown + realized vol
Artifacts:

- [artifacts/regime_supervision_study_common404_v2_20260404](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_study_common404_v2_20260404)

Result:

- adding realized vol did not materially fix the problem
- default v2 was not clearly better than v1

#### V3: trend + drawdown + realized vol + cross-asset spread
Artifacts:

- [artifacts/regime_supervision_study_common404_v3_20260404](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_study_common404_v3_20260404)

Cross-asset inputs came from:

- [artifacts/c_risk_tuning_regime_window_20260330/proxy_asset_window_returns_common404.csv](/E:/CodeX/StockMachine-260321/artifacts/c_risk_tuning_regime_window_20260330/proxy_asset_window_returns_common404.csv)

Result:

- cross-asset input is directionally sensible
- but the default v3 detector still did not materially outperform v1

### Detector Parameter Sweeps

#### Sweep 1: v1 and v2 family
Artifacts:

- [artifacts/regime_supervision_sweep_20260404/detector_grid_summary.csv](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_sweep_20260404/detector_grid_summary.csv)
- [artifacts/regime_supervision_sweep_20260404/top20_by_accuracy.csv](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_sweep_20260404/top20_by_accuracy.csv)
- [artifacts/regime_supervision_sweep_20260404/top20_by_bull_bear_gap.csv](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_sweep_20260404/top20_by_bull_bear_gap.csv)

Headline result:

- best observed accuracy was still from the simpler v1 family
- best v1 accuracy: about `0.564`
- best v2 accuracy: about `0.531`

#### Sweep 2: cross-asset detector family
Artifacts:

- [artifacts/regime_supervision_cross_asset_sweep_20260404/cross_asset_grid_summary.csv](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_cross_asset_sweep_20260404/cross_asset_grid_summary.csv)
- [artifacts/regime_supervision_cross_asset_sweep_20260404/top20_by_accuracy.csv](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_cross_asset_sweep_20260404/top20_by_accuracy.csv)
- [artifacts/regime_supervision_cross_asset_sweep_20260404/top20_by_bull_bear_gap.csv](/E:/CodeX/StockMachine-260321/artifacts/regime_supervision_cross_asset_sweep_20260404/top20_by_bull_bear_gap.csv)

Headline result:

- cross-asset detector family did not beat the best v1 detector on accuracy
- best observed cross-asset accuracy: about `0.499`

### Current Interpretation

The regime research direction is valid, but the current detector family is not yet strong enough.

What we know:

- regime infrastructure now exists and is reusable
- a simple gate can slightly help `C + 6% lightgbm_ranker`
- however, current benchmark-only and simple cross-asset detectors are not yet good enough to justify making regime gating a mainline product dependency

### Recommended Decision

Pause regime work on the main product path for now.

Specifically:

- do not enable regime gating in `C policy + lightgbm_ranker` production research by default
- do not spend more time tuning gate policy on top of the current detector family
- continue to treat `C policy + lightgbm_ranker` as the primary candidate without regime gating

### Recommended Next Thread

If regime research is resumed in a dedicated thread, the next priorities should be:

1. improve detector inputs, not gate policy
2. continue using benchmark-centric supervision, not strategy-PnL-driven tuning
3. test richer market-state inputs, such as:
   - broader cross-asset relative strength
   - defensive leadership breadth
   - volatility regime proxies
   - macro or credit proxies if reliable point-in-time data is available
4. only revisit gate policy optimization after detector quality materially improves

### Validation

Relevant tests:

- [tests/test_risk/test_regime.py](/E:/CodeX/StockMachine-260321/tests/test_risk/test_regime.py)
- [tests/test_apps/test_run_regime_supervision_study.py](/E:/CodeX/StockMachine-260321/tests/test_apps/test_run_regime_supervision_study.py)
- [tests/test_apps/test_run_c_policy_sleeve_experiment.py](/E:/CodeX/StockMachine-260321/tests/test_apps/test_run_c_policy_sleeve_experiment.py)

Last verified status:

- `15 passed`
- `python -m compileall src tests` passed
