# Evaluation Protocol

## Goal

Make research results trustworthy before they are interesting.

## Timing Rules

The first model follows this exact timing:

1. build features using information available by session `T` close
2. freeze the stock universe at the same cutoff
3. generate signals after the close of `T`
4. place simulated entries at the open of `T+1`
5. exit after 5 held sessions at the open of `T+6`

Same-session close execution is not allowed for this first workflow.

## Data Validation Gates

Every dataset build must verify:

- no duplicate primary keys
- no missing sessions relative to the trading calendar
- no negative prices or volumes
- no active-symbol bars after delisting date
- no rows using future adjustment factors

## Split Policy

Use a rolling walk-forward schedule:

- train window: 36 months
- validation window: 6 months
- test window: 6 months
- roll frequency: monthly
- purge window: 6 sessions

The final held-out test period must never be used for model or parameter
selection.

## First-Line Model Metrics

The first report should include:

- rank IC
- ICIR
- top-minus-bottom spread return
- decile monotonicity
- turnover
- hit rate

## Strategy Metrics

The first backtest report should include:

- annualized return
- annualized volatility
- Sharpe
- Calmar
- max drawdown
- turnover
- average holding period
- post-cost performance

## Stress Checks

The first pass of robustness testing should include:

- slippage doubled versus base assumptions
- one-session execution delay
- tighter liquidity filter
- alternate holding windows of 3 and 10 sessions

## Promotion Rule

A model can move forward only if:

- it beats simple baselines on the same test periods
- post-cost results remain positive
- performance is not concentrated in one short time block
- results survive basic stress checks
- audit outputs can reconstruct signals, targets, and simulated fills

## Reproducibility and Audit Minimums

Every experiment that may be cited in a report should preserve a minimal replay packet. At minimum, record:

- the exact command or notebook entrypoint used to launch the run
- all runtime parameters, including market, universe, date range, holding period, cost assumptions, risk filters, and random seed
- the repository commit hash, branch name, and run timestamp
- the Python version and the key dependency versions used for the run
- the dataset snapshot identifiers for raw inputs, normalized features, labels, and universe membership
- the file paths and hashes for model artifacts, prediction outputs, backtest summaries, and report files
- the rule set used for timing, execution, and risk gating

Suggested replay order:

1. restore the recorded code version and environment
2. verify the data snapshot hashes
3. rerun feature generation or load the preserved feature snapshot
4. rerun inference with the recorded parameters
5. rerun backtest or paper simulation with the same cost and risk assumptions
6. compare the new outputs against the preserved artifacts

If any of the following are missing, the result should be treated as a research note rather than an audit-grade experiment:

- command line
- dependency snapshot
- dataset snapshot
- parameter snapshot
- artifact hash
