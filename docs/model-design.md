# Model Design Notes

## First Principle

The predictive model should answer:

- which names look attractive
- in what direction
- over what horizon
- with what relative strength or confidence

It should not answer:

- how many shares to buy
- whether a portfolio-wide risk limit is already full
- what order type to submit

## Start With The Target

Before selecting a model family, define the target clearly.

Candidate targets:

1. next `N` day raw return
2. next `N` day excess return versus benchmark
3. next `N` day rank within universe
4. probability of positive return over horizon
5. probability of beating transaction costs by a margin

For a first equity system, a rank-style target is often easier to use than
pure price forecasting because portfolio construction mainly needs relative
ordering.

## Recommended V1 Framing

If the first version is a daily-bar stock strategy, a pragmatic setup is:

- task: cross-sectional ranking
- target: next 5 trading day excess return
- output: normalized score plus confidence
- rebalance: daily or every few days
- objective: select top names, not predict exact prices

This usually aligns better with portfolio construction than forecasting exact
close prices.

## Feature Families

Start with a compact, interpretable set:

- price momentum over multiple windows
- volatility and drawdown features
- turnover and liquidity features
- relative strength versus benchmark or sector
- market regime context
- sector or style neutralized factors

Add fundamentals only after the daily-bar pipeline is clean and point-in-time
safe.

## Model Family Options

### Linear or rank-based baseline

Examples:

- weighted factor score
- linear regression
- logistic regression

Strengths:

- fast to build
- easy to interpret
- easier to debug leakage and regime issues

Weaknesses:

- weaker non-linear expression

### Tree-based models

Examples:

- LightGBM
- XGBoost
- CatBoost

Strengths:

- strong tabular performance
- handles mixed feature interactions well
- common default choice for medium-frequency equity modeling

Weaknesses:

- easier to overfit if data splitting is sloppy
- feature leakage can look deceptively good

### Sequence models

Examples:

- TCN
- LSTM
- Transformer variants

Strengths:

- useful when sequential path information matters strongly

Weaknesses:

- more data hungry
- more moving parts
- harder to validate early in the project

For a first production-capable version, tree models are often the best balance.

## Evaluation Checklist

Model quality should be judged on trading usefulness, not just loss curves.

Primary metrics:

- rank IC and IC stability
- top-decile minus bottom-decile spread
- turnover after signal refresh
- post-cost Sharpe and drawdown in backtest
- hit rate by regime and sector

## Leakage Traps

Watch for:

- future-adjusted prices used in features
- survivorship bias in the universe
- point-in-time fundamentals errors
- label overlap without proper walk-forward validation
- using same-day close data to trade at same close

## Decisions Still Open

- target market and universe
- holding horizon
- long-only or long-short
- daily, hourly, or event-driven frequency
- whether to optimize for rank, return, or probability
