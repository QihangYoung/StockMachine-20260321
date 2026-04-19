# Non-Price Two-Factor Alpha Memo

Date: 2026-04-19

## Decision

The first non-price alpha experiment should not force one scalar score to
serve both the long book and the short book.

Instead, split the research target into two independent factor channels:

- `long_evidence_score`: positive evidence for owning a stock
- `short_evidence_score`: negative evidence for shorting or avoiding a stock

This is a better fit for the actual information structure of non-price data.
Many positive signals are not symmetric with negative signals:

- insider buying is more informative than routine insider selling
- absence of filing red flags is not a strong long signal
- high short interest can be bearish or bullish depending on squeeze context
- fundamental quality improvement is not the exact mirror image of accounting
  red flags

The goal is therefore not to build one universal "good minus bad" score. The
goal is to build two auditable scores that can be evaluated separately and then
combined only at the portfolio-construction layer.

## Window Alignment

This line inherits the Beta / Alpha thread research discipline.

Canonical windows:

- validation / research window: `2013-08-01` through `2019-12-31`
- pure-alpha trading-session start: `2013-08-05` when daily stock data allows
- final test lockbox: `2020-01-02` through `2026-04-08`

Operational rule:

- feature discovery, source selection, parser design, and factor selection must
  use the validation window only;
- the test lockbox must remain closed until a candidate is frozen in a
  validation memo;
- if a source cannot cover the full validation window, it must be labeled as a
  later-start exploratory source, not promoted into the first aligned factor
  experiment.

The current pure-alpha data audit also matters:

- broad top1000 price data is strongest from `2016-01-04` onward;
- the `2013-08-05` through `2015-12-31` segment is currently supported by a
  provisional Yahoo gap fill and current-top1000 bootstrap scope;
- therefore non-price features may cover `2013+`, but final claims still depend
  on the stock universe and price layer becoming research-grade.

## Factor Split

### Long Evidence Score

Purpose:

Identify stocks with positive company-specific evidence that can justify long
selection after controlling for market, sector, liquidity, and existing price
features.

First candidate components:

- fundamental quality improvement
- revenue growth with margin confirmation
- operating cash-flow quality improvement
- receivables and inventory not outrunning revenue
- declining dilution or balance-sheet stress
- open-market insider buying
- cluster buying by multiple insiders
- CEO/CFO purchases

Initial formula shape:

```text
 z(fundamental_quality_change)
+ z(open_market_insider_buy_intensity)
+ z(cluster_buy_flag)
+ z(positive_filing_event_context)
- z(accounting_quality_deterioration)
```

Interpretation:

- high score: eligible long boost
- neutral score: no long evidence
- low score: weak negative, not automatically a short

### Short Evidence Score

Purpose:

Identify stocks with negative company-specific evidence that can justify short
selection or exclusion from long books.

First candidate components:

- late filing / NT 10-K / NT 10-Q
- amendment frequency
- restatement / non-reliance language
- auditor change or accounting control weakness
- negative 8-K event categories
- deteriorating accounting quality
- high short pressure only when not in squeeze context

Initial formula shape:

```text
+ z(filing_red_flag_intensity)
+ z(restatement_or_non_reliance_flag)
+ z(material_weakness_flag)
+ z(accounting_quality_deterioration)
+ z(short_pressure_without_positive_momentum)
- z(insider_cluster_buy)
```

Interpretation:

- high score: eligible short candidate or long exclusion
- neutral score: no short evidence
- low score: weak positive, not automatically a long

## Why This Is Cleaner

One scalar long/short factor assumes evidence is symmetric. Non-price data is
usually asymmetric.

Examples:

- Insider buy is a credible positive event. Insider sell is often tax,
  diversification, or pre-planned selling, so it is not equally credible as a
  negative event.
- Filing red flags are credible negative events. Clean filings do not create
  equally strong positive evidence.
- Short interest is not directional by itself. It can indicate informed
  bearishness or crowded squeeze risk.
- Fundamental improvement and accounting deterioration can both be true in
  different dimensions, so forcing them into one number too early can hide the
  mechanism.

By separating the channels, we can answer clearer questions:

- Does the long factor improve the long leg?
- Does the short factor improve the short leg?
- Do both legs contribute to long-short spread?
- Does either channel merely reduce risk rather than add return?
- Are long and short channels stable across sectors and subperiods?

## Initial Data Source Feasibility

### Strong First Sources

SEC filing metadata:

- covers the required window
- structured and auditable
- suitable for `short_evidence_score`
- good first implementation target

SEC insider transactions:

- covers the required window
- structured Form 3/4/5 data
- suitable for `long_evidence_score`
- sparse but high information density

SEC structured fundamentals:

- covers the required window with enough warm-up
- suitable for both long and short channels
- higher implementation complexity due to taxonomy and restatements

### Later Sources

13F:

- mostly covers the aligned window from 2013 onward
- better as crowding/risk context than a first directional score

FINRA short data:

- daily short-sale volume does not fully cover the validation window
- exchange-listed historical short interest coverage is not clean from the
  free FINRA archive before 2021
- useful later if a better vendor source is available

Attention/news:

- Wikipedia and GDELT 2.0 start too late for full-window alignment
- useful for later-start exploratory work
- not first-line for aligned Beta/Alpha research

## Evaluation Design

Evaluate the two channels separately before combining them.

Long-channel tests:

- long bucket return
- long bucket excess return versus median eligible stock
- sector-neutral rank IC
- beta-residual rank IC
- contribution to current h5 long-only implementation

Short-channel tests:

- short bucket return, where lower realized return is better
- short-minus-median under sign convention
- bottom-leg contribution in long-short pure-alpha spread
- long-exclusion value
- tail-risk reduction

Combined long-short tests:

- long book selected by `long_evidence_score`
- short book selected by `short_evidence_score`
- beta-matched and dollar-neutral construction
- sector and industry exposure diagnostics
- cost stress and turnover review

Do not tune either factor on `C policy + sleeve` results. Sleeve tests should
come only after standalone validation proves that at least one channel has
stock-selection value.

## First Implementation Path

### Phase 0: Data Preparation

Prepare reference and coverage data before building factors:

- load current pure-alpha top1000 bootstrap universe
- download SEC ticker-to-CIK reference
- map universe symbols to CIKs
- produce coverage and missing-mapping artifacts
- write source feasibility manifest for the aligned windows

Exit criteria:

- CIK mapping coverage is measurable
- missing symbols are listed and reviewable
- source coverage is documented against the Beta/Alpha windows
- no feature or performance result is computed

### Phase 1: Short Evidence MVP

Start with filing metadata red flags:

- filing delay
- NT 10-K / NT 10-Q
- amended filings
- 8-K red-flag item categories
- restatement / non-reliance keywords if structured metadata is insufficient

Reason:

This is the fastest path to an auditable negative channel. It should help both
short selection and long exclusion.

### Phase 2: Long Evidence MVP

Add insider open-market buying:

- Form 4 open-market purchases
- role-weighted purchases
- cluster buying
- 20-session and 60-session decays

Reason:

This is behavior-based, sparse, and plausibly orthogonal to price-only
features.

### Phase 3: Fundamental Quality

Add structured accounting quality:

- growth quality
- margin confirmation
- cash-flow quality
- receivables/inventory stress
- debt and dilution

Reason:

This is likely the most important long-run source, but it needs careful
normalization and restatement handling.

## Current Recommendation

Proceed with two separate factors:

- first short channel: `filing_red_flag_score`
- first long channel: `insider_net_buy_score`
- first fundamental shared channel: `fundamental_quality_change_score`

Prepare data in that order:

1. SEC symbol-to-CIK mapping and coverage.
2. Filing metadata availability.
3. Insider transaction availability.
4. Structured fundamentals availability.

This sequence keeps the research disciplined: first prove that the data can be
aligned to our windows and universe, then build one leg at a time.
