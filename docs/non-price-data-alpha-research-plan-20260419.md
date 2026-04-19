# Non-Price Data Alpha Research Plan

Date: 2026-04-19

## Purpose

This plan defines the first research path for using non-price data in the U.S.
equities alpha stack.

The product goal is not simply to add more data. The goal is to introduce
information that is not already reducible to recent price, volume, beta,
sector, or liquidity effects, and to test whether that information improves:

- h5 stock ranking quality
- pure-alpha diagnostics
- long-only production-like performance
- correlation and sleeve value versus `C policy`
- robustness after realistic transaction costs

The first principle is point-in-time correctness. Non-price data is useful only
if we know when it became available to us, not merely which accounting period,
event date, or reporting period it describes.

## Core Thesis

Our current h5 stack is mostly built from market-derived information. That is
fast, clean, and powerful, but it risks learning signals that are already
embedded in price behavior.

Non-price data can add value through slower and more discrete information
channels:

- fundamentals changing before price fully reflects them
- filings revealing new risks, management tone, or event categories
- insiders revealing behavior that is not a public price feature
- short interest and short-sale flow revealing crowding or squeeze pressure
- ownership data revealing concentration and institutional crowding
- attention/news data revealing mismatch between public attention and actual
  fundamentals

The expected edge is not a generic "sentiment factor". The expected edge is
conditional information: unusual non-price information combined with price,
liquidity, sector, beta, and existing model context.

## Research Questions

Primary question:

Can non-price features improve out-of-sample stock-selection alpha after we
control for market beta, sector exposure, liquidity, and transaction costs?

Secondary questions:

- Which non-price source family has the best standalone signal quality?
- Which family adds the most incremental value after current h5 features?
- Do non-price features help `lightgbm_ranker` more than tree ensembles or
  linear models?
- Do the signals work in long-only selection, pure-alpha long-short selection,
  or only as risk filters?
- Do they improve `C policy + sleeve` portfolios because of low correlation, or
  do they merely raise standalone stock alpha Sharpe?
- Are the strongest effects event-driven or persistent daily panel effects?

## Data Source Priority

### Tier 1: Do First

#### SEC Structured Fundamentals And Filing Metadata

Scope:

- SEC `companyfacts` / XBRL facts
- SEC `submissions` metadata
- 10-K / 10-Q / 8-K filing timestamps
- later: filing text derived from 10-K, 10-Q, and 8-K

Why first:

- public source
- structured fields
- strong auditability
- long enough history for research
- directly tied to company fundamentals
- lower correlation with pure price features than most technical indicators

Expected useful signals:

- revenue, margin, inventory, receivables, capex, depreciation, debt, share
  count, and cash-flow abnormal changes
- 8-K event type and event intensity
- filing delay, amendment frequency, and reporting irregularity
- risk factor / MD&A language change
- guidance, uncertainty, and management tone proxies

Expected character:

- medium latency
- high reliability for structured records
- best for event-time features and slow fundamental drift
- likely more useful for h5/h20 confirmation and risk filtering than for
  single-day reversal

#### Insider Transactions

Scope:

- Form 4 transactions
- open-market buy/sell classification
- officer role
- transaction size relative to market cap and insider ownership
- cluster behavior across insiders

Why first:

- structured
- timely
- behavior-based rather than price-derived
- naturally suitable for event-conditioned alpha

Expected useful signals:

- cluster open-market buys
- CEO/CFO purchases
- insider buying after drawdown
- insider behavior combined with improving fundamentals
- insider buying while short interest is high

Expected character:

- low to medium latency
- sparse but high information density
- must filter option exercises, grants, planned transactions, and automatic
  sales
- likely works better as a conditional boost than as a dense daily feature

### Tier 2: Do After Tier 1 Plumbing Is Stable

#### FINRA Short Interest And Daily Short Sale Volume

Scope:

- semi-monthly short interest
- daily short sale volume
- days-to-cover
- short interest change / acceleration
- short-sale volume share

Why useful:

- captures crowding, positioning, and squeeze pressure
- complements insider and fundamentals data
- more about pressure and inventory than classic directional alpha

Expected character:

- short interest is lagged and slow
- daily short-sale volume is faster but noisy
- high short interest is not automatically bearish
- most useful when interacted with price trend, attention, insider behavior,
  and fundamental revision

#### 13F Ownership And Crowding

Scope:

- institutional holdings
- ownership concentration
- crowding
- holder churn
- crowded long plus short interest interaction

Why useful:

- identifies risk concentration and crowded trades
- useful for avoiding fragile longs and detecting squeeze risk

Expected character:

- heavily lagged
- not suitable for fast reactions
- better as a risk/crowding layer than a primary h5 alpha source

### Tier 3: Research Later, After Stronger PIT Infrastructure

#### Public Attention And News

Scope:

- Google Trends
- Wikipedia pageviews
- GDELT news/event volume and tone

Why useful:

- captures attention, narrative pressure, and information diffusion
- can detect attention/fundamental mismatch

Expected character:

- fast, broad, and noisy
- harder entity mapping
- more false positives
- normalization is non-trivial
- should be treated as a conditional feature family, not a standalone
  sentiment oracle

## Point-In-Time Contract

Every normalized non-price table must preserve at least four timestamps:

- `event_date`: when the underlying business event happened
- `period_end`: accounting/reporting period end, if applicable
- `filed_at`: when the source says the record was filed or published
- `available_at`: when our system could reasonably have used the record

Every feature table must include:

- `effective_session`: first trading session where the feature is allowed to be
  visible to the model
- `source`: upstream source name
- `source_version`: raw snapshot or parser version
- `asof_timestamp`: timestamp used for point-in-time reconstruction

Default trading rule:

If data arrives after the decision cutoff for session `T`, it cannot affect
orders for `T`. It becomes eligible no earlier than the next tradable session.

This rule should be conservative at first. We can relax it later only after
the execution schedule and source delivery times are explicitly modeled.

## Data Architecture

The non-price stack should follow the existing internal layer pattern:

### Raw

Immutable source payloads:

- SEC JSON / XML / text files
- FINRA CSV or API payloads
- 13F holdings files
- Wikipedia / GDELT / Trends payloads

No modeling code should read `raw` directly.

### Bronze

Source-specific parsed records, preserving source fields and identifiers.

Examples:

- `sec_companyfacts_bronze`
- `sec_submissions_bronze`
- `sec_form4_transactions_bronze`
- `finra_short_interest_bronze`
- `sec_13f_holdings_bronze`

### Silver

Normalized point-in-time tables keyed to internal identifiers.

Minimum requirements:

- `symbol`
- `cik` or source entity id
- `effective_session`
- normalized fields
- source timestamps
- data quality flags

### Gold

Model-ready feature tables:

- daily panel features keyed by `(session, symbol)`
- event-time features keyed by `(effective_session, symbol, event_id)`
- decayed event features for daily models
- feature-family metadata for routing and ablation

## Entity Mapping

Entity mapping is a first-class research risk.

Required mapping tables:

- ticker to CIK
- symbol history
- CIK to company identity
- active listing interval
- merger / delisting / ticker-change flags where available

Research rule:

No non-price feature is eligible until its entity mapping can be reconstructed
point-in-time or conservatively approximated without survivorship leakage.

Early MVP can start with the current tradable universe, but every result must
be labeled as provisional until mapping coverage is audited.

## Feature Design

### SEC Structured Features

Initial features:

- year-over-year and quarter-over-quarter change in revenue
- gross margin and operating margin change
- inventory growth versus revenue growth
- receivables growth versus revenue growth
- capex to depreciation
- debt growth and interest burden proxies
- cash-flow quality
- share count dilution
- reporting delay and amendment count
- 8-K item category counts

Modeling form:

- raw level
- change rate
- z-score within sector
- percentile rank within sector
- event-day flag
- exponential decay after filing

### SEC Text Features

Initial features should be simple and auditable:

- section length changes
- risk factor text similarity versus prior filing
- MD&A text similarity versus prior filing
- uncertainty / litigation / liquidity keyword intensity
- guidance-related keyword intensity
- negative-change flags

Avoid first:

- large opaque LLM summaries
- complicated sentiment classifiers with unclear training data
- features whose timestamp or parser behavior is hard to reproduce

### Insider Features

Initial features:

- open-market buy flag
- open-market sell flag
- net insider buy dollar
- net insider buy dollar divided by market cap
- number of unique insiders buying
- CEO/CFO buy flag
- cluster buy flag
- insider buy after drawdown
- insider buy plus improving fundamentals
- insider buy plus high short interest

Filtering rules:

- separate open-market trades from option exercises and grants
- flag planned or automatic transactions where detectable
- separate executives, directors, and beneficial owners
- winsorize large outliers

### Short Interest Features

Initial features:

- short interest level
- short interest change
- short interest acceleration
- days-to-cover
- short interest percentile within stock history
- short-sale volume share
- short-sale volume abnormality
- high short interest plus positive price momentum
- high short interest plus insider buying

Interpretation rule:

Short data should not be treated as simply bearish. It is more naturally a
crowding, pressure, and convexity feature family.

### Ownership / 13F Features

Initial features:

- institutional ownership concentration
- top holder concentration
- number of reporting holders
- holder churn
- new institutional ownership
- crowdedness rank
- crowding plus short interest

Use case:

- risk filter
- position sizing input
- squeeze/crowding interaction
- not a primary fast alpha feature

### Attention And News Features

Initial features:

- abnormal Wikipedia pageviews
- Google Trends relative search abnormality
- GDELT news volume abnormality
- GDELT tone abnormality
- attention spike without fundamental improvement
- negative news plus insider buy
- attention spike plus high short interest

Use case:

- conditional interactions
- event confirmation
- risk warning
- not a standalone sentiment alpha in the first version

## Modeling Integration

Initial model priority:

- first: `lightgbm_ranker`
- second: `extra_trees`
- third: ridge / linear diagnostics

Reason:

`lightgbm_ranker` is best positioned to use sparse event flags, nonlinear
interactions, and conditional effects. Ridge remains useful as a diagnostic
tool: if non-price features only help ridge through linear exposure, the signal
may be simpler than expected; if they help trees/rankers more, interaction
effects may matter.

Integration path:

1. Add non-price feature families to the h5 feature router.
2. Keep each source family independently switchable.
3. Run family-level ablation before joint optimization.
4. Run model-level diagnostics before portfolio sleeve experiments.
5. Only after standalone alpha improves, test `C policy + alpha sleeve`.

Do not optimize non-price features directly on `C policy + sleeve` Sharpe.
That can overfit the interaction with one portfolio policy rather than measure
true stock-selection alpha.

## Evaluation Protocol

### Data Quality Layer

Required checks:

- symbol-day coverage
- event count by year
- source timestamp availability
- filing delay distribution
- missing entity mapping rate
- duplicated record rate
- restatement / amendment handling
- feature staleness distribution

### Signal Layer

Required diagnostics:

- rank IC
- sector-neutral rank IC
- beta-neutral top-bottom spread
- top bucket versus median
- bottom bucket versus median
- information decay from event day
- performance by event age
- performance by sector
- performance by liquidity bucket

### Portfolio Layer

Required backtests:

- current h5 long-only implementation
- beta-matched long-short diagnostic
- sector-neutral long-short diagnostic
- cost stress at current default cost
- higher-cost stress
- turnover and capacity analysis
- correlation versus current single models
- correlation versus `C policy`

### Sleeve Layer

Only after standalone validation:

- `C policy + 3% non-price alpha`
- `C policy + 5% non-price alpha`
- `C policy + 6% non-price alpha`
- `C policy + 8% non-price alpha`
- volatility-targeted comparison at 20%

Sleeve acceptance should require both:

- improved validation Sharpe or drawdown-adjusted return
- clear explanation of whether the improvement comes from higher alpha Sharpe
  or lower correlation

## Validation Discipline

The final test lockbox must not be used for:

- source selection
- feature selection
- parser design decisions based on performance
- model selection
- hyperparameter selection
- sleeve weight selection
- cost assumption tuning

Initial work should stay in the validation/research window already used by the
pure-alpha line. Test-window results are allowed only after a candidate is
frozen in a validation memo.

## Phase Plan

### Phase 0: Source And PIT Audit

Goal:

Confirm exact availability, historical coverage, schema, and timestamp behavior
for each candidate source.

Tasks:

- verify source access path
- document historical coverage
- document publication latency
- define `available_at` rules
- map source identifiers to internal symbols
- identify licensing or usage constraints
- define raw snapshot format
- define normalized silver schemas

Deliverables:

- source audit memo
- entity mapping coverage report
- PIT timestamp contract
- first-source implementation recommendation

Exit criteria:

- no known timestamp ambiguity for Tier 1 source
- entity mapping coverage is measurable
- first feature family can be built without future leakage

### Phase 1: SEC Structured Fundamentals MVP

Goal:

Build the first model-ready non-price feature table from SEC structured data.

Tasks:

- ingest raw SEC company facts and submissions metadata
- normalize CIK, symbol, filing date, period end, and fields
- construct core accounting change features
- construct filing event features
- build daily h5-compatible gold table
- add feature-family routing
- run `lightgbm_ranker` ablation

Deliverables:

- SEC structured silver tables
- SEC structured gold features
- data quality report
- h5 ablation result
- pure-alpha diagnostic result

Exit criteria:

- model can train with SEC features switched on/off
- validation diagnostics show whether the feature family adds value
- no test lockbox usage

### Phase 2: Insider Transactions MVP

Goal:

Add sparse but high-information event behavior features.

Tasks:

- ingest Form 4 transaction records
- classify open-market buys and sells
- classify role and cluster behavior
- generate decayed event features
- test standalone and combined with SEC structured features

Deliverables:

- Form 4 feature table
- event-time signal decay report
- h5 and pure-alpha ablation report

Exit criteria:

- sparse event features are correctly aligned to trading sessions
- top-bucket conditional return is measurable
- interaction with SEC fundamentals is tested

### Phase 3: Short Interest And Short-Sale Flow

Goal:

Add crowding and pressure features.

Tasks:

- ingest short interest and daily short-sale volume
- model reporting lag conservatively
- create level, change, acceleration, and abnormality features
- test interactions with insider and attention features when available

Deliverables:

- short-data feature table
- crowding diagnostic memo
- long-only and long-short impact report

Exit criteria:

- signal is interpreted as crowding/pressure, not naive bearishness
- high-short candidates are evaluated under drawdown and squeeze risk

### Phase 4: Filing Text Features

Goal:

Extract simple, auditable text-change features from filings.

Tasks:

- parse 10-K / 10-Q / 8-K text
- segment risk factor and MD&A where feasible
- compute similarity and keyword-change features
- avoid opaque LLM-only features in first pass
- test text features after structured SEC baseline

Deliverables:

- filing text parsing coverage report
- text feature table
- text ablation report

Exit criteria:

- parser coverage is high enough for research
- text features add value beyond filing metadata and structured facts

### Phase 5: 13F Crowding

Goal:

Use slow ownership data as risk and crowding context.

Tasks:

- ingest holdings
- map issuers to symbols
- compute concentration and ownership changes
- test as risk filter and interaction layer

Deliverables:

- 13F feature table
- crowding risk memo
- portfolio exposure diagnostic

Exit criteria:

- feature latency is correctly modeled
- value is tested as risk/crowding, not fast alpha

### Phase 6: Attention And News

Goal:

Test whether attention and news proxies add conditional information.

Tasks:

- build entity mapping for attention/news sources
- create abnormal attention and news-volume features
- test mismatch signals against fundamentals, insider behavior, and short data
- run strict false-positive and robustness checks

Deliverables:

- attention/news feature table
- entity matching report
- attention mismatch ablation report

Exit criteria:

- attention features show incremental value after price and fundamentals
- false positives are documented by sector and event type

### Phase 7: Combined Non-Price Alpha Candidate

Goal:

Build a frozen candidate that can be compared against current h5 models and
used in sleeve experiments.

Tasks:

- combine accepted feature families
- keep family routing and ablation switches
- retrain `lightgbm_ranker`
- compare against baseline h5 model
- compare against top single models
- measure correlation to `C policy`
- run `C policy + alpha sleeve` only after standalone validation

Deliverables:

- combined non-price candidate report
- feature-family contribution table
- current model correlation matrix
- sleeve candidate update
- decision memo: promote, iterate, or shelve

Exit criteria:

- candidate improves validation alpha metrics
- incremental information is visible in ablation
- implementation is reproducible
- candidate is frozen before any lockbox test

## Initial Success Criteria

A source family is considered promising if it satisfies at least two of:

- improves validation rank IC after existing h5 features
- improves beta-neutral or sector-neutral top-bottom spread
- improves long-only h5 Sharpe after costs without raising turnover
  excessively
- improves pure-alpha long-short diagnostics
- lowers correlation versus existing single models while preserving positive
  standalone Sharpe
- improves `C policy + sleeve` validation results after standalone alpha has
  already been established

A source family should be shelved or demoted if:

- value disappears after sector/beta neutralization
- value appears only in one lucky subperiod
- value depends on aggressive timestamp assumptions
- coverage is too sparse for stable training
- feature construction is not reproducible
- the improvement comes mainly from increased hidden beta

## Main Risks

### Leakage

The biggest risk is using information before it was tradable:

- restated financial facts
- amended filings
- filing date versus acceptance timestamp confusion
- period-end fields treated as availability timestamps
- ticker mappings built from future identities
- attention data normalized with future windows

Mitigation:

- preserve all source timestamps
- use conservative `effective_session`
- snapshot raw payloads
- test with delayed-availability stress

### Sparse Events

Many useful non-price events are sparse. Dense daily models can overfit if we
forward-fill event features incorrectly.

Mitigation:

- use explicit event-age features
- use decay functions
- report event-conditioned diagnostics
- separate event features from stale panel features

### Multiple Testing

Non-price data creates many plausible features. This can produce accidental
alpha.

Mitigation:

- pre-register feature families
- run family ablations before fine tuning
- keep validation-only discipline
- promote candidates only with clear contribution evidence

### Entity Mapping

Ticker changes, mergers, delistings, CIK relationships, and ADR/security-class
issues can create hidden survivorship bias.

Mitigation:

- build mapping coverage reports
- require mapping confidence flags
- start with high-confidence common stocks
- keep lower-confidence names out of first claims

## Recommended First Execution

The first concrete execution should be:

1. Phase 0 audit for SEC structured data and Form 4.
2. Implement SEC structured fundamentals plus filing metadata.
3. Add h5 feature routing for `sec_structured`.
4. Run `lightgbm_ranker` feature-family ablation.
5. Run pure-alpha diagnostics before any sleeve experiment.
6. If SEC structured features pass, add Form 4 and test interactions.

This order gives us the best balance of:

- information quality
- implementation feasibility
- PIT auditability
- expected orthogonality to price-only features
- compatibility with the current h5 and pure-alpha research lines

## Current Stance

Non-price data should become a separate, reusable infrastructure layer, not a
one-off feature patch inside a single model.

The near-term goal is not to prove that "more data helps". The goal is to find
which non-price information channels survive strict point-in-time handling,
neutralized diagnostics, and cost-aware portfolio tests. If a source family
cannot pass those gates, it should be treated as research context or risk
metadata rather than promoted into the main alpha engine.
