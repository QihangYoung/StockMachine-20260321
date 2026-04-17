# US Equities Pure Alpha Phase 0 Data Audit

Date: 2026-04-17

## Purpose

This note records the Phase 0 pre-universe audit for the
`us_equities_pure_alpha_h5` research line.

Phase 0 is intentionally pre-alpha and pre-universe. It does not choose the
final universe, tune a signal, or inspect test-window strategy performance.

The question is narrower:

Can the current local `silver` data generate and compare credible
high-liquidity U.S. stock universe candidates for a beta-matched long-short
pure-alpha product?

## Artifact Root

Generated Phase 0 artifacts:

- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_pre_universe_data_audit_20260417/table_coverage.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_pre_universe_data_audit_20260417/candidate_universe_field_map.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_pre_universe_data_audit_20260417/candidate_feasibility_summary_validation.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_pre_universe_data_audit_20260417/candidate_feasibility_daily_validation.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_pre_universe_data_audit_20260417/available_stock_symbol_summary.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_pre_universe_data_audit_20260417/phase0_summary.json`

Supplemental top1000 backfill artifacts:

- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/top1000_manifest.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/candidate_asset_filter_audit.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/candidate_liquidity_ranking.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/daily_bar_backfill_chunks.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/adj_factor_backfill_chunks.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/top1000_backfill_coverage_by_symbol.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/top1000_data_backfill_20260417/top1000_backfill_coverage_rollup.json`

## Executive Decision

At the start of this audit, current data was sufficient for a small mechanics
baseline on the existing
`us_equities_research_v1` 66-stock basket.

That starting data was not sufficient to select the intended production-style
high-liquidity universe from top `500`, top `1000`, top `1500`, top `2000`, or
top `3000` U.S. stocks.

The reason is simple:

- `symbol_master` contains broad metadata, with `13,663` symbols.
- `daily_bar` contains only `77` symbols across stocks and ETFs.
- isolating the existing stock basket through `us_equities_research_v1` leaves
  only `66` stock price histories.

Therefore Phase 1 should split into two tracks:

- Track A: use the existing 66-stock basket only to validate mechanics such as
  beta estimation, beta matching, artifact shape, and robustness integration.
- Track B: build or ingest broader point-in-time daily bars before selecting the
  real high-liquidity pure-alpha universe.

Do not treat the 66-stock basket as the final product universe.

After the initial audit, a supplemental top1000 data backfill was completed.
This improves the available broad-stock data layer, but it still does not
solve point-in-time universe membership. The backfilled top1000 list is a
current-date liquidity-ranked bootstrap universe used to populate data, not a
final historical membership table.

Because Alpaca SIP did not return broad daily bars before 2016 during Phase 0
probing, this supplemental top1000 data starts on `2016-01-04`. Any broad
top1000 validation work based on this backfill must start no earlier than
`2016-01-04` plus the required beta and feature warm-up, unless an alternate
vendor fills `2013-08-05` through `2015-12-31`.

## Supplemental Top1000 Backfill

Selection method:

- source: Alpaca active `us_equity` assets
- liquidity feed: SIP
- ranking window: `2026-03-02` through `2026-04-16`
- ranking metric: median daily dollar volume
- target count: top `1000`
- eligibility: tradable, marginable, shortable, NYSE/NASDAQ/AMEX listed
- exclusions: obvious ETFs, ETNs, funds, warrants, rights, units, preferreds,
  depositary instruments, notes/bonds, SPAC/acquisition shells, and common ETF
  sponsor patterns

Backfill result:

| Item | Result |
|---|---:|
| active Alpaca `us_equity` assets scanned | `13,552` |
| common-like shortable candidates after filters | `3,567` |
| selected liquidity-ranked symbols | `1,000` |
| daily-bar window | `2016-01-04` to `2026-04-16` |
| top1000 `daily_bar` rows written | `2,338,647` |
| top1000 `adj_factor` rows written | `2,338,647` |
| corporate-action records captured | `23,582` |
| SPY benchmark rows written | `2,586` |
| symbols with any daily bars | `1,000` |
| symbols with matching `daily_bar` and `adj_factor` rows | `1,000` |
| symbols with latest session `2026-04-16` | `1,000` |
| symbols with at least `252` sessions | `985` |
| symbols with at least `756` sessions | `962` |
| symbols with near-full 2016-2026 coverage | `788` |

Interpretation:

- the top1000 data layer is now broad enough for Phase 1 mechanics on a real
  large-cap / mid-cap style stock set;
- the current top1000 list must not be used as a historical point-in-time
  universe without rebuilding membership from lagged liquidity;
- shorter-coverage symbols are expected from IPOs and listing changes and must
  be handled by warm-up and eligibility filters;
- shortability is still only current Alpaca metadata, not a historical borrow
  series.

## Table Coverage

Loaded table coverage:

| Table | Rows | Symbols | Date Column | Min Date | Max Date | Notes |
|---|---:|---:|---|---|---|---|
| `universe_membership` | `203,299` | `78` | `session_date` | `2014-01-02` | `2026-04-08` | Includes stock and ETF universe memberships. |
| `daily_bar` | `231,215` | `77` | `session_date` | `2013-08-01` | `2026-04-16` | Only 77 traded symbols locally; not broad enough for top500/top1000 selection. |
| `adj_factor` | `234,411` | `78` | `session_date` | `2013-08-01` | `2026-04-16` | Adjustment path exists. |
| `benchmark_index` | `3,196` | `1` | `session_date` | `2013-08-01` | `2026-04-16` | `SPY` benchmark exists. |
| `industry_membership` | `203,299` | `78` | `as_of_date` | `2014-01-02` | `2026-04-08` | Static backfill, not true historical industry events. |
| `symbol_master` | `377,987` | `13,663` | `as_of_date` | `2014-01-02` | `2026-04-16` | Broad metadata exists, but price bars do not. |

Important data-quality note:

`symbol_master` currently labels known ETFs such as `AGG`, `BIL`, `GLD`, `IEF`,
`LQD`, `VXUS`, and others as `COMMON_STOCK / us_equity` in the latest Alpaca
snapshot. Therefore ETF exclusion cannot rely only on `security_type` or
`asset_class`. For this audit, the stock-only set was isolated through
`universe_membership == us_equities_research_v1`.

## Field Availability

Fields available for candidate universe construction:

- session date
- symbol
- OHLCV
- dollar volume
- adjustment factor
- adjusted-open derivation path
- benchmark `SPY`
- static sector / industry classification for the current basket
- explicit membership for the legacy stock basket

Fields missing or incomplete:

- borrow fee
- short availability / locate
- short interest
- robust halt or suspension status
- true historical industry classification
- broad historical daily bars for top500/top1000/top1500 U.S. stocks
- reliable ETF/common-stock classification using `symbol_master` alone

## Validation-Window Feasibility

The feasibility check used:

- validation window: `2013-08-05` through `2019-12-31`
- current stock basket: `us_equities_research_v1`
- capital assumption: `500,000 USD`
- price floor: `10 USD`
- trailing median dollar volume window: `20` sessions
- beta lookback: `252` sessions
- beta minimum observations: `126`
- benchmark: `SPY`

Because current stock bars start on `2014-01-02` and beta needs warm-up, the
first nonzero beta-ready candidate session is `2014-07-03`.

Summary:

| Candidate | Status | Median Eligible Names | Max Eligible Names | Share Sessions Feasible For `20/20` | Share Sessions Feasible For `30/30` |
|---|---|---:|---:|---:|---:|
| `ADV >= 20M` | legacy-basket only | `64` | `65` | `85.58%` | `85.58%` |
| `ADV >= 30M` | legacy-basket only | `64` | `65` | `85.58%` | `85.58%` |
| `ADV >= 50M` | legacy-basket only | `64` | `65` | `85.58%` | `85.58%` |
| top `500` | blocked by current bars | `64` | `65` | `85.58%` | `85.58%` |
| top `1000` | blocked by current bars | `64` | `65` | `85.58%` | `85.58%` |
| top `1500` | blocked by current bars | `64` | `65` | `85.58%` | `85.58%` |

The identical counts across `ADV` thresholds are not evidence that all broad
U.S. stocks pass those thresholds. They only show that the current 66-stock
legacy basket is already highly liquid.

## Capacity Read

At `500,000 USD` capital, `100/100` gross, and `20` names per side:

- per-name notional is roughly `25,000 USD`
- median ADV participation is about `0.0044%`

At `150/150` gross and `20` names per side:

- per-name notional is roughly `37,500 USD`
- median ADV participation is about `0.0066%`

So capacity is not the binding constraint for the existing liquid basket. The
binding constraints are breadth, universe quality, shortability data, and
whether there is enough cross-sectional diversity to support real pure-alpha
research.

## Missing-Data Risks

### 1. Broad price coverage is missing

The starting local data could not construct top500/top1000/top1500 U.S. stock
universes because only 66 stock symbols had local daily bars.

The supplemental Alpaca backfill improves this from 2016 onward, but it still
does not fill the Beta-aligned validation start. The broad-stock data gap is now
specifically `2013-08-05` through `2015-12-31`.

Impact:

- no full-window broad high-liquidity universe selection yet
- no full-window universe breadth comparison yet
- no full-window evaluation of mid-large cap alpha outside the existing basket
- no claim of complete Beta-window alignment until an alternate vendor fills the
  2013-2015 gap

### 2. Shortability data is missing

No borrow fee, locate, short availability, or hard-to-borrow table was found in
the loaded `silver` dataset.

Impact:

- short book feasibility must be proxied through liquidity at first
- any short alpha result must be borrow-stressed
- hard-to-borrow alpha cannot be claimed as production-ready

### 3. Security-type classification is noisy

Latest `symbol_master` labels ETFs as `COMMON_STOCK / us_equity`.

Impact:

- ETF exclusion needs stronger logic
- Phase 1 should not rely only on `security_type`
- a curated ETF exclusion list or better asset classification source is needed

### 4. Industry history is static

`industry_membership` exists but is static backfill.

Impact:

- sector diagnostics are usable for rough validation
- true point-in-time industry neutrality is not fully solved
- industry drift should be interpreted carefully

### 5. Halt and suspension fields are missing

No robust halt, suspension, or tradability-status series was found.

Impact:

- v1 can use stale-price checks and liquidity filters
- production-quality trading eligibility needs more data

## Phase 0 Answer

Can we generate and compare credible high-liquidity universe candidates with
current data?

Answer:

Partially.

We can generate feasibility diagnostics for the existing 66-stock high-liquidity
legacy basket. We cannot yet generate the intended broad high-liquidity
universe candidates because broad daily price history is missing.

## Phase 1 Recommendation

Proceed to Phase 1 with two clearly separated tracks.

### Track A: Mechanics Baseline

Use `us_equities_research_v1` only as a mechanics baseline.

Allowed uses:

- beta estimator development
- beta-matched weight solver
- long-short artifact contract
- robustness-suite integration
- cost and turnover plumbing

Not allowed:

- declaring final product universe
- claiming broad high-liquidity alpha
- optimizing universe based on strategy returns

### Track B: Real Universe Build

Before real pure-alpha universe selection, build broader point-in-time daily bar
coverage.

Required next data target:

- at least top `500` U.S. high-liquidity common stocks by trailing dollar volume
- preferably enough coverage to evaluate top `1000`, top `1500`, top `2000`,
  and top `3000`
- daily OHLCV
- adjustment factors
- reliable common-stock / ETF classification
- sector / industry classification
- enough history for validation-window beta estimation

Universe-selection note:

Top `1000` and top `1500` should be treated as cleaner core candidates. Top
`2000` and top `3000` should also be included because the target capacity is
only `<= 500,000 USD`, but they must be evaluated as expansion candidates with
stricter cost, shortability, stale-price, gap-risk, and borrow-stress checks.

The Phase 1 design should also allow asymmetric universes:

- broader long-side candidate set
- more conservative short-side candidate set

This preserves the small-capacity alpha opportunity without pretending the
short book has the same borrow and squeeze risk as the long book.

Only after this broader coverage exists should Phase 1 choose a validation
default universe.

### Track C: 2013-2015 Vendor Gap Fill

The top1000 Alpaca backfill should not be stretched beyond its observed
coverage. To fully align with the Beta thread, add a vendor bake-off for broad
U.S. stock data from `2013-08-05` through `2015-12-31`.

Recommended candidate order:

| Vendor | Role | Reason |
|---|---|---|
| Norgate Data | preferred independent-research source | Strong survivorship-bias-free U.S. equities coverage, delisted stocks, and useful historical membership/context. |
| Sharadar / Nasdaq Data Link | preferred API-style source | Active plus delisted equities with repeatable ingestion and corporate-action support. |
| CRSP / WRDS | gold standard if accessible | Research-grade stock, distribution, and delisting data, but access is usually institutional. |
| Polygon.io | API fallback | Good market-data API and corporate actions, but dividend-adjusted total-return handling must be rebuilt and audited. |
| QuantQuote / HistoricalData.net / EODHD | low-cost backup | Potentially useful, but only after strict quality checks. |
| Yahoo / Stooq | sanity check only | Not robust enough as the primary source for final pure-alpha claims. |

Minimum acceptance checks:

- daily bars exist on `2013-08-05`, `2014-01-02`, and `2015-12-31`;
- delisted names are present or survivorship bias is explicitly quantified;
- split and dividend adjustments reconcile on sampled corporate-action events;
- common stocks can be separated from ETFs, ETNs, units, warrants, preferreds,
  and SPAC-like instruments;
- lagged trailing dollar-volume topN membership can be generated without future
  leakage;
- overlapping 2016+ samples reconcile against the current Alpaca SIP backfill.

## Lockbox Status

No test-window strategy performance was used in this Phase 0 audit.

The final test lockbox remains closed.
