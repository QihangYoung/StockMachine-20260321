# US Equities Pure Alpha Research Plan

Date: 2026-04-17

## Purpose

This plan defines the first execution path for the `us_equities_pure_alpha_h5`
research line.

The product goal is a beta-matched long-short U.S. equities strategy that can
run up to `500,000 USD` of capital and generate attractive pure stock-selection
alpha without depending on broad market direction.

This is a small-capacity product by institutional standards. That is an
advantage. We do not need to build a strategy that can absorb billions of
dollars. We can target higher alpha density, more focused positions, and
moderate turnover, while still enforcing strict beta neutrality and realistic
execution assumptions.

## Research Objective

Build and validate a U.S. high-liquidity stock cross-sectional strategy with
the following properties:

- long-short
- dollar neutral or close to dollar neutral
- beta matched between long and short books
- low realized beta versus `SPY`
- low market correlation
- cost-aware
- borrow-aware where data exists
- capacity-aware for `<= 500,000 USD`
- evaluated without opening the final test lockbox

The primary question is:

Can our selected long book reliably outperform our selected short book after
costs, while broad market direction explains little or none of the return?

## Product Shape Assumption

Initial product profile:

| Dimension | First Assumption |
|---|---:|
| Target capital | `<= 500,000 USD` |
| Frequency | daily decision, `h5` default holding horizon |
| Portfolio | beta-matched long-short |
| Starting gross | `100/100` |
| Later gross candidates | `125/125`, `150/150` |
| Long book size | `10 ~ 30` names |
| Short book size | `10 ~ 30` names |
| Initial universe | U.S. high-liquidity common stocks |
| Primary benchmark for beta | `SPY` |
| Test set | final lockbox only |

The product should not be optimized for institutional mega-capacity. It should
be optimized for clean alpha, tradability at small scale, and a return profile
that is meaningfully different from long-only equity exposure.

## Time Window Discipline

The timeline follows the Beta-thread validation architecture:

- validation / research window: `2013-08-05` through `2019-12-31`
- final test lockbox: `2020-01-02` through `2026-04-08`

If stock-level universe coverage starts after `2013-08-05`, the usable research
start may move later. The test start must not move earlier.

Phase 0 supplemental top1000 data currently starts on `2016-01-04`. If Phase 1
uses that broad top1000 backfill, the broad-stock validation start must move to
`2016-01-04` plus the required beta and feature warm-up. Keeping the Beta-thread
`2013-08-05` validation start requires an alternate vendor to fill broad stock
bars for `2013-08-05` through `2015-12-31`.

A provisional Yahoo chart gap fill now covers the missing `2013-08-05` through
`2015-12-31` segment for current top1000 bootstrap symbols. This allows
Beta-window plumbing and coverage checks, but it is not yet the final
survivorship-bias-free source for research claims.

A provisional point-in-time membership artifact also exists for the validation
window, built from lagged 20-session median dollar volume. It supports top500
and current-top1000-scope mechanics, but not true full-market top1500/top2000/
top3000 comparisons.

The final test window must not be used for:

- feature discovery
- signal selection
- model selection
- universe selection
- beta-matching parameter tuning
- gross exposure tuning
- cost assumption tuning
- borrow filter tuning
- deciding whether a candidate is promising

Routine work is validation-only. A test run is allowed only after a candidate
has been frozen in a validation memo.

## Capacity Implication

Because target capital is `<= 500,000 USD`, the strategy can use a more focused
and higher-conviction design than a large institutional fund.

Example at `100/100` gross:

- capital: `500,000 USD`
- long notional: `500,000 USD`
- short notional: `500,000 USD`
- total gross: `1,000,000 USD`
- `20` long names and `20` short names
- average position: `25,000 USD`

If each traded name has trailing median dollar volume above `50,000,000 USD`,
the average position is only `0.05%` of daily dollar volume.

This supports:

- more focused books
- moderate turnover
- use of names outside only mega-cap indices
- alpha ideas that would be too small for a multi-billion-dollar manager

It does not justify:

- ignoring borrow constraints
- ignoring bid-ask and slippage
- trading illiquid small caps
- running concentrated short books without risk caps
- opening the test lockbox early

## Phase 0: Pre-Universe Data Audit And Candidate Design

Goal:

Define what data is needed to choose a universe, then check whether the current
`silver` layer can support candidate-universe construction and feasibility
comparison.

Phase 0 does not decide that the final strategy data is sufficient. That would
be premature because the universe is not selected yet. Phase 0 only answers:

Can we generate and compare credible high-liquidity universe candidates without
future leakage?

Tasks:

- define candidate-universe rules before looking at performance
- map which fields are required to construct those candidates
- confirm whether current `silver` tables contain those fields
- check daily bars, adjusted factors, benchmark index, symbol master, and
  industry membership coverage for candidate generation
- identify whether any explicit high-liquidity universe table already exists
- identify which universe fields must be derived from trailing data
- document gaps in shortability, borrow fee, and corporate action data
- define the default validation-only artifact root under
  `us_equities_pure_alpha_h5`

Candidate universe families to evaluate in Phase 1:

- `ADV >= 50M`
- `ADV >= 30M`
- `ADV >= 20M`
- `ADV >= 10M` as an expansion candidate only
- top `500` by trailing dollar volume
- top `1000` by trailing dollar volume
- top `1500` by trailing dollar volume
- top `2000` by trailing dollar volume
- top `3000` by trailing dollar volume
- stricter short-side variant for each candidate when borrow data is missing
- asymmetric variants where the long book can draw from a broader universe than
  the short book

Fields needed for candidate construction:

- session date
- symbol
- open, close, volume, and dollar volume
- adjusted open or adjustment factor
- security type / listing metadata
- active or tradable status
- sector and industry metadata
- benchmark `SPY` return
- enough trailing history for liquidity and beta estimation
- borrow fee, short availability, or proxy fields when available

Feasibility checks:

- daily eligible-name count by candidate rule
- daily beta-estimable count by candidate rule
- sector and industry breadth
- missing adjusted-price rate
- missing metadata rate
- minimum feasible long and short book size
- expected single-name notional at `500,000 USD` capital
- expected ADV participation at `100/100`, `125/125`, and `150/150` gross

Deliverables:

- pre-universe data audit note
- candidate-universe field map
- table coverage summary for candidate construction
- missing-data risk list
- proposed Phase 1 candidate universe set
- decision on whether Phase 1 should use explicit membership, derived trailing
  liquidity membership, or both
- supplemental top1000 data backfill manifest and coverage rollup

Exit criteria:

- candidate-universe rules are defined before performance research
- required construction fields are present or explicitly marked missing
- no known future-dated field is required for candidate generation
- adjusted open-to-open return path appears available for feasibility checks
- benchmark return path appears available for beta feasibility checks
- shortability and borrow gaps are documented
- test lockbox remains untouched

Phase 0 status as of `2026-04-18`:

- a current-date liquidity-ranked top1000 bootstrap manifest exists;
- top1000 raw daily bars and adjustment factors were backfilled from Alpaca SIP
  for `2016-01-04` through `2026-04-16`;
- provisional Yahoo chart data fills `2013-08-05` through `2015-12-31` for
  current top1000 bootstrap symbols;
- provisional validation-window membership artifacts were generated using
  lagged 20-session median dollar volume;
- a repeatable Phase 0 utility exists at
  `stockmachine.apps.run_pure_alpha_phase0`;
- asset-class QA found `0` obvious ETF/fund-like names in the selected top1000,
  but `96` symbols require review, mostly foreign / ADR-like names;
- the default Phase 1 clean core excludes those review-required names, leaving
  `904` bootstrap symbols;
- current metadata shows `1,000 / 1,000` selected symbols as shortable and
  easy-to-borrow, but this remains only a current-date proxy;
- vendor bake-off artifacts exist with Norgate, Sharadar, CRSP, and Polygon as
  the first evaluation order;
- a Phase 0 closure packet now exists under
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase0_closure_20260418`;
- the top1000 data backfill is suitable for Phase 1 data engineering and
  validation-only mechanics;
- it is not a point-in-time historical universe and must be converted into
  lagged, session-scoped membership before alpha research;
- full Beta-window plumbing is now possible, but final research claims remain
  blocked until survivorship-bias-free membership and adjustment quality are
  solved by a primary vendor or stricter audit.

## Phase 0B: Broad Data Gap Fill

Goal:

Fill and validate the broad U.S. stock data gap from `2013-08-05` through
`2015-12-31` so the pure-alpha line can align with the Beta-thread validation
window instead of starting broad-stock research in 2016.

Current status:

- provisional Yahoo chart gap fill exists for current top1000 symbols;
- `470,021` `daily_bar` rows and `470,021` `adj_factor` rows were written;
- `801` current top1000 symbols have at least one gap-fill row;
- `747` current top1000 symbols have full `608`-session coverage;
- raw OHLCV scale reconciles tightly to Alpaca on a 2016 sample after split
  reconstruction;
- adjusted factors remain provisional because complex corporate actions can
  diverge from Alpaca.

Provisional membership status:

- validation-window lagged-liquidity artifacts exist for `2013-08-05` through
  `2019-12-31`;
- membership uses trailing median dollar volume over `20` prior sessions with
  at least `15` observations and a lagged `10 USD` price floor;
- median liquidity-eligible count inside current top1000 scope is `744.5`;
- top500 mechanics are supported inside the current top1000 scope;
- top1000 is only a current-scope top1000 artifact, not a full-market
  historical top1000;
- top1500/top2000/top3000 remain blocked until broader data is backfilled.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase0 all
```

This command regenerates:

- provisional validation-window lagged-liquidity membership;
- current top1000 asset-class QA;
- vendor bake-off matrix and acceptance checks;
- Phase 0 closure memo, local policy decisions, and Phase 1 blocker list.

Phase 1 entry decision:

- start mechanics work with validation-only top500 and current-top1000-scope
  diagnostics;
- use the `904`-symbol clean core as the first default universe candidate;
- keep top1500/top2000/top3000 as blocked expansion candidates until broader
  historical bars are available;
- reuse the Beta-line robustness framework before promoting any pure-alpha
  result;
- keep final performance claims gated on primary-vendor selection, PIT
  survivorship-bias-free membership, adjustment reconciliation, and short-book
  cost realism.

Required data:

- daily OHLCV for active and delisted U.S. common stocks
- split and dividend adjustment path, or enough corporate actions to rebuild it
- historical listing status and ticker-change handling
- reliable common-stock / ETF / ETN / unit / warrant filtering
- enough breadth to form top `1000`, top `1500`, top `2000`, and top `3000`
  candidates from lagged trailing dollar volume

Preferred vendor order:

| Vendor | Use Case | Research Note |
|---|---|---|
| Norgate Data | preferred independent-research source | Strong fit for survivorship-bias-free U.S. equities, delisted stocks, and historical membership/context. |
| Sharadar / Nasdaq Data Link | preferred API-style source | Strong fit for active plus delisted equities and repeatable ingestion. |
| CRSP / WRDS | gold-standard source if available | Best research quality, but usually requires institutional access. |
| Polygon.io | API fallback | Good coverage and corporate actions, but dividend adjustment must be rebuilt and audited. |
| QuantQuote / HistoricalData.net / EODHD | low-cost backup | Only acceptable after strict split, dividend, delisting, and ETF-filter validation. |
| Yahoo / Stooq | sanity-check only | Not acceptable as the primary research source for final claims. |

Vendor bake-off acceptance checks:

- can load daily bars for `2013-08-05`, `2014-01-02`, and `2015-12-31`;
- can generate lagged topN membership without using future liquidity;
- includes delisted names or otherwise documents survivorship bias clearly;
- adjustment factors match known split/dividend events on sampled names;
- ETF, ETN, preferred, warrant, unit, and SPAC-like instruments can be filtered;
- overlap and return samples reconcile against the existing Alpaca 2016+ data;
- ingestion cost and update mechanics are acceptable for repeated research runs.

Exit criteria:

- one primary vendor is selected for the 2013-2015 gap;
- broad-stock `daily_bar` and `adj_factor` coverage is available from
  `2013-08-05`;
- the first Phase 1 universe construction run can use the full validation
  window without pretending the current Alpaca top1000 manifest is historical
  membership.

Until those exit criteria are met, the Yahoo gap fill may be used for plumbing,
coverage, and beta-mechanics experiments, but not as the final evidence layer
for production-grade alpha claims.

## Phase 1: Point-In-Time High-Liquidity Universe

Goal:

Build and compare point-in-time candidate universes that match the product's
small-capacity but high-quality execution target, then choose a validation-only
default universe for the first baseline cycle.

Initial eligibility rules:

- U.S.-listed common equity or liquid primary share class
- price above `10 USD`
- trailing median dollar volume above `50,000,000 USD`
- enough lagged history for beta estimation and features
- no stale price
- no missing adjusted open needed for entry or exit
- no known non-tradable status

Short-side conservative filters:

- higher liquidity preference than long side
- avoid extreme short-interest names unless explicitly modeled
- avoid hard-to-borrow names when borrow data is available
- apply borrow-cost stress if real borrow cost is unavailable

Candidate variants:

- `ADV >= 50M`
- `ADV >= 30M`
- `ADV >= 20M`
- `ADV >= 10M` as an expansion candidate only
- top `500`, top `1000`, top `1500`, top `2000`, and top `3000` by liquidity
- core universe variants: top `1000` / top `1500`
- expansion universe variants: top `2000` / top `3000`
- asymmetric variants: broader long-side universe with a more conservative
  short-side universe

Important rule:

Universe variants are compared on feasibility, tradability, breadth, and
robustness readiness before alpha performance. Once a strategy candidate is
frozen, the universe definition must also be frozen before any test run.

Do not pick the universe because it maximizes backtest return. Pick the Phase 1
default because it is point-in-time safe, liquid enough, broad enough for
balanced long-short construction, and appropriate for `<= 500,000 USD`.

For this product size, top `2000` and top `3000` are legitimate research
candidates. A `500,000 USD` strategy may be able to trade smaller high-liquidity
names that are unattractive to large institutions. The trade-off is that wider
universes require stricter execution and short-side controls.

Phase 1 should therefore treat the universe as liquidity buckets:

| Bucket | Intended Use |
|---|---|
| top `500` | safest liquidity and borrow baseline, but likely most crowded |
| top `1000` | conservative core candidate |
| top `1500` | wider core candidate with still-manageable liquidity |
| top `2000` | small-capacity expansion candidate |
| top `3000` | highest breadth / highest data-quality and shortability burden |

Expansion buckets must not become the default unless they pass additional
checks:

- higher cost stress
- borrow or shortability stress
- jump and gap-risk diagnostics
- stale-price and halt-risk diagnostics
- stricter short-side liquidity filters
- sector and size exposure attribution

The long and short universes do not have to be identical. One reasonable small-
capacity design is:

- long side: allow top `2000` or top `3000` if liquidity and data quality pass
- short side: restrict to top `1000` or top `1500`, or easy-to-borrow names when
  borrow data exists

The final portfolio must still satisfy beta matching, exposure controls, and
cost realism.

Deliverables:

- universe construction module or script
- universe membership artifact
- universe coverage report
- universe stability manifest for robustness suite
- recommended validation-default universe
- secondary universe variants reserved for robustness checks

Exit criteria:

- membership is session-scoped
- no use of future liquidity or future membership
- enough names per day for balanced long-short construction
- short-side eligibility is documented
- a validation-default universe is chosen without using test-window performance

Immediate implementation note:

The Phase 0 top1000 backfill should be treated as a raw data lake for Phase 1.
The next module should derive daily topN and ADV-threshold membership from
lagged trailing dollar volume, rather than reading `top1000_manifest.csv` as a
fixed historical universe.

The provisional Phase 0 membership artifact is acceptable as the template for
that module, but Phase 1 should re-run it from a repeatable ingestion pipeline
and with the final selected data source.

Phase 1 builder status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase1`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase1_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase1_universe_builder_20260418`;
- recommended mechanics default: `top500_clean_core_beta_full`;
- supported diagnostics: `top1000_clean_core_beta_full`, `adv50m`,
  `adv30m`, `adv20m`, and `adv10m` clean-core beta-full variants;
- blocked variants: top1500/top2000/top3000, because the current backfill has
  only current-top1000-scope symbols;
- first 20-by-20 long/short readiness date: `2014-08-05`;
- `top500_clean_core_beta_full` median members: `454.0`, minimum members:
  `446`;
- no alpha signal, return, model selection, or test-window performance was
  computed.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase1
```

Generated artifacts:

- `phase1_candidate_universe_membership_validation.csv.gz`;
- `phase1_candidate_universe_daily_counts_validation.csv`;
- `phase1_candidate_universe_summary_validation.csv`;
- `phase1_universe_stability_manifest.csv`;
- `phase1_universe_builder_memo.md`;
- `phase1_universe_builder_rollup.json`.

## Phase 2: Beta Estimation Layer

Goal:

Estimate lagged stock betas robustly enough to support beta-matched portfolio
construction.

Default beta specification:

- benchmark: `SPY`
- return input: adjusted daily returns
- lookback: `252` sessions
- minimum observations: `126`
- shrinkage target: `1.0`
- clip: `[0.0, 3.0]`
- update frequency: daily

Research variants:

- lookback `126`
- lookback `252`
- lookback `504`
- shrinkage versus no shrinkage
- beta clip `[0.0, 2.5]` versus `[0.0, 3.0]`

The beta estimator must use only data available before the portfolio decision.

Deliverables:

- beta estimate panel
- beta coverage report
- beta stability diagnostics
- validation-only comparison of beta specifications

Exit criteria:

- beta estimates exist for most eligible names
- missing-beta handling is explicit
- ex-ante net beta can be measured every session
- no future return leakage

Phase 2 beta panel status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase2`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase2_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase2_beta_panel_20260418`;
- beta panel rows: `923,359`;
- beta-covered symbols: `763`;
- supported variants with 100% beta coverage: `top500_clean_core_beta_full`,
  `top1000_clean_core_beta_full`, `adv50m`, `adv30m`, `adv20m`, and `adv10m`;
- first beta coverage date: `2014-08-05`;
- default beta spec: SPY adjusted close-to-close returns, `252` lookback,
  `126` minimum observations, one-session as-of lag, `10%` shrinkage toward
  `1.0`, and clip `[0.0, 3.0]`;
- median beta for the recommended `top500_clean_core_beta_full` variant:
  `1.0112480372524817`;
- no alpha signal, portfolio return, model selection, or test-window
  performance was computed.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase2
```

Generated artifacts:

- `phase2_beta_panel_validation.csv.gz`;
- `phase2_variant_beta_coverage_validation.csv`;
- `phase2_beta_coverage_summary_validation.csv`;
- `phase2_symbol_beta_stability_validation.csv`;
- `phase2_beta_panel_memo.md`;
- `phase2_beta_panel_rollup.json`.

## Phase 3: Baseline Signals

Goal:

Start with simple transparent signals before adding complex models.

Candidate signal families:

- short-term reversal
- medium-term momentum
- residual momentum after market beta adjustment
- volatility-adjusted momentum
- quality or profitability proxies if available
- liquidity and volume pressure features
- gap and overnight behavior
- simple earnings or event avoidance filters if data exists

First baseline ladder:

1. random ranking sanity check
2. simple momentum and reversal factors
3. z-scored composite of transparent factors
4. rank-based model using existing tree models only after simple baselines are
   understood

Signal diagnostics:

- daily IC
- daily RankIC
- IC information ratio
- top-minus-bottom spread
- long-leg return
- short-leg return
- sector contribution
- beta-adjusted spread

Deliverables:

- validation-only factor leaderboard
- IC report
- signal correlation matrix
- simple composite candidate

Exit criteria:

- at least one simple signal shows stable validation spread
- short leg contributes economically
- signal is not just market beta or sector tilt
- signal survives basic cost assumptions

Phase 3 baseline status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase3`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase3_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase3_baseline_signals_20260418`;
- validation span used by the labeled signal panel: `2014-08-05` through
  `2019-12-20`;
- signal panel rows: `4,572,955`;
- daily diagnostic rows: `65,088`;
- leaderboard rows: `48`;
- primary target: `forward_beta_residual_return_5d`;
- forward label: adjusted open-to-open return from the next session open to the
  open after the configured holding period;
- baseline signal families: random control, 5-session reversal, 20-session
  momentum, 60-session momentum, beta-residual momentum, volatility-adjusted
  momentum, liquidity rank, and a transparent composite;
- strongest simple validation baseline across supported variants:
  `reversal_5d`;
- recommended `top500_clean_core_beta_full` snapshot: mean RankIC `0.011937`,
  mean top-minus-bottom beta-residual spread `0.000818`, spread hit rate
  `0.519174`;
- random control sanity check is near zero on the same variant: mean RankIC
  `0.000062`, mean spread `-0.000100`;
- no beta-matched portfolio construction, model selection, production claim, or
  test-window performance was computed.

Interpretation:

The first transparent result says "there is a small but measurable validation
reversal effect worth challenging." It does not yet say "we have a product."
The next step must translate the score into a beta-matched long-short book,
charge realistic costs, and verify that both legs contribute after constraints.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase3
```

Generated artifacts:

- `phase3_baseline_signal_panel_validation.csv.gz`;
- `phase3_baseline_signal_daily_diagnostics_validation.csv`;
- `phase3_baseline_signal_leaderboard_validation.csv`;
- `phase3_signal_correlation_validation.csv`;
- `phase3_baseline_signal_memo.md`;
- `phase3_baseline_signal_rollup.json`.

## Phase 4: Beta-Matched Long-Short Portfolio Constructor

Goal:

Translate cross-sectional scores into a tradable beta-neutral long-short book.

Default construction:

- choose top-ranked names for long book
- choose bottom-ranked names for short book
- start with equal weights inside each side
- solve or rescale weights to match long and short beta exposure
- enforce gross and single-name caps
- record skip reason when constraints cannot be satisfied

Initial constraints:

- gross: `100/100`
- max single-name weight: `5%` of gross per side
- minimum names per side: `10`
- target names per side: `20`
- max names per side: `30`
- ex-ante net beta: `<= 0.05`
- net dollar exposure: close to `0`

Research variants:

- `10/10`, `20/20`, `30/30` books
- equal weight versus score weight
- beta-rescaled equal weight
- industry-neutral version
- volatility-scaled position caps

Deliverables:

- portfolio construction module
- per-session weights
- beta matching diagnostics
- skipped-session report

Exit criteria:

- book can be constructed on most validation sessions
- realized beta remains within gate
- long and short books are both economically active
- constraints are visible in artifacts

Phase 4 beta-matched portfolio status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4_beta_matched_portfolios_20260418`;
- input signal panel rows loaded from Phase 3: `4,572,955`;
- baseline signal: `reversal_5d`;
- validation span: `2014-08-05` through `2019-12-20`;
- constructed session rows: `6,493`;
- skipped session rows: `1,643`;
- position rows: `363,186`;
- construction setting: `100/100` gross, `20` target names per side, up to
  `30` candidates per side, `5%` max single-name side weight, and `0.05` net
  beta tolerance;
- required nonzero names per side is effectively `20`, because a `5%` single-
  name cap needs at least `20` positions to fill one side;
- all skipped sessions were due to `beta_range_no_overlap`, meaning the top and
  bottom candidate sets could not be beta-matched under the first strict weight
  constraints;
- constructed books achieved near-zero ex-ante net beta; mean absolute net beta
  is numerical-noise level across variants;
- recommended `top500_clean_core_beta_full` snapshot: construction rate
  `0.776549`, mean overlapping 5-session spread `0.000925`, spread hit rate
  `0.518519`, and median `29` nonzero names per side;
- `top1000_clean_core_beta_full` snapshot: construction rate `0.825959`, mean
  overlapping 5-session spread `0.001097`, spread hit rate `0.508036`, and
  median `29` nonzero names per side;
- `adv10m_clean_core_beta_full` has the strongest raw Phase 4 diagnostic spread
  at `0.001270`, but it carries the highest execution and shortability burden
  among the supported variants;
- the first pass does not yet satisfy the "both legs contribute" product
  standard: average short-leg contribution is negative across variants, so the
  positive spread is currently driven mainly by the long leg;
- no transaction costs, borrow costs, turnover accounting, non-overlapping
  backtest conversion, production claim, candidate freeze, or test-window
  performance was computed.

Interpretation:

The first portfolio-construction pass shows that the signal can be turned into
a mechanically beta-matched long-short book on most validation sessions. The
remaining hard problem is not "can we solve beta"; it is "can we keep enough
high-alpha names while satisfying beta, borrow, cost, turnover realism, and a
short book that actually contributes." Phase 5 should therefore convert these
overlapping diagnostics into a standard backtest packet and charge realistic
costs before treating any spread as an economic result.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4
```

Generated artifacts:

- `phase4_beta_matched_positions_validation.csv.gz`;
- `phase4_portfolio_daily_diagnostics_validation.csv`;
- `phase4_skipped_sessions_validation.csv`;
- `phase4_portfolio_summary_validation.csv`;
- `phase4_beta_matched_portfolio_memo.md`;
- `phase4_beta_matched_portfolio_rollup.json`.

## Phase 4B: Short Book Diagnosis

Goal:

Pause before Phase 5 and diagnose why the first beta-matched short book does
not yet contribute economically.

Phase 4B exists because a positive long-short spread is not enough for this
product. A pure-alpha strategy should not simply hide a long-side edge inside a
short overlay. Before building the formal backtest artifact, we must understand
whether the short-leg failure is caused by the signal, the market regime, beta-
matching constraints, symmetric long/short design, missing borrow data, or some
combination of these.

Diagnostic questions:

- does the short side fail before beta matching, using equal-weight raw signal
  candidates?
- does the short side only fail after Phase 4 beta-matching weights are applied?
- is the short problem visible in beta-residual returns, or only in raw returns?
- do other transparent signals produce better short candidates than
  `reversal_5d`?
- are skipped Phase 4 sessions mostly caused by beta feasibility rather than
  insufficient universe breadth?
- should the next construction pass use asymmetric long and short signals?

Deliverables:

- equal-weight long/short candidate side diagnostics by signal;
- Phase 4 weighted long/short side diagnostics;
- short-failure matrix labeling likely failure modes;
- Phase 4 skip-reason summary;
- short-book diagnosis memo;
- recommendation for whether to continue with symmetric `reversal_5d`, switch
  to asymmetric short selection, or return to signal research.

Exit criteria:

- short-leg failure is attributed to a specific mechanism or marked unknown;
- at least one candidate short-side remedy is specified without using test data;
- Phase 5 is not started until the short-book caveat is explicitly accepted or
  addressed.

Phase 4B short-book diagnosis status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4b`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4b_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4b_short_book_diagnosis_20260418`;
- input signal panel rows loaded from Phase 3: `4,572,955`;
- side diagnostic rows: `143,162`;
- side summary rows: `108`;
- failure matrix rows: `48`;
- skip summary rows: `6`;
- candidate count for equal-weight side diagnostics: `30` names per side;
- diagnosed signals: random control, `reversal_5d`, `momentum_20d`,
  `momentum_60d`, beta-residual momentum, volatility-adjusted momentum,
  liquidity rank, and transparent composite;
- no transaction costs, borrow costs, turnover costs, candidate freeze, or
  test-window performance was computed.

Main finding:

The Phase 4 short-book problem is more subtle than "the short alpha does not
exist." In raw return terms, the short leg loses money on average across
variants. In beta-residual terms, however, `reversal_5d` short candidates are
positive in all six supported variants before beta matching, and the actual
Phase 4 weighted short book remains beta-residual positive in all six variants.

Key `reversal_5d` diagnostics:

- mean equal-weight short beta-residual contribution across variants:
  `0.000415`;
- equal-weight short beta-residual contribution is positive in `6 / 6`
  variants;
- Phase 4 weighted short beta-residual contribution is also positive in `6 / 6`
  variants;
- raw short contribution remains negative across variants, for example
  `-0.001641` in `top500_clean_core_beta_full` and `-0.002089` in
  `top1000_clean_core_beta_full`;
- weighted short beta-residual contribution is modest: `0.000242` in
  `top500_clean_core_beta_full` and only `0.000063` in
  `top1000_clean_core_beta_full`;
- weighted long beta-residual contribution is still larger than short
  contribution in `top500`, `top1000`, and `adv10m`, so the book is not yet
  balanced from an alpha-contribution perspective.

Signal comparison:

- `reversal_5d` is the strongest short-side candidate in aggregate, with mean
  equal-weight short beta-residual contribution `0.000415`;
- `momentum_60d` is a weaker secondary short candidate, viable in `4 / 6`
  variants with mean equal-weight short beta-residual contribution `0.000124`;
- `momentum_20d`, beta-residual momentum, transparent composite, volatility-
  adjusted momentum, and liquidity-rank short candidates are negative on
  average in this diagnostic;
- random control remains negative on average, which supports the diagnosis that
  the `reversal_5d` short residual edge is not just mechanical noise.

Constraint finding:

All Phase 4 skipped sessions remain `beta_range_no_overlap`:

- `top500_clean_core_beta_full`: `303` skipped sessions;
- `top1000_clean_core_beta_full`: `236` skipped sessions;
- `adv50m_clean_core_beta_full`: `298` skipped sessions;
- `adv30m_clean_core_beta_full`: `280` skipped sessions;
- `adv20m_clean_core_beta_full`: `273` skipped sessions;
- `adv10m_clean_core_beta_full`: `253` skipped sessions.

Interpretation:

The original Phase 4 caveat should be refined. The short leg is not yet
attractive on raw P&L, but it does show positive beta-residual contribution.
For a pure-alpha product, beta-residual contribution is the cleaner diagnostic,
while raw short P&L will often look bad in a rising market. The next research
step should therefore avoid treating raw short loss as automatic failure, but
should still challenge whether the short residual edge is large enough after
costs, borrow, turnover, and skipped-session handling.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4b
```

Generated artifacts:

- `phase4b_short_book_side_diagnostics_validation.csv`;
- `phase4b_short_book_side_summary_validation.csv`;
- `phase4b_short_failure_matrix_validation.csv`;
- `phase4b_phase4_skip_summary_validation.csv`;
- `phase4b_short_book_diagnosis_memo.md`;
- `phase4b_short_book_diagnosis_rollup.json`.

## Phase 4C: Residual Loser Selector Lab

Goal:

Research how to better identify future beta-residual losers without expanding
the universe.

Phase 4C deliberately keeps the same supported Phase 1 universe variants. It
does not solve the short-book problem by reaching into a broader stock pool.
Instead, it asks whether the current feature set can be transformed into better
short-side selectors, and which new feature families should be sourced later.

Diagnostic questions:

- are future residual losers present inside the current universe?
- how much headroom exists versus an oracle that knows future residual losers?
- does the current `reversal_5d` short rule miss most future residual losers?
- do price-only transformations such as overextension, weakness continuation,
  beta fragility, price fragility, and liquidity fragility improve selection?
- which feature families require new data before they can plausibly improve the
  short book?

Candidate selector families:

- current Phase 4 short rule: recent 5-session winner;
- weakness continuation: weak 20-session, weak 60-session, and weak residual
  momentum;
- overextension: 20-session winner plus 5-session extension;
- residual overextension: 20-session residual winner plus 5-session extension;
- breakdown after strength: 60-session strength plus recent 5-session weakness;
- fragility proxies: high beta, lower price, lower ADV, lower liquidity;
- fragile winner proxy: recent winner plus high beta and lower ADV;
- random control.

Deliverables:

- daily selector diagnostics by variant and candidate selector;
- selector summary table;
- improvement-versus-current-reversal table;
- same-universe residual-loser opportunity summary;
- feature roadmap separating currently derivable features from new-data
  feature families;
- residual-loser lab memo.

Exit criteria:

- at least one candidate short selector is identified for a future Phase 4
  construction rerun, or the current feature set is declared inadequate;
- the need for new data is documented explicitly;
- no universe expansion or test-window use occurs.

Phase 4C residual-loser lab status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4c`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4c_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4c_residual_loser_lab_20260418`;
- input signal panel rows loaded from Phase 3: `4,572,955`;
- daily selector rows: `113,904`;
- selector summary rows: `84`;
- improvement-versus-reversal rows: `84`;
- opportunity rows: `6`;
- feature-roadmap rows: `8`;
- candidate count per selector: `30` names;
- loser quantile for capture diagnostics: bottom `20%` by future beta-residual
  return;
- no universe expansion, transaction costs, borrow costs, portfolio
  construction, model selection, candidate freeze, or test-window performance
  was computed.

Same-universe opportunity:

- future beta-residual losers are present inside the current supported
  universes: the mean negative residual share is about `49.3% ~ 49.4%` across
  variants;
- the oracle bottom-30 short residual contribution is very large, around
  `0.0697` in `top500_clean_core_beta_full`, `0.0851` in
  `top1000_clean_core_beta_full`, and `0.0814` in `adv10m_clean_core_beta_full`;
- this oracle is not tradable and uses future labels, but it proves the problem
  is not simply "there are no losers in the current universe."

Main selector finding:

The best same-universe price-only direction is not generic weakness
continuation. It is overextension:

- `short_exhausted_winner_20_5` is the strongest aggregate selector, with mean
  short beta-residual contribution `0.000995` across supported variants;
- `short_residual_overextension_20_5` is second, with mean `0.000932`;
- the current `short_reversal_winner` baseline averages only `0.000415`;
- `short_exhausted_winner_20_5` improves over the current reversal short rule
  in all six variants;
- `top500_clean_core_beta_full` is an exception where `short_fragile_winner_proxy`
  is best at `0.001149`, slightly ahead of `short_exhausted_winner_20_5` at
  `0.001117`;
- `top1000_clean_core_beta_full` improves from `0.000225` under current
  reversal short selection to `0.000576` under `short_exhausted_winner_20_5`;
- `adv20m_clean_core_beta_full` improves from `0.000696` to `0.001181`;
- `adv30m_clean_core_beta_full` improves from `0.000545` to `0.001155`;
- `adv50m_clean_core_beta_full` improves from `0.000229` to `0.000986`;
- `adv10m_clean_core_beta_full` improves from `0.000471` to `0.000953`.

Negative selector finding:

- weak 20-session momentum and weak residual 20-session momentum are negative
  on average as short selectors in this diagnostic;
- lower ADV / lower liquidity proxy is also negative on average;
- this argues against the naive idea that the short book should simply short
  already-weak or lower-liquidity names;
- current evidence points instead to "overextended winners that start to mean
  revert" as the better same-universe residual-loser direction.

New-data roadmap:

Currently derivable features can improve the short selector, but the oracle gap
remains enormous. Before a product candidate freeze, the short book likely
needs richer information:

- fundamental quality: profitability, accruals, leverage, dilution, and margin
  deterioration;
- earnings and revisions: estimate cuts, negative surprises, guidance cuts, and
  post-earnings drift;
- valuation and growth mismatch: expensive names with slowing growth or margin
  pressure;
- borrow and short-interest data: borrow fee, utilization, shares available,
  short interest, and days to cover;
- event and news risk: fraud, regulatory, legal, downgrade, and product-failure
  flags.

Interpretation:

Phase 4C refines the short-book path. We should not expand the universe yet.
The current universe already contains many residual losers, but the existing
short selector is too blunt. The next construction pass should test an
asymmetric short side using `short_exhausted_winner_20_5` and possibly
`short_residual_overextension_20_5` or `short_fragile_winner_proxy`, while
keeping the long side separately evaluated. If these price-only improvements do
not survive beta matching and costs, the next research bottleneck is feature
quality, not universe width.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4c
```

Generated artifacts:

- `phase4c_residual_loser_selector_daily_validation.csv`;
- `phase4c_residual_loser_selector_summary_validation.csv`;
- `phase4c_residual_loser_improvement_vs_reversal_validation.csv`;
- `phase4c_residual_loser_universe_opportunity_validation.csv`;
- `phase4c_residual_loser_feature_roadmap.csv`;
- `phase4c_residual_loser_lab_memo.md`;
- `phase4c_residual_loser_lab_rollup.json`.

## Phase 4D: Tree Residual-Loser Selector

Goal:

Test whether a simple tree model can outperform the same-universe price-only
short selectors from Phase 4C.

Phase 4D is still a selector diagnostic, not a portfolio construction pass. The
model is allowed to combine existing Phase 3 features nonlinearly, but it must
respect time ordering. It trains only on older validation sessions, leaves a
label embargo before each prediction block, and does not expand the universe.

Default model:

- model: `sklearn.ensemble.ExtraTreesRegressor`;
- target: `forward_beta_residual_return_5d`;
- selector score: negative predicted residual return;
- higher selector score means more likely residual loser;
- default variant: `top1000_clean_core_beta_full`;
- initial train window: `252` sessions;
- label embargo: `5` sessions;
- prediction block: `126` sessions;
- maximum training rows per fold: `150,000`;
- estimators: `96`;
- max depth: `5`;
- minimum samples per leaf: `200`.

Features:

- lagged price and trailing dollar-volume logs;
- liquidity rank and beta;
- 5-session return, 20-session momentum, 60-session momentum;
- beta-residual 20-session momentum;
- volatility-adjusted 20-session momentum;
- cross-sectional z-scores of return, momentum, beta, and liquidity;
- Phase 4C overextension features: `exhausted_winner_20_5`,
  `residual_overextension_20_5`, and `fragile_winner_proxy`.

Deliverables:

- tree selector predictions;
- daily selector diagnostics;
- selector summary table;
- rolling fold manifest;
- feature-importance summary;
- tree selector memo.

Exit criteria:

- tree selector beats the best Phase 4C price-only selector on the same date
  span, or is rejected;
- fold manifest proves the model uses only prior sessions plus embargo;
- no model output is promoted into a portfolio without a new Phase 4
  construction pass.

Phase 4D tree-selector status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4d`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4d_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4d_tree_residual_loser_selector_20260418`;
- variant tested: `top1000_clean_core_beta_full`;
- feature panel rows loaded: `918,968`;
- prediction rows: `755,241`;
- daily diagnostic rows: `1,099`;
- rolling folds: `9`;
- prediction span: `2015-08-12` through `2019-12-20`;
- no universe expansion, portfolio construction, transaction costs, borrow
  costs, candidate freeze, or test-window performance was computed.

Result:

The first tree selector is rejected.

- mean short beta-residual contribution: `-0.001210`;
- short residual hit rate: `0.478617`;
- mean short raw contribution: `-0.005399`;
- mean oracle overlap rate: `0.093479`;
- mean RankIC of loser score versus forward residual: `-0.010697`;
- same-date-span Phase 4C `short_exhausted_winner_20_5`: `0.000891`;
- same-date-span Phase 4C `short_residual_overextension_20_5`: `0.000753`;
- same-date-span current reversal short rule: `0.000489`.

Feature-importance diagnosis:

The model places large importance on beta and liquidity proxies rather than on
the overextension features that performed best in Phase 4C:

- `beta_z`: `0.136675`;
- `beta`: `0.120422`;
- `liquidity_rank_z`: `0.095597`;
- `liquidity_rank`: `0.087999`;
- `vol_adjusted_momentum_20d`: `0.070821`;
- `momentum_60d_z`: `0.066215`.

Interpretation:

The tree model learned some weak cross-sectional direction, but it failed as a
top-30 residual-loser selector. The issue is tail calibration: the model's top
short candidates have positive realized residual returns on average, producing
negative short contribution. A quick classifier-style bottom-20% loser check
showed the same pattern, so the immediate bottleneck is not merely regression
versus classification. With the current price-only feature set, the transparent
Phase 4C overextension rule is better than the tree model. Tree models should
not be promoted until richer features, such as fundamentals, revisions, borrow,
short interest, or event data, are available and timestamp-safe.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4d
```

Generated artifacts:

- `phase4d_tree_selector_predictions_validation.csv.gz`;
- `phase4d_tree_selector_daily_validation.csv`;
- `phase4d_tree_selector_summary_validation.csv`;
- `phase4d_tree_selector_fold_manifest_validation.csv`;
- `phase4d_tree_selector_feature_importance_validation.csv`;
- `phase4d_tree_selector_memo.md`;
- `phase4d_tree_selector_rollup.json`.

## Phase 4E: Feature Independence Analysis

Goal:

Measure whether the current Phase 4D feature set is independently informative
about future 5-session beta-residual returns, and measure how independent the
features are from each other.

This phase is explicitly about independence, not linear or rank correlation.
The diagnostic uses quantile-discretized normalized mutual information (`NMI`)
with a permutation baseline. A feature can have low Pearson/Spearman
correlation but still be dependent in a nonlinear way; NMI is used to catch
that class of relationship. Excess NMI near `0` means approximately independent
after subtracting finite-sample permutation bias.

Default method:

- feature set: the same `19` Phase 4D selector features;
- target: `forward_beta_residual_return_5d`;
- default variant: `top1000_clean_core_beta_full`;
- metric: quantile-discretized normalized mutual information;
- bins: `10`;
- maximum sampled rows per variant: `200,000`;
- permutation baseline count: `5`;
- feature-cluster threshold: `0.05` excess NMI;
- no Pearson or Spearman correlation is used;
- no test-window performance is computed.

Deliverables:

- feature-target independence table;
- feature-pair independence table;
- strong feature-dependence cluster table;
- sample manifest;
- feature independence memo.

Exit criteria:

- identify whether any existing feature has meaningful standalone nonlinear
  dependence with future 5-session beta-residual returns;
- identify redundant feature families before adding more tree models;
- keep test lockbox closed.

Phase 4E feature-independence status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4e`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4e_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4e_feature_independence_20260418`;
- variant tested: `top1000_clean_core_beta_full`;
- feature panel rows loaded: `918,968`;
- complete-case rows: `918,968`;
- sampled rows: `200,000`;
- sample span: `2014-08-05` through `2019-12-20`;
- feature-target rows: `19`;
- feature-pair rows: `171`;
- strong feature-dependence clusters: `1`;
- no test-window performance was computed.

Feature-target result:

The current features are close to independent from future 5-session
beta-residual returns on a standalone basis. The highest excess NMI values are
small:

- `beta_z`: `0.007072`;
- `beta`: `0.005538`;
- `momentum_60d`: `0.005501`;
- `fragile_winner_proxy`: `0.005178`;
- `lagged_close_log`: `0.005108`;
- `beta_residual_momentum_20d`: `0.004540`;
- `return_5d`: `0.004277`;
- `momentum_20d`: `0.004192`;
- `residual_overextension_20_5`: `0.003748`;
- `exhausted_winner_20_5`: `0.003684`.

Interpretation:

This supports the Phase 4D failure diagnosis. The existing price/liquidity/beta/
momentum feature set does not contain strong single-feature nonlinear
information about future residual losers. Phase 4C works better because it is a
simple tail heuristic, not because the current feature set contains a rich,
learnable residual-loser map.

Feature-feature result:

The features are not mutually independent. They form one large same-source
dependence cluster covering `18 / 19` features. The strongest pairwise excess
NMI values are:

- `liquidity_rank` vs `liquidity_rank_z`: `0.768569`;
- `exhausted_winner_20_5` vs `residual_overextension_20_5`: `0.749127`;
- `trailing_median_dollar_volume_20_log` vs `liquidity_rank_z`: `0.656557`;
- `momentum_20d_z` vs `beta_residual_momentum_20d_z`: `0.640106`;
- `trailing_median_dollar_volume_20_log` vs `liquidity_rank`: `0.584077`;
- `beta_residual_momentum_20d` vs `beta_residual_momentum_20d_z`: `0.538551`;
- `momentum_20d` vs `vol_adjusted_momentum_20d`: `0.535294`;
- `beta` vs `beta_z`: `0.519535`.

Interpretation:

The model input matrix is wide but not diverse. Many fields are transformed
versions of the same price, liquidity, beta, and momentum ingredients. This
means a tree model can easily spend splits on redundant proxies without gaining
new information about residual losers.

Research implication:

Before another flexible selector is promoted, Phase 4 should add genuinely new
timestamp-safe information families or intentionally compress the current
feature set. Candidate new families include fundamentals, analyst revisions,
earnings/event timing, short interest, borrow cost and availability, sector/
industry residual context, and richer intraday/liquidity microstructure. If new
data is not available, the safer path is to keep the transparent Phase 4C
overextension short rule as the current diagnostic baseline.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4e
```

Generated artifacts:

- `phase4e_feature_target_independence_validation.csv`;
- `phase4e_feature_pair_independence_validation.csv`;
- `phase4e_feature_dependence_clusters_validation.csv`;
- `phase4e_feature_independence_sample_manifest_validation.csv`;
- `phase4e_feature_independence_memo.md`;
- `phase4e_feature_independence_rollup.json`.

## Phase 4F: Feature Posterior Shape Diagnostics

Goal:

Translate Phase 4E feature-target non-independence into conditional target
distributions that are easier to reason about as selectors.

The information-theory interpretation is:

If feature `X` and target `Y` are not independent, then knowing `X = x` should
change the posterior distribution `P(Y | X = x)`. Phase 4F makes that concrete
by binning each feature and measuring the future 5-session beta-residual return
distribution inside each bin.

Default method:

- feature set: the same `19` Phase 4D selector features;
- target: `forward_beta_residual_return_5d`;
- default variant: `top1000_clean_core_beta_full`;
- feature buckets: `10` validation-sample quantile bins;
- residual-loser tail: bottom `20%` of validation targets;
- regimes: `all`, `market_up_5d`, and `market_down_5d`;
- diagnostic statistics: conditional mean, median, standard deviation, q10,
  q25, q75, q90, negative residual share, bottom-loser share, and hypothetical
  equal-weight short contribution for each bin;
- no beta-matched portfolio, transaction-cost model, borrow model, or test
  window is used.

Important guardrail:

The regime split uses the realized future 5-session benchmark return only to
diagnose what happened. It is not a tradable real-time regime label.

Deliverables:

- feature-bin posterior distribution table;
- high-minus-low extreme-bin comparison table;
- sample manifest;
- posterior-shape memo.

Exit criteria:

- determine whether the Phase 4E non-independence is directionally useful for
  residual-loser selection;
- identify whether `beta` changes the posterior mean, the left-tail
  probability, or both;
- keep test lockbox closed.

Phase 4F posterior-shape status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4f`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4f_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4f_feature_posterior_shapes_20260418`;
- variant tested: `top1000_clean_core_beta_full`;
- feature panel rows loaded: `918,968`;
- complete-case rows: `918,968`;
- sample span: `2014-08-05` through `2019-12-20`;
- posterior-bin rows: `570`;
- extreme-comparison rows: `57`;
- no test-window performance was computed.

Beta posterior result:

`beta` does change the target posterior, which is consistent with Phase 4E's
weak non-independence result. The effect is directionally meaningful in tail
probability, but only modest in average return.

For absolute `beta`, comparing the highest beta decile with the lowest beta
decile:

- all regimes: mean residual moves from `0.001006` to `-0.000058`;
- all regimes: high-minus-low short contribution is `0.001064`;
- all regimes: bottom-20% residual-loser share rises from `16.90%` to `28.70%`;
- all regimes: negative residual share rises from `47.24%` to `51.35%`;
- market-up 5-session windows: high-minus-low short contribution is `0.001265`;
- market-down 5-session windows: high-minus-low short contribution is only
  `0.000565`.

For cross-sectional `beta_z`, comparing the highest beta-z decile with the
lowest beta-z decile:

- all regimes: mean residual moves from `0.001001` to `-0.000435`;
- all regimes: high-minus-low short contribution is `0.001436`;
- all regimes: bottom-20% residual-loser share rises from `16.89%` to `30.12%`;
- market-down 5-session windows: high-minus-low short contribution is
  `0.002976`;
- market-up 5-session windows: high-minus-low short contribution is only
  `0.000534`.

Interpretation:

The user's information-theory intuition is confirmed: knowing `beta` or
`beta_z` narrows or reshapes the posterior distribution of future beta-residual
return. But the shape matters. The strongest beta effect is in left-tail
probability, not in a large stable negative mean. A highest-beta decile short
has elevated loser odds, but its unconditional average residual short
contribution is small.

This means `beta` is better interpreted as a residual-tail-risk conditioner
than as a standalone residual-loser selector. It may be useful as an interaction
or risk overlay, especially combined with overextension and regime-aware
features, but it does not by itself solve the short-book selection problem.

Feature posterior result beyond beta:

The largest high-minus-low short-contribution contrasts appear mostly in
momentum / overextension features during market-up windows:

- `momentum_20d`, market-up: `0.006683`;
- `beta_residual_momentum_20d`, market-up: `0.005779`;
- `vol_adjusted_momentum_20d`, market-up: `0.005716`;
- `momentum_20d_z`, market-up: `0.005504`;
- `momentum_60d`, market-up: `0.005264`;
- `exhausted_winner_20_5`, market-up: `0.005185`;
- `residual_overextension_20_5`, market-up: `0.005000`.

This reinforces the Phase 4C result: the current usable short-side signal is
still closer to an overextension / exhausted-winner tail heuristic than to a
general learned residual-loser model.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4f
```

Generated artifacts:

- `phase4f_feature_posterior_bins_validation.csv`;
- `phase4f_feature_posterior_extremes_validation.csv`;
- `phase4f_feature_posterior_sample_manifest_validation.csv`;
- `phase4f_feature_posterior_memo.md`;
- `phase4f_feature_posterior_rollup.json`.

## Phase 4G: Robust Feature Utility Re-Ranking

Goal:

Re-rank the current feature set using robust posterior metrics so that a
feature is not promoted merely because it increases left-tail hit rate or gets
lucky on a few extreme residual losers.

Phase 4G keeps the nonlinear shape view from Phase 4F, but changes the pass
criteria. A useful short-side feature bucket should show robust weakness in the
ordinary part of the distribution, not just in the far left tail.

Default method:

- feature set: the same `19` Phase 4D selector features;
- target: `forward_beta_residual_return_5d`;
- default variant: `top1000_clean_core_beta_full`;
- feature buckets: `10` validation-sample quantile bins;
- selected bin per feature: the bin with the lowest `10/90` trimmed residual
  mean;
- left-tail residual-loser cutoff: bottom `20%` of validation targets;
- right-tail residual-winner cutoff: top `20%` of validation targets;
- right-tail tolerance versus universe: `2` percentage points;
- primary robust metrics: median residual, `10/90` trimmed mean, `5/95`
  winsorized mean, bottom-loser share, top-winner share, and tail balance;
- regimes: `all`, `market_up_5d`, and `market_down_5d`;
- no beta-matched portfolio, transaction-cost model, borrow model, or test
  window is used.

Interpretation rules:

- strong candidate: median, trimmed mean, and winsorized mean all support the
  short side; right-tail winner risk is controlled; robust edge is present in
  at least two regimes;
- promising but risky: robust center-of-distribution edge exists, but right
  tail or regime stability is not clean enough;
- tail-only candidate: left-tail hit rate is high, but center-of-distribution
  robustness is weak;
- weak candidate: no robust short-side edge.

Deliverables:

- robust feature-bin table;
- universe robust baseline table;
- robust feature summary table;
- sample manifest;
- robust feature utility memo.

Exit criteria:

- identify which current features remain useful after median / trimmed / right
  tail checks;
- explicitly demote features whose value is mostly volatility or left-tail
  lottery exposure;
- keep test lockbox closed.

Phase 4G robust feature utility status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4g`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4g_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4g_robust_feature_utility_20260418`;
- variant tested: `top1000_clean_core_beta_full`;
- feature panel rows loaded: `918,968`;
- complete-case rows: `918,968`;
- sample span: `2014-08-05` through `2019-12-20`;
- robust feature-bin rows: `570`;
- robust feature summary rows: `19`;
- no test-window performance was computed.

Universe robust baseline:

- all validation rows: median residual `0.000382`;
- all validation rows: `10/90` trimmed residual mean `0.000345`;
- all validation rows: `5/95` winsorized residual mean `0.000331`;
- all validation rows: bottom-loser share `20.00%`;
- all validation rows: top-winner share `20.00%`.

Cleanest robust short-side candidates:

The strongest robust candidates are high momentum / overextension buckets, not
standalone beta:

| Feature | Selected Bin | Trimmed Short | Median Short | Winsor Short | Bottom Share | Top Share | Tail Balance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `momentum_20d` | `10` | `0.001208` | `0.001288` | `0.001188` | `24.56%` | `21.86%` | `2.70%` |
| `beta_residual_momentum_20d_z` | `10` | `0.000839` | `0.000747` | `0.000896` | `24.19%` | `21.98%` | `2.21%` |
| `vol_adjusted_momentum_20d` | `10` | `0.000659` | `0.000458` | `0.000712` | `18.64%` | `16.48%` | `2.16%` |

These are the only `1_robust_short_candidate` labels in the first robust pass.
They pass because the center of the distribution is weaker, not only because
the left tail is fatter.

Promising but less clean candidates:

Several features have robust short-side center metrics, but either right-tail
winner risk is too high or regime behavior is less clean:

- `exhausted_winner_20_5`: trimmed short `0.001229`, median short `0.001149`,
  top-winner share `22.43%`;
- `residual_overextension_20_5`: trimmed short `0.001137`, median short
  `0.001046`, top-winner share `22.36%`;
- `beta_z`: trimmed short `0.001122`, median short `0.001399`, top-winner share
  `27.85%`;
- `momentum_20d_z`: trimmed short `0.000956`, median short `0.000861`,
  top-winner share `22.14%`;
- `beta_residual_momentum_20d`: trimmed short `0.000918`, median short
  `0.000883`, top-winner share `22.38%`;
- `beta`: trimmed short `0.000713`, median short `0.001212`, top-winner share
  `26.92%`.

Beta interpretation after robust checks:

`beta` and `beta_z` are not useless. They do shift the posterior and their
highest deciles have robust negative center metrics. But they also carry much
higher right-tail residual-winner risk:

- `beta_z` highest decile: bottom-loser share `30.12%`, but top-winner share
  `27.85%`;
- `beta` highest decile: bottom-loser share `28.70%`, but top-winner share
  `26.92%`.

That is too symmetric for a clean standalone short alpha. Beta remains better
classified as a tail-risk / volatility conditioner than as a primary
residual-loser selector.

Research implication:

The current strongest same-data-family short-side evidence is still the
overextended-winner family. A future selector should emphasize robust center
weakness and controlled right-tail risk. Concretely, the next short-side rule
should prioritize:

- high `momentum_20d`;
- high residual or z-scored residual momentum;
- high exhausted-winner / residual-overextension score;
- optional beta conditioning only when right-tail winner risk is capped.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4g
```

Generated artifacts:

- `phase4g_robust_feature_bins_validation.csv`;
- `phase4g_robust_feature_baseline_validation.csv`;
- `phase4g_robust_feature_summary_validation.csv`;
- `phase4g_robust_feature_sample_manifest_validation.csv`;
- `phase4g_robust_feature_utility_memo.md`;
- `phase4g_robust_feature_utility_rollup.json`.

## Phase 4H: Universe Prior Strength Study

Goal:

Test whether the current short-book difficulty is partly caused by the selected
universe being too strong. The current `top1000_clean_core_beta_full` validation
sample has a positive 5-session beta-residual prior:

- mean residual: `0.000579`;
- median residual: `0.000382`;
- `10/90` trimmed residual mean: `0.000345`;
- `5/95` winsorized residual mean: `0.000331`.

If the universe prior is positive, then the short book is not merely trying to
find stocks with negative residual returns. It is trying to overcome a positive
drift inside a high-liquidity, survivorship-prone stock pool.

Primary hypothesis:

`top1000` is too strong for short selection. A broader or differently filtered
tradable universe may have a flatter residual prior and more reliable residual
loser candidates, even if the final product still needs to be capacity-aware.

Secondary hypotheses:

- current top1000 membership may embed survivorship and quality bias;
- high-liquidity names may have stronger institutional support and cleaner
  financing than broader stocks;
- the long side benefits from the top1000 prior, while the short side is
  structurally disadvantaged;
- the best short universe may need to be broader or asymmetrically filtered
  relative to the long universe.

Experiment design:

1. Build validation-only universe-prior panels for existing Phase 1 variants.
2. Compute unconditional residual-return priors for each variant:
   - `forward_return_5d`;
   - `forward_market_relative_return_5d`;
   - `forward_beta_residual_return_5d`;
   - median;
   - `10/90` trimmed mean;
   - `5/95` winsorized mean;
   - negative residual share;
   - bottom-20% residual-loser share;
   - top-20% residual-winner share;
   - daily cross-sectional residual mean and median.
3. Split priors by:
   - calendar year;
   - market-up and market-down 5-session benchmark windows;
   - liquidity bucket;
   - beta bucket;
   - price bucket;
   - sector / industry if metadata is available.
4. Re-run Phase 4G robust feature utility on each feasible universe variant.
5. Compare whether the first- and second-tier short features remain useful
   when the universe prior changes.

Universe variants to test first:

- `top500_clean_core_beta_full`;
- `top1000_clean_core_beta_full`;
- `adv50m_clean_core_beta_full`;
- `adv30m_clean_core_beta_full`;
- `adv20m_clean_core_beta_full`;
- `adv10m_clean_core_beta_full` as an expansion diagnostic only;
- available top1500/top2000/top3000 proxies only if membership coverage is
  timestamp-safe enough for validation mechanics.

Asymmetric universe diagnostic:

Also test a research-only long/short split:

- long candidates: current high-quality top1000 or stricter subset;
- short candidates: broader `adv20m` / `adv10m` clean core, with stronger
  borrow and liquidity constraints.

This is not a product decision yet. It is a diagnostic to see whether short
alpha is being suppressed by the current universe prior.

Primary outputs:

- `phase4h_universe_prior_summary_validation.csv`;
- `phase4h_universe_prior_by_year_validation.csv`;
- `phase4h_universe_prior_by_regime_validation.csv`;
- `phase4h_universe_prior_by_bucket_validation.csv`;
- `phase4h_feature_utility_by_universe_validation.csv`;
- `phase4h_universe_prior_memo.md`;
- `phase4h_universe_prior_rollup.json`.

Decision criteria:

- If broader universes reduce median / trimmed beta-residual prior toward `0`
  and improve robust short-feature utility without unacceptable liquidity or
  borrow risk, the short side should not be constrained to top1000.
- If all feasible universes have similarly positive residual priors, the
  bottleneck is more likely residual definition, feature quality, or borrow /
  shortability rather than universe breadth.
- If broader universes improve short diagnostics but harm long diagnostics, use
  asymmetric long/short candidate pools.

Guardrails:

- no test-window performance;
- no final universe selection based on test data;
- no expansion into illiquid or hard-to-borrow names without explicit capacity
  and borrow stress;
- universe comparison must report tradability counts, median dollar volume,
  single-name participation at `500,000 USD`, and missing-data rates.

Phase 4H universe-prior status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4h`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4h_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4h_universe_prior_strength_20260418`;
- panel rows loaded: `4,572,955`;
- variants tested: `top500`, `top1000`, `adv50m`, `adv30m`, `adv20m`,
  `adv10m` clean-core beta-full variants;
- universe-prior summary rows: `6`;
- year rows: `36`;
- regime rows: `18`;
- bucket rows: `240`;
- feature-utility rows: `114`;
- no test-window performance was computed.

Universe-prior result:

`top1000_clean_core_beta_full` has the strongest positive residual prior among
the tested variants:

| Variant | Median Residual | Trimmed Mean | Winsor Mean | Daily Mean Positive Share |
|---|---:|---:|---:|---:|
| `top500_clean_core_beta_full` | `0.000325` | `0.000236` | `0.000102` | `50.52%` |
| `adv50m_clean_core_beta_full` | `0.000342` | `0.000243` | `0.000107` | `50.59%` |
| `adv30m_clean_core_beta_full` | `0.000389` | `0.000300` | `0.000195` | `52.58%` |
| `adv20m_clean_core_beta_full` | `0.000383` | `0.000308` | `0.000218` | `52.06%` |
| `adv10m_clean_core_beta_full` | `0.000396` | `0.000345` | `0.000289` | `53.02%` |
| `top1000_clean_core_beta_full` | `0.000382` | `0.000345` | `0.000331` | `53.47%` |

Interpretation:

The hypothesis is partly supported. The short side is operating against a
positive residual prior, and `top1000` is the most positively biased by
winsorized mean and daily positive-prior share. However, the pattern is not a
simple "broader is always weaker" story. `adv10m` remains almost as positive on
trimmed mean, while stricter `top500` and `adv50m` have flatter priors. This
suggests that the current top1000 construction, not just breadth, may be
embedding a quality / survivorship / liquidity-strength bias.

Feature utility by universe:

The first-tier robust short features transfer across universes and are often
cleaner outside `top1000`:

- `momentum_20d` is a `1_robust_short_candidate` in all six variants;
- `exhausted_winner_20_5` is first-tier in `top500`, `adv50m`, `adv30m`, and
  `adv20m`, but only second-tier in `top1000` and `adv10m`;
- `residual_overextension_20_5` is first-tier in `top500`, `adv50m`, `adv30m`,
  `adv20m`, and `adv10m`, but only second-tier in `top1000`;
- `beta_residual_momentum_20d_z` is first-tier in all variants except it is
  below the top displayed `top1000` group because `top1000` has more right-tail
  pressure.

The strongest first-pass robust short utility appears in `top500`, `adv50m`,
`adv30m`, and `adv20m`, not in `top1000`.

Phase 4H conclusion:

Do not assume `top1000` is the best short-side universe. For short selection,
the next constructor should compare at least `top500`, `adv50m`, `adv30m`, and
`adv20m` against `top1000`, and should consider asymmetric long/short candidate
pools. The result does not yet justify expanding to weak-liquidity names; it
does justify not treating top1000 as a settled choice.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4h
```

Generated artifacts:

- `phase4h_universe_prior_summary_validation.csv`;
- `phase4h_universe_prior_by_year_validation.csv`;
- `phase4h_universe_prior_by_regime_validation.csv`;
- `phase4h_universe_prior_by_bucket_validation.csv`;
- `phase4h_feature_utility_by_universe_validation.csv`;
- `phase4h_universe_prior_memo.md`;
- `phase4h_universe_prior_rollup.json`.

## Phase 4I: Multi-Factor Residual Target Study

Goal:

Test whether the current `forward_beta_residual_return_5d` target is too
coarse because it removes only market beta:

```text
forward_beta_residual_return_5d
= forward_return_5d - beta * benchmark_forward_return_5d
```

This is not a true multi-factor idiosyncratic residual. It can still contain
sector, size, quality, momentum, liquidity, volatility, and other common-factor
effects. If those effects remain inside the target, a short selector may appear
to predict residual losers while actually loading on unrewarded or unstable
style risk.

Primary hypothesis:

The short side looks weak partly because the target is not a clean
multi-factor residual. The current short signals may be predicting a mixture of
idiosyncratic reversal, factor reversal, sector drift, and beta-estimation
error.

Experiment design:

Build alternative validation-only residual targets using only timestamp-safe
features available at selection time.

Target A: current beta residual baseline

```text
y_beta = forward_return_5d - beta * benchmark_forward_return_5d
```

Target B: daily cross-sectional demeaned beta residual

```text
y_cs_beta = y_beta - cross_section_mean(y_beta)
```

Purpose:

Remove the positive daily universe prior without adding new factors.

Target C: sector-neutral beta residual

```text
y_sector = y_beta - sector_day_mean(y_beta)
```

Purpose:

Separate stock selection from sector drift. Use only sector metadata that is
known or safely lagged for the session.

Target D: style-neutral residual

Run a daily cross-sectional regression:

```text
forward_return_5d
~ beta * benchmark_forward_return_5d
+ sector dummies
+ log dollar volume / liquidity rank
+ beta
+ return_5d
+ momentum_20d
+ momentum_60d
+ volatility proxy or vol_adjusted_momentum proxy
```

The residual is:

```text
y_multifactor = forward_return_5d - fitted_forward_return_5d
```

Purpose:

Remove broad market, sector, liquidity / size proxy, beta, momentum, and
volatility-style effects before asking whether our features predict true
stock-specific residual losers.

Target E: factor-neutral residual with no signal leakage

Same as Target D, but exclude any candidate selector feature being evaluated
from the neutralization regression when possible.

Purpose:

Avoid accidentally neutralizing away the exact alpha feature we want to test.

Minimum viable implementation:

- start with Target B and Target C because they are simple and easy to audit;
- add Target D after confirming sector and style fields are timestamp-safe;
- compare every target against the current `y_beta` baseline using the same
  Phase 4E / Phase 4F / Phase 4G diagnostics.

Primary diagnostics:

- target prior distribution:
  - mean;
  - median;
  - `10/90` trimmed mean;
  - `5/95` winsorized mean;
  - negative residual share;
  - bottom / top tail balance;
- daily prior distribution:
  - daily mean and median residual;
  - fraction of days with positive residual prior;
  - correlation of daily residual prior with benchmark return;
- feature-target independence:
  - Phase 4E NMI and permutation baseline for each target;
- posterior shape:
  - Phase 4F decile curves for top features;
- robust utility:
  - Phase 4G first- and second-tier feature rankings under each target;
- selector transfer:
  - whether high `momentum_20d`, high `beta_residual_momentum_20d_z`, and high
    `vol_adjusted_momentum_20d` remain useful after multi-factor neutralization.

Primary outputs:

- `phase4i_residual_target_prior_summary_validation.csv`;
- `phase4i_residual_target_daily_prior_validation.csv`;
- `phase4i_feature_independence_by_target_validation.csv`;
- `phase4i_robust_feature_utility_by_target_validation.csv`;
- `phase4i_target_comparison_memo.md`;
- `phase4i_residual_target_rollup.json`.

Decision criteria:

- If the positive residual prior disappears under cross-sectional or
  multi-factor residuals, then current short-book weakness is partly a target
  definition problem.
- If Phase 4G first-tier features remain robust under multi-factor residuals,
  they are stronger short-alpha candidates.
- If first-tier features collapse after neutralization, they were likely
  capturing factor reversal or universe drift rather than pure stock-specific
  alpha.
- If multi-factor residuals improve short diagnostics but reduce long-side
  signal quality, long and short targets may need separate treatment.

Guardrails:

- no future features inside residual neutralization;
- no target construction using test-window data;
- no feature should be neutralized with information unavailable at selection
  time;
- every target must record exact regression fields, missing-data treatment,
  winsorization, and minimum cross-sectional sample sizes;
- test lockbox remains closed.

Phase 4I residual-target status as of `2026-04-18`:

- repeatable app: `stockmachine.apps.run_pure_alpha_phase4i`;
- project entrypoint: `configs/strategy_projects/us_equities_pure_alpha_h5.json`
  now includes `phase4i_app`;
- validation-only artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4i_residual_target_study_20260418`;
- variant tested: `top1000_clean_core_beta_full`;
- rows loaded: `918,968`;
- residual targets tested:
  - `beta_residual`;
  - `cs_demeaned_beta_residual`;
  - `risk_neutral_beta_residual`;
  - `style_neutral_beta_residual`;
- feature-independence rows: `76`;
- robust feature-utility rows: `76`;
- no test-window performance was computed.

Residual target priors:

| Target | Mean | Median | Trimmed Mean | Winsor Mean | Negative Share |
|---|---:|---:|---:|---:|---:|
| `beta_residual` | `0.000579` | `0.000382` | `0.000345` | `0.000331` | `49.36%` |
| `cs_demeaned_beta_residual` | `-0.000000` | `-0.000152` | `-0.000194` | `-0.000227` | `50.25%` |
| `risk_neutral_beta_residual` | `0.000000` | `-0.000142` | `-0.000197` | `-0.000220` | `50.25%` |
| `style_neutral_beta_residual` | `0.000000` | `-0.000084` | `-0.000163` | `-0.000198` | `50.16%` |

Interpretation:

The target-definition hypothesis is supported. The positive residual prior
largely disappears after daily cross-sectional demeaning and remains gone after
simple risk/style neutralization. This means part of the short-book difficulty
comes from evaluating selectors against a beta-only residual target that still
contains a positive daily universe drift.

Feature transfer across targets:

The core overextension / high-momentum short features survive the cleaner
targets:

- under `cs_demeaned_beta_residual`, `momentum_20d`,
  `momentum_20d_z`, and `beta_residual_momentum_20d_z` remain first-tier
  robust short candidates;
- under `risk_neutral_beta_residual`, overextension and high-momentum features
  remain top-ranked, though most are labeled second-tier because right-tail
  risk remains present;
- under `style_neutral_beta_residual`, high-momentum features weaken because
  the target deliberately neutralizes return, momentum, residual momentum, and
  volatility-adjusted momentum. This is expected and should be treated as a
  transfer diagnostic, not proof that momentum exhaustion has no alpha.

Top robust features by target:

- `beta_residual`: `momentum_20d`, `exhausted_winner_20_5`,
  `residual_overextension_20_5`, `beta_z`, `momentum_20d_z`;
- `cs_demeaned_beta_residual`: `exhausted_winner_20_5`,
  `residual_overextension_20_5`, `momentum_20d`, `momentum_20d_z`,
  `beta_residual_momentum_20d_z`;
- `risk_neutral_beta_residual`: `exhausted_winner_20_5`,
  `residual_overextension_20_5`, `momentum_20d`,
  `beta_residual_momentum_20d`, `beta_residual_momentum_20d_z`;
- `style_neutral_beta_residual`: `lagged_close_log`, `liquidity_rank`,
  `liquidity_rank_z`, `fragile_winner_proxy`, `return_5d_z`,
  `exhausted_winner_20_5`, `momentum_20d`, `beta_z`.

Phase 4I conclusion:

The next short-side selector should not be evaluated only against the current
beta residual. At minimum, Phase 4 construction should report both:

- absolute `beta_residual` contribution;
- relative / cross-section-demeaned residual contribution.

Before a final selector is frozen, the team should add timestamp-safe sector
metadata and rerun a sector-neutral target. If the momentum / overextension
family survives daily demeaning, risk neutralization, and sector-neutral
residuals, it becomes a much stronger pure-alpha short candidate.

Repeatable command:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase4i
```

Generated artifacts:

- `phase4i_residual_target_prior_summary_validation.csv`;
- `phase4i_residual_target_daily_prior_validation.csv`;
- `phase4i_feature_independence_by_target_validation.csv`;
- `phase4i_robust_feature_utility_by_target_validation.csv`;
- `phase4i_target_comparison_memo.md`;
- `phase4i_residual_target_rollup.json`.

## Phase 5: Backtest And Artifact Contract

Goal:

Produce a standard backtest output that the shared robustness framework can
consume.

Required outputs:

- `backtest_records.csv`
- `summary_metrics.csv`
- `predictions.csv`
- `protocol.json`
- optional `parameter_manifest.csv`
- optional `universe_stability_manifest.csv`

Required shared record columns:

- `entry_date`
- `exit_date`
- `net_return`
- `benchmark_return`
- `turnover`
- `cost_bps`
- `positions`

Pure-alpha extension columns:

- `long_return`
- `short_return`
- `spread_return`
- `long_beta`
- `short_beta`
- `net_beta`
- `gross_exposure`
- `net_exposure`
- `long_count`
- `short_count`
- `beta_match_error`
- `borrow_cost_bps`
- `skip_reason`

Summary metrics:

- annualized return
- annualized volatility
- Sharpe
- max drawdown
- realized beta versus `SPY`
- market correlation
- beta-adjusted alpha
- mean turnover
- mean cost bps
- long-leg contribution
- short-leg contribution

Exit criteria:

- artifact contract passes shared robustness suite input checks
- all metrics are validation-only
- no test-window metrics are produced by default

## Phase 6: Robustness And Challenge

Goal:

Challenge candidate strategies using the shared robustness framework and
pure-alpha-specific diagnostics.

Shared robustness analyzers to run:

- time stability
- tail dependence
- cost and execution stress
- turnover concentration
- selection-bias diagnostics
- universe stability diagnostics
- parameter stability when sweep output exists

Pure-alpha-specific diagnostics:

- rolling realized beta
- up-market beta
- down-market beta
- beta by crisis subperiod
- long and short leg attribution
- sector and industry drift
- style exposure drift
- borrow-cost stress
- ADV capacity stress

Initial robustness gates:

- validation Sharpe greater than `0.7`
- annualized return after costs greater than `5%`
- absolute realized beta no greater than `0.05`
- absolute market correlation no greater than `0.10`
- positive contribution from both long and short legs
- no top five days explain most of return
- performance remains positive under plausible cost stress
- parameter leader is not an isolated point

Exit criteria:

- candidate survives validation-only robustness review
- failure modes are documented
- no hidden market or sector exposure explains the result

## Phase 7: Product Candidate Freeze

Goal:

Freeze one candidate before any final test-window evaluation.

Freeze packet:

- universe definition
- signal definition
- beta estimator specification
- portfolio construction rules
- gross exposure
- cost assumptions
- borrow assumptions
- rebalance and holding rules
- validation metrics
- robustness outputs
- expected failure modes

No parameter can be changed after freeze unless a new research cycle is
declared.

Exit criteria:

- validation memo signed off
- artifact paths recorded
- test run command documented but not executed during routine research

## Phase 8: Final Test Lockbox Evaluation

Goal:

Evaluate the frozen candidate once on the final test window.

Rules:

- use only the frozen candidate
- do not tune after viewing the result
- write a pass/fail lockbox memo
- if the result fails, record failure honestly
- any post-test redesign starts a new research cycle

Expected outputs:

- final test summary
- final test robustness review
- comparison versus validation expectations
- decision: observe, reject, or restart research cycle

## Initial Product Targets

First validation-stage targets:

| Metric | Target |
|---|---:|
| Net annualized return | `8% ~ 15%` |
| Net Sharpe | `0.8 ~ 1.5` |
| Absolute realized beta | `<= 0.05` |
| Absolute market correlation | `<= 0.10` |
| Max drawdown | target below `10% ~ 15%` |
| Gross exposure | start `100/100`, test up to `150/150` |
| Long / short names | `10 ~ 30` per side |
| Capacity | `<= 500,000 USD` |

These are research targets, not promises. If a candidate reaches return targets
by violating beta neutrality, shortability, or cost realism, it should fail.

## Work Order

Recommended implementation order:

1. Create the high-liquidity universe builder.
2. Add lagged beta estimation.
3. Run random and simple-factor baselines.
4. Build a simple beta-matched portfolio constructor.
5. Diagnose short-book failure before formalizing the backtest.
6. Emit a minimal long-short backtest artifact bundle.
7. Add cost, turnover, and borrow stress.
8. Plug outputs into the shared robustness suite.
9. Freeze the best validation-only candidate.
10. Only then consider a test lockbox run.

## Immediate Next Deliverables

Near-term deliverables:

- `us_equities_pure_alpha_h5` universe coverage report: started with the
  `2026-04-18` Phase 1 universe builder
- lagged beta panel and diagnostics: generated with the `2026-04-18` Phase 2
  beta builder
- validation-only transparent signal diagnostics: generated with the
  `2026-04-18` Phase 3 baseline signal builder
- first beta-matched long-short baseline: generated with the `2026-04-18`
  Phase 4 portfolio constructor
- short-book failure diagnosis before Phase 5: generated with the `2026-04-18`
  Phase 4B diagnostic builder
- same-universe residual-loser selector lab: generated with the `2026-04-18`
  Phase 4C diagnostic builder
- tree residual-loser selector check: generated with the `2026-04-18` Phase 4D
  diagnostic builder and rejected for the first top1000 pass
- feature independence analysis: generated with the `2026-04-18` Phase 4E
  diagnostic builder, showing weak standalone feature-target dependence and
  high feature-feature redundancy
- feature posterior-shape diagnostics: generated with the `2026-04-18` Phase
  4F diagnostic builder, showing that beta changes residual-loser tail odds but
  is not a strong standalone short selector
- robust feature utility re-ranking: generated with the `2026-04-18` Phase 4G
  diagnostic builder, showing that high momentum / overextension features are
  cleaner short-side candidates than standalone beta after median, trimmed
  mean, and right-tail checks
- universe-prior strength study: next diagnostic Phase 4H, designed to test
  whether `top1000` is structurally too strong for short selection
- multi-factor residual target study: next diagnostic Phase 4I, designed to
  test whether beta-only residuals leave too much sector / style / liquidity
  drift in the target
- robustness-compatible artifact manifest
- validation-only portfolio baseline memo

The first baseline does not need to be impressive. It needs to be clean,
leakage-safe, beta-aware, and easy to challenge.
