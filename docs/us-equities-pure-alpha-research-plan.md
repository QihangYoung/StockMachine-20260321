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
3. Build a simple beta-matched portfolio constructor.
4. Emit a minimal long-short backtest artifact bundle.
5. Run random and simple-factor baselines.
6. Add cost, turnover, and borrow stress.
7. Plug outputs into the shared robustness suite.
8. Freeze the best validation-only candidate.
9. Only then consider a test lockbox run.

## Immediate Next Deliverables

Near-term deliverables:

- `us_equities_pure_alpha_h5` universe coverage report: started with the
  `2026-04-18` Phase 1 universe builder
- lagged beta panel and diagnostics
- first beta-matched long-short baseline
- robustness-compatible artifact manifest
- validation-only baseline memo

The first baseline does not need to be impressive. It needs to be clean,
leakage-safe, beta-aware, and easy to challenge.
