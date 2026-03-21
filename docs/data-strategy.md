# Data Strategy

## Position

The project should not depend on a single open-source connector as its only
dataset. A better approach is:

1. use public and open-source data to move fast in research
2. preserve raw source snapshots under our own control
3. normalize everything into internal schemas
4. let models train only on internal canonical tables

This gives us speed early and reproducibility later.

## Recommended Source Mix For A-Share Research

### Structured daily market data

Use a structured connector first for quick iteration:

- Tushare Pro for stock list, daily bars, adjustment factors, trading calendar,
  daily indicators, and limit information
- AKShare as a useful supplement for public-source coverage and quick access to
  additional datasets

### Official primary sources

Use official websites where correctness matters most:

- Shanghai Stock Exchange for exchange-published market information
- Shenzhen exchange and disclosure systems for filings and notices
- CNINFO operator pages for centralized listed-company disclosures

### Custom crawled information

Add our own crawlers for information that is useful but not always available in
stable, machine-friendly APIs:

- company announcements
- exchange notices
- suspension and resumption notices
- sector membership snapshots
- shareholder or management change events
- news or event calendars if we later decide to use them

## Why Not Rely Only On Open-Source Connectors

Open-source connectors are excellent for research velocity, but they have
limits:

- upstream websites may change without notice
- some interfaces have rate limits or IP blocking risk
- schema changes may not be versioned
- licensing and commercial-use boundaries may matter later
- historical point-in-time reconstruction may still need our own snapshots

Because of that, open-source connectors should be inputs into our ingestion
layer, not the final source of truth for modeling.

## Internal Data Layers

We should keep four layers:

### `raw`

Unmodified payloads from API responses, HTML pages, CSV files, or PDFs.

### `bronze`

Parsed source records with source-specific fields preserved.

### `silver`

Normalized internal tables with a shared schema such as `daily_bar`,
`corporate_action`, `trading_calendar`, `announcement`, and `symbol_master`.

### `gold`

Model-ready datasets and feature tables used by training and backtesting.

## Suggested First Scope

The first version only needs enough data to support daily-bar equity modeling:

- stock master and listing status history
- daily OHLCV bars
- adjustment factors
- trading calendar
- suspension and limit-up or limit-down information
- benchmark index data
- sector or industry classification

Announcements can be ingested in parallel, but they do not need to block the
first predictive model.

## Ingestion Module Responsibilities

The new ingestion module should own:

- source definitions
- crawling and API pull jobs
- retry and rate limiting
- raw payload snapshotting
- schema normalization
- incremental update checkpoints
- data quality checks

The rest of the system should read from normalized tables instead of scraping
directly.
