# Canonical Schemas

## Purpose

These are the first internal silver-table contracts. External sources can vary,
but normalized internal tables should keep these names and semantics stable.

## Shared Metadata Fields

Every silver table should include:

- `source_name`: upstream data source identifier
- `load_time_utc`: when we loaded the record
- `effective_time_utc`: when the record became effective if applicable
- `source_version`: optional upstream schema or snapshot version

## `symbol_master`

Primary key:

- `as_of_date`
- `symbol`

Required business fields:

- `security_id`
- `company_name`
- `exchange_mic`
- `currency`
- `security_type`
- `asset_class`
- `is_active`
- `list_date`
- `delist_date`
- `sector`
- `industry`
- `country_of_listing`
- `primary_share_class`

## `trading_calendar`

Primary key:

- `calendar_name`
- `session_date`

Required business fields:

- `is_open`
- `session_open_utc`
- `session_close_utc`
- `previous_open_session`
- `next_open_session`

## `daily_bar`

Primary key:

- `session_date`
- `symbol`

Required business fields:

- `open`
- `high`
- `low`
- `close`
- `volume`
- `vwap`
- `dollar_volume`
- `trade_count`

## `adj_factor`

Primary key:

- `session_date`
- `symbol`

Required business fields:

- `split_factor`
- `cash_dividend`
- `price_adjust_factor`

## `benchmark_index`

Primary key:

- `session_date`
- `symbol`

Required business fields:

- `open`
- `high`
- `low`
- `close`
- `volume`
- `return_1d`

## `daily_basic`

Primary key:

- `session_date`
- `symbol`

Required business fields:

- `shares_outstanding`
- `free_float_shares`
- `market_cap`
- `free_float_market_cap`
- `turnover_rate`
- `median_dollar_volume_20d`
- `median_dollar_volume_60d`

## `industry_membership`

Primary key:

- `as_of_date`
- `symbol`
- `industry_system`

Required business fields:

- `sector_name`
- `industry_group_name`
- `industry_name`
- `subindustry_name`

## `suspend_resume`

Primary key:

- `symbol`
- `status_start_time_utc`

Required business fields:

- `status_code`
- `status_reason`
- `status_end_time_utc`
- `is_halted`

## `price_limit`

Primary key:

- `session_date`
- `symbol`

Required business fields:

- `limit_type`
- `lower_band`
- `upper_band`

For US equities this is intended to store limit-up limit-down style guardrails
when available.
