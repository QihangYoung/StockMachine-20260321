# Historical Universe Contract

This note defines the first P2 contract for historical universe handling.

## Goal

Research and backtests should prefer an explicit session-scoped universe table
instead of inferring membership indirectly from late `symbol_master` snapshots.

## Preferred Table

- canonical table: `universe_membership`
- primary key: `session_date + universe_name + symbol`

Required fields:

- `session_date`
- `universe_name`
- `symbol`
- `is_member`

Recommended fields:

- `membership_source`
- `entry_date`
- `exit_date`
- `source_name`
- `load_time_utc`
- `source_version`

## Resolution Rules

For one research session:

1. look for the latest visible `universe_membership` snapshot on or before the
   session date
2. if found, treat it as the primary membership filter
3. only then join `symbol_master` and `industry_membership`
4. if no explicit membership exists, fall back to the current conservative
   `symbol_master` active snapshot logic

## Current State

The repository now supports this contract in code, but current silver data still
mostly relies on:

- static backfilled `universe_membership`
- static backfilled `symbol_master`
- static backfilled `industry_membership`

So the contract is in place, but the historical source quality still needs to be
improved.
