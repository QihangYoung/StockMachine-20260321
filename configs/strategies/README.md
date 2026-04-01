## Strategy Profiles

Built-in paper strategy profiles are organized by horizon-scoped subprojects.

- `h5/`: current US equities swing profiles built around the 5-session rebalance stack
- `h1/`: reserved for future 1-session / daily-rebalance strategy lines

Operator-facing legacy aliases such as `us_extra_trees_daily` remain supported.
Internally they resolve to the new nested profile ids such as `h5/us_extra_trees_daily`.

Project-level metadata for these horizon lines lives separately under
`configs/strategy_projects/`, so a strategy line can exist as an explicit project
before it has runnable built-in profiles.

Default runtime workspaces resolve from the selected `strategy_project`:

- `artifacts/strategy_projects/<strategy_project>/research`
- `artifacts/strategy_projects/<strategy_project>/paper`

For the active h5 line, operator CLIs and paper runtimes therefore default to:

- `artifacts/strategy_projects/us_equities_h5/paper/paper_ledger.sqlite3`
- `artifacts/strategy_projects/us_equities_h5/paper/paper_daily.kill`
