## Strategy Projects

These project specs describe horizon-scoped strategy lines at the governance level.

- Use `configs/strategies/*` for runnable paper profile defaults.
- Use `configs/strategy_projects/*` for project identity, maturity, and entrypoint metadata.

This split lets `h1` exist as an explicit subproject before it has production-ready
paper profiles.

Project specs also anchor the default filesystem layout used by research and
paper/operator tooling:

- `artifacts/strategy_projects/<project>/research`
- `artifacts/strategy_projects/<project>/paper`

Research entrypoints such as `run_us_equities_model_sweep` and
`run_p1_rigor_suite` now default to the project-scoped `research` workspace
when `--output-root` is not supplied. Paper/operator CLIs resolve ledger and
kill-switch defaults from the same project id.
