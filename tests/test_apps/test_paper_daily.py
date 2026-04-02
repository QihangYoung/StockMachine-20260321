from __future__ import annotations

from argparse import Namespace
from datetime import date, datetime, timezone
from types import SimpleNamespace

from stockmachine.apps import paper_daily
from stockmachine.apps.run_us_equities_paper import PaperRunConfig
from stockmachine.domain.project_paths import build_strategy_project_paths
from stockmachine.monitoring.reports import PaperRunFailure, build_paper_run_report


def _make_run_args(**overrides) -> Namespace:
    args = {
        "session_date": "2026-03-22",
        "universe": ["AAPL"],
        "run_name": "daily-smoke",
        "execute": False,
        "demo_mode": False,
        "model": "hist_gbm",
        "top_k": 10,
        "horizon": 5,
        "data_root": "data",
        "ledger_path": "artifacts/paper_demo/paper_ledger.sqlite3",
        "min_close": 10.0,
        "min_median_dollar_volume_20": 50_000_000.0,
        "max_vol_20": 0.04,
        "max_positions_per_sector": 2,
        "disable_sector_neutral": False,
        "require_market_open": False,
        "min_buying_power_buffer": 0.0,
        "max_order_notional": None,
        "max_total_notional": None,
        "max_total_orders": None,
        "execution_equity_cap": 500.0,
        "post_submit_poll_seconds": 15.0,
        "post_submit_poll_interval_seconds": 2.0,
        "allow_unhealthy": False,
        "artifact_dir": None,
        "artifact_root": "artifacts",
        "strategy_profile": None,
        "strategy_family": None,
        "strategy_project": None,
        "strategy_horizon_bucket": None,
        "strategy_track": None,
        "include_silver_symbol_master_refresh": None,
        "skip_silver_symbol_master_refresh": False,
    }
    args.update(overrides)
    return Namespace(**args)


def test_run_command_happy_path_uses_builder_and_runner(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _Runner:
        def run(self, config):
            calls["config"] = config
            return build_paper_run_report(
                session_date=config.session_date,
                dry_run=config.dry_run,
                stage="completed",
                counts={"signals": 1, "targets": 1, "orders": 1},
                run_id="run-123",
                meta={"source": "test"},
            )

        def close(self):
            calls["closed"] = True

    def _build_runner(**kwargs):
        calls["builder_kwargs"] = kwargs
        return _Runner(), PaperRunConfig(session_date=date(2026, 3, 22), dry_run=True, universe=("AAPL",))

    monkeypatch.setattr(paper_daily, "build_alpaca_paper_runner", _build_runner)
    monkeypatch.setattr(paper_daily, "build_demo_runner", lambda universe: (_Runner(), PaperRunConfig(session_date=date(2026, 3, 22), dry_run=True, universe=tuple(universe))))
    monkeypatch.setattr(
        paper_daily,
        "_safe_build_post_run_payload",
        lambda **kwargs: {"ok": True, "reconciliation": {"run_id": kwargs["run_id"]}},
    )
    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: paper_daily.PaperDailyPreflightResult(
            policy_allowed=True,
            allowed=True,
            override_used=False,
            reasons=(),
            healthcheck={"healthy": True, "reasons": []},
            session_guard=None,
            kill_switch={"active": False, "path": "artifacts/paper_demo/paper_daily.kill", "reason": "absent", "payload": None},
            effective_session_date=date(2026, 3, 22),
            data_freshness_meta={"resolution": "exact"},
        ),
    )
    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        lambda **kwargs: {"ok": True, "performed": False, "skipped": True, "reason": "test_stub"},
    )

    payload = paper_daily.run_command(_make_run_args())

    assert payload["ok"] is True
    assert payload["command"] == "run"
    assert payload["silver_refresh"]["reason"] == "test_stub"
    assert payload["preflight"]["allowed"] is True
    assert payload["run"]["report"]["run_id"] == "run-123"
    assert payload["post_run"]["ok"] is True
    assert payload["summary"]["decision"] == "executed"
    assert payload["summary"]["report_status"] == "success"
    assert calls["builder_kwargs"]["ledger_path"] == "artifacts/paper_demo/paper_ledger.sqlite3"
    assert calls["config"].run_name == "daily-smoke"
    assert calls["config"].dry_run is True
    assert calls["closed"] is True


def test_run_command_carries_strategy_lineage_into_config_and_summary(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _Runner:
        def run(self, config):
            calls["config"] = config
            return build_paper_run_report(
                session_date=config.session_date,
                dry_run=config.dry_run,
                stage="completed",
                counts={"signals": 1, "targets": 1, "orders": 1},
                run_id="run-lineage",
                meta={"source": "test"},
            )

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(
        paper_daily,
        "build_alpaca_paper_runner",
        lambda **kwargs: (_Runner(), PaperRunConfig(session_date=date(2026, 3, 22), dry_run=True, universe=("AAPL",))),
    )
    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: paper_daily.PaperDailyPreflightResult(
            policy_allowed=True,
            allowed=True,
            override_used=False,
            reasons=(),
            healthcheck={"healthy": True, "reasons": []},
            session_guard=None,
            kill_switch={"active": False, "path": "artifacts/paper_demo/paper_daily.kill", "reason": "absent", "payload": None},
            effective_session_date=date(2026, 3, 22),
            data_freshness_meta={"resolution": "exact"},
        ),
    )
    monkeypatch.setattr(
        paper_daily,
        "_safe_build_post_run_payload",
        lambda **kwargs: {"ok": True, "reconciliation": {"run_id": kwargs["run_id"]}},
    )
    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        lambda **kwargs: {"ok": True, "performed": False, "skipped": True, "reason": "test_stub"},
    )

    payload = paper_daily.run_command(
        _make_run_args(
            strategy_profile="us_extra_trees_daily",
            strategy_family="us_equities",
            strategy_project="us_equities_h5",
            strategy_horizon_bucket="h5",
            strategy_track="swing_rebalance",
        )
    )

    assert payload["ok"] is True
    assert calls["config"].strategy_profile == "us_extra_trees_daily"
    assert calls["config"].strategy_project == "us_equities_h5"
    assert payload["summary"]["strategy_profile"] == "us_extra_trees_daily"
    assert payload["summary"]["strategy_lineage"] == {
        "strategy_family": "us_equities",
        "strategy_project": "us_equities_h5",
        "strategy_horizon_bucket": "h5",
        "strategy_track": "swing_rebalance",
    }
    workspace = build_strategy_project_paths("us_equities_h5")
    assert payload["summary"]["strategy_workspace"]["ledger_path"] == str(workspace.ledger_path)
    assert calls["closed"] is True


def test_run_command_blocks_on_unhealthy_preflight_without_override(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: paper_daily.PaperDailyPreflightResult(
            policy_allowed=False,
            allowed=False,
            override_used=False,
            reasons=("missing_silver_data",),
            healthcheck={"healthy": False, "reasons": ["missing_latest_run"]},
            session_guard=None,
            kill_switch={"active": False, "path": "artifacts/paper_demo/paper_daily.kill", "reason": "absent", "payload": None},
            effective_session_date=None,
            data_freshness_meta={"resolution": "missing_silver_data"},
        ),
    )
    monkeypatch.setattr(
        paper_daily,
        "build_alpaca_paper_runner",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("runner should not be built")),
    )
    monkeypatch.setattr(
        paper_daily,
        "_safe_build_post_run_payload",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("post-run should not be built")),
    )
    monkeypatch.setattr(
        paper_daily,
        "build_demo_runner",
        lambda universe: (_ for _ in ()).throw(AssertionError("demo runner should not be built")),
    )
    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        lambda **kwargs: {"ok": True, "performed": False, "skipped": True, "reason": "test_stub"},
    )

    payload = paper_daily.run_command(_make_run_args())

    assert payload["ok"] is False
    assert payload["summary"]["decision"] == "blocked_preflight"
    assert payload["summary"]["preflight_allowed"] is False
    assert payload["preflight"]["allowed"] is False
    assert payload["preflight"]["reasons"] == ["missing_silver_data"]


def test_run_command_allows_override_for_unhealthy_preflight(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _Runner:
        def run(self, config):
            calls["config"] = config
            return build_paper_run_report(
                session_date=config.session_date,
                dry_run=config.dry_run,
                stage="completed",
                counts={"signals": 1, "targets": 1, "orders": 1},
                run_id="run-456",
            )

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: paper_daily.PaperDailyPreflightResult(
            policy_allowed=False,
            allowed=True,
            override_used=True,
            reasons=("missing_silver_data",),
            healthcheck={"healthy": False, "reasons": ["missing_latest_run"]},
            session_guard=None,
            kill_switch={"active": False, "path": "artifacts/paper_demo/paper_daily.kill", "reason": "absent", "payload": None},
            effective_session_date=date(2026, 3, 22),
            data_freshness_meta={"resolution": "missing_silver_data"},
        ),
    )
    monkeypatch.setattr(
        paper_daily,
        "_safe_build_post_run_payload",
        lambda **kwargs: {"ok": True, "reconciliation": {"run_id": kwargs["run_id"]}},
    )
    monkeypatch.setattr(paper_daily, "build_alpaca_paper_runner", lambda **kwargs: (_Runner(), PaperRunConfig(session_date=date(2026, 3, 22), dry_run=True, universe=("AAPL",))))
    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        lambda **kwargs: {"ok": True, "performed": False, "skipped": True, "reason": "test_stub"},
    )

    payload = paper_daily.run_command(_make_run_args(allow_unhealthy=True))

    assert payload["ok"] is True
    assert payload["summary"]["decision"] == "executed_with_override"
    assert payload["preflight"]["override_used"] is True
    assert payload["run"]["report"]["run_id"] == "run-456"
    assert calls["closed"] is True


def test_run_command_marks_failed_report_as_not_ok(monkeypatch) -> None:
    class _Runner:
        def run(self, config):
            return build_paper_run_report(
                session_date=config.session_date,
                dry_run=config.dry_run,
                stage="completed_with_warnings",
                counts={"signals": 1, "targets": 1, "orders": 1},
                run_id="run-failed-123",
                failures=(
                    PaperRunFailure(
                        stage="submit_orders",
                        reason="alpaca_rejected_order",
                        details={"message": "opg orders must be submitted before 9:28am"},
                    ),
                ),
                meta={"source": "test"},
            )

        def close(self):
            return None

    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: paper_daily.PaperDailyPreflightResult(
            policy_allowed=True,
            allowed=True,
            override_used=False,
            reasons=(),
            healthcheck={"healthy": True, "reasons": []},
            session_guard=None,
            kill_switch={"active": False, "path": "artifacts/paper_demo/paper_daily.kill", "reason": "absent", "payload": None},
            effective_session_date=date(2026, 3, 22),
            data_freshness_meta={"resolution": "exact"},
        ),
    )
    monkeypatch.setattr(
        paper_daily,
        "build_alpaca_paper_runner",
        lambda **kwargs: (_Runner(), PaperRunConfig(session_date=date(2026, 3, 22), dry_run=False, universe=("AAPL",))),
    )
    monkeypatch.setattr(
        paper_daily,
        "_safe_build_post_run_payload",
        lambda **kwargs: {"ok": True, "reconciliation": {"run_id": kwargs["run_id"]}},
    )
    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        lambda **kwargs: {"ok": True, "performed": False, "skipped": True, "reason": "test_stub"},
    )

    payload = paper_daily.run_command(
        _make_run_args(
            run_name="daily-failed",
            execute=True,
            require_market_open=True,
            execution_equity_cap=None,
        )
    )

    assert payload["ok"] is False
    assert payload["summary"]["decision"] == "failed"
    assert payload["summary"]["report_status"] == "failed"
    assert payload["run"]["report"]["run_id"] == "run-failed-123"


def test_run_command_fails_when_silver_refresh_cannot_reach_latest_completed_session(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        lambda **kwargs: {
            "ok": False,
            "performed": True,
            "skipped": False,
            "reason": "refresh_left_silver_stale",
            "latest_local_session_before_refresh": "2026-03-20",
            "latest_local_session_after_refresh": "2026-03-24",
        },
    )
    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("preflight should not run when refresh fails")),
    )

    payload = paper_daily.run_command(_make_run_args())

    assert payload["ok"] is False
    assert payload["summary"]["stage"] == "silver_refresh_failed"
    assert payload["error"]["type"] == "SilverRefreshError"
    assert payload["silver_refresh"]["reason"] == "refresh_left_silver_stale"


def test_maybe_refresh_silver_before_run_refreshes_to_latest_completed_session(monkeypatch) -> None:
    observed: dict[str, object] = {}
    session_dates = iter([date(2026, 3, 20), date(2026, 3, 23)])
    industry_dates = iter([date(2026, 3, 20), date(2026, 3, 23)])
    universe_dates = iter([date(2026, 3, 20), date(2026, 3, 23)])
    coverage_states = iter(
        [
            {"ok": False, "reason": "universe_membership_incomplete"},
            {"ok": True, "reason": "exact_coverage"},
        ]
    )

    monkeypatch.setattr(
        paper_daily,
        "_latest_local_daily_bar_session_date",
        lambda storage: next(session_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_industry_membership_snapshot_date",
        lambda storage: next(industry_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_universe_membership_session_date",
        lambda storage: next(universe_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_evaluate_universe_membership_coverage",
        lambda storage, *, expected_latest_session: next(coverage_states),
    )

    def _collect_research_seed(**kwargs):
        observed["collector_kwargs"] = kwargs
        return {"normalized_rows": 123}

    monkeypatch.setattr(paper_daily, "collect_research_seed", _collect_research_seed)

    payload = paper_daily.maybe_refresh_silver_before_run(
        session_date=date(2026, 3, 24),
        data_root="data",
        demo_mode=False,
        skip_refresh=False,
        feed="iex",
        adjustment="raw",
        chunk_size=25,
        include_symbol_master=False,
    )

    assert payload["ok"] is True
    assert payload["reason"] == "refresh_completed"
    assert observed["collector_kwargs"]["start_date"] == date(2026, 3, 21)
    assert observed["collector_kwargs"]["end_date"] == date(2026, 3, 23)
    assert observed["collector_kwargs"]["membership_start_date"] == date(2026, 3, 21)
    assert observed["collector_kwargs"]["include_daily_bars"] is True


def test_run_command_defaults_to_symbol_master_refresh(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _refresh_stub(**kwargs):
        observed["refresh_kwargs"] = kwargs
        return {
            "ok": True,
            "performed": False,
            "skipped": True,
            "reason": "test_stub",
        }

    monkeypatch.setattr(
        paper_daily,
        "maybe_refresh_silver_before_run",
        _refresh_stub,
    )
    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_preflight",
        lambda **kwargs: paper_daily.PaperDailyPreflightResult(
            policy_allowed=False,
            allowed=False,
            override_used=False,
            reasons=("blocked_for_test",),
            healthcheck={"healthy": True, "reasons": []},
            session_guard=None,
            kill_switch={"active": False, "path": "artifacts/paper_demo/paper_daily.kill", "reason": "absent", "payload": None},
            effective_session_date=None,
            data_freshness_meta={"resolution": "test"},
        ),
    )

    payload = paper_daily.run_command(_make_run_args())

    assert payload["ok"] is False
    assert observed["refresh_kwargs"]["include_symbol_master"] is True


def test_maybe_refresh_silver_before_run_refreshes_when_symbol_master_lags(monkeypatch) -> None:
    observed: dict[str, object] = {}
    daily_dates = iter([date(2026, 3, 23), date(2026, 3, 23)])
    symbol_dates = iter([date(2026, 3, 20), date(2026, 3, 23)])

    monkeypatch.setattr(
        paper_daily,
        "_latest_local_daily_bar_session_date",
        lambda storage: next(daily_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_symbol_master_snapshot_date",
        lambda storage: next(symbol_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_industry_membership_snapshot_date",
        lambda storage: date(2026, 3, 23),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_universe_membership_session_date",
        lambda storage: date(2026, 3, 23),
    )
    monkeypatch.setattr(
        paper_daily,
        "_evaluate_universe_membership_coverage",
        lambda storage, *, expected_latest_session: {"ok": True, "reason": "exact_coverage"},
    )

    def _collect_research_seed(**kwargs):
        observed["collector_kwargs"] = kwargs
        return {"normalized_rows": 123}

    monkeypatch.setattr(paper_daily, "collect_research_seed", _collect_research_seed)

    payload = paper_daily.maybe_refresh_silver_before_run(
        session_date=date(2026, 3, 24),
        data_root="data",
        demo_mode=False,
        skip_refresh=False,
        feed="iex",
        adjustment="raw",
        chunk_size=25,
        include_symbol_master=True,
    )

    assert payload["ok"] is True
    assert payload["reason"] == "refresh_completed"
    assert payload["latest_symbol_master_snapshot_before_refresh"] == "2026-03-20"
    assert payload["latest_symbol_master_snapshot_after_refresh"] == "2026-03-23"
    assert observed["collector_kwargs"]["start_date"] == date(2026, 3, 23)
    assert observed["collector_kwargs"]["end_date"] == date(2026, 3, 23)
    assert observed["collector_kwargs"]["include_symbol_master"] is True
    assert observed["collector_kwargs"]["membership_start_date"] is None


def test_maybe_refresh_silver_before_run_fails_when_symbol_master_remains_stale(monkeypatch) -> None:
    daily_dates = iter([date(2026, 3, 23), date(2026, 3, 23)])
    symbol_dates = iter([date(2026, 3, 20), date(2026, 3, 20)])

    monkeypatch.setattr(
        paper_daily,
        "_latest_local_daily_bar_session_date",
        lambda storage: next(daily_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_symbol_master_snapshot_date",
        lambda storage: next(symbol_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_industry_membership_snapshot_date",
        lambda storage: date(2026, 3, 23),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_universe_membership_session_date",
        lambda storage: date(2026, 3, 23),
    )
    monkeypatch.setattr(
        paper_daily,
        "_evaluate_universe_membership_coverage",
        lambda storage, *, expected_latest_session: {"ok": True, "reason": "exact_coverage"},
    )
    monkeypatch.setattr(
        paper_daily,
        "collect_research_seed",
        lambda **kwargs: {"normalized_rows": 0},
    )

    payload = paper_daily.maybe_refresh_silver_before_run(
        session_date=date(2026, 3, 24),
        data_root="data",
        demo_mode=False,
        skip_refresh=False,
        feed="iex",
        adjustment="raw",
        chunk_size=25,
        include_symbol_master=True,
    )

    assert payload["ok"] is False
    assert payload["reason"] == "refresh_left_symbol_master_stale"
    assert payload["latest_local_session_after_refresh"] == "2026-03-23"
    assert payload["latest_symbol_master_snapshot_after_refresh"] == "2026-03-20"


def test_maybe_refresh_silver_before_run_fails_when_dataset_remains_stale(monkeypatch) -> None:
    session_dates = iter([date(2026, 3, 20), date(2026, 3, 20)])

    monkeypatch.setattr(
        paper_daily,
        "_latest_local_daily_bar_session_date",
        lambda storage: next(session_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_industry_membership_snapshot_date",
        lambda storage: date(2026, 3, 20),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_universe_membership_session_date",
        lambda storage: date(2026, 3, 20),
    )
    monkeypatch.setattr(
        paper_daily,
        "_evaluate_universe_membership_coverage",
        lambda storage, *, expected_latest_session: {"ok": False, "reason": "universe_membership_incomplete"},
    )
    monkeypatch.setattr(
        paper_daily,
        "collect_research_seed",
        lambda **kwargs: {"normalized_rows": 0},
    )

    payload = paper_daily.maybe_refresh_silver_before_run(
        session_date=date(2026, 3, 24),
        data_root="data",
        demo_mode=False,
        skip_refresh=False,
        feed="iex",
        adjustment="raw",
        chunk_size=25,
        include_symbol_master=False,
    )

    assert payload["ok"] is False
    assert payload["reason"] == "refresh_left_silver_stale"
    assert payload["refresh_end_date"] == "2026-03-23"


def test_maybe_refresh_silver_before_run_refreshes_membership_without_repulling_bars(monkeypatch) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        paper_daily,
        "_latest_local_daily_bar_session_date",
        lambda storage: date(2026, 3, 23),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_symbol_master_snapshot_date",
        lambda storage: date(2026, 3, 23),
    )
    industry_dates = iter([date(2026, 3, 20), date(2026, 3, 23)])
    universe_dates = iter([date(2026, 3, 20), date(2026, 3, 23)])
    coverage_states = iter(
        [
            {"ok": False, "reason": "universe_membership_incomplete"},
            {"ok": True, "reason": "exact_coverage"},
        ]
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_industry_membership_snapshot_date",
        lambda storage: next(industry_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_latest_local_universe_membership_session_date",
        lambda storage: next(universe_dates),
    )
    monkeypatch.setattr(
        paper_daily,
        "_evaluate_universe_membership_coverage",
        lambda storage, *, expected_latest_session: next(coverage_states),
    )

    def _collect_research_seed(**kwargs):
        observed["collector_kwargs"] = kwargs
        return {"normalized_rows": 42}

    monkeypatch.setattr(paper_daily, "collect_research_seed", _collect_research_seed)

    payload = paper_daily.maybe_refresh_silver_before_run(
        session_date=date(2026, 3, 24),
        data_root="data",
        demo_mode=False,
        skip_refresh=False,
        feed="iex",
        adjustment="raw",
        chunk_size=25,
        include_symbol_master=True,
    )

    assert payload["ok"] is True
    assert observed["collector_kwargs"]["include_daily_bars"] is False
    assert observed["collector_kwargs"]["include_adj_factor"] is False
    assert observed["collector_kwargs"]["include_symbol_master"] is False
    assert observed["collector_kwargs"]["start_date"] == date(2026, 3, 23)
    assert observed["collector_kwargs"]["membership_start_date"] == date(2026, 3, 21)
    assert payload["latest_industry_membership_snapshot_after_refresh"] == "2026-03-23"
    assert payload["latest_universe_membership_session_after_refresh"] == "2026-03-23"


def test_healthcheck_command_uses_helper(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_daily,
        "build_paper_daily_healthcheck",
        lambda ledger_path: SimpleNamespace(
            healthy=True,
            to_dict=lambda: {
                "healthy": True,
                "ledger_path": str(ledger_path),
                "reasons": [],
            },
        ),
    )

    payload = paper_daily.healthcheck_command(Namespace(ledger_path="artifacts/paper_demo/paper_ledger.sqlite3"))

    assert payload["command"] == "healthcheck"
    assert payload["ok"] is True
    assert payload["preflight"]["ledger_path"] == "artifacts/paper_demo/paper_ledger.sqlite3"
    assert payload["summary"]["healthy"] is True


def test_parse_args_applies_strategy_profile_defaults_and_allows_cli_override() -> None:
    args = paper_daily.parse_args(
        [
            "run",
            "--strategy-profile",
            "us_hist_gbm_daily",
            "--top-k",
            "3",
        ]
    )

    assert args.strategy_profile == "us_hist_gbm_daily"
    assert args.run_name == "us-hist-gbm-daily"
    assert args.model == "hist_gbm"
    assert args.top_k == 3
    assert args.horizon == 5
    assert args.strategy_project == "us_equities_h5"
    assert args.strategy_horizon_bucket == "h5"
    assert args.strategy_track == "swing_rebalance"
    workspace = build_strategy_project_paths("us_equities_h5")
    assert args.ledger_path == str(workspace.ledger_path)
    assert args.kill_switch_path == str(workspace.kill_switch_path)


def test_parse_args_reads_sys_argv_when_not_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["paper_daily", "run", "--strategy-profile", "us_hist_gbm_ridge_mean_daily"],
    )

    args = paper_daily.parse_args()

    assert args.strategy_profile == "us_hist_gbm_ridge_mean_daily"
    assert args.model == "ensemble_hist_gbm_ridge_mean"
    assert args.run_name == "us-hist-gbm-ridge-mean-daily"


def test_parse_args_supports_random_forest_rank_strategy_profile() -> None:
    args = paper_daily.parse_args(
        [
            "run",
            "--strategy-profile",
            "us_hist_gbm_random_forest_rank_daily",
        ]
    )

    assert args.strategy_profile == "us_hist_gbm_random_forest_rank_daily"
    assert args.model == "ensemble_hist_gbm_random_forest_rank"
    assert args.run_name == "us-hist-gbm-random-forest-rank-daily"


def test_parse_args_supports_random_forest_lightgbm_rank_strategy_profile() -> None:
    args = paper_daily.parse_args(
        [
            "run",
            "--strategy-profile",
            "us_random_forest_lightgbm_rank_daily",
        ]
    )

    assert args.strategy_profile == "us_random_forest_lightgbm_rank_daily"
    assert args.model == "ensemble_random_forest_lightgbm_regressor_rank"
    assert args.run_name == "us-random-forest-lightgbm-rank-daily"


def test_parse_args_supports_hist_gbm_random_forest_lightgbm_rank_strategy_profile() -> None:
    args = paper_daily.parse_args(
        [
            "run",
            "--strategy-profile",
            "us_hist_gbm_random_forest_lightgbm_rank_daily",
        ]
    )

    assert args.strategy_profile == "us_hist_gbm_random_forest_lightgbm_rank_daily"
    assert args.model == "ensemble_hist_gbm_random_forest_lightgbm_regressor_rank"
    assert args.run_name == "us-hist-gbm-random-forest-lightgbm-rank-daily"


def test_parse_args_supports_extra_trees_strategy_profile() -> None:
    args = paper_daily.parse_args(
        [
            "run",
            "--strategy-profile",
            "us_extra_trees_daily",
        ]
    )

    assert args.strategy_profile == "us_extra_trees_daily"
    assert args.model == "extra_trees"
    assert args.run_name == "us-extra-trees-daily"


def test_parse_args_supports_nested_strategy_profile_paths() -> None:
    args = paper_daily.parse_args(
        [
            "run",
            "--strategy-profile",
            "h5/us_extra_trees_daily",
        ]
    )

    assert args.strategy_profile == "h5/us_extra_trees_daily"
    assert args.model == "extra_trees"
    assert args.run_name == "us-extra-trees-daily"
    assert args.strategy_project == "us_equities_h5"
