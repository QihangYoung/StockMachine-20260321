from __future__ import annotations

from argparse import Namespace
from datetime import date, datetime, timezone
from types import SimpleNamespace

from stockmachine.apps import paper_daily
from stockmachine.apps.run_us_equities_paper import PaperRunConfig
from stockmachine.monitoring.reports import build_paper_run_report


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

    payload = paper_daily.run_command(
        Namespace(
            session_date="2026-03-22",
            universe=["AAPL"],
            run_name="daily-smoke",
            execute=False,
            demo_mode=False,
            model="hist_gbm",
            top_k=10,
            horizon=5,
            data_root="data",
            ledger_path="artifacts/paper_demo/paper_ledger.sqlite3",
            min_close=10.0,
            min_median_dollar_volume_20=50_000_000.0,
            max_vol_20=0.04,
            max_positions_per_sector=2,
            disable_sector_neutral=False,
            require_market_open=False,
            min_buying_power_buffer=0.0,
            max_order_notional=None,
            max_total_notional=None,
            max_total_orders=None,
            execution_equity_cap=500.0,
            post_submit_poll_seconds=15.0,
            post_submit_poll_interval_seconds=2.0,
            allow_unhealthy=False,
            artifact_dir=None,
        )
    )

    assert payload["ok"] is True
    assert payload["command"] == "run"
    assert payload["preflight"]["allowed"] is True
    assert payload["run"]["report"]["run_id"] == "run-123"
    assert payload["post_run"]["ok"] is True
    assert payload["summary"]["decision"] == "executed"
    assert payload["summary"]["report_status"] == "success"
    assert calls["builder_kwargs"]["ledger_path"] == "artifacts/paper_demo/paper_ledger.sqlite3"
    assert calls["config"].run_name == "daily-smoke"
    assert calls["config"].dry_run is True
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

    payload = paper_daily.run_command(
        Namespace(
            session_date="2026-03-22",
            universe=["AAPL"],
            run_name="daily-smoke",
            execute=False,
            demo_mode=False,
            model="hist_gbm",
            top_k=10,
            horizon=5,
            data_root="data",
            ledger_path="artifacts/paper_demo/paper_ledger.sqlite3",
            min_close=10.0,
            min_median_dollar_volume_20=50_000_000.0,
            max_vol_20=0.04,
            max_positions_per_sector=2,
            disable_sector_neutral=False,
            require_market_open=False,
            min_buying_power_buffer=0.0,
            max_order_notional=None,
            max_total_notional=None,
            max_total_orders=None,
            execution_equity_cap=500.0,
            post_submit_poll_seconds=15.0,
            post_submit_poll_interval_seconds=2.0,
            allow_unhealthy=False,
            artifact_dir=None,
        )
    )

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

    payload = paper_daily.run_command(
        Namespace(
            session_date="2026-03-22",
            universe=["AAPL"],
            run_name="daily-smoke",
            execute=False,
            demo_mode=False,
            model="hist_gbm",
            top_k=10,
            horizon=5,
            data_root="data",
            ledger_path="artifacts/paper_demo/paper_ledger.sqlite3",
            min_close=10.0,
            min_median_dollar_volume_20=50_000_000.0,
            max_vol_20=0.04,
            max_positions_per_sector=2,
            disable_sector_neutral=False,
            require_market_open=False,
            min_buying_power_buffer=0.0,
            max_order_notional=None,
            max_total_notional=None,
            max_total_orders=None,
            execution_equity_cap=500.0,
            post_submit_poll_seconds=15.0,
            post_submit_poll_interval_seconds=2.0,
            allow_unhealthy=True,
            artifact_dir=None,
        )
    )

    assert payload["ok"] is True
    assert payload["summary"]["decision"] == "executed_with_override"
    assert payload["preflight"]["override_used"] is True
    assert payload["run"]["report"]["run_id"] == "run-456"
    assert calls["closed"] is True


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
