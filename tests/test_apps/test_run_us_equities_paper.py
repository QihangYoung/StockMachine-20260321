from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import SimpleNamespace

from stockmachine.apps.run_us_equities_paper import (
    AlpacaOrderSubmitter,
    PaperRunConfig,
    PaperRunDependencies,
    PaperRunner,
    build_arg_parser,
    build_alpaca_paper_runner,
    build_demo_runner,
    parse_session_date,
)
from stockmachine.backtest.protocols import AccountSnapshot, MarketBar
from stockmachine.domain.models import OrderIntent, Signal, TargetPosition
from stockmachine.execution import SameSessionMarketOrderExecutionPolicy
from stockmachine.execution.brokers import AlpacaBrokerError
from stockmachine.live import PollingOrderReconciler
from stockmachine.live.trade_updates import TradeUpdateMessageSource
from stockmachine.risk import validate_client_order_id
from stockmachine.state import LocalLedger, OrderRecord, RunManifestRecord, RunRecord


def test_paper_runner_cli_parses_core_flags() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--session-date",
            "2026-03-21",
            "--universe",
            "AAPL",
            "MSFT",
            "--run-name",
            "smoke-test",
            "--execution-equity-cap",
            "500",
            "--execute",
        ]
    )

    assert args.session_date == "2026-03-21"
    assert args.universe == ["AAPL", "MSFT"]
    assert args.run_name == "smoke-test"
    assert args.execution_equity_cap == 500.0
    assert args.max_order_notional is None
    assert args.max_total_notional is None
    assert args.max_total_orders is None
    assert args.post_submit_poll_seconds == 15.0
    assert args.post_submit_poll_interval_seconds == 2.0
    assert args.execute is True


def test_build_alpaca_paper_runner_rejects_non_paper_endpoint(monkeypatch, tmp_path) -> None:
    class _FakeBroker:
        def __init__(self) -> None:
            self.credentials = SimpleNamespace(trading_base_url="https://api.alpaca.markets")

        def is_paper_trading_environment(self) -> bool:
            return False

    monkeypatch.setattr(
        "stockmachine.apps.run_us_equities_paper.AlpacaTradingAdapter.from_env",
        lambda: _FakeBroker(),
    )

    try:
        build_alpaca_paper_runner(
            universe=("AAPL",),
            session_date=date(2026, 3, 22),
            ledger_path=tmp_path / "paper-ledger.sqlite3",
        )
    except RuntimeError as exc:
        assert "non-paper trading endpoint" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected non-paper endpoint guard to raise.")


def test_parse_session_date_round_trips_iso_string() -> None:
    assert parse_session_date("2026-03-21") == date(2026, 3, 21)


def test_same_session_market_execution_policy_generates_day_market_orders() -> None:
    policy = SameSessionMarketOrderExecutionPolicy()
    account = AccountSnapshot(session_date=date(2026, 3, 23), cash=10_000.0, equity=10_000.0, gross_exposure=0.0)
    targets = [
        TargetPosition(
            symbol="AAPL",
            target_weight=0.1,
            max_weight=0.1,
            reason="test",
            timestamp=datetime(2026, 3, 23, tzinfo=timezone.utc),
            meta={},
        )
    ]
    bars = {
        "AAPL": MarketBar(
            session_date=date(2026, 3, 23),
            symbol="AAPL",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000_000.0,
        )
    }

    orders = policy.generate_orders(date(2026, 3, 23), targets, bars, account)

    assert len(orders) == 1
    assert orders[0].order_type == "market"
    assert orders[0].quantity == 10


def test_demo_runner_produces_report_shape() -> None:
    runner, config = build_demo_runner(["AAPL", "SPY"])
    report = runner.run(config)

    payload = report.to_dict()
    assert payload["status"] == "success"
    assert payload["dry_run"] is True
    assert payload["counts"]["universe_size"] == 2
    assert payload["counts"]["signals"] == 0
    assert payload["counts"]["orders"] == 0


def test_execute_mode_without_submitter_surfaces_failure() -> None:
    runner, config = build_demo_runner(["AAPL"])
    report = runner.run(
        type(config)(
            session_date=config.session_date,
            dry_run=False,
            universe=config.universe,
            run_name=config.run_name,
            execution_equity_cap=config.execution_equity_cap,
        )
    )

    payload = report.to_dict()
    assert payload["status"] == "failed"
    assert payload["stage"] == "completed_with_warnings"
    assert payload["failures"][0]["reason"] == "missing_order_submitter"


@dataclass(slots=True)
class _OneSignalModel:
    def predict(self, session_date: date, universe: list[str]) -> list[Signal]:
        return [
            Signal(
                symbol="AAPL",
                side="LONG",
                score=1.0,
                confidence=0.9,
                horizon_bars=5,
                timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
                meta={"close": 100.0, "reference_price": 100.0},
            )
        ]


@dataclass(slots=True)
class _OneTargetPolicy:
    def build_targets(self, session_date: date, signals: list[Signal], account: AccountSnapshot) -> list[TargetPosition]:
        return [
            TargetPosition(
                symbol="AAPL",
                target_weight=0.5,
                max_weight=0.5,
                reason="test",
                timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
                meta={"close": 100.0, "reference_price": 100.0},
            )
        ]


@dataclass(slots=True)
class _OneExecutionPolicy:
    def generate_orders(
        self,
        session_date: date,
        targets: list[TargetPosition],
        bars: dict[str, MarketBar],
        account: AccountSnapshot,
    ) -> list[OrderIntent]:
        return [
            OrderIntent(
                symbol="AAPL",
                side="BUY",
                quantity=5,
                order_type="market",
                limit_price=None,
                timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
                meta={"close": 100.0, "reference_price": 100.0},
            )
        ]


@dataclass(slots=True)
class _OneUniverseProvider:
    def get_universe(self, session_date: date) -> list[str]:
        return ["AAPL"]


@dataclass(slots=True)
class _OneAccountProvider:
    def get_account_snapshot(self, session_date: date) -> AccountSnapshot:
        return AccountSnapshot(session_date=session_date, cash=1000.0, equity=1000.0, gross_exposure=0.0)


@dataclass(slots=True)
class _OneMarketDataProvider:
    def get_bars(self, session_date: date, universe: list[str]) -> dict[str, MarketBar]:
        return {
            "AAPL": MarketBar(
                session_date=session_date,
                symbol="AAPL",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000_000.0,
            )
        }


@dataclass(slots=True)
class _FilledOrderSubmitter:
    def submit_orders(self, orders: list[OrderIntent]) -> list[dict[str, object]]:
        return [
            {
                "id": "ord-001",
                "client_order_id": orders[0].meta["client_order_id"],
                "symbol": "AAPL",
                "side": "buy",
                "status": "filled",
                "qty": 5,
                "filled_qty": 5,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:01:00+00:00",
                "avg_fill_price": 100.25,
            }
        ]


@dataclass(slots=True)
class _AcceptedOrderSubmitter:
    def submit_orders(self, orders: list[OrderIntent]) -> list[dict[str, object]]:
        return [
            {
                "id": "ord-accepted-001",
                "client_order_id": orders[0].meta["client_order_id"],
                "symbol": "AAPL",
                "side": "buy",
                "status": "accepted",
                "qty": 5,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:00:00+00:00",
                "avg_fill_price": None,
            }
        ]


@dataclass(slots=True)
class _RetryingBroker:
    attempts: int = 0

    def submit_order(self, *, symbol, side, quantity, order_type, time_in_force, limit_price=None, client_order_id=None):
        self.attempts += 1
        if self.attempts == 1:
            raise AlpacaBrokerError("HTTP 422: insufficient buying power")
        return {
            "id": "ord-retry-001",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "status": "accepted",
            "qty": str(int(quantity)),
            "filled_qty": "0",
            "type": order_type,
            "time_in_force": time_in_force,
            "filled_avg_price": None,
            "limit_price": limit_price,
            "submitted_at": "2026-03-22T01:00:00+00:00",
            "updated_at": "2026-03-22T01:00:00+00:00",
        }


@dataclass(slots=True)
class _TransitioningOrderStatusProvider:
    states: list[dict[str, object]]
    index: int = 0

    def get_order(self, order_id: str) -> dict[str, object]:
        current = self.states[min(self.index, len(self.states) - 1)]
        self.index += 1
        return current


@dataclass(slots=True)
class _FakeAccountSyncProvider:
    def sync(self, session_date: date | None = None):
        effective_date = session_date or date(2026, 3, 22)
        snapshot = AccountSnapshot(session_date=effective_date, cash=1000.0, equity=1000.0, gross_exposure=0.0)
        return SimpleNamespace(
            snapshot=snapshot,
            broker_account=SimpleNamespace(
                status="ACTIVE",
                buying_power=1000.0,
                cash=1000.0,
                equity=1000.0,
                trading_blocked=False,
                account_blocked=False,
                observed_at=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            ),
            broker_positions=(),
            clock=SimpleNamespace(
                is_open=False,
                timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            ),
        )


def test_runner_execute_path_records_orders_and_fills(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            order_submitter=_FilledOrderSubmitter(),
            ledger=ledger,
            reconciler=PollingOrderReconciler(ledger),
        )
    )
    config = PaperRunConfig(
        session_date=date(2026, 3, 22),
        dry_run=False,
        universe=("AAPL",),
        run_name="integration-test",
        execution_equity_cap=500.0,
    )

    report = runner.run(config)
    payload = report.to_dict()
    run = ledger.get_run(payload["run_id"])
    manifest = ledger.get_run_manifest(payload["run_id"])
    order = ledger.get_order("ord-001")
    decisions = ledger.list_order_decisions(run_id=payload["run_id"])
    fills = ledger.list_fills(order_id="ord-001")
    fill_audits = ledger.list_fill_audits(order_id="ord-001")

    assert payload["counts"]["signals"] == 1
    assert payload["counts"]["targets"] == 1
    assert payload["counts"]["orders"] == 1
    assert payload["counts"]["submitted_batches"] == 1
    assert run is not None
    assert manifest is not None
    assert payload["meta"]["run_manifest"]["client_order_id_prefix"] == "smk"
    assert order is not None
    assert order.run_id == payload["run_id"]
    assert order.status == "filled"
    assert len(decisions) == 1
    assert validate_client_order_id(decisions[0].client_order_id)
    assert len(fills) == 1
    assert fills[0].price == 100.25
    assert len(fill_audits) == 1
    assert fill_audits[0].run_id == payload["run_id"]
    assert fill_audits[0].expected_price == 100.0
    assert fill_audits[0].slippage == 0.25


def test_runner_records_submission_retry_metadata_on_buy_rejection(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    broker = _RetryingBroker()
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            order_submitter=AlpacaOrderSubmitter(broker),
            ledger=ledger,
            reconciler=PollingOrderReconciler(ledger),
        )
    )
    config = PaperRunConfig(
        session_date=date(2026, 3, 22),
        dry_run=False,
        universe=("AAPL",),
        run_name="retry-test",
        execution_equity_cap=500.0,
    )

    report = runner.run(config)
    payload = report.to_dict()
    order = ledger.get_order("ord-retry-001")

    assert broker.attempts == 2
    assert payload["status"] == "success"
    assert payload["meta"]["submission_retry"]["attempted_orders"] == 1
    assert payload["meta"]["submission_retry"]["submitted_orders"] == 1
    assert payload["meta"]["submission_retry"]["retried_orders"] == 1
    assert payload["meta"]["submission_retry"]["records"][0]["retry_used"] is True
    assert payload["meta"]["submission_retry"]["records"][0]["initial_error"] == "HTTP 422: insufficient buying power"
    assert payload["meta"]["submission_retry"]["records"][0]["final_quantity"] == 2
    assert order is not None
    assert order.quantity == 2


def test_runner_dry_run_serializes_account_sync_payloads(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            account_sync_provider=_FakeAccountSyncProvider(),
            market_data_provider=_OneMarketDataProvider(),
            ledger=ledger,
        )
    )

    report = runner.run(
        PaperRunConfig(
            session_date=date(2026, 3, 22),
            dry_run=True,
            universe=("AAPL",),
            run_name="account-sync-json",
            execution_equity_cap=500.0,
        )
    )

    payload = report.to_dict()
    snapshots = ledger.list_equity_snapshots(run_id=payload["run_id"])
    manifest = ledger.get_run_manifest(payload["run_id"])

    assert payload["stage"] == "completed"
    assert payload["status"] == "success"
    assert manifest is not None
    assert manifest.dry_run is True
    assert len(snapshots) == 1
    assert snapshots[0].payload["clock"]["timestamp"] == "2026-03-22T01:00:00+00:00"


def test_runner_post_submit_poll_reconciles_follow_up_status(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            order_submitter=_AcceptedOrderSubmitter(),
            order_status_provider=_TransitioningOrderStatusProvider(
                states=[
                    {
                        "id": "ord-accepted-001",
                        "client_order_id": "cid-accepted-001",
                        "symbol": "AAPL",
                        "side": "buy",
                        "status": "accepted",
                        "qty": 5,
                        "filled_qty": 0,
                        "type": "market",
                        "submitted_at": "2026-03-22T01:00:00+00:00",
                        "updated_at": "2026-03-22T01:00:01+00:00",
                    },
                    {
                        "id": "ord-accepted-001",
                        "client_order_id": "cid-accepted-001",
                        "symbol": "AAPL",
                        "side": "buy",
                        "status": "filled",
                        "qty": 5,
                        "filled_qty": 5,
                        "type": "market",
                        "submitted_at": "2026-03-22T01:00:00+00:00",
                        "updated_at": "2026-03-22T01:00:02+00:00",
                        "avg_fill_price": 100.5,
                    },
                ]
            ),
            ledger=ledger,
            reconciler=PollingOrderReconciler(ledger),
        )
    )

    report = runner.run(
        PaperRunConfig(
            session_date=date(2026, 3, 22),
            dry_run=False,
            universe=("AAPL",),
            run_name="poll-follow-up",
            post_submit_poll_seconds=1.0,
            post_submit_poll_interval_seconds=0.0,
        )
    )

    payload = report.to_dict()
    order = ledger.get_order("ord-accepted-001")
    fills = ledger.list_fills(order_id="ord-accepted-001")
    fill_audits = ledger.list_fill_audits(order_id="ord-accepted-001")

    assert payload["status"] == "success"
    assert payload["meta"]["post_submit_poll"]["attempts"] >= 2
    assert payload["meta"]["post_submit_poll"]["remaining_open_orders"] == 0
    assert order is not None
    assert order.status == "filled"
    assert len(fills) == 1
    assert fills[0].price == 100.5
    assert len(fill_audits) == 1
    assert fill_audits[0].slippage == 0.5


def test_runner_post_submit_poll_reports_hybrid_stream_mode(tmp_path) -> None:
    class _IdleStream(TradeUpdateMessageSource):
        def iter_messages(self):
            if False:  # pragma: no cover - protocol-only generator shape
                yield None
            return

    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            order_submitter=_AcceptedOrderSubmitter(),
            order_status_provider=_TransitioningOrderStatusProvider(
                states=[
                    {
                        "id": "ord-accepted-002",
                        "client_order_id": "cid-accepted-002",
                        "symbol": "AAPL",
                        "side": "buy",
                        "status": "accepted",
                        "qty": 5,
                        "filled_qty": 0,
                        "type": "market",
                        "submitted_at": "2026-03-22T01:00:00+00:00",
                        "updated_at": "2026-03-22T01:00:01+00:00",
                    },
                    {
                        "id": "ord-accepted-002",
                        "client_order_id": "cid-accepted-002",
                        "symbol": "AAPL",
                        "side": "buy",
                        "status": "filled",
                        "qty": 5,
                        "filled_qty": 5,
                        "type": "market",
                        "submitted_at": "2026-03-22T01:00:00+00:00",
                        "updated_at": "2026-03-22T01:00:02+00:00",
                        "avg_fill_price": 100.5,
                    },
                ]
            ),
            trade_update_stream=_IdleStream(),
            ledger=ledger,
            reconciler=PollingOrderReconciler(ledger),
        )
    )

    report = runner.run(
        PaperRunConfig(
            session_date=date(2026, 3, 22),
            dry_run=False,
            universe=("AAPL",),
            run_name="hybrid-follow-up",
            post_submit_poll_seconds=1.0,
            post_submit_poll_interval_seconds=0.0,
        )
    )

    payload = report.to_dict()
    assert payload["meta"]["post_submit_poll"]["source_mode"] == "hybrid"
    assert payload["meta"]["post_submit_poll"]["fallback_used"] is True


def test_runner_blocks_duplicate_completed_run_when_guard_fails(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    ledger.record_run(
        RunRecord(
            run_id="prior-run",
            strategy_name="guarded-run",
            market="US",
            created_at_utc=datetime(2026, 3, 22, 0, 0, tzinfo=timezone.utc),
            status="running",
            meta={"session_date": "2026-03-22", "dry_run": True},
        )
    )
    ledger.record_run_manifest(
        RunManifestRecord(
            run_id="prior-run",
            session_date=date(2026, 3, 22),
            strategy_name="guarded-run",
            model_name="hist_gbm",
            generated_at_utc=datetime(2026, 3, 22, 0, 0, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=True,
            data_snapshot={"market": "US"},
            risk_policy={},
            execution_policy={},
            meta={},
        )
    )
    ledger.finish_run("prior-run", finished_at_utc=datetime(2026, 3, 22, 0, 5, tzinfo=timezone.utc))

    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            ledger=ledger,
        )
    )
    monkeypatch.setattr(
        PaperRunner,
        "_available_silver_session_dates",
        lambda self: (date(2026, 3, 22),),
    )

    report = runner.run(
        PaperRunConfig(
            session_date=date(2026, 3, 22),
            dry_run=True,
            universe=("AAPL",),
            run_name="guarded-run",
        )
    )

    payload = report.to_dict()
    assert payload["stage"] == "blocked"
    assert payload["failures"][0]["reason"] == "duplicate_completed_run"
    assert payload["meta"]["session_guard"]["duplicate_run_ids"] == ["prior-run"]


def test_runner_uses_effective_session_date_when_guard_falls_back(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            ledger=ledger,
        )
    )
    monkeypatch.setattr(
        PaperRunner,
        "_available_silver_session_dates",
        lambda self: (date(2026, 3, 20),),
    )

    report = runner.run(
        PaperRunConfig(
            session_date=date(2026, 3, 22),
            dry_run=True,
            universe=("AAPL",),
            run_name="fallback-run",
        )
    )

    payload = report.to_dict()
    snapshots = ledger.list_equity_snapshots(run_id=payload["run_id"])
    assert payload["meta"]["effective_session_date"] == "2026-03-20"
    assert payload["meta"]["session_guard"]["data_freshness_meta"]["resolution"] == "fallback_previous_available_session"
    assert len(snapshots) == 1
    assert snapshots[0].session_date == date(2026, 3, 20)


@dataclass(slots=True)
class _OpenOrderProvider:
    def list_orders(self, *, status: str | None = None, symbols: list[str] | None = None):
        return [
            {
                "id": "existing-order-1",
                "client_order_id": "existing-client-1",
                "symbol": "AAPL",
                "side": "buy",
                "status": "accepted",
                "qty": 5,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:00:00+00:00",
            }
        ]


def test_runner_records_recovery_meta_before_new_work(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "paper-ledger.sqlite3")
    ledger.initialize()
    ledger.upsert_order(
        OrderRecord(
            order_id="existing-order-1",
            run_id="old-run",
            session_date=date(2026, 3, 22),
            client_order_id="existing-client-1",
            symbol="AAPL",
            side="buy",
            quantity=5,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    runner = PaperRunner(
        PaperRunDependencies(
            signal_model=_OneSignalModel(),
            portfolio_policy=_OneTargetPolicy(),
            execution_policy=_OneExecutionPolicy(),
            universe_provider=_OneUniverseProvider(),
            account_provider=_OneAccountProvider(),
            market_data_provider=_OneMarketDataProvider(),
            open_order_provider=_OpenOrderProvider(),
            ledger=ledger,
            reconciler=PollingOrderReconciler(ledger),
        )
    )
    monkeypatch.setattr(
        PaperRunner,
        "_available_silver_session_dates",
        lambda self: (date(2026, 3, 22),),
    )

    report = runner.run(
        PaperRunConfig(
            session_date=date(2026, 3, 22),
            dry_run=True,
            universe=("AAPL",),
            run_name="recovery-run",
        )
    )

    payload = report.to_dict()
    assert payload["meta"]["recovery"]["aligned_open_orders"] == 1
    assert payload["meta"]["recovery"]["orphan_broker_orders"] == 0
    assert payload["meta"]["recovery"]["stale_ledger_orders"] == 0
