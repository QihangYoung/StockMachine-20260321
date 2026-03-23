from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from stockmachine.domain.datetime_utils import parse_iso_datetime_like
from stockmachine.state.models import (
    EquitySnapshotRecord,
    FillRecord,
    FillAuditRecord,
    OrderDecisionRecord,
    OrderRecord,
    RunManifestRecord,
    RunRecord,
    SignalRecord,
    TargetRecord,
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_iso_datetime(value: datetime) -> str:
    return _ensure_utc(value).isoformat()


def _to_iso_date(value: date) -> str:
    return value.isoformat()


def _json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _row_to_datetime(value: str | None) -> datetime:
    if value is None:
        raise ValueError("datetime column cannot be null")
    parsed = parse_iso_datetime_like(value)
    return _ensure_utc(parsed)


def _row_to_date(value: str | None) -> date:
    if value is None:
        raise ValueError("date column cannot be null")
    return date.fromisoformat(value)


class LocalLedger:
    """Small SQLite ledger for paper trading runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LocalLedger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                market TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                finished_at_utc TEXT
            );

            CREATE TABLE IF NOT EXISTS run_manifests (
                run_id TEXT PRIMARY KEY,
                session_date TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                generated_at_utc TEXT NOT NULL,
                client_order_id_prefix TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                data_snapshot_json TEXT NOT NULL,
                risk_policy_json TEXT NOT NULL,
                execution_policy_json TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                session_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                horizon_bars INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                session_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                target_weight REAL NOT NULL,
                max_weight REAL NOT NULL,
                reason TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS order_decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT,
                session_date TEXT NOT NULL,
                client_order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                decision_price REAL,
                estimated_notional REAL NOT NULL,
                approved INTEGER NOT NULL,
                reason TEXT NOT NULL,
                decision_at_utc TEXT NOT NULL,
                slippage_bps REAL,
                fee_estimate REAL,
                meta_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                run_id TEXT,
                session_date TEXT,
                client_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                limit_price REAL,
                status TEXT NOT NULL,
                filled_quantity INTEGER NOT NULL,
                avg_fill_price REAL,
                submitted_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                broker_payload_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fill_audits (
                audit_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                run_id TEXT,
                session_date TEXT,
                client_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                expected_price REAL,
                price REAL NOT NULL,
                slippage REAL,
                fee REAL,
                filled_at_utc TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                run_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                filled_at_utc TEXT NOT NULL,
                broker_payload_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                session_date TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                gross_exposure REAL NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_run_session ON signals (run_id, session_date);
            CREATE INDEX IF NOT EXISTS idx_targets_run_session ON targets (run_id, session_date);
            CREATE INDEX IF NOT EXISTS idx_run_manifests_session ON run_manifests (session_date, run_id);
            CREATE INDEX IF NOT EXISTS idx_order_decisions_run_session ON order_decisions (run_id, session_date);
            CREATE INDEX IF NOT EXISTS idx_orders_run_status ON orders (run_id, status);
            CREATE INDEX IF NOT EXISTS idx_orders_client_id ON orders (client_order_id);
            CREATE INDEX IF NOT EXISTS idx_fill_audits_order_id ON fill_audits (order_id);
            CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills (order_id);
            CREATE INDEX IF NOT EXISTS idx_equity_run_session ON equity_snapshots (run_id, session_date);
            """
        )
        self._conn.commit()

    def record_run(self, run: RunRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, strategy_name, market, created_at_utc, status, meta_json, finished_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                strategy_name = excluded.strategy_name,
                market = excluded.market,
                created_at_utc = excluded.created_at_utc,
                status = excluded.status,
                meta_json = excluded.meta_json
            """,
            (
                run.run_id,
                run.strategy_name,
                run.market,
                _to_iso_datetime(run.created_at_utc),
                run.status,
                _json_dump(run.meta),
                None,
            ),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, *, finished_at_utc: datetime, status: str = "finished") -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at_utc = ? WHERE run_id = ?",
            (status, _to_iso_datetime(finished_at_utc), run_id),
        )
        self._conn.commit()

    def record_run_manifest(self, record: RunManifestRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO run_manifests (
                run_id, session_date, strategy_name, model_name, generated_at_utc,
                client_order_id_prefix, dry_run, data_snapshot_json, risk_policy_json,
                execution_policy_json, meta_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                session_date = excluded.session_date,
                strategy_name = excluded.strategy_name,
                model_name = excluded.model_name,
                generated_at_utc = excluded.generated_at_utc,
                client_order_id_prefix = excluded.client_order_id_prefix,
                dry_run = excluded.dry_run,
                data_snapshot_json = excluded.data_snapshot_json,
                risk_policy_json = excluded.risk_policy_json,
                execution_policy_json = excluded.execution_policy_json,
                meta_json = excluded.meta_json
            """,
            (
                record.run_id,
                _to_iso_date(record.session_date),
                record.strategy_name,
                record.model_name,
                _to_iso_datetime(record.generated_at_utc),
                record.client_order_id_prefix,
                1 if record.dry_run else 0,
                _json_dump(record.data_snapshot),
                _json_dump(record.risk_policy),
                _json_dump(record.execution_policy),
                _json_dump(record.meta),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()

    def record_order_decision(self, record: OrderDecisionRecord) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO order_decisions (
                decision_id, run_id, session_date, client_order_id, symbol, side,
                decision_type, decision_price, estimated_notional, approved, reason,
                decision_at_utc, slippage_bps, fee_estimate, meta_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.decision_id,
                record.run_id,
                _to_iso_date(record.session_date),
                record.client_order_id,
                record.symbol,
                record.side,
                record.decision_type,
                record.decision_price,
                record.estimated_notional,
                1 if record.approved else 0,
                record.reason,
                _to_iso_datetime(record.decision_at_utc),
                record.slippage_bps,
                record.fee_estimate,
                _json_dump(record.meta),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def record_fill_audit(self, record: FillAuditRecord) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO fill_audits (
                audit_id, order_id, run_id, session_date, client_order_id, symbol, side,
                quantity, expected_price, price, slippage, fee, filled_at_utc, meta_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.audit_id,
                record.order_id,
                record.run_id,
                _to_iso_date(record.session_date) if record.session_date else None,
                record.client_order_id,
                record.symbol,
                record.side,
                record.quantity,
                record.expected_price,
                record.price,
                record.slippage,
                record.fee,
                _to_iso_datetime(record.filled_at_utc),
                _json_dump(record.meta),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def append_signal(self, record: SignalRecord) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO signals (
                run_id, session_date, symbol, side, score, confidence, horizon_bars,
                timestamp_utc, meta_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                _to_iso_date(record.session_date),
                record.symbol,
                record.side,
                record.score,
                record.confidence,
                record.horizon_bars,
                _to_iso_datetime(record.timestamp_utc),
                _json_dump(record.meta),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def append_target(self, record: TargetRecord) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO targets (
                run_id, session_date, symbol, target_weight, max_weight, reason,
                timestamp_utc, meta_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                _to_iso_date(record.session_date),
                record.symbol,
                record.target_weight,
                record.max_weight,
                record.reason,
                _to_iso_datetime(record.timestamp_utc),
                _json_dump(record.meta),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def upsert_order(self, record: OrderRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO orders (
                order_id, run_id, session_date, client_order_id, symbol, side, quantity,
                order_type, limit_price, status, filled_quantity, avg_fill_price,
                submitted_at_utc, updated_at_utc, broker_payload_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                run_id = excluded.run_id,
                session_date = excluded.session_date,
                client_order_id = excluded.client_order_id,
                symbol = excluded.symbol,
                side = excluded.side,
                quantity = excluded.quantity,
                order_type = excluded.order_type,
                limit_price = excluded.limit_price,
                status = excluded.status,
                filled_quantity = excluded.filled_quantity,
                avg_fill_price = excluded.avg_fill_price,
                submitted_at_utc = excluded.submitted_at_utc,
                updated_at_utc = excluded.updated_at_utc,
                broker_payload_json = excluded.broker_payload_json
            """,
            (
                record.order_id,
                record.run_id,
                _to_iso_date(record.session_date) if record.session_date else None,
                record.client_order_id,
                record.symbol,
                record.side,
                record.quantity,
                record.order_type,
                record.limit_price,
                record.status,
                record.filled_quantity,
                record.avg_fill_price,
                _to_iso_datetime(record.submitted_at_utc),
                _to_iso_datetime(record.updated_at_utc),
                _json_dump(record.broker_payload),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()

    def record_fill(self, record: FillRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, run_id, symbol, side, quantity, price,
                filled_at_utc, broker_payload_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fill_id) DO UPDATE SET
                order_id = excluded.order_id,
                run_id = excluded.run_id,
                symbol = excluded.symbol,
                side = excluded.side,
                quantity = excluded.quantity,
                price = excluded.price,
                filled_at_utc = excluded.filled_at_utc,
                broker_payload_json = excluded.broker_payload_json
            """,
            (
                record.fill_id,
                record.order_id,
                record.run_id,
                record.symbol,
                record.side,
                record.quantity,
                record.price,
                _to_iso_datetime(record.filled_at_utc),
                _json_dump(record.broker_payload),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()

    def record_equity_snapshot(self, record: EquitySnapshotRecord) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO equity_snapshots (
                run_id, session_date, timestamp_utc, cash, equity, gross_exposure,
                payload_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                _to_iso_date(record.session_date),
                _to_iso_datetime(record.timestamp_utc),
                record.cash,
                record.equity,
                record.gross_exposure,
                _json_dump(record.payload),
                _to_iso_datetime(datetime.now(timezone.utc)),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            strategy_name=row["strategy_name"],
            market=row["market"],
            created_at_utc=_row_to_datetime(row["created_at_utc"]),
            status=row["status"],
            meta=_json_load(row["meta_json"]),
        )

    def get_order(self, order_id: str) -> OrderRecord | None:
        row = self._conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_order(row)

    def get_order_by_client_order_id(self, client_order_id: str) -> OrderRecord | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ? ORDER BY updated_at_utc DESC LIMIT 1",
            (client_order_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_order(row)

    def get_run_manifest(self, run_id: str) -> RunManifestRecord | None:
        row = self._conn.execute("SELECT * FROM run_manifests WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_manifest(row)

    def list_orders(self, *, run_id: str | None = None, status: str | None = None) -> list[OrderRecord]:
        query = "SELECT * FROM orders"
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at_utc DESC, order_id DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_order(row) for row in rows]

    def list_run_manifests(self, *, run_id: str | None = None) -> list[RunManifestRecord]:
        query = "SELECT * FROM run_manifests"
        params: list[Any] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY generated_at_utc DESC, run_id DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_manifest(row) for row in rows]

    def list_order_decisions(
        self,
        *,
        run_id: str | None = None,
        client_order_id: str | None = None,
    ) -> list[OrderDecisionRecord]:
        query = "SELECT * FROM order_decisions"
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if client_order_id is not None:
            clauses.append("client_order_id = ?")
            params.append(client_order_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY decision_at_utc ASC, decision_id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_order_decision(row) for row in rows]

    def list_fill_audits(
        self,
        *,
        order_id: str | None = None,
        run_id: str | None = None,
    ) -> list[FillAuditRecord]:
        query = "SELECT * FROM fill_audits"
        clauses: list[str] = []
        params: list[Any] = []
        if order_id is not None:
            clauses.append("order_id = ?")
            params.append(order_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY filled_at_utc ASC, audit_id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_fill_audit(row) for row in rows]

    def list_open_orders(self, *, run_id: str | None = None) -> list[OrderRecord]:
        terminal = {"filled", "canceled", "cancelled", "rejected", "expired"}
        return [
            order
            for order in self.list_orders(run_id=run_id)
            if order.status.lower() not in terminal
        ]

    def list_fills(self, *, order_id: str | None = None, run_id: str | None = None) -> list[FillRecord]:
        query = "SELECT * FROM fills"
        clauses: list[str] = []
        params: list[Any] = []
        if order_id is not None:
            clauses.append("order_id = ?")
            params.append(order_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY filled_at_utc ASC, fill_id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_fill(row) for row in rows]

    def list_equity_snapshots(self, *, run_id: str | None = None) -> list[EquitySnapshotRecord]:
        query = "SELECT * FROM equity_snapshots"
        params: list[Any] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY timestamp_utc ASC, id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_equity_snapshot(row) for row in rows]

    def _row_to_order(self, row: sqlite3.Row) -> OrderRecord:
        return OrderRecord(
            order_id=row["order_id"],
            run_id=row["run_id"],
            session_date=_row_to_date(row["session_date"]) if row["session_date"] else None,
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            order_type=row["order_type"],
            limit_price=row["limit_price"],
            status=row["status"],
            filled_quantity=row["filled_quantity"],
            avg_fill_price=row["avg_fill_price"],
            submitted_at_utc=_row_to_datetime(row["submitted_at_utc"]),
            updated_at_utc=_row_to_datetime(row["updated_at_utc"]),
            broker_payload=_json_load(row["broker_payload_json"]),
        )

    def _row_to_manifest(self, row: sqlite3.Row) -> RunManifestRecord:
        return RunManifestRecord(
            run_id=row["run_id"],
            session_date=_row_to_date(row["session_date"]),
            strategy_name=row["strategy_name"],
            model_name=row["model_name"],
            generated_at_utc=_row_to_datetime(row["generated_at_utc"]),
            client_order_id_prefix=row["client_order_id_prefix"],
            dry_run=bool(row["dry_run"]),
            data_snapshot=_json_load(row["data_snapshot_json"]),
            risk_policy=_json_load(row["risk_policy_json"]),
            execution_policy=_json_load(row["execution_policy_json"]),
            meta=_json_load(row["meta_json"]),
        )

    def _row_to_order_decision(self, row: sqlite3.Row) -> OrderDecisionRecord:
        return OrderDecisionRecord(
            decision_id=row["decision_id"],
            run_id=row["run_id"],
            session_date=_row_to_date(row["session_date"]),
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            decision_type=row["decision_type"],
            decision_price=row["decision_price"],
            estimated_notional=row["estimated_notional"],
            approved=bool(row["approved"]),
            reason=row["reason"],
            decision_at_utc=_row_to_datetime(row["decision_at_utc"]),
            slippage_bps=row["slippage_bps"],
            fee_estimate=row["fee_estimate"],
            meta=_json_load(row["meta_json"]),
        )

    def _row_to_fill_audit(self, row: sqlite3.Row) -> FillAuditRecord:
        return FillAuditRecord(
            audit_id=row["audit_id"],
            order_id=row["order_id"],
            run_id=row["run_id"],
            session_date=_row_to_date(row["session_date"]) if row["session_date"] else None,
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            expected_price=row["expected_price"],
            price=row["price"],
            slippage=row["slippage"],
            fee=row["fee"],
            filled_at_utc=_row_to_datetime(row["filled_at_utc"]),
            meta=_json_load(row["meta_json"]),
        )

    def _row_to_fill(self, row: sqlite3.Row) -> FillRecord:
        return FillRecord(
            fill_id=row["fill_id"],
            order_id=row["order_id"],
            run_id=row["run_id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            price=row["price"],
            filled_at_utc=_row_to_datetime(row["filled_at_utc"]),
            broker_payload=_json_load(row["broker_payload_json"]),
        )

    def _row_to_equity_snapshot(self, row: sqlite3.Row) -> EquitySnapshotRecord:
        return EquitySnapshotRecord(
            run_id=row["run_id"],
            session_date=_row_to_date(row["session_date"]),
            timestamp_utc=_row_to_datetime(row["timestamp_utc"]),
            cash=row["cash"],
            equity=row["equity"],
            gross_exposure=row["gross_exposure"],
            payload=_json_load(row["payload_json"]),
        )


@contextmanager
def open_ledger(path: str | Path) -> Iterator[LocalLedger]:
    ledger = LocalLedger(path)
    try:
        ledger.initialize()
        yield ledger
    finally:
        ledger.close()
