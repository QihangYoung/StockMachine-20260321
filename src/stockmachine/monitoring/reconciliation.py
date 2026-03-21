from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import pandas as pd

from stockmachine.state import LocalLedger
from stockmachine.state.models import FillAuditRecord, FillRecord, OrderDecisionRecord, OrderRecord, RunManifestRecord


_TERMINAL_ORDER_STATUSES = {"filled", "canceled", "cancelled", "rejected", "expired"}
_FILLED_ORDER_STATUSES = {"filled", "partially_filled", "partially filled"}
_SUMMARY_COUNT_FIELDS = (
    "decision_count",
    "approved_count",
    "submitted_count",
    "filled_count",
    "open_order_count",
    "rejected_count",
)


@dataclass(slots=True, frozen=True)
class ExpectedReconciliationSnapshot:
    """Optional external benchmark for future backtest/research comparisons."""

    reference: str = "expected_snapshot"
    counts: Mapping[str, int] = field(default_factory=dict)
    symbol_status: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | object) -> ExpectedReconciliationSnapshot:
        if isinstance(payload, ExpectedReconciliationSnapshot):
            return payload
        if isinstance(payload, Mapping):
            return cls(
                reference=str(payload.get("reference", payload.get("name", "expected_snapshot"))),
                counts=_normalize_count_mapping(payload.get("counts") or payload.get("summary_counts") or {}),
                symbol_status={
                    str(symbol): _normalize_count_mapping(status)
                    for symbol, status in dict(payload.get("symbol_status") or {}).items()
                },
                meta=dict(payload.get("meta") or {}),
            )
        reference = getattr(payload, "reference", getattr(payload, "name", "expected_snapshot"))
        counts = getattr(payload, "counts", getattr(payload, "summary_counts", {}))
        symbol_status = getattr(payload, "symbol_status", {})
        meta = getattr(payload, "meta", {})
        return cls(
            reference=str(reference),
            counts=_normalize_count_mapping(counts),
            symbol_status={str(symbol): _normalize_count_mapping(status) for symbol, status in dict(symbol_status).items()},
            meta=dict(meta),
        )


@dataclass(slots=True, frozen=True)
class CountDelta:
    """Expected-versus-actual delta for one count field."""

    expected: int | None
    actual: int | None
    delta: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected,
            "actual": self.actual,
            "delta": self.delta,
        }


@dataclass(slots=True, frozen=True)
class SymbolDelta:
    """Expected-versus-actual delta for one symbol."""

    expected: Mapping[str, int]
    actual: Mapping[str, int]
    delta: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": dict(self.expected),
            "actual": dict(self.actual),
            "delta": dict(self.delta),
        }


@dataclass(slots=True, frozen=True)
class SlippageSummary:
    """Aggregate fill slippage statistics."""

    count: int = 0
    mean_bps: float | None = None
    median_bps: float | None = None
    min_bps: float | None = None
    max_bps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_bps": self.mean_bps,
            "median_bps": self.median_bps,
            "min_bps": self.min_bps,
            "max_bps": self.max_bps,
        }


@dataclass(slots=True, frozen=True)
class SymbolReconciliationSummary:
    """Run-level execution summary grouped by symbol."""

    decision_count: int = 0
    approved_count: int = 0
    submitted_count: int = 0
    filled_count: int = 0
    open_order_count: int = 0
    rejected_count: int = 0
    fill_rate: float | None = None
    slippage: SlippageSummary = field(default_factory=SlippageSummary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_count": self.decision_count,
            "approved_count": self.approved_count,
            "submitted_count": self.submitted_count,
            "filled_count": self.filled_count,
            "open_order_count": self.open_order_count,
            "rejected_count": self.rejected_count,
            "fill_rate": self.fill_rate,
            "slippage": self.slippage.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class PaperReconciliationSummary:
    """Run-level reconciliation summary built from the local ledger."""

    found: bool
    run_id: str | None
    session_date: date | None
    strategy_name: str | None
    model_name: str | None
    dry_run: bool | None
    decision_count: int = 0
    approved_count: int = 0
    submitted_count: int = 0
    filled_count: int = 0
    fill_rate: float | None = None
    open_order_count: int = 0
    rejected_count: int = 0
    slippage: SlippageSummary = field(default_factory=SlippageSummary)
    symbol_status: Mapping[str, SymbolReconciliationSummary] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)
    expected_snapshot: ExpectedReconciliationSnapshot | None = None
    comparison: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "run_id": self.run_id,
            "session_date": self.session_date.isoformat() if self.session_date is not None else None,
            "strategy_name": self.strategy_name,
            "model_name": self.model_name,
            "dry_run": self.dry_run,
            "decision_count": self.decision_count,
            "approved_count": self.approved_count,
            "submitted_count": self.submitted_count,
            "filled_count": self.filled_count,
            "fill_rate": self.fill_rate,
            "open_order_count": self.open_order_count,
            "rejected_count": self.rejected_count,
            "slippage": self.slippage.to_dict(),
            "symbol_status": {symbol: summary.to_dict() for symbol, summary in self.symbol_status.items()},
            "manifest": dict(self.manifest),
            "expected_snapshot": _expected_snapshot_to_dict(self.expected_snapshot),
            "comparison": _comparison_to_dict(self.comparison),
            "notes": list(self.notes),
        }


def build_paper_reconciliation_summary(
    ledger: LocalLedger,
    *,
    run_id: str | None = None,
    expected_snapshot: ExpectedReconciliationSnapshot | Mapping[str, Any] | object | None = None,
) -> PaperReconciliationSummary:
    """Summarize paper-run execution and optional external expectations."""

    resolved_run_id = run_id or _resolve_latest_run_id(ledger)
    if resolved_run_id is None:
        return PaperReconciliationSummary(
            found=False,
            run_id=None,
            session_date=None,
            strategy_name=None,
            model_name=None,
            dry_run=None,
            expected_snapshot=_normalize_expected_snapshot(expected_snapshot),
        )

    manifest = ledger.get_run_manifest(resolved_run_id)
    orders = ledger.list_orders(run_id=resolved_run_id)
    decisions = ledger.list_order_decisions(run_id=resolved_run_id)
    fills = ledger.list_fills(run_id=resolved_run_id)
    fill_audits = ledger.list_fill_audits(run_id=resolved_run_id)

    symbol_status = _build_symbol_status(decisions, orders, fills, fill_audits)
    slippage = _build_slippage_summary(fill_audits, fills)
    submitted_count = len(orders)
    filled_count = sum(1 for order in orders if _is_filled_order(order))
    rejected_count = sum(1 for order in orders if order.status.lower() == "rejected")
    open_order_count = sum(1 for order in orders if order.status.lower() not in _TERMINAL_ORDER_STATUSES)
    approved_count = sum(1 for decision in decisions if decision.approved)
    fill_rate = (filled_count / submitted_count) if submitted_count else None

    normalized_expected = _normalize_expected_snapshot(expected_snapshot)
    comparison = _build_comparison(expected_snapshot=normalized_expected, summary_counts={
        "decision_count": len(decisions),
        "approved_count": approved_count,
        "submitted_count": submitted_count,
        "filled_count": filled_count,
        "open_order_count": open_order_count,
        "rejected_count": rejected_count,
    }, actual_symbol_status=symbol_status)

    return PaperReconciliationSummary(
        found=True,
        run_id=resolved_run_id,
        session_date=manifest.session_date if manifest is not None else _infer_session_date(decisions, orders),
        strategy_name=manifest.strategy_name if manifest is not None else None,
        model_name=manifest.model_name if manifest is not None else None,
        dry_run=manifest.dry_run if manifest is not None else None,
        decision_count=len(decisions),
        approved_count=approved_count,
        submitted_count=submitted_count,
        filled_count=filled_count,
        fill_rate=fill_rate,
        open_order_count=open_order_count,
        rejected_count=rejected_count,
        slippage=slippage,
        symbol_status=symbol_status,
        manifest=_manifest_to_dict(manifest),
        expected_snapshot=normalized_expected,
        comparison=comparison,
        notes=_build_notes(manifest, orders, decisions, fills, fill_audits) + tuple(comparison.get("notes", ())),
    )


def _resolve_latest_run_id(ledger: LocalLedger) -> str | None:
    manifests = ledger.list_run_manifests()
    if manifests:
        return manifests[0].run_id

    orders = ledger.list_orders()
    if orders and orders[0].run_id is not None:
        return orders[0].run_id

    decisions = ledger.list_order_decisions()
    if decisions and decisions[0].run_id is not None:
        return decisions[0].run_id
    return None


def _build_symbol_status(
    decisions: Sequence[OrderDecisionRecord],
    orders: Sequence[OrderRecord],
    fills: Sequence[FillRecord],
    fill_audits: Sequence[FillAuditRecord],
) -> dict[str, SymbolReconciliationSummary]:
    symbol_names = sorted(
        {decision.symbol for decision in decisions}
        | {order.symbol for order in orders}
        | {fill.symbol for fill in fills}
        | {audit.symbol for audit in fill_audits}
    )
    summaries: dict[str, SymbolReconciliationSummary] = {}
    for symbol in symbol_names:
        symbol_decisions = [decision for decision in decisions if decision.symbol == symbol]
        symbol_orders = [order for order in orders if order.symbol == symbol]
        symbol_fills = [fill for fill in fills if fill.symbol == symbol]
        symbol_audits = [audit for audit in fill_audits if audit.symbol == symbol]
        submitted_count = len(symbol_orders)
        filled_count = sum(1 for order in symbol_orders if _is_filled_order(order))
        rejected_count = sum(1 for order in symbol_orders if order.status.lower() == "rejected")
        open_order_count = sum(1 for order in symbol_orders if order.status.lower() not in _TERMINAL_ORDER_STATUSES)
        slippage = _build_slippage_summary(symbol_audits, symbol_fills)
        summaries[symbol] = SymbolReconciliationSummary(
            decision_count=len(symbol_decisions),
            approved_count=sum(1 for decision in symbol_decisions if decision.approved),
            submitted_count=submitted_count,
            filled_count=filled_count,
            open_order_count=open_order_count,
            rejected_count=rejected_count,
            fill_rate=(filled_count / submitted_count) if submitted_count else None,
            slippage=slippage,
        )
    return summaries


def _build_slippage_summary(
    fill_audits: Sequence[FillAuditRecord],
    fills: Sequence[FillRecord],
) -> SlippageSummary:
    values = [_coerce_slippage_value(audit) for audit in fill_audits]
    if not values:
        values = [_coerce_fill_slippage_value(fill) for fill in fills]
    values = [value for value in values if value is not None]
    if not values:
        return SlippageSummary(count=0)
    return SlippageSummary(
        count=len(values),
        mean_bps=float(mean(values)),
        median_bps=float(median(values)),
        min_bps=float(min(values)),
        max_bps=float(max(values)),
    )


def _coerce_slippage_value(record: FillAuditRecord) -> float | None:
    if record.slippage is not None:
        return float(record.slippage)
    if record.expected_price in (None, 0):
        return None
    return float((record.price - record.expected_price) / record.expected_price * 10_000.0)


def _coerce_fill_slippage_value(record: FillRecord) -> float | None:
    if record.expected_price is None or record.expected_price == 0:
        return None
    if record.slippage is not None:
        return float(record.slippage)
    return float((record.price - record.expected_price) / record.expected_price * 10_000.0)


def _is_filled_order(order: OrderRecord) -> bool:
    status = order.status.lower()
    return status in _FILLED_ORDER_STATUSES or order.filled_quantity > 0


def _build_notes(
    manifest: RunManifestRecord | None,
    orders: Sequence[OrderRecord],
    decisions: Sequence[OrderDecisionRecord],
    fills: Sequence[FillRecord],
    fill_audits: Sequence[FillAuditRecord],
) -> tuple[str, ...]:
    notes: list[str] = []
    if manifest is None:
        notes.append("run_manifest_missing")
    if decisions and not orders:
        notes.append("decisions_without_orders")
    if orders and not fills and not fill_audits:
        notes.append("orders_without_fill_evidence")
    return tuple(notes)


def _infer_session_date(
    decisions: Sequence[OrderDecisionRecord],
    orders: Sequence[OrderRecord],
) -> date | None:
    if decisions:
        return decisions[0].session_date
    if orders and orders[0].session_date is not None:
        return orders[0].session_date
    return None


def _manifest_to_dict(manifest: RunManifestRecord | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    return {
        "run_id": manifest.run_id,
        "session_date": manifest.session_date.isoformat(),
        "strategy_name": manifest.strategy_name,
        "model_name": manifest.model_name,
        "generated_at_utc": manifest.generated_at_utc.astimezone(timezone.utc).isoformat(),
        "client_order_id_prefix": manifest.client_order_id_prefix,
        "dry_run": manifest.dry_run,
        "data_snapshot": dict(manifest.data_snapshot),
        "risk_policy": dict(manifest.risk_policy),
        "execution_policy": dict(manifest.execution_policy),
        "meta": dict(manifest.meta),
    }


def _normalize_expected_snapshot(
    expected_snapshot: ExpectedReconciliationSnapshot | Mapping[str, Any] | object | None,
) -> ExpectedReconciliationSnapshot | None:
    if expected_snapshot is None:
        return None
    if isinstance(expected_snapshot, ExpectedReconciliationSnapshot):
        return expected_snapshot
    return ExpectedReconciliationSnapshot.from_payload(expected_snapshot)


def _expected_snapshot_to_dict(expected_snapshot: ExpectedReconciliationSnapshot | None) -> dict[str, Any]:
    if expected_snapshot is None:
        return {}
    return {
        "reference": expected_snapshot.reference,
        "counts": dict(expected_snapshot.counts),
        "symbol_status": {symbol: dict(status) for symbol, status in expected_snapshot.symbol_status.items()},
        "meta": dict(expected_snapshot.meta),
    }


def _comparison_to_dict(comparison: Mapping[str, Any] | None) -> dict[str, Any]:
    if not comparison:
        return {}
    payload: dict[str, Any] = {}
    for key, value in comparison.items():
        if hasattr(value, "to_dict"):
            payload[key] = value.to_dict()  # type: ignore[assignment]
        elif isinstance(value, Mapping):
            payload[key] = {sub_key: sub_value.to_dict() if hasattr(sub_value, "to_dict") else sub_value for sub_key, sub_value in value.items()}
        else:
            payload[key] = value
    return payload


def _normalize_count_mapping(value: Mapping[str, Any] | object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        if hasattr(value, "items"):
            value = dict(value.items())  # type: ignore[assignment]
        else:
            return {}
    return {str(key): int(val) for key, val in value.items()}


def load_expected_snapshot_from_artifact_dir(
    artifact_dir: str | Mapping[str, Any] | object,
    *,
    model_name: str | None = None,
    top_k: int | None = None,
    target_session_date: date | str | None = None,
) -> ExpectedReconciliationSnapshot:
    """Load an expected snapshot from research/backtest artifacts."""

    artifact_path = _coerce_artifact_path(artifact_dir)
    summary_path = artifact_path / "backtest_summary.csv"
    records_path = artifact_path / "backtest_records.csv"
    predictions_path = artifact_path / "predictions.csv"
    if not summary_path.exists() or not records_path.exists() or not predictions_path.exists():
        missing = [str(path.name) for path in (summary_path, records_path, predictions_path) if not path.exists()]
        raise FileNotFoundError(f"Artifact directory '{artifact_path}' is missing required files: {missing}.")

    summary_frame = pd.read_csv(summary_path)
    records_frame = pd.read_csv(records_path)
    predictions_frame = pd.read_csv(predictions_path)

    resolved_model_name = _resolve_artifact_model_name(summary_frame, predictions_frame, model_name=model_name)
    if "model" in predictions_frame.columns:
        predictions_frame = predictions_frame[predictions_frame["model"] == resolved_model_name].copy()

    if top_k is None:
        top_k = _infer_top_k(records_frame)

    selected_frame, selection_meta = _select_expected_positions(
        predictions_frame,
        top_k=top_k,
        target_session_date=target_session_date,
    )
    selected_symbol_counts = selected_frame["symbol"].value_counts().sort_index() if not selected_frame.empty else pd.Series(dtype=int)
    selected_total = int(len(selected_frame))
    summary_row = _first_summary_row(summary_frame, model_name=resolved_model_name)

    counts = {
        "decision_count": selected_total,
        "approved_count": selected_total,
        "submitted_count": selected_total,
        "filled_count": selected_total,
        "open_order_count": 0,
        "rejected_count": 0,
        "selected_rows": selected_total,
        "prediction_rows": int(len(predictions_frame)),
        "backtest_sessions": int(summary_row.get("sessions", len(records_frame))),
    }
    symbol_status = {
        str(symbol): {
            "decision_count": int(count),
            "approved_count": int(count),
            "submitted_count": int(count),
            "filled_count": int(count),
            "open_order_count": 0,
            "rejected_count": 0,
        }
        for symbol, count in selected_symbol_counts.items()
    }
    meta = {
        "artifact_dir": str(artifact_path),
        "model_name": resolved_model_name,
        "backtest_summary": summary_row,
        "backtest_records_rows": int(len(records_frame)),
        "prediction_rows": int(len(predictions_frame)),
        "top_k": top_k,
        "selected_symbols": int(selected_frame["symbol"].nunique()) if not selected_frame.empty else 0,
        **selection_meta,
        "source_files": {
            "backtest_summary": str(summary_path),
            "backtest_records": str(records_path),
            "predictions": str(predictions_path),
        },
    }
    return ExpectedReconciliationSnapshot(
        reference=artifact_path.name,
        counts=counts,
        symbol_status=symbol_status,
        meta=meta,
    )


def infer_expected_snapshot_from_manifest(
    manifest: RunManifestRecord | Mapping[str, Any] | object | None,
    *,
    top_k: int | None = None,
    model_name: str | None = None,
    target_session_date: date | str | None = None,
) -> ExpectedReconciliationSnapshot | None:
    """Infer the expected snapshot artifact dir from a run manifest if available."""

    manifest_payload = _manifest_payload_to_mapping(manifest)
    if not manifest_payload:
        return None
    meta = dict(manifest_payload.get("meta") or {})
    artifact_dir = (
        meta.get("artifact_dir")
        or meta.get("artifacts_dir")
        or meta.get("backtest_artifact_dir")
        or meta.get("research_artifact_dir")
    )
    if artifact_dir is None:
        return None
    artifact_model_name = model_name or meta.get("model_name") or manifest_payload.get("model_name")
    resolved_target_session_date = (
        target_session_date
        or meta.get("effective_session_date")
        or meta.get("requested_session_date")
        or manifest_payload.get("session_date")
    )
    return load_expected_snapshot_from_artifact_dir(
        artifact_dir,
        model_name=artifact_model_name,
        top_k=top_k,
        target_session_date=resolved_target_session_date,
    )


def _build_comparison(
    *,
    expected_snapshot: ExpectedReconciliationSnapshot | None,
    summary_counts: Mapping[str, int | float | None],
    actual_symbol_status: Mapping[str, SymbolReconciliationSummary],
) -> dict[str, Any]:
    if expected_snapshot is None:
        return {}
    ordered_count_keys = list(summary_counts.keys()) + [key for key in expected_snapshot.counts.keys() if key not in summary_counts]
    count_deltas = {
        key: CountDelta(
            expected=_safe_int(expected_snapshot.counts.get(key)),
            actual=_safe_int(summary_counts.get(key)),
            delta=_safe_int(summary_counts.get(key)) - _safe_int(expected_snapshot.counts.get(key))
            if _safe_int(summary_counts.get(key)) is not None and _safe_int(expected_snapshot.counts.get(key)) is not None
            else None,
        )
        for key in ordered_count_keys
    }

    symbol_names = sorted(set(expected_snapshot.symbol_status) | set(actual_symbol_status))
    symbol_deltas: dict[str, SymbolDelta] = {}
    for symbol in symbol_names:
        expected_counts = expected_snapshot.symbol_status.get(symbol, {})
        actual_counts = actual_symbol_status.get(symbol)
        actual_counts_payload = (
            {field: getattr(actual_counts, field) for field in _SUMMARY_COUNT_FIELDS}
            if actual_counts is not None
            else {}
        )
        all_keys = list(actual_counts_payload.keys()) + [key for key in expected_counts.keys() if key not in actual_counts_payload]
        delta_counts: dict[str, int] = {}
        actual_payload: dict[str, int] = {}
        for key in all_keys:
            expected_value = _safe_int(expected_counts.get(key))
            actual_value = _safe_int(actual_counts_payload.get(key))
            if actual_value is not None:
                actual_payload[key] = actual_value
            if expected_value is not None and actual_value is not None:
                delta_counts[key] = actual_value - expected_value
        symbol_deltas[symbol] = SymbolDelta(
            expected={key: value for key, value in expected_counts.items()},
            actual=actual_payload,
            delta=delta_counts,
        )

    notes = []
    if expected_snapshot.meta:
        notes.append(f"artifact_reference={expected_snapshot.reference}")
        if expected_snapshot.meta.get("artifact_dir"):
            notes.append(f"artifact_dir={expected_snapshot.meta['artifact_dir']}")
    return {
        "counts": count_deltas,
        "symbol_status": symbol_deltas,
        "notes": tuple(notes),
    }


def _safe_int(value: int | float | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_artifact_path(artifact_dir: str | Mapping[str, Any] | object) -> Path:
    return Path(str(artifact_dir))


def _resolve_artifact_model_name(
    summary_frame: pd.DataFrame,
    predictions_frame: pd.DataFrame,
    *,
    model_name: str | None = None,
) -> str:
    if model_name is not None:
        return model_name
    if "model" in summary_frame.columns and not summary_frame.empty:
        return str(summary_frame.iloc[0]["model"])
    if "model" in predictions_frame.columns and not predictions_frame.empty:
        model_values = [str(value) for value in predictions_frame["model"].dropna().unique()]
        if len(model_values) == 1:
            return model_values[0]
    return "hist_gbm"


def _first_summary_row(summary_frame: pd.DataFrame, *, model_name: str) -> dict[str, Any]:
    if summary_frame.empty:
        return {}
    if "model" in summary_frame.columns:
        filtered = summary_frame[summary_frame["model"] == model_name]
        if not filtered.empty:
            return filtered.iloc[0].to_dict()
    return summary_frame.iloc[0].to_dict()


def _infer_top_k(records_frame: pd.DataFrame) -> int:
    if "positions" in records_frame.columns and not records_frame.empty:
        values = records_frame["positions"].dropna().astype(int)
        if not values.empty:
            return int(values.mode().iloc[0])
    return 10


def _select_expected_positions(
    predictions_frame: pd.DataFrame,
    *,
    top_k: int,
    target_session_date: date | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if predictions_frame.empty:
        return pd.DataFrame(columns=list(predictions_frame.columns)), {"prediction_date_resolution": "no_predictions"}
    date_column = "date" if "date" in predictions_frame.columns else "signal_date"
    score_column = "score" if "score" in predictions_frame.columns else "final_score"
    if date_column not in predictions_frame.columns or score_column not in predictions_frame.columns:
        return pd.DataFrame(columns=list(predictions_frame.columns)), {"prediction_date_resolution": "missing_required_columns"}

    frame = predictions_frame.copy()
    frame["_prediction_date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    available_dates = sorted(current for current in frame["_prediction_date"].dropna().unique())
    selection_meta: dict[str, Any] = {
        "requested_session_date": _coerce_date_like(target_session_date).isoformat()
        if _coerce_date_like(target_session_date) is not None
        else None,
        "available_prediction_start": available_dates[0].isoformat() if available_dates else None,
        "available_prediction_end": available_dates[-1].isoformat() if available_dates else None,
    }

    selected_date, resolution = _resolve_prediction_date(available_dates, target_session_date=target_session_date)
    selection_meta["prediction_date_resolution"] = resolution
    selection_meta["selected_prediction_date"] = selected_date.isoformat() if selected_date is not None else None
    if selected_date is None:
        return pd.DataFrame(columns=list(predictions_frame.columns)), {
            key: value for key, value in selection_meta.items() if value is not None
        }

    selected_frame = frame.loc[frame["_prediction_date"] == selected_date].copy()
    if selected_frame.empty:
        return pd.DataFrame(columns=list(predictions_frame.columns)), {
            **{key: value for key, value in selection_meta.items() if value is not None},
            "prediction_date_resolution": "selected_prediction_date_empty",
        }
    selected_frame = selected_frame.sort_values(score_column, ascending=False).head(top_k).drop(columns="_prediction_date")
    return selected_frame.reset_index(drop=True), {
        key: value for key, value in selection_meta.items() if value is not None
    }


def _resolve_prediction_date(
    available_dates: Sequence[date],
    *,
    target_session_date: date | str | None = None,
) -> tuple[date | None, str]:
    if not available_dates:
        return None, "no_predictions"

    requested_date = _coerce_date_like(target_session_date)
    if requested_date is None:
        return available_dates[-1], "latest_available"

    candidates = [current for current in available_dates if current <= requested_date]
    if candidates:
        selected = candidates[-1]
        resolution = "exact" if selected == requested_date else "fallback_latest_on_or_before_target"
        return selected, resolution
    return None, "no_prediction_on_or_before_target"


def _coerce_date_like(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _manifest_payload_to_mapping(manifest: RunManifestRecord | Mapping[str, Any] | object | None) -> Mapping[str, Any]:
    if manifest is None:
        return {}
    if isinstance(manifest, Mapping):
        return manifest
    if hasattr(manifest, "meta"):
        return {
            "meta": getattr(manifest, "meta", {}),
            "model_name": getattr(manifest, "model_name", None),
            "session_date": getattr(manifest, "session_date", None),
        }
    return {}
