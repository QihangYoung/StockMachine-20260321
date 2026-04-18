from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_MEMBERSHIP_PATH = (
    RESEARCH_ROOT
    / "phase0_provisional_pit_universe_feasibility_20260417"
    / "provisional_current_top1000_lagged_liquidity_membership_validation.csv.gz"
)
DEFAULT_ASSET_QA_BY_SYMBOL = (
    RESEARCH_ROOT
    / "phase0_asset_class_qa_20260417"
    / "top1000_asset_class_qa_by_symbol.csv"
)
DEFAULT_PHASE0_CLOSURE_ROLLUP = (
    RESEARCH_ROOT / "phase0_closure_20260418" / "phase0_closure_rollup.json"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase1_universe_builder_20260418"


SUPPORTED_VARIANTS = (
    {
        "variant": "top500_clean_core_beta_full",
        "rule_column": "in_top500",
        "rule_value": True,
        "variant_role": "recommended_mechanics_default",
        "data_status": "supported_current_top1000_scope",
    },
    {
        "variant": "top1000_clean_core_beta_full",
        "rule_column": "in_top1000",
        "rule_value": True,
        "variant_role": "current_scope_diagnostic",
        "data_status": "supported_current_top1000_scope_not_full_market_top1000",
    },
    {
        "variant": "adv50m_clean_core_beta_full",
        "rule_column": "adv_ge_50m",
        "rule_value": True,
        "variant_role": "liquidity_threshold_core",
        "data_status": "supported_current_top1000_scope",
    },
    {
        "variant": "adv30m_clean_core_beta_full",
        "rule_column": "adv_ge_30m",
        "rule_value": True,
        "variant_role": "liquidity_threshold_core",
        "data_status": "supported_current_top1000_scope",
    },
    {
        "variant": "adv20m_clean_core_beta_full",
        "rule_column": "adv_ge_20m",
        "rule_value": True,
        "variant_role": "liquidity_threshold_expansion",
        "data_status": "supported_current_top1000_scope",
    },
    {
        "variant": "adv10m_clean_core_beta_full",
        "rule_column": "adv_ge_10m",
        "rule_value": True,
        "variant_role": "liquidity_threshold_expansion",
        "data_status": "supported_current_top1000_scope",
    },
)
BLOCKED_VARIANTS = (
    {
        "variant": "top1500_clean_core_beta_full",
        "rule_column": "in_top1500",
        "variant_role": "blocked_expansion_candidate",
        "data_status": "blocked_current_backfill_has_only_top1000_symbols",
    },
    {
        "variant": "top2000_clean_core_beta_full",
        "rule_column": "in_top2000",
        "variant_role": "blocked_expansion_candidate",
        "data_status": "blocked_current_backfill_has_only_top1000_symbols",
    },
    {
        "variant": "top3000_clean_core_beta_full",
        "rule_column": "in_top3000",
        "variant_role": "blocked_expansion_candidate",
        "data_status": "blocked_current_backfill_has_only_top1000_symbols",
    },
)


def build_phase1_universe_artifacts(
    *,
    membership_path: str | Path = DEFAULT_MEMBERSHIP_PATH,
    asset_qa_by_symbol_path: str | Path = DEFAULT_ASSET_QA_BY_SYMBOL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    phase0_closure_rollup_path: str | Path = DEFAULT_PHASE0_CLOSURE_ROLLUP,
    min_long_short_names: int = 20,
) -> dict[str, Any]:
    """Build validation-only Phase 1 universe candidates from lagged membership."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(membership_path)
    asset_qa = pd.read_csv(asset_qa_by_symbol_path)
    if membership.empty:
        raise ValueError("Membership artifact is empty.")

    prepared = _prepare_membership(membership, asset_qa)
    supported_membership = _build_supported_membership(prepared)
    daily = _daily_variant_counts(supported_membership)
    summary = _variant_summary(daily, prepared, supported_membership, min_long_short_names)
    blocked = _blocked_variant_summary(prepared)
    full_summary = pd.concat([summary, blocked], ignore_index=True)
    sessions_total = int(prepared["session_date"].nunique())
    stability = _stability_manifest(daily, sessions_total, min_long_short_names)

    membership_out = output_dir / "phase1_candidate_universe_membership_validation.csv.gz"
    daily_out = output_dir / "phase1_candidate_universe_daily_counts_validation.csv"
    summary_out = output_dir / "phase1_candidate_universe_summary_validation.csv"
    stability_out = output_dir / "phase1_universe_stability_manifest.csv"
    memo_out = output_dir / "phase1_universe_builder_memo.md"
    rollup_out = output_dir / "phase1_universe_builder_rollup.json"

    supported_membership.to_csv(membership_out, index=False, compression="gzip")
    daily.to_csv(daily_out, index=False)
    full_summary.to_csv(summary_out, index=False)
    stability.to_csv(stability_out, index=False)
    phase0_closure = _read_json_if_exists(phase0_closure_rollup_path)
    memo_out.write_text(
        _universe_memo(full_summary, stability, phase0_closure, min_long_short_names),
        encoding="utf-8",
    )

    recommended = _recommended_variant(full_summary, min_long_short_names)
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "membership_path": Path(membership_path).as_posix(),
        "asset_qa_by_symbol_path": Path(asset_qa_by_symbol_path).as_posix(),
        "phase0_closure_rollup_path": Path(phase0_closure_rollup_path).as_posix(),
        "validation_start": str(prepared["session_date"].min()),
        "validation_end": str(prepared["session_date"].max()),
        "input_membership_rows": int(len(membership)),
        "input_symbols": int(membership["symbol"].nunique()),
        "clean_core_symbols": int(asset_qa.loc[~asset_qa["review_required"].map(_is_true), "symbol"].nunique()),
        "supported_variants": [variant["variant"] for variant in SUPPORTED_VARIANTS],
        "blocked_variants": [variant["variant"] for variant in BLOCKED_VARIANTS],
        "recommended_validation_default": recommended,
        "min_long_short_names": int(min_long_short_names),
        "membership_artifact": membership_out.as_posix(),
        "daily_counts_artifact": daily_out.as_posix(),
        "summary_artifact": summary_out.as_posix(),
        "stability_manifest_artifact": stability_out.as_posix(),
        "memo_artifact": memo_out.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "Built from current-top1000-scope Phase 0 membership, not a final full-market PIT universe.",
            "Short eligibility uses current shortable/easy-to-borrow metadata as a proxy only.",
            "Top1500/top2000/top3000 remain blocked until broader historical data exists.",
            "No alpha signal, return, model selection, or test-window performance is computed.",
        ],
    }
    rollup_out.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _prepare_membership(membership: pd.DataFrame, asset_qa: pd.DataFrame) -> pd.DataFrame:
    required_membership = {
        "session_date",
        "symbol",
        "liquidity_rank",
        "trailing_median_dollar_volume_20",
        "lagged_close",
        "beta_full_lookback_proxy_ready",
    }
    missing = sorted(required_membership.difference(membership.columns))
    if missing:
        raise ValueError(f"Membership artifact is missing required columns: {missing}")
    required_asset = {"symbol", "review_required", "tradable", "shortable", "easy_to_borrow"}
    missing_asset = sorted(required_asset.difference(asset_qa.columns))
    if missing_asset:
        raise ValueError(f"Asset QA artifact is missing required columns: {missing_asset}")

    qa = asset_qa.copy()
    qa["clean_core"] = ~qa["review_required"].map(_is_true)
    qa["tradable_current"] = qa["tradable"].map(_is_true)
    qa["shortable_current"] = qa["shortable"].map(_is_true)
    qa["easy_to_borrow_current"] = qa["easy_to_borrow"].map(_is_true)
    keep_cols = [
        "symbol",
        "classification",
        "flags",
        "clean_core",
        "tradable_current",
        "shortable_current",
        "easy_to_borrow_current",
    ]
    prepared = membership.merge(qa[keep_cols], on="symbol", how="left")
    prepared["session_date"] = pd.to_datetime(prepared["session_date"]).dt.date.astype(str)
    prepared["clean_core"] = prepared["clean_core"].fillna(False).astype(bool)
    prepared["tradable_current"] = prepared["tradable_current"].fillna(False).astype(bool)
    prepared["shortable_current"] = prepared["shortable_current"].fillna(False).astype(bool)
    prepared["easy_to_borrow_current"] = prepared["easy_to_borrow_current"].fillna(False).astype(bool)
    prepared["beta_full_lookback_proxy_ready"] = prepared["beta_full_lookback_proxy_ready"].map(_is_true)
    prepared["long_eligible"] = (
        prepared["clean_core"]
        & prepared["tradable_current"]
        & prepared["beta_full_lookback_proxy_ready"]
    )
    prepared["short_eligible_proxy"] = (
        prepared["long_eligible"]
        & prepared["shortable_current"]
        & prepared["easy_to_borrow_current"]
    )
    return prepared


def _build_supported_membership(prepared: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_cols = [
        "session_date",
        "symbol",
        "source_segment",
        "lagged_close",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
        "clean_core",
        "tradable_current",
        "shortable_current",
        "easy_to_borrow_current",
        "beta_full_lookback_proxy_ready",
        "long_eligible",
        "short_eligible_proxy",
    ]
    existing_base_cols = [column for column in base_cols if column in prepared.columns]
    for variant in SUPPORTED_VARIANTS:
        rule_column = variant["rule_column"]
        if rule_column not in prepared.columns:
            continue
        mask = prepared[rule_column].map(_is_true) & prepared["long_eligible"]
        selected = prepared.loc[mask, existing_base_cols].copy()
        selected.insert(0, "variant", variant["variant"])
        selected["variant_role"] = variant["variant_role"]
        selected["data_status"] = variant["data_status"]
        rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=["variant", *existing_base_cols, "variant_role", "data_status"])
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["variant", "session_date", "liquidity_rank", "symbol"]).reset_index(drop=True)


def _daily_variant_counts(membership: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "variant",
                "members",
                "long_eligible_members",
                "short_eligible_proxy_members",
                "median_liquidity_rank",
                "median_trailing_median_dollar_volume_20",
            ]
        )
    daily = (
        membership.groupby(["session_date", "variant"])
        .agg(
            members=("symbol", "nunique"),
            long_eligible_members=("long_eligible", "sum"),
            short_eligible_proxy_members=("short_eligible_proxy", "sum"),
            median_liquidity_rank=("liquidity_rank", "median"),
            median_trailing_median_dollar_volume_20=(
                "trailing_median_dollar_volume_20",
                "median",
            ),
        )
        .reset_index()
        .sort_values(["variant", "session_date"])
    )
    for column in ("members", "long_eligible_members", "short_eligible_proxy_members"):
        daily[column] = daily[column].astype(int)
    return daily


def _variant_summary(
    daily: pd.DataFrame,
    prepared: pd.DataFrame,
    membership: pd.DataFrame,
    min_long_short_names: int,
) -> pd.DataFrame:
    rows = []
    sessions_total = int(prepared["session_date"].nunique())
    membership_by_variant = {
        variant: frame for variant, frame in membership.groupby("variant", sort=False)
    }
    for variant in SUPPORTED_VARIANTS:
        name = variant["variant"]
        daily_variant = daily[daily["variant"] == name]
        members = daily_variant["members"] if not daily_variant.empty else pd.Series(dtype=float)
        short_members = (
            daily_variant["short_eligible_proxy_members"]
            if not daily_variant.empty
            else pd.Series(dtype=float)
        )
        enough = (
            (daily_variant["long_eligible_members"] >= min_long_short_names)
            & (daily_variant["short_eligible_proxy_members"] >= min_long_short_names)
            if not daily_variant.empty
            else pd.Series(dtype=bool)
        )
        sessions_with_members = int(daily_variant["session_date"].nunique())
        sessions_with_20x20 = int(enough.sum()) if len(enough) else 0
        frame = membership_by_variant.get(name, pd.DataFrame())
        rows.append(
            {
                "variant": name,
                "variant_role": variant["variant_role"],
                "data_status": variant["data_status"],
                "sessions_total": sessions_total,
                "sessions_with_members": sessions_with_members,
                "first_session_with_members": _first_or_none(daily_variant.get("session_date")),
                "first_session_with_min_20x20": _first_or_none(daily_variant.loc[enough, "session_date"])
                if not daily_variant.empty
                else None,
                "share_total_sessions_with_members": _safe_share(sessions_with_members, sessions_total),
                "share_total_sessions_with_min_20x20": _safe_share(sessions_with_20x20, sessions_total),
                "share_sessions_with_min_20x20": float(enough.mean()) if len(enough) else 0.0,
                "median_members": float(members.median()) if len(members) else 0.0,
                "p10_members": float(members.quantile(0.10)) if len(members) else 0.0,
                "min_members": int(members.min()) if len(members) else 0,
                "max_members": int(members.max()) if len(members) else 0,
                "median_short_eligible_proxy_members": float(short_members.median()) if len(short_members) else 0.0,
                "unique_symbols": int(frame["symbol"].nunique()) if not frame.empty else 0,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _blocked_variant_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    sessions_total = int(prepared["session_date"].nunique())
    rows = []
    for variant in BLOCKED_VARIANTS:
        rows.append(
            {
                "variant": variant["variant"],
                "variant_role": variant["variant_role"],
                "data_status": variant["data_status"],
                "sessions_total": sessions_total,
                "sessions_with_members": 0,
                "first_session_with_members": None,
                "first_session_with_min_20x20": None,
                "share_total_sessions_with_members": 0.0,
                "share_total_sessions_with_min_20x20": 0.0,
                "share_sessions_with_min_20x20": 0.0,
                "median_members": 0.0,
                "p10_members": 0.0,
                "min_members": 0,
                "max_members": 0,
                "median_short_eligible_proxy_members": 0.0,
                "unique_symbols": 0,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _stability_manifest(
    daily: pd.DataFrame,
    sessions_total: int,
    min_long_short_names: int,
) -> pd.DataFrame:
    rows = []
    for variant, frame in daily.groupby("variant", sort=False):
        stable = (
            (frame["long_eligible_members"] >= min_long_short_names)
            & (frame["short_eligible_proxy_members"] >= min_long_short_names)
        )
        stable_sessions = int(stable.sum()) if len(stable) else 0
        total_share = _safe_share(stable_sessions, sessions_total)
        rows.append(
            {
                "variant": variant,
                "robustness_ready": bool(total_share >= 0.80),
                "share_total_sessions_min_20x20": total_share,
                "share_sessions_min_20x20": float(stable.mean()) if len(stable) else 0.0,
                "min_long_eligible_members": int(frame["long_eligible_members"].min()) if len(frame) else 0,
                "min_short_eligible_proxy_members": int(frame["short_eligible_proxy_members"].min())
                if len(frame)
                else 0,
                "median_members": float(frame["members"].median()) if len(frame) else 0.0,
                "robustness_note": (
                    "Ready for validation-only robustness plumbing."
                    if total_share >= 0.80
                    else "Keep as diagnostic until coverage improves."
                ),
            }
        )
    for variant in BLOCKED_VARIANTS:
        rows.append(
            {
                "variant": variant["variant"],
                "robustness_ready": False,
                "share_total_sessions_min_20x20": 0.0,
                "share_sessions_min_20x20": 0.0,
                "min_long_eligible_members": 0,
                "min_short_eligible_proxy_members": 0,
                "median_members": 0.0,
                "robustness_note": "Blocked until broader historical data exists.",
            }
        )
    return pd.DataFrame(rows)


def _recommended_variant(summary: pd.DataFrame, min_long_short_names: int) -> str | None:
    candidates = summary[
        (summary["variant_role"] == "recommended_mechanics_default")
        & (summary["share_total_sessions_with_min_20x20"] >= 0.80)
        & (summary["median_short_eligible_proxy_members"] >= min_long_short_names)
    ]
    if not candidates.empty:
        return str(candidates.iloc[0]["variant"])
    supported = summary[
        summary["data_status"].astype(str).str.startswith("supported")
        & (summary["share_total_sessions_with_min_20x20"] >= 0.80)
    ]
    if supported.empty:
        return None
    return str(supported.sort_values(["median_members", "variant"], ascending=[False, True]).iloc[0]["variant"])


def _universe_memo(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    phase0_closure: dict[str, Any],
    min_long_short_names: int,
) -> str:
    recommended = _recommended_variant(summary, min_long_short_names)
    lines = [
        "# Pure Alpha Phase 1 Universe Builder Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only universe construction artifact. It compares feasibility, "
        "tradability, breadth, and robustness readiness before any alpha performance.",
        "",
        "No test-window performance, signal return, model selection, or universe-by-return "
        "selection is used.",
        "",
        "## Recommendation",
        "",
        f"- recommended validation default: `{recommended or 'none'}`",
        f"- minimum long/short names per session gate: `{min_long_short_names}`",
        f"- Phase 0 status: `{phase0_closure.get('phase0_status', 'missing')}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Status | Median Members | Min Members | First 20x20 Date |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['data_status']} | {row['median_members']} | "
            f"{row['min_members']} | {row['first_session_with_min_20x20']} |"
        )
    lines.extend(
        [
            "",
            "## Robustness Readiness",
            "",
            "| Variant | Ready | Total-Window Share 20x20 | Note |",
            "|---|---:|---:|---|",
        ]
    )
    for _, row in stability.iterrows():
        lines.append(
            f"| {row['variant']} | {row['robustness_ready']} | "
            f"{row['share_total_sessions_min_20x20']} | {row['robustness_note']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Current top1000 scope is a bootstrap data lake, not final historical full-market membership.",
            "- Short-side eligibility uses current shortable/easy-to-borrow metadata only as a proxy.",
            "- Top1500, top2000, and top3000 are intentionally blocked until broader data is available.",
            "- Final universe selection remains gated on a survivorship-bias-free primary vendor.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {"path": json_path.as_posix(), "missing": True}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["path"] = json_path.as_posix()
        payload["missing"] = False
        return payload
    return {"path": json_path.as_posix(), "missing": False, "value": payload}


def _first_or_none(series: pd.Series | None) -> str | None:
    if series is None or len(series) == 0:
        return None
    return str(series.iloc[0])


def _safe_share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _is_true(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "t", "yes", "y", "1"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 1 universe builder.")
    parser.add_argument("--membership-path", default=str(DEFAULT_MEMBERSHIP_PATH))
    parser.add_argument("--asset-qa-by-symbol-path", default=str(DEFAULT_ASSET_QA_BY_SYMBOL))
    parser.add_argument("--phase0-closure-rollup-path", default=str(DEFAULT_PHASE0_CLOSURE_ROLLUP))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min-long-short-names", type=int, default=20)
    args = parser.parse_args(argv)
    result = build_phase1_universe_artifacts(
        membership_path=args.membership_path,
        asset_qa_by_symbol_path=args.asset_qa_by_symbol_path,
        phase0_closure_rollup_path=args.phase0_closure_rollup_path,
        output_root=args.output_root,
        min_long_short_names=args.min_long_short_names,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
