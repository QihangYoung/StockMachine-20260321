from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_MANIFEST = RESEARCH_ROOT / "top1000_data_backfill_20260417" / "top1000_manifest.csv"
DEFAULT_ASSET_AUDIT = (
    RESEARCH_ROOT / "top1000_data_backfill_20260417" / "candidate_asset_filter_audit.csv"
)
DEFAULT_PIT_ROOT = RESEARCH_ROOT / "phase0_provisional_pit_universe_feasibility_20260417"
DEFAULT_ASSET_QA_ROOT = RESEARCH_ROOT / "phase0_asset_class_qa_20260417"
DEFAULT_VENDOR_ROOT = RESEARCH_ROOT / "phase0_vendor_bakeoff_20260417"
DEFAULT_CLOSURE_ROOT = RESEARCH_ROOT / "phase0_closure_20260418"
DEFAULT_TOP1000_COVERAGE_ROLLUP = (
    DEFAULT_MANIFEST.parent / "top1000_backfill_coverage_rollup.json"
)
DEFAULT_YAHOO_GAPFILL_SUMMARY = (
    RESEARCH_ROOT
    / "top1000_gapfill_yahoo_20130805_20151231_20260417"
    / "yahoo_gapfill_summary.json"
)
DEFAULT_YAHOO_RECONCILIATION_SUMMARY = (
    RESEARCH_ROOT
    / "top1000_gapfill_yahoo_20130805_20151231_20260417"
    / "yahoo_vs_alpaca_reconciliation_20160104_top120_summary.json"
)
DEFAULT_PIT_ROLLUP = DEFAULT_PIT_ROOT / "phase0_provisional_pit_universe_feasibility_rollup.json"
DEFAULT_ASSET_QA_ROLLUP = DEFAULT_ASSET_QA_ROOT / "top1000_asset_class_qa_rollup.json"
DEFAULT_ASSET_QA_BY_SYMBOL = DEFAULT_ASSET_QA_ROOT / "top1000_asset_class_qa_by_symbol.csv"
DEFAULT_VENDOR_ROLLUP = DEFAULT_VENDOR_ROOT / "vendor_bakeoff_rollup.json"
DEFAULT_DAILY_GLOBS = (
    "data/silver/daily_bar/phase0_top1000_yahoo_gap_20130805_20151231_chunk*.jsonl",
    "data/silver/daily_bar/phase0_top1000_sip_raw_20160104_20260416_chunk*.jsonl",
)
DEFAULT_TOP_BUCKETS = (500, 1000, 1500, 2000, 3000)
DEFAULT_ADV_THRESHOLDS = (10_000_000, 20_000_000, 30_000_000, 50_000_000)

ETF_PATTERN = re.compile(
    r"\b(ETF|ETN|FUND|FUNDS|INDEX FUND|EXCHANGE[- ]TRADED|SPDR|ISHARES|VANGUARD|"
    r"PROSHARES|DIREXION|GLOBAL X|WISDOMTREE|FIRST TRUST|VANECK)\b",
    re.IGNORECASE,
)
NON_COMMON_PATTERN = re.compile(
    r"\b(WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PREF|DEBENTURES?|NOTES?|BONDS?|"
    r"ACQUISITION|BLANK CHECK|SPAC)\b",
    re.IGNORECASE,
)
ADR_PATTERN = re.compile(
    r"\b(ADR|ADS|AMERICAN DEPOSITARY|DEPOSITARY SHARES?|PLC|P\.L\.C\.|N\.V\.|"
    r"S\.A\.|SE|LTD|LIMITED|COMPANHIA|BANCO|HOLDING[S]? PLC)\b",
    re.IGNORECASE,
)
TRUST_PATTERN = re.compile(r"\b(REIT|REAL ESTATE INVESTMENT TRUST|ROYALTY TRUST)\b", re.IGNORECASE)


def build_pit_universe_feasibility(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    output_root: str | Path = DEFAULT_PIT_ROOT,
    validation_start: str = "2013-08-05",
    validation_end: str = "2019-12-31",
    lookback_sessions: int = 20,
    min_observations: int = 15,
    price_floor: float = 10.0,
    beta_warmup_observations: int = 126,
    beta_full_observations: int = 252,
    top_buckets: Sequence[int] = DEFAULT_TOP_BUCKETS,
    adv_thresholds: Sequence[int] = DEFAULT_ADV_THRESHOLDS,
) -> dict[str, Any]:
    """Build validation-only lagged-liquidity universe artifacts."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    symbols = set(manifest["symbol"].astype(str))
    files = _expand_globs(daily_globs)
    if not files:
        raise FileNotFoundError(f"No daily files matched: {list(map(str, daily_globs))}")

    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in files:
        source = _source_segment(path)
        source_counts.setdefault(source, 0)
        for row in _iter_jsonl(path):
            session_date = row.get("session_date")
            symbol = row.get("symbol")
            if not isinstance(session_date, str) or not isinstance(symbol, str):
                continue
            if session_date < validation_start or session_date > validation_end or symbol not in symbols:
                continue
            rows.append(
                {
                    "session_date": session_date,
                    "symbol": symbol,
                    "close": _to_float(row.get("close")),
                    "volume": _to_float(row.get("volume")),
                    "dollar_volume": _to_float(row.get("dollar_volume")),
                    "source_segment": source,
                }
            )
            source_counts[source] += 1
    if not rows:
        raise ValueError("No rows loaded for the requested validation window.")

    frame = pd.DataFrame(rows)
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame = frame.sort_values(["symbol", "session_date"]).drop_duplicates(
        ["symbol", "session_date"], keep="last"
    )
    frame = frame.reset_index(drop=True)

    eligible = _lagged_liquidity(
        frame,
        lookback_sessions=lookback_sessions,
        min_observations=min_observations,
        price_floor=price_floor,
        beta_warmup_observations=beta_warmup_observations,
        beta_full_observations=beta_full_observations,
    )
    for bucket in top_buckets:
        eligible[f"in_top{bucket}"] = eligible["liquidity_rank"] <= bucket
    for threshold in adv_thresholds:
        eligible[_adv_label(threshold)] = eligible["trailing_median_dollar_volume_20"] >= threshold

    membership_path = output_dir / "provisional_current_top1000_lagged_liquidity_membership_validation.csv.gz"
    eligible[_membership_columns(top_buckets, adv_thresholds)].to_csv(
        membership_path, index=False, compression="gzip"
    )
    daily = _daily_counts(frame, eligible, top_buckets=top_buckets, adv_thresholds=adv_thresholds)
    daily_path = output_dir / "candidate_universe_daily_counts_validation.csv"
    daily.to_csv(daily_path, index=False)
    summary = _candidate_summary(daily, top_buckets=top_buckets, adv_thresholds=adv_thresholds)
    summary_path = output_dir / "candidate_universe_feasibility_summary_validation.csv"
    summary.to_csv(summary_path, index=False)
    source_path = output_dir / "source_segment_daily_counts_validation.csv"
    _source_by_date(eligible).to_csv(source_path, index=False)

    rollup = {
        "created_at_utc": _utc_now(),
        "validation_start": validation_start,
        "validation_end": validation_end,
        "lookback_sessions": lookback_sessions,
        "min_liquidity_observations": min_observations,
        "price_floor": price_floor,
        "input_daily_files": [path.as_posix() for path in files],
        "manifest_path": Path(manifest_path).as_posix(),
        "artifact_dir": output_dir.as_posix(),
        "source_rows_loaded": source_counts,
        "validation_rows_loaded": int(len(frame)),
        "symbols_loaded": int(frame["symbol"].nunique()),
        "validation_sessions": int(len(daily)),
        "eligible_membership_rows": int(len(eligible)),
        "membership_artifact": membership_path.as_posix(),
        "daily_counts_artifact": daily_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "source_segment_daily_counts_artifact": source_path.as_posix(),
        "key_findings": _pit_key_findings(daily, summary),
        "limitations": [
            "Provisional and limited to the current top1000 bootstrap symbol set.",
            "Top1500/top2000/top3000 cannot be evaluated from a top1000-only backfill.",
            "No alpha signal, portfolio return, or test-window performance was computed.",
        ],
    }
    (output_dir / "phase0_provisional_pit_universe_feasibility_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return rollup


def run_asset_class_qa(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    asset_audit_path: str | Path = DEFAULT_ASSET_AUDIT,
    output_root: str | Path = DEFAULT_ASSET_QA_ROOT,
) -> dict[str, Any]:
    """Generate current top1000 common-stock cleanliness QA artifacts."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    assets = pd.read_csv(asset_audit_path)
    frame = manifest.merge(
        assets.drop_duplicates("symbol"),
        on="symbol",
        how="left",
        suffixes=("_manifest", "_asset"),
    )
    rows = []
    for _, row in frame.iterrows():
        symbol = str(row["symbol"])
        name = _row_str(row, "name_manifest") or _row_str(row, "name_asset")
        classification, flags = classify_asset(symbol=symbol, name=name, row=row)
        tradable = _row_bool_any(row, "tradable")
        marginable = _row_bool_any(row, "marginable")
        shortable = _row_bool_any(row, "shortable")
        easy_to_borrow = _row_bool_any(row, "easy_to_borrow")
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": _row_str_any(row, "exchange"),
                "liquidity_rank": row.get("liquidity_rank"),
                "classification": classification,
                "flags": "|".join(flags),
                "review_required": classification != "common_stock_candidate" or bool(flags),
                "tradable": tradable,
                "marginable": marginable,
                "shortable": shortable,
                "easy_to_borrow": easy_to_borrow,
                "asset_filter_reason": _row_value_any(row, "phase0_filter_reason"),
            }
        )
    qa = pd.DataFrame(rows)
    by_symbol_path = output_dir / "top1000_asset_class_qa_by_symbol.csv"
    qa.to_csv(by_symbol_path, index=False)
    summary = (
        qa.groupby("classification")
        .agg(symbols=("symbol", "nunique"), review_required=("review_required", "sum"))
        .reset_index()
        .sort_values(["review_required", "symbols"], ascending=[False, False])
    )
    summary_path = output_dir / "top1000_asset_class_qa_summary.csv"
    summary.to_csv(summary_path, index=False)
    flag_counts = _flag_counts(qa)
    flag_counts_path = output_dir / "top1000_asset_class_flag_counts.csv"
    flag_counts.to_csv(flag_counts_path, index=False)
    review_path = output_dir / "top1000_asset_class_review_queue.csv"
    qa[qa["review_required"]].sort_values("liquidity_rank").to_csv(review_path, index=False)

    rollup = {
        "created_at_utc": _utc_now(),
        "manifest_path": Path(manifest_path).as_posix(),
        "asset_audit_path": Path(asset_audit_path).as_posix(),
        "artifact_dir": output_dir.as_posix(),
        "symbols_checked": int(qa["symbol"].nunique()),
        "common_stock_candidates": int((qa["classification"] == "common_stock_candidate").sum()),
        "review_required_symbols": int(qa["review_required"].sum()),
        "blocked_non_common_like_symbols": int((qa["classification"] == "blocked_non_common_like").sum()),
        "foreign_or_adr_review_symbols": int((qa["classification"] == "review_foreign_or_adr_like").sum()),
        "trust_or_reit_review_symbols": int((qa["classification"] == "review_trust_or_reit_like").sum()),
        "by_symbol_artifact": by_symbol_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "flag_counts_artifact": flag_counts_path.as_posix(),
        "review_queue_artifact": review_path.as_posix(),
        "limitations": [
            "Name/metadata QA is not a legal security master.",
            "Review flags do not automatically exclude a symbol.",
            "ADR and foreign issuer policy must be chosen before final universe freeze.",
        ],
    }
    (output_dir / "top1000_asset_class_qa_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return rollup


def classify_asset(*, symbol: str, name: str, row: pd.Series | None = None) -> tuple[str, list[str]]:
    flags: list[str] = []
    if ETF_PATTERN.search(name):
        flags.append("etf_or_fund_name_pattern")
    if NON_COMMON_PATTERN.search(name):
        flags.append("non_common_security_name_pattern")
    if ADR_PATTERN.search(name):
        flags.append("adr_or_foreign_issuer_name_pattern")
    if TRUST_PATTERN.search(name):
        flags.append("trust_or_reit_name_pattern")
    if "." in symbol:
        flags.append("class_share_symbol")
    if row is not None:
        if _row_bool_any(row, "tradable") is False:
            flags.append("not_tradable_current_metadata")
        if _row_bool_any(row, "shortable") is False:
            flags.append("not_shortable_current_metadata")
        if _row_bool_any(row, "easy_to_borrow") is False:
            flags.append("not_easy_to_borrow_current_metadata")

    if "etf_or_fund_name_pattern" in flags or "non_common_security_name_pattern" in flags:
        classification = "blocked_non_common_like"
    elif "adr_or_foreign_issuer_name_pattern" in flags:
        classification = "review_foreign_or_adr_like"
    elif "trust_or_reit_name_pattern" in flags:
        classification = "review_trust_or_reit_like"
    else:
        classification = "common_stock_candidate"
    return classification, flags


def write_vendor_bakeoff_artifacts(
    *,
    output_root: str | Path = DEFAULT_VENDOR_ROOT,
) -> dict[str, Any]:
    """Write the Phase 0 vendor bake-off matrix and acceptance checks."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    vendors = _vendor_rows()
    checks = _vendor_checks()
    vendors_path = output_dir / "vendor_bakeoff_candidates.csv"
    checks_path = output_dir / "vendor_bakeoff_acceptance_checks.csv"
    _write_csv(vendors_path, vendors)
    _write_csv(checks_path, checks)
    memo_path = output_dir / "vendor_bakeoff_plan.md"
    memo_path.write_text(_vendor_bakeoff_markdown(vendors, checks), encoding="utf-8")
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "candidate_vendors": len(vendors),
        "acceptance_checks": len(checks),
        "recommended_order": ["Norgate Data", "Sharadar / Nasdaq Data Link", "CRSP / WRDS", "Polygon.io"],
        "vendor_candidates_artifact": vendors_path.as_posix(),
        "acceptance_checks_artifact": checks_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "decision_rule": (
            "Select the first vendor that passes survivorship-bias, adjustment, "
            "common-stock classification, 2013-2015 coverage, and repeatable-ingestion checks."
        ),
    }
    (output_dir / "vendor_bakeoff_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return rollup


def write_phase0_closure_artifacts(
    *,
    output_root: str | Path = DEFAULT_CLOSURE_ROOT,
    top1000_coverage_rollup_path: str | Path = DEFAULT_TOP1000_COVERAGE_ROLLUP,
    yahoo_gapfill_summary_path: str | Path = DEFAULT_YAHOO_GAPFILL_SUMMARY,
    yahoo_reconciliation_summary_path: str | Path = DEFAULT_YAHOO_RECONCILIATION_SUMMARY,
    pit_rollup_path: str | Path = DEFAULT_PIT_ROLLUP,
    asset_qa_rollup_path: str | Path = DEFAULT_ASSET_QA_ROLLUP,
    asset_qa_by_symbol_path: str | Path = DEFAULT_ASSET_QA_BY_SYMBOL,
    vendor_rollup_path: str | Path = DEFAULT_VENDOR_ROLLUP,
) -> dict[str, Any]:
    """Write the local Phase 0 closure packet and Phase 1 entry gates."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "top1000_coverage": _read_json_if_exists(top1000_coverage_rollup_path),
        "yahoo_gapfill": _read_json_if_exists(yahoo_gapfill_summary_path),
        "yahoo_reconciliation": _read_json_if_exists(yahoo_reconciliation_summary_path),
        "pit_universe": _read_json_if_exists(pit_rollup_path),
        "asset_qa": _read_json_if_exists(asset_qa_rollup_path),
        "vendor_bakeoff": _read_json_if_exists(vendor_rollup_path),
    }
    asset_policy_counts = _asset_policy_counts(asset_qa_by_symbol_path)
    decisions = _phase0_local_decisions(asset_policy_counts)
    blockers = _phase0_phase1_blockers()

    decisions_path = output_dir / "phase0_local_policy_decisions.csv"
    blockers_path = output_dir / "phase0_phase1_blockers.csv"
    memo_path = output_dir / "phase0_closure_memo.md"
    rollup_path = output_dir / "phase0_closure_rollup.json"
    _write_csv(decisions_path, decisions)
    _write_csv(blockers_path, blockers)
    memo_path.write_text(
        _phase0_closure_markdown(inputs, asset_policy_counts, decisions, blockers),
        encoding="utf-8",
    )

    missing_inputs = [name for name, payload in inputs.items() if payload.get("missing")]
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "phase0_status": "local_phase0_complete_for_phase1_plumbing",
        "phase1_entry_recommendation": (
            "Start Phase 1 mechanics with validation-only current-top1000-scope top500/top1000 "
            "experiments, while treating broader universes and final performance claims as gated "
            "on a research-grade vendor."
        ),
        "test_lockbox": "No test-window strategy performance should be used for Phase 0 closure.",
        "asset_policy_counts": asset_policy_counts,
        "missing_inputs": missing_inputs,
        "local_policy_decisions_artifact": decisions_path.as_posix(),
        "phase1_blockers_artifact": blockers_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "inputs": {name: payload.get("path") for name, payload in inputs.items()},
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _lagged_liquidity(
    frame: pd.DataFrame,
    *,
    lookback_sessions: int,
    min_observations: int,
    price_floor: float,
    beta_warmup_observations: int,
    beta_full_observations: int,
) -> pd.DataFrame:
    grouped = frame.groupby("symbol", group_keys=False)
    frame["lagged_close"] = grouped["close"].shift(1)
    frame["prior_bar_count"] = grouped.cumcount()
    frame["trailing_median_dollar_volume_20"] = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(lookback_sessions, min_periods=min_observations).median()
    )
    frame["trailing_obs_20"] = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(lookback_sessions, min_periods=1).count()
    )
    frame["beta_warmup_proxy_ready"] = frame["prior_bar_count"] >= beta_warmup_observations
    frame["beta_full_lookback_proxy_ready"] = frame["prior_bar_count"] >= beta_full_observations
    frame["liquidity_eligible"] = (
        (frame["trailing_obs_20"] >= min_observations)
        & frame["trailing_median_dollar_volume_20"].notna()
        & (frame["trailing_median_dollar_volume_20"] > 0)
        & frame["lagged_close"].notna()
        & (frame["lagged_close"] >= price_floor)
    )
    eligible = frame[frame["liquidity_eligible"]].copy()
    eligible["liquidity_rank"] = eligible.groupby("session_date")[
        "trailing_median_dollar_volume_20"
    ].rank(method="first", ascending=False)
    eligible["liquidity_rank"] = eligible["liquidity_rank"].astype(int)
    return eligible


def _daily_counts(
    frame: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_buckets: Sequence[int],
    adv_thresholds: Sequence[int],
) -> pd.DataFrame:
    dates = pd.DataFrame({"session_date": sorted(frame["session_date"].unique())})
    daily = frame.groupby("session_date").agg(
        symbols_with_bar=("symbol", "nunique"),
        raw_rows=("symbol", "size"),
    ).reset_index()
    base = eligible.groupby("session_date").agg(
        liquidity_eligible_count=("symbol", "nunique"),
        beta_warmup_proxy_count=("beta_warmup_proxy_ready", "sum"),
        beta_full_lookback_proxy_count=("beta_full_lookback_proxy_ready", "sum"),
    ).reset_index()
    daily = dates.merge(daily, on="session_date", how="left").merge(base, on="session_date", how="left")
    for column in daily.columns:
        if column != "session_date":
            daily[column] = daily[column].fillna(0).astype(int)
    for bucket in top_buckets:
        daily = _merge_counts(daily, eligible[eligible[f"in_top{bucket}"]], f"top{bucket}")
    for threshold in adv_thresholds:
        label = _adv_label(threshold)
        daily = _merge_counts(daily, eligible[eligible[label]], label)
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.date.astype(str)
    return daily


def _merge_counts(daily: pd.DataFrame, subset: pd.DataFrame, prefix: str) -> pd.DataFrame:
    counts = subset.groupby("session_date").agg(
        **{
            f"{prefix}_count": ("symbol", "nunique"),
            f"{prefix}_beta_warmup_proxy_count": ("beta_warmup_proxy_ready", "sum"),
            f"{prefix}_beta_full_lookback_proxy_count": ("beta_full_lookback_proxy_ready", "sum"),
        }
    ).reset_index()
    daily = daily.merge(counts, on="session_date", how="left")
    for suffix in ("count", "beta_warmup_proxy_count", "beta_full_lookback_proxy_count"):
        column = f"{prefix}_{suffix}"
        daily[column] = daily[column].fillna(0).astype(int)
    return daily


def _candidate_summary(
    daily: pd.DataFrame,
    *,
    top_buckets: Sequence[int],
    adv_thresholds: Sequence[int],
) -> pd.DataFrame:
    rows = [
        _summarize(
            daily,
            "liquidity_eligible_all_current_top1000_scope",
            "liquidity_eligible_count",
            "beta_warmup_proxy_count",
            "beta_full_lookback_proxy_count",
            "provisional_current_top1000_scope",
        )
    ]
    for bucket in top_buckets:
        status = (
            "supported_within_current_top1000_scope"
            if bucket <= 1000
            else "blocked_current_backfill_has_only_top1000_symbols"
        )
        rows.append(
            _summarize(
                daily,
                f"top{bucket}_by_lagged_trailing_median_dollar_volume_20d",
                f"top{bucket}_count",
                f"top{bucket}_beta_warmup_proxy_count",
                f"top{bucket}_beta_full_lookback_proxy_count",
                status,
            )
        )
    for threshold in adv_thresholds:
        label = _adv_label(threshold)
        rows.append(
            _summarize(
                daily,
                f"{label}_lagged_trailing_median_dollar_volume_20d",
                f"{label}_count",
                f"{label}_beta_warmup_proxy_count",
                f"{label}_beta_full_lookback_proxy_count",
                "supported_only_within_current_top1000_scope",
            )
        )
    return pd.DataFrame(rows)


def _summarize(
    daily: pd.DataFrame,
    name: str,
    count_col: str,
    beta_col: str,
    full_beta_col: str,
    status: str,
) -> dict[str, Any]:
    counts = daily[count_col]
    beta = daily[beta_col]
    full_beta = daily[full_beta_col]
    nonzero = daily.loc[counts > 0, "session_date"]
    beta40 = daily.loc[beta >= 40, "session_date"]
    beta60 = daily.loc[beta >= 60, "session_date"]
    fullbeta40 = daily.loc[full_beta >= 40, "session_date"]
    return {
        "candidate": name,
        "data_status": status,
        "sessions": int(len(daily)),
        "first_nonzero_date": nonzero.iloc[0] if len(nonzero) else None,
        "median_members": float(counts.median()),
        "p10_members": float(counts.quantile(0.10)),
        "min_members": int(counts.min()),
        "max_members": int(counts.max()),
        "median_beta_warmup_proxy_members": float(beta.median()),
        "first_beta_warmup_proxy_20_20_date": beta40.iloc[0] if len(beta40) else None,
        "first_beta_warmup_proxy_30_30_date": beta60.iloc[0] if len(beta60) else None,
        "share_sessions_beta_warmup_proxy_20_20": float((beta >= 40).mean()),
        "share_sessions_beta_warmup_proxy_30_30": float((beta >= 60).mean()),
        "median_full_beta_lookback_proxy_members": float(full_beta.median()),
        "first_full_beta_lookback_proxy_20_20_date": fullbeta40.iloc[0] if len(fullbeta40) else None,
        "share_sessions_full_beta_lookback_proxy_20_20": float((full_beta >= 40).mean()),
    }


def _pit_key_findings(daily: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    findings: dict[str, Any] = {
        "median_symbols_with_bar": float(daily["symbols_with_bar"].median()),
        "min_symbols_with_bar": int(daily["symbols_with_bar"].min()),
        "median_liquidity_eligible_count": float(daily["liquidity_eligible_count"].median()),
        "min_liquidity_eligible_count": int(daily["liquidity_eligible_count"].min()),
        "median_top500_count": float(daily["top500_count"].median()) if "top500_count" in daily else None,
        "median_top1000_count": float(daily["top1000_count"].median()) if "top1000_count" in daily else None,
    }
    for bucket in (500, 1000):
        row = summary[summary["candidate"].str.startswith(f"top{bucket}_")]
        if not row.empty:
            findings[f"first_top{bucket}_beta_warmup_proxy_30_30_date"] = row[
                "first_beta_warmup_proxy_30_30_date"
            ].iloc[0]
    return findings


def _source_by_date(eligible: pd.DataFrame) -> pd.DataFrame:
    frame = eligible.groupby(["session_date", "source_segment"]).size().reset_index(name="eligible_rows")
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    return frame


def _membership_columns(top_buckets: Sequence[int], adv_thresholds: Sequence[int]) -> list[str]:
    columns = [
        "session_date",
        "symbol",
        "source_segment",
        "lagged_close",
        "trailing_obs_20",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
        "beta_warmup_proxy_ready",
        "beta_full_lookback_proxy_ready",
    ]
    columns.extend(f"in_top{bucket}" for bucket in top_buckets)
    columns.extend(_adv_label(threshold) for threshold in adv_thresholds)
    return columns


def _flag_counts(qa: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for raw_flags in qa["flags"].fillna(""):
        for flag in str(raw_flags).split("|"):
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return pd.DataFrame([{"flag": flag, "symbols": count} for flag, count in sorted(counts.items())])


def _vendor_rows() -> list[dict[str, str]]:
    return [
        {
            "vendor": "Norgate Data",
            "priority": "1",
            "role": "preferred independent-research source",
            "expected_strength": "survivorship-bias-free US equities, delisted stocks, historical membership/context",
            "main_risk": "Windows/local tooling integration and subscription workflow",
            "trial_task": "Export 2013-08-05, 2014-01-02, 2015-12-31 bars plus delisted coverage sample.",
            "source_url": "https://norgatedata.com/data-content-tables.php",
        },
        {
            "vendor": "Sharadar / Nasdaq Data Link",
            "priority": "2",
            "role": "preferred API-style source",
            "expected_strength": "active plus delisted equity prices, repeatable ingestion, corporate actions",
            "main_risk": "subscription access and point-in-time membership still built by us",
            "trial_task": "Pull price/ticker/action tables for 2013-2015 coverage and corporate-action QA.",
            "source_url": "https://docs.data.nasdaq.com/",
        },
        {
            "vendor": "CRSP / WRDS",
            "priority": "3",
            "role": "gold standard if accessible",
            "expected_strength": "research-grade returns, distributions, delistings, identifiers",
            "main_risk": "institutional access requirement and ingestion complexity",
            "trial_task": "Confirm access and export daily stock plus delisting-return sample.",
            "source_url": "https://wrds-www.wharton.upenn.edu/",
        },
        {
            "vendor": "Polygon.io",
            "priority": "4",
            "role": "API fallback",
            "expected_strength": "stock aggregates, tickers, splits/dividends endpoints, automation-friendly API",
            "main_risk": "total-return adjustment must be rebuilt and audited",
            "trial_task": "Pull grouped daily bars and corporate actions for a 2013-2015 sample.",
            "source_url": "https://polygon.io/docs/rest/stocks/aggregates/daily-market-summary",
        },
        {
            "vendor": "QuantQuote / HistoricalData.net / EODHD",
            "priority": "5",
            "role": "low-cost backup",
            "expected_strength": "bulk historical coverage may fill practical gaps quickly",
            "main_risk": "quality and metadata must be independently audited",
            "trial_task": "Run split/dividend/delisting and ETF-filter QA before research use.",
            "source_url": "https://quantquote.com/stock-data",
        },
        {
            "vendor": "Yahoo / Stooq",
            "priority": "6",
            "role": "sanity-check or provisional gap fill only",
            "expected_strength": "quick access for plumbing and price-scale reconciliation",
            "main_risk": "not a final survivorship-bias-free institutional source",
            "trial_task": "Keep as cross-check and temporary data-engineering bridge only.",
            "source_url": "https://finance.yahoo.com/",
        },
    ]


def _vendor_checks() -> list[dict[str, str]]:
    checks = [
        ("coverage_2013_2015", "Daily bars exist on 2013-08-05, 2014-01-02, and 2015-12-31."),
        ("active_and_delisted", "Active and delisted names are present, or survivorship bias is quantified."),
        ("corporate_actions", "Split and dividend adjustment factors reconcile on sampled events."),
        ("common_stock_filter", "Common stocks can be separated from ETFs, units, warrants, preferreds, and SPACs."),
        ("lagged_membership", "Lagged dollar-volume topN membership can be generated without future leakage."),
        ("alpaca_overlap", "2016+ overlap sample reconciles against current Alpaca SIP backfill."),
        ("automation", "Ingestion can be repeated from a documented command or export procedure."),
        ("cost_fit", "Subscription cost and operational burden are acceptable for this research line."),
    ]
    return [
        {"check_id": check_id, "acceptance_check": text, "required": "yes"}
        for check_id, text in checks
    ]


def _vendor_bakeoff_markdown(vendors: list[dict[str, str]], checks: list[dict[str, str]]) -> str:
    lines = [
        "# Pure Alpha Phase 0 Vendor Bake-Off Plan",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Decision Rule",
        "",
        "Select the first vendor that passes survivorship-bias, adjustment, common-stock "
        "classification, 2013-2015 coverage, and repeatable-ingestion checks.",
        "",
        "## Candidate Order",
        "",
        "| Priority | Vendor | Role | Main Risk |",
        "|---:|---|---|---|",
    ]
    for vendor in vendors:
        lines.append(
            f"| {vendor['priority']} | {vendor['vendor']} | {vendor['role']} | {vendor['main_risk']} |"
        )
    lines.extend(["", "## Acceptance Checks", "", "| Check | Required |", "|---|---|"])
    for check in checks:
        lines.append(f"| {check['acceptance_check']} | {check['required']} |")
    lines.extend(
        [
            "",
            "Do not use test-window strategy performance during vendor selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _phase0_local_decisions(asset_policy_counts: dict[str, Any]) -> list[dict[str, str]]:
    core_symbols = asset_policy_counts.get("default_core_symbols")
    review_symbols = asset_policy_counts.get("review_required_symbols")
    if core_symbols is None or review_symbols is None:
        asset_policy = (
            "Default core excludes review-required names until a security-master decision is made."
        )
    else:
        asset_policy = (
            f"Default core excludes {review_symbols} review-required names, leaving "
            f"{core_symbols} symbols for the cleanest bootstrap core."
        )
    return [
        {
            "decision_id": "window_alignment",
            "status": "closed_locally",
            "policy": "Use Beta-line validation window 2013-08-05 through 2019-12-31.",
            "phase1_effect": "Universe and signal experiments must stay validation-only until final lockbox review.",
            "phase1_gate": "No test-window strategy performance in Phase 0 or early Phase 1.",
        },
        {
            "decision_id": "bootstrap_universe",
            "status": "closed_for_plumbing",
            "policy": "Use current top1000 only as a data bootstrap, not as historical membership truth.",
            "phase1_effect": "Daily membership must be recomputed from lagged price/volume on each date.",
            "phase1_gate": "Final claims require survivorship-bias-free active plus delisted coverage.",
        },
        {
            "decision_id": "asset_class_policy",
            "status": "closed_for_default_core",
            "policy": asset_policy,
            "phase1_effect": "Run the first mechanics on the clean core, then test ADR/REIT/class-share extensions separately.",
            "phase1_gate": "Freeze inclusion rules before alpha model selection.",
        },
        {
            "decision_id": "short_side_policy",
            "status": "closed_as_proxy_only",
            "policy": "Current shortable/easy-to-borrow metadata is a sanity check, not a historical borrow dataset.",
            "phase1_effect": "Short book needs liquidity, borrow stress, and no hard-to-borrow alpha assumptions.",
            "phase1_gate": "Historical borrow/locate data or conservative proxy stress is required before production sizing.",
        },
        {
            "decision_id": "adjustment_policy",
            "status": "closed_as_provisional",
            "policy": "Yahoo 2013-2015 gap-fill is acceptable for plumbing but not final adjustment truth.",
            "phase1_effect": "Outlier corporate-action names must be excluded or reconciled in sensitivity runs.",
            "phase1_gate": "A primary vendor must pass split/dividend/delisting reconciliation.",
        },
        {
            "decision_id": "topn_policy",
            "status": "closed_for_top500_current_scope",
            "policy": "Top500 mechanics are supported inside current top1000 scope; top1500+ is blocked by data breadth.",
            "phase1_effect": "Start with top500 and current-scope top1000 diagnostics before expanding.",
            "phase1_gate": "Top1500/top2000/top3000 need broader historical bars and PIT membership construction.",
        },
        {
            "decision_id": "robustness_framework",
            "status": "closed_to_reuse_beta_framework",
            "policy": "Reuse the Beta-line robustness frame for windows, costs, turnover, parameter, and sub-universe stress.",
            "phase1_effect": "Pure alpha claims must survive robustness checks, not just one selected validation run.",
            "phase1_gate": "Promote only signals that pass predeclared robustness gates.",
        },
    ]


def _phase0_phase1_blockers() -> list[dict[str, str]]:
    return [
        {
            "blocker_id": "primary_vendor_selection",
            "priority": "P0",
            "blocker": "No research-grade primary vendor has been selected yet.",
            "why_it_matters": "Final claims need active plus delisted coverage and audited corporate actions.",
            "exit_criterion": "One vendor passes the Phase 0 vendor bake-off acceptance checks.",
        },
        {
            "blocker_id": "survivorship_bias_free_universe",
            "priority": "P0",
            "blocker": "Current top1000 membership is a present-day bootstrap list.",
            "why_it_matters": "Using it as history would overweight survivors and distort alpha estimates.",
            "exit_criterion": "Build lagged PIT membership from a broad active plus delisted security universe.",
        },
        {
            "blocker_id": "broad_top1500_plus_data",
            "priority": "P1",
            "blocker": "Top1500/top2000/top3000 cannot be tested from a top1000-only backfill.",
            "why_it_matters": "Capacity and liquidity trade-offs require broader universe comparisons.",
            "exit_criterion": "Acquire or backfill broader historical bars and rerun PIT feasibility.",
        },
        {
            "blocker_id": "historical_shortability_borrow",
            "priority": "P1",
            "blocker": "Historical borrow cost and locate availability are not available locally.",
            "why_it_matters": "The short book can look great before borrow fees and hard-to-borrow constraints.",
            "exit_criterion": "Use vendor borrow data or a documented conservative proxy stress suite.",
        },
        {
            "blocker_id": "corporate_action_reconciliation",
            "priority": "P1",
            "blocker": "Yahoo gap-fill adjustment factors have known outliers versus Alpaca overlap.",
            "why_it_matters": "Bad adjustments can manufacture false reversal/momentum signals.",
            "exit_criterion": "Reconcile split/dividend/delist events or quarantine outlier symbols.",
        },
        {
            "blocker_id": "pit_industry_and_security_master",
            "priority": "P2",
            "blocker": "PIT industry, ADR, REIT, and class-share metadata are not fully frozen.",
            "why_it_matters": "Beta-matched long/short portfolios can still carry hidden sector or issuer bets.",
            "exit_criterion": "Freeze security-master rules and sector/industry metadata before model comparison.",
        },
    ]


def _phase0_closure_markdown(
    inputs: dict[str, dict[str, Any]],
    asset_policy_counts: dict[str, Any],
    decisions: list[dict[str, str]],
    blockers: list[dict[str, str]],
) -> str:
    top1000 = inputs["top1000_coverage"]
    yahoo = inputs["yahoo_gapfill"]
    recon = inputs["yahoo_reconciliation"]
    pit = inputs["pit_universe"]
    asset = inputs["asset_qa"]
    vendor = inputs["vendor_bakeoff"]
    lines = [
        "# Pure Alpha Phase 0 Closure Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Closure State",
        "",
        "Phase 0 is locally complete for Phase 1 data engineering and strategy plumbing. "
        "It is not a final claim that the research data is production-grade, because the "
        "primary survivorship-bias-free vendor is still an external gate.",
        "",
        "No test-window strategy performance was used.",
        "",
        "## Evidence Snapshot",
        "",
        f"- Top1000 Alpaca SIP rows: {_payload_value(top1000, 'daily_bar_rows_total')}",
        f"- Top1000 Alpaca coverage: {_payload_value(top1000, 'daily_first_date_min')} through "
        f"{_payload_value(top1000, 'daily_last_date_max')}",
        f"- Yahoo gap-fill rows: {_payload_value(yahoo, 'daily_rows_total')}",
        f"- Yahoo full-coverage symbols: {_payload_value(yahoo, 'symbols_with_full_608_rows')}",
        f"- Alpaca/Yahoo overlap median raw-close diff: {_payload_value(recon, 'median_abs_close_rel_diff')}",
        f"- PIT validation rows loaded: {_payload_value(pit, 'validation_rows_loaded')}",
        f"- PIT median liquidity-eligible members: "
        f"{_payload_value(pit.get('key_findings', {}), 'median_liquidity_eligible_count')}",
        f"- Asset QA checked symbols: {_payload_value(asset, 'symbols_checked')}",
        f"- Asset QA review-required symbols: {_payload_value(asset, 'review_required_symbols')}",
        f"- Vendor candidates: {_payload_value(vendor, 'candidate_vendors')}",
        "",
        "## Default Phase 1 Entry",
        "",
        "Start with validation-only top500 and current-top1000-scope mechanics. Use lagged "
        "daily liquidity membership, beta-matched long/short construction, conservative "
        "transaction-cost and borrow assumptions, and the Beta-line robustness frame.",
        "",
        "Asset policy counts:",
        "",
        f"- default core symbols: {asset_policy_counts.get('default_core_symbols', 'n/a')}",
        f"- review-required symbols: {asset_policy_counts.get('review_required_symbols', 'n/a')}",
        f"- current easy-to-borrow symbols: {asset_policy_counts.get('easy_to_borrow_symbols', 'n/a')}",
        "",
        "## Local Decisions",
        "",
        "| Decision | Status | Phase 1 Gate |",
        "|---|---|---|",
    ]
    for decision in decisions:
        lines.append(
            f"| {decision['decision_id']} | {decision['status']} | {decision['phase1_gate']} |"
        )
    lines.extend(["", "## Remaining Blockers", "", "| Priority | Blocker | Exit Criterion |", "|---|---|---|"])
    for blocker in blockers:
        lines.append(
            f"| {blocker['priority']} | {blocker['blocker']} | {blocker['exit_criterion']} |"
        )
    lines.extend(
        [
            "",
            "Phase 1 can begin on mechanics now, but final product claims remain gated on "
            "vendor selection, survivorship-bias-free PIT membership, and short-book cost realism.",
        ]
    )
    return "\n".join(lines) + "\n"


def _asset_policy_counts(asset_qa_by_symbol_path: str | Path) -> dict[str, Any]:
    path = Path(asset_qa_by_symbol_path)
    counts: dict[str, Any] = {"artifact_path": path.as_posix(), "available": False}
    if not path.exists():
        return counts
    qa = pd.read_csv(path)
    review_required = (
        qa["review_required"].map(_is_true) if "review_required" in qa else pd.Series(False, index=qa.index)
    )
    shortable = qa["shortable"].map(_is_true) if "shortable" in qa else pd.Series(False, index=qa.index)
    easy_to_borrow = (
        qa["easy_to_borrow"].map(_is_true)
        if "easy_to_borrow" in qa
        else pd.Series(False, index=qa.index)
    )
    flags = qa["flags"].fillna("").astype(str) if "flags" in qa else pd.Series("", index=qa.index)
    classification = (
        qa["classification"] if "classification" in qa else pd.Series("", index=qa.index)
    )
    counts.update(
        {
            "available": True,
            "symbols": int(len(qa)),
            "default_core_symbols": int((~review_required).sum()),
            "review_required_symbols": int(review_required.sum()),
            "shortable_symbols": int(shortable.sum()),
            "easy_to_borrow_symbols": int(easy_to_borrow.sum()),
            "foreign_or_adr_review_symbols": int(
                (classification == "review_foreign_or_adr_like").sum()
            ),
            "trust_or_reit_review_symbols": int(
                (classification == "review_trust_or_reit_like").sum()
            ),
            "class_share_review_symbols": int(flags.str.contains("class_share_symbol", regex=False).sum()),
        }
    )
    return counts


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {"path": json_path.as_posix(), "missing": True}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        result = dict(payload)
    else:
        result = {"value": payload}
    result["path"] = json_path.as_posix()
    result["missing"] = False
    return result


def _payload_value(payload: dict[str, Any], key: str) -> Any:
    if payload.get("missing"):
        return "missing"
    return payload.get(key, "n/a")


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _adv_label(threshold: int) -> str:
    return f"adv_ge_{int(threshold / 1_000_000)}m"


def _expand_globs(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(dict.fromkeys(paths))


def _source_segment(path: Path) -> str:
    name = path.name.lower()
    if "yahoo_gap" in name:
        return "yahoo_gapfill"
    if "sip_raw" in name or "alpaca" in name:
        return "alpaca_sip"
    return "unknown"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_str(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _row_str_any(row: pd.Series, column: str) -> str:
    value = _row_value_any(row, column)
    if value is None:
        return ""
    return str(value)


def _row_bool(row: pd.Series, column: str) -> bool | None:
    return _coerce_bool(row.get(column))


def _row_bool_any(row: pd.Series, column: str) -> bool | None:
    return _coerce_bool(_row_value_any(row, column))


def _row_value_any(row: pd.Series, column: str) -> Any:
    for candidate in (f"{column}_asset", f"{column}_manifest", column):
        value = row.get(candidate)
        if value is not None and not pd.isna(value):
            return value
    return None


def _is_true(value: Any) -> bool:
    return _coerce_bool(value) is True


def _coerce_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if str(value).strip().lower() in {"true", "t", "yes", "y", "1"}:
        return True
    if str(value).strip().lower() in {"false", "f", "no", "n", "0"}:
        return False
    return str(value).strip().lower() == "true"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 0 utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pit = subparsers.add_parser("pit-universe", help="Build lagged-liquidity universe artifacts.")
    pit.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    pit.add_argument("--daily-glob", action="append", dest="daily_globs")
    pit.add_argument("--output-root", default=str(DEFAULT_PIT_ROOT))
    pit.add_argument("--validation-start", default="2013-08-05")
    pit.add_argument("--validation-end", default="2019-12-31")
    pit.add_argument("--lookback-sessions", type=int, default=20)
    pit.add_argument("--min-observations", type=int, default=15)
    pit.add_argument("--price-floor", type=float, default=10.0)

    qa = subparsers.add_parser("asset-qa", help="Run current top1000 asset-class QA.")
    qa.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    qa.add_argument("--asset-audit-path", default=str(DEFAULT_ASSET_AUDIT))
    qa.add_argument("--output-root", default=str(DEFAULT_ASSET_QA_ROOT))

    vendor = subparsers.add_parser("vendor-bakeoff", help="Write vendor bake-off artifacts.")
    vendor.add_argument("--output-root", default=str(DEFAULT_VENDOR_ROOT))

    closure = subparsers.add_parser("closure", help="Write Phase 0 closure artifacts.")
    closure.add_argument("--output-root", default=str(DEFAULT_CLOSURE_ROOT))
    closure.add_argument("--top1000-coverage-rollup-path", default=str(DEFAULT_TOP1000_COVERAGE_ROLLUP))
    closure.add_argument("--yahoo-gapfill-summary-path", default=str(DEFAULT_YAHOO_GAPFILL_SUMMARY))
    closure.add_argument(
        "--yahoo-reconciliation-summary-path",
        default=str(DEFAULT_YAHOO_RECONCILIATION_SUMMARY),
    )
    closure.add_argument("--pit-rollup-path", default=str(DEFAULT_PIT_ROLLUP))
    closure.add_argument("--asset-qa-rollup-path", default=str(DEFAULT_ASSET_QA_ROLLUP))
    closure.add_argument("--asset-qa-by-symbol-path", default=str(DEFAULT_ASSET_QA_BY_SYMBOL))
    closure.add_argument("--vendor-rollup-path", default=str(DEFAULT_VENDOR_ROLLUP))

    all_parser = subparsers.add_parser(
        "all",
        help="Run PIT universe, asset QA, vendor bake-off, and closure artifacts.",
    )
    all_parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    all_parser.add_argument("--asset-audit-path", default=str(DEFAULT_ASSET_AUDIT))
    all_parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    all_parser.add_argument("--pit-output-root", default=str(DEFAULT_PIT_ROOT))
    all_parser.add_argument("--asset-qa-output-root", default=str(DEFAULT_ASSET_QA_ROOT))
    all_parser.add_argument("--vendor-output-root", default=str(DEFAULT_VENDOR_ROOT))
    all_parser.add_argument("--closure-output-root", default=str(DEFAULT_CLOSURE_ROOT))

    args = parser.parse_args(argv)
    if args.command == "pit-universe":
        result = build_pit_universe_feasibility(
            manifest_path=args.manifest_path,
            daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
            output_root=args.output_root,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            lookback_sessions=args.lookback_sessions,
            min_observations=args.min_observations,
            price_floor=args.price_floor,
        )
    elif args.command == "asset-qa":
        result = run_asset_class_qa(
            manifest_path=args.manifest_path,
            asset_audit_path=args.asset_audit_path,
            output_root=args.output_root,
        )
    elif args.command == "vendor-bakeoff":
        result = write_vendor_bakeoff_artifacts(output_root=args.output_root)
    elif args.command == "closure":
        result = write_phase0_closure_artifacts(
            output_root=args.output_root,
            top1000_coverage_rollup_path=args.top1000_coverage_rollup_path,
            yahoo_gapfill_summary_path=args.yahoo_gapfill_summary_path,
            yahoo_reconciliation_summary_path=args.yahoo_reconciliation_summary_path,
            pit_rollup_path=args.pit_rollup_path,
            asset_qa_rollup_path=args.asset_qa_rollup_path,
            asset_qa_by_symbol_path=args.asset_qa_by_symbol_path,
            vendor_rollup_path=args.vendor_rollup_path,
        )
    else:
        pit_output_root = Path(args.pit_output_root)
        asset_qa_output_root = Path(args.asset_qa_output_root)
        vendor_output_root = Path(args.vendor_output_root)
        result = {
            "pit_universe": build_pit_universe_feasibility(
                manifest_path=args.manifest_path,
                daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
                output_root=pit_output_root,
            ),
            "asset_qa": run_asset_class_qa(
                manifest_path=args.manifest_path,
                asset_audit_path=args.asset_audit_path,
                output_root=asset_qa_output_root,
            ),
            "vendor_bakeoff": write_vendor_bakeoff_artifacts(output_root=vendor_output_root),
        }
        result["closure"] = write_phase0_closure_artifacts(
            output_root=args.closure_output_root,
            pit_rollup_path=pit_output_root / "phase0_provisional_pit_universe_feasibility_rollup.json",
            asset_qa_rollup_path=asset_qa_output_root / "top1000_asset_class_qa_rollup.json",
            asset_qa_by_symbol_path=asset_qa_output_root / "top1000_asset_class_qa_by_symbol.csv",
            vendor_rollup_path=vendor_output_root / "vendor_bakeoff_rollup.json",
        )

    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
