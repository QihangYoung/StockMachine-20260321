from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase1 import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PHASE1_ROOT,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4f import DEFAULT_PHASE3_SIGNAL_PANEL


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4l_top2000_feasibility_audit_20260418"
DEFAULT_PHASE1_SUMMARY = DEFAULT_PHASE1_ROOT / "phase1_candidate_universe_summary_validation.csv"
DEFAULT_PHASE4K_SUMMARY = (
    RESEARCH_ROOT
    / "phase4k_asymmetric_universe_constructor_20260418"
    / "phase4k_asymmetric_summary_validation.csv"
)
EXPANSION_VARIANTS = (
    "top1500_clean_core_beta_full",
    "top2000_clean_core_beta_full",
    "top3000_clean_core_beta_full",
)
PANEL_USECOLS = (
    "session_date",
    "variant",
    "symbol",
    "lagged_close",
    "trailing_median_dollar_volume_20",
    "liquidity_rank",
    "beta",
    "forward_return_5d",
)


def build_phase4l_top2000_feasibility_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    phase1_summary_path: str | Path = DEFAULT_PHASE1_SUMMARY,
    phase4k_summary_path: str | Path = DEFAULT_PHASE4K_SUMMARY,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    expansion_variants: Sequence[str] = EXPANSION_VARIANTS,
) -> dict[str, Any]:
    """Audit whether top2000-style expansion can be safely researched today."""

    if not expansion_variants:
        raise ValueError("At least one expansion variant is required.")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel(signal_panel_path)
    phase1_summary = pd.read_csv(phase1_summary_path)
    available = _available_variant_summary(panel)
    gate = _gate_table(
        expansion_variants=expansion_variants,
        phase1_summary=phase1_summary,
        available=available,
    )
    gaps = _data_gap_manifest(gate)
    proxy = _proxy_table(phase4k_summary_path)

    available_path = output_dir / "phase4l_available_variant_summary_validation.csv"
    gate_path = output_dir / "phase4l_top2000_gate_validation.csv"
    gaps_path = output_dir / "phase4l_data_gap_manifest_validation.csv"
    proxy_path = output_dir / "phase4l_proxy_universe_evidence_validation.csv"
    memo_path = output_dir / "phase4l_top2000_feasibility_memo.md"
    rollup_path = output_dir / "phase4l_top2000_feasibility_rollup.json"

    available.to_csv(available_path, index=False)
    gate.to_csv(gate_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    proxy.to_csv(proxy_path, index=False)
    memo_path.write_text(_memo(gate, gaps, proxy), encoding="utf-8")
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "phase1_summary_path": Path(phase1_summary_path).as_posix(),
        "phase4k_summary_path": Path(phase4k_summary_path).as_posix(),
        "expansion_variants": list(expansion_variants),
        "available_variant_rows": int(len(available)),
        "gate_rows": int(len(gate)),
        "data_gap_rows": int(len(gaps)),
        "proxy_rows": int(len(proxy)),
        "top2000_status": _variant_status(gate, "top2000_clean_core_beta_full"),
        "available_variant_artifact": available_path.as_posix(),
        "top2000_gate_artifact": gate_path.as_posix(),
        "data_gap_artifact": gaps_path.as_posix(),
        "proxy_evidence_artifact": proxy_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "top2000_feasibility_audit_no_performance_test",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_panel(signal_panel_path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(signal_panel_path, usecols=list(PANEL_USECOLS), low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    for column in PANEL_USECOLS:
        if column not in ("session_date", "variant", "symbol"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel


def _available_variant_summary(panel: pd.DataFrame) -> pd.DataFrame:
    daily = (
        panel.groupby(["variant", "session_date"], sort=True)
        .agg(
            members=("symbol", "nunique"),
            median_adv=("trailing_median_dollar_volume_20", "median"),
            median_price=("lagged_close", "median"),
            median_beta=("beta", "median"),
        )
        .reset_index()
    )
    rows = []
    for variant, group in daily.groupby("variant", sort=True):
        rows.append(
            {
                "variant": variant,
                "sessions": int(group["session_date"].nunique()),
                "first_session": str(group["session_date"].min()),
                "last_session": str(group["session_date"].max()),
                "median_members": float(group["members"].median()),
                "min_members": int(group["members"].min()),
                "p10_members": float(group["members"].quantile(0.10)),
                "median_adv": float(group["median_adv"].median()),
                "median_price": float(group["median_price"].median()),
                "median_beta": float(group["median_beta"].median()),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _gate_table(
    *,
    expansion_variants: Sequence[str],
    phase1_summary: pd.DataFrame,
    available: pd.DataFrame,
) -> pd.DataFrame:
    phase1_by_variant = phase1_summary.set_index("variant", drop=False)
    available_variants = set(available["variant"].astype(str))
    rows = []
    for variant in expansion_variants:
        phase1_row = phase1_by_variant.loc[variant] if variant in phase1_by_variant.index else None
        available_now = variant in available_variants
        data_status = "" if phase1_row is None else str(phase1_row["data_status"])
        blocked = (not available_now) or data_status.startswith("blocked")
        rows.append(
            {
                "variant": variant,
                "audit_status": "blocked" if blocked else "research_ready",
                "available_in_phase1": phase1_row is not None,
                "available_in_phase3_panel": available_now,
                "phase1_data_status": data_status or "missing_from_phase1",
                "phase1_median_members": _phase1_number(phase1_row, "median_members"),
                "phase1_sessions_with_members": _phase1_number(phase1_row, "sessions_with_members"),
                "primary_blocker": _blocker(data_status, available_now),
                "allowed_next_action": (
                    "build timestamp-safe full-market membership first"
                    if blocked
                    else "eligible for validation-only diagnostics"
                ),
                "recommended_proxy_before_unblock": "top1000 long + adv30m short from Phase4K",
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _data_gap_manifest(gate: pd.DataFrame) -> pd.DataFrame:
    top2000_blocked = bool(
        gate.loc[gate["variant"] == "top2000_clean_core_beta_full", "audit_status"]
        .eq("blocked")
        .any()
    )
    rows = [
        (
            "timestamp_safe_full_market_membership",
            "blocked" if top2000_blocked else "ready",
            "Need daily historical rank membership beyond the current top1000 bootstrap symbols.",
        ),
        (
            "full_market_price_volume_history",
            "blocked" if top2000_blocked else "ready",
            "Current validation signal panel has no top2000/top3000 rows to score.",
        ),
        (
            "point_in_time_asset_class_filter",
            "partial",
            "Current asset QA is useful plumbing but still based on bootstrap scope/current metadata.",
        ),
        (
            "point_in_time_shortability_borrow",
            "partial",
            "Current shortability/easy-to-borrow fields are proxies, not historical borrow availability.",
        ),
        (
            "sector_industry_style_fields",
            "missing",
            "Needed for stronger multi-factor or sector-neutral residual targets.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "requirement": requirement,
                "status": status,
                "note": note,
                "test_window_used": False,
            }
            for requirement, status, note in rows
        ]
    )


def _proxy_table(phase4k_summary_path: str | Path) -> pd.DataFrame:
    path = Path(phase4k_summary_path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "proxy_rank",
                "pair",
                "mean_spread_cs_demeaned_residual",
                "mean_short_cs_demeaned_residual_contribution",
                "mean_abs_net_beta",
                "test_window_used",
            ]
        )
    summary = pd.read_csv(path)
    keep = summary.sort_values(
        ["mean_spread_cs_demeaned_residual", "mean_spread_return"],
        ascending=[False, False],
    ).head(8)
    out = keep[
        [
            "pair",
            "mean_spread_cs_demeaned_residual",
            "mean_short_cs_demeaned_residual_contribution",
            "mean_abs_net_beta",
            "test_window_used",
        ]
    ].copy()
    out.insert(0, "proxy_rank", np.arange(1, len(out) + 1))
    return out


def _memo(gate: pd.DataFrame, gaps: pd.DataFrame, proxy: pd.DataFrame) -> str:
    lines = [
        "# Pure Alpha Phase 4L Top2000 Feasibility Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Decision",
        "",
        "Do not run top2000 alpha or portfolio performance yet. The current panel has no timestamp-safe top2000 membership rows, so using top2000 now would be a universe/data experiment, not a fair alpha test.",
        "",
        "## Expansion Gate",
        "",
        "| Variant | Status | Phase1 Status | Blocker | Next Action |",
        "|---|---|---|---|---|",
    ]
    for _, row in gate.iterrows():
        lines.append(
            f"| {row['variant']} | {row['audit_status']} | {row['phase1_data_status']} | "
            f"{row['primary_blocker']} | {row['allowed_next_action']} |"
        )
    lines.extend(["", "## Data Gaps", "", "| Requirement | Status | Note |", "|---|---|---|"])
    for _, row in gaps.iterrows():
        lines.append(f"| {row['requirement']} | {row['status']} | {row['note']} |")
    lines.extend(["", "## Current Proxy Evidence", ""])
    if proxy.empty:
        lines.append("No Phase4K proxy summary was available.")
    else:
        lines.extend(
            [
                "| Rank | Pair | CS Resid Spread | Short CS Contrib | Abs Net Beta |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for _, row in proxy.iterrows():
            lines.append(
                f"| {row['proxy_rank']} | {row['pair']} | "
                f"{row['mean_spread_cs_demeaned_residual']} | "
                f"{row['mean_short_cs_demeaned_residual_contribution']} | "
                f"{row['mean_abs_net_beta']} |"
            )
    lines.extend(
        [
            "",
            "## Practical Implication",
            "",
            "The reasonable near-term path is to keep top1000 as the long book and use ADV proxy variants, especially adv30m, as the short-side expansion sandbox until a real top2000 history is built.",
        ]
    )
    return "\n".join(lines) + "\n"


def _blocker(data_status: str, available_now: bool) -> str:
    if not available_now:
        return "missing_from_phase3_signal_panel"
    if data_status.startswith("blocked"):
        return data_status
    return "none"


def _phase1_number(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return 0.0
    value = pd.to_numeric(row[column], errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def _variant_status(gate: pd.DataFrame, variant: str) -> str:
    match = gate.loc[gate["variant"] == variant, "audit_status"]
    return "missing" if match.empty else str(match.iloc[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 4L top2000 audit.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--phase1-summary-path", default=str(DEFAULT_PHASE1_SUMMARY))
    parser.add_argument("--phase4k-summary-path", default=str(DEFAULT_PHASE4K_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--expansion-variant", action="append", dest="expansion_variants")
    args = parser.parse_args(argv)
    result = build_phase4l_top2000_feasibility_artifacts(
        signal_panel_path=args.signal_panel_path,
        phase1_summary_path=args.phase1_summary_path,
        phase4k_summary_path=args.phase4k_summary_path,
        output_root=args.output_root,
        expansion_variants=tuple(args.expansion_variants or EXPANSION_VARIANTS),
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
