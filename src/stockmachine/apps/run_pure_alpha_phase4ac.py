from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4ac_hybrid_vs_lead_path_compare_20260419"
DEFAULT_LEAD_CURVE = (
    RESEARCH_ROOT
    / "phase4z_strict_h10_daily_paths_20260419"
    / "phase4z_strict_h10_daily_curve.csv"
)
DEFAULT_HYBRID_POSITIONS = (
    RESEARCH_ROOT
    / "phase4ab_hybrid_short_veto_20260419"
    / "phase4ab_beta_matched_positions_validation.csv.gz"
)
DEFAULT_HYBRID_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "hybrid_strict_h10_rebuild"
DEFAULT_LEAD_PORTFOLIO = "sic2_soft_neutral"
DEFAULT_HYBRID_PORTFOLIO = "sic2_soft_neutral__short_hybrid_soft_fw_overlay"
DEFAULT_HOLDING_PERIOD_SESSIONS = 10


def build_phase4ac_hybrid_vs_lead_path_compare(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    lead_curve_path: str | Path = DEFAULT_LEAD_CURVE,
    hybrid_positions_path: str | Path = DEFAULT_HYBRID_POSITIONS,
    hybrid_strict_root: str | Path = DEFAULT_HYBRID_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    lead_portfolio: str = DEFAULT_LEAD_PORTFOLIO,
    hybrid_portfolio: str = DEFAULT_HYBRID_PORTFOLIO,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
    """Compare strict h10 daily paths for the old lead, hybrid selector, and scaled SPY."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    hybrid_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=hybrid_positions_path,
        output_root=hybrid_strict_root,
        phase3_rollup_path=phase3_rollup_path,
        existing_curve_path=None,
        holding_period_sessions=holding_period_sessions,
        lead_portfolio=hybrid_portfolio,
    )

    lead_curve = _load_curve(lead_curve_path, lead_portfolio)
    hybrid_curve = _load_curve(
        Path(hybrid_strict_root) / "phase4z_strict_h10_daily_curve.csv",
        hybrid_portfolio,
    )
    comparison = _comparison_curve(
        lead_curve,
        hybrid_curve,
        lead_portfolio=lead_portfolio,
        hybrid_portfolio=hybrid_portfolio,
    )
    metrics = _metrics_table(comparison)
    annual = _annual_table(comparison)

    curve_path = output_dir / "phase4ac_hybrid_vs_lead_curve.csv"
    metrics_path = output_dir / "phase4ac_hybrid_vs_lead_metrics.csv"
    annual_path = output_dir / "phase4ac_hybrid_vs_lead_annual_returns.csv"
    plot_path = output_dir / "phase4ac_hybrid_vs_lead_spy_scaled.png"
    memo_path = output_dir / "phase4ac_hybrid_vs_lead_path_compare_memo.md"
    rollup_path = output_dir / "phase4ac_rollup.json"

    comparison.to_csv(curve_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    annual.to_csv(annual_path, index=False)
    _plot_comparison(comparison, output_path=plot_path)
    memo_path.write_text(_memo(metrics, annual, plot_path=plot_path), encoding="utf-8")

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "lead_curve_path": Path(lead_curve_path).as_posix(),
        "hybrid_positions_path": Path(hybrid_positions_path).as_posix(),
        "hybrid_strict_root": Path(hybrid_strict_root).as_posix(),
        "phase3_rollup_path": Path(phase3_rollup_path).as_posix(),
        "lead_portfolio": lead_portfolio,
        "hybrid_portfolio": hybrid_portfolio,
        "holding_period_sessions": int(holding_period_sessions),
        "hybrid_strict_rollup": hybrid_rollup,
        "comparison_rows": int(len(comparison)),
        "metrics_rows": int(len(metrics)),
        "annual_rows": int(len(annual)),
        "curve_artifact": curve_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "annual_artifact": annual_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "strict_h10_hybrid_vs_previous_lead_and_endpoint_scaled_spy",
        "spy_scaling": "SPY open-to-open log returns are endpoint-scaled to the previous lead final equity over the aligned comparison window.",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_curve(path: str | Path, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["return_date"] = pd.to_datetime(frame["return_date"])
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame = frame[frame["portfolio"].eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"Portfolio {portfolio!r} not found in {path}.")
    if frame["test_window_used"].map(_is_true).any():
        raise ValueError(f"Curve {path} includes test-window rows; refusing to compare.")
    needed = ["return_date", "gross_return", "long_gross_return", "short_gross_return", "benchmark_oto_return"]
    return frame[needed].sort_values("return_date").reset_index(drop=True)


def _comparison_curve(
    lead: pd.DataFrame,
    hybrid: pd.DataFrame,
    *,
    lead_portfolio: str,
    hybrid_portfolio: str,
) -> pd.DataFrame:
    merged = lead.merge(
        hybrid,
        on="return_date",
        how="inner",
        suffixes=("_lead", "_hybrid"),
    )
    if merged.empty:
        raise ValueError("Lead and hybrid curves have no overlapping return dates.")

    merged["lead_return"] = merged["gross_return_lead"]
    merged["hybrid_return"] = merged["gross_return_hybrid"]
    merged["lead_long_return"] = merged["long_gross_return_lead"]
    merged["lead_short_return"] = merged["short_gross_return_lead"]
    merged["hybrid_long_return"] = merged["long_gross_return_hybrid"]
    merged["hybrid_short_return"] = merged["short_gross_return_hybrid"]
    merged["spy_return"] = merged["benchmark_oto_return_lead"].fillna(
        merged["benchmark_oto_return_hybrid"]
    )

    lead_equity = _equity(merged["lead_return"])
    hybrid_equity = _equity(merged["hybrid_return"])
    spy_scaled_equity = _endpoint_scaled_equity(
        merged["spy_return"],
        target_final_equity=float(lead_equity.iloc[-1]),
    )

    out = pd.DataFrame(
        {
            "return_date": merged["return_date"],
            "lead_portfolio": lead_portfolio,
            "hybrid_portfolio": hybrid_portfolio,
            "lead_return": merged["lead_return"],
            "hybrid_return": merged["hybrid_return"],
            "spy_return": merged["spy_return"],
            "lead_long_return": merged["lead_long_return"],
            "lead_short_return": merged["lead_short_return"],
            "hybrid_long_return": merged["hybrid_long_return"],
            "hybrid_short_return": merged["hybrid_short_return"],
            "lead_equity": lead_equity,
            "hybrid_equity": hybrid_equity,
            "spy_endpoint_scaled_to_lead_equity": spy_scaled_equity,
        }
    )
    for prefix in ("lead", "hybrid"):
        out[f"{prefix}_drawdown"] = _drawdown(out[f"{prefix}_equity"])
        out[f"{prefix}_rolling_60_return"] = out[f"{prefix}_equity"] / out[
            f"{prefix}_equity"
        ].shift(60) - 1.0
    out["spy_endpoint_scaled_drawdown"] = _drawdown(out["spy_endpoint_scaled_to_lead_equity"])
    out["spy_endpoint_scaled_rolling_60_return"] = (
        out["spy_endpoint_scaled_to_lead_equity"]
        / out["spy_endpoint_scaled_to_lead_equity"].shift(60)
        - 1.0
    )
    out["hybrid_minus_lead_return"] = out["hybrid_return"] - out["lead_return"]
    out["test_window_used"] = False
    return out


def _metrics_table(curve: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _metric_row(
            curve,
            label="previous_lead_sic2_soft_neutral",
            return_column="lead_return",
            equity_column="lead_equity",
            drawdown_column="lead_drawdown",
            rolling_60_column="lead_rolling_60_return",
        ),
        _metric_row(
            curve,
            label="hybrid_soft_fw_overlay",
            return_column="hybrid_return",
            equity_column="hybrid_equity",
            drawdown_column="hybrid_drawdown",
            rolling_60_column="hybrid_rolling_60_return",
        ),
        _metric_row(
            curve,
            label="spy_endpoint_scaled_to_lead",
            return_column=None,
            equity_column="spy_endpoint_scaled_to_lead_equity",
            drawdown_column="spy_endpoint_scaled_drawdown",
            rolling_60_column="spy_endpoint_scaled_rolling_60_return",
        ),
    ]
    metrics = pd.DataFrame(rows)
    lead_return = curve["lead_return"]
    hybrid_return = curve["hybrid_return"]
    spy_return = curve["spy_return"]
    metrics.loc[metrics["series"].eq("previous_lead_sic2_soft_neutral"), "corr_to_spy"] = (
        lead_return.corr(spy_return)
    )
    metrics.loc[metrics["series"].eq("hybrid_soft_fw_overlay"), "corr_to_spy"] = (
        hybrid_return.corr(spy_return)
    )
    metrics.loc[metrics["series"].eq("spy_endpoint_scaled_to_lead"), "corr_to_spy"] = 1.0
    metrics.loc[
        metrics["series"].eq("hybrid_soft_fw_overlay"), "mean_daily_excess_vs_lead_bps"
    ] = float((hybrid_return - lead_return).mean() * 10000.0)
    return metrics


def _metric_row(
    curve: pd.DataFrame,
    *,
    label: str,
    return_column: str | None,
    equity_column: str,
    drawdown_column: str,
    rolling_60_column: str,
) -> dict[str, Any]:
    equity = curve[equity_column].dropna()
    if return_column is None:
        returns = equity.pct_change().dropna()
    else:
        returns = curve[return_column].dropna()
    final_equity = float(equity.iloc[-1])
    annualized_return = float(final_equity ** (252.0 / len(equity)) - 1.0)
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(252.0))
    rolling_60 = curve[rolling_60_column].dropna()
    return {
        "series": label,
        "start": str(pd.to_datetime(curve["return_date"].min()).date()),
        "end": str(pd.to_datetime(curve["return_date"].max()).date()),
        "daily_rows": int(len(equity)),
        "final_equity": final_equity,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe_no_rf": annualized_return / annualized_vol if annualized_vol > 0 else np.nan,
        "max_drawdown": float(curve[drawdown_column].min()),
        "rolling_60_positive_rate": float((rolling_60 > 0).mean()) if len(rolling_60) else np.nan,
        "mean_daily_return_bps": float(returns.mean() * 10000.0),
        "worst_daily_return_bps": float(returns.min() * 10000.0),
        "best_daily_return_bps": float(returns.max() * 10000.0),
        "test_window_used": False,
    }


def _annual_table(curve: pd.DataFrame) -> pd.DataFrame:
    frame = curve.copy()
    frame["year"] = pd.to_datetime(frame["return_date"]).dt.year
    rows = []
    for year, group in frame.groupby("year", sort=True):
        rows.append(
            {
                "year": int(year),
                "days": int(len(group)),
                "lead_return": _compound(group["lead_return"]),
                "hybrid_return": _compound(group["hybrid_return"]),
                "spy_endpoint_scaled_return": _compound(
                    group["spy_endpoint_scaled_to_lead_equity"].pct_change().fillna(
                        group["spy_endpoint_scaled_to_lead_equity"].iloc[0] - 1.0
                    )
                ),
                "hybrid_minus_lead": _compound(group["hybrid_return"])
                - _compound(group["lead_return"]),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _plot_comparison(curve: pd.DataFrame, *, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10), sharex=True)
    dates = pd.to_datetime(curve["return_date"])
    colors = {
        "lead": "#0B6B5E",
        "hybrid": "#1F5EFF",
        "spy": "#111111",
    }
    axes[0].plot(dates, curve["lead_equity"], label="Previous lead", color=colors["lead"], linewidth=2.0)
    axes[0].plot(dates, curve["hybrid_equity"], label="Hybrid soft overlay", color=colors["hybrid"], linewidth=2.0)
    axes[0].plot(
        dates,
        curve["spy_endpoint_scaled_to_lead_equity"],
        label="SPY endpoint-scaled",
        color=colors["spy"],
        linewidth=1.5,
        linestyle="--",
    )
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Phase4AC Strict h10 Gross Path: Hybrid vs Previous Lead vs Scaled SPY")

    axes[1].plot(dates, curve["lead_drawdown"] * 100.0, label="Previous lead", color=colors["lead"], linewidth=1.8)
    axes[1].plot(dates, curve["hybrid_drawdown"] * 100.0, label="Hybrid soft overlay", color=colors["hybrid"], linewidth=1.8)
    axes[1].plot(
        dates,
        curve["spy_endpoint_scaled_drawdown"] * 100.0,
        label="SPY endpoint-scaled",
        color=colors["spy"],
        linewidth=1.2,
        linestyle="--",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Drawdown %")

    axes[2].plot(
        dates,
        curve["lead_rolling_60_return"] * 100.0,
        label="Previous lead",
        color=colors["lead"],
        linewidth=1.6,
    )
    axes[2].plot(
        dates,
        curve["hybrid_rolling_60_return"] * 100.0,
        label="Hybrid soft overlay",
        color=colors["hybrid"],
        linewidth=1.6,
    )
    axes[2].plot(
        dates,
        curve["spy_endpoint_scaled_rolling_60_return"] * 100.0,
        label="SPY endpoint-scaled",
        color=colors["spy"],
        linewidth=1.1,
        linestyle="--",
    )
    axes[2].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Rolling 60-session %")
    axes[2].set_xlabel("Date")

    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _memo(metrics: pd.DataFrame, annual: pd.DataFrame, *, plot_path: Path) -> str:
    lines = [
        "# Phase4AC Hybrid vs Previous Lead Path Compare",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "- Validation-only strict h10 daily path.",
        "- Previous lead: `sic2_soft_neutral` from Phase4Z.",
        "- Hybrid: `sic2_soft_neutral__short_hybrid_soft_fw_overlay` from Phase4AB.",
        "- SPY is endpoint-scaled to the previous lead final equity over the aligned comparison window.",
        f"- Plot: `{plot_path.as_posix()}`.",
        "",
        "## Metrics",
        "",
        _table(_display_metrics(metrics)),
        "",
        "## Calendar-Year Returns",
        "",
        _table(_display_annual(annual)),
    ]
    return "\n".join(lines) + "\n"


def _display_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    for column in ("annualized_return", "annualized_vol", "max_drawdown", "rolling_60_positive_rate"):
        out[column] = out[column].map(_fmt_pct)
    for column in ("final_equity", "sharpe_no_rf", "corr_to_spy"):
        out[column] = out[column].map(_fmt_float)
    for column in ("mean_daily_return_bps", "worst_daily_return_bps", "best_daily_return_bps", "mean_daily_excess_vs_lead_bps"):
        if column in out:
            out[column] = out[column].map(_fmt_bps_from_bps)
    return out


def _display_annual(annual: pd.DataFrame) -> pd.DataFrame:
    out = annual.copy()
    for column in ("lead_return", "hybrid_return", "spy_endpoint_scaled_return", "hybrid_minus_lead"):
        out[column] = out[column].map(_fmt_pct)
    return out


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _endpoint_scaled_equity(returns: pd.Series, *, target_final_equity: float) -> pd.Series:
    log_path = np.log1p(returns.fillna(0.0)).cumsum()
    endpoint = float(log_path.iloc[-1]) if len(log_path) else 0.0
    scale = np.log(target_final_equity) / endpoint if abs(endpoint) > 1e-12 else 0.0
    return np.exp(log_path * scale)


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _compound(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def _fmt_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _fmt_bps_from_bps(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Phase4AB hybrid strict h10 path with previous lead and scaled SPY."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--lead-curve-path", default=str(DEFAULT_LEAD_CURVE))
    parser.add_argument("--hybrid-positions-path", default=str(DEFAULT_HYBRID_POSITIONS))
    parser.add_argument("--hybrid-strict-root", default=str(DEFAULT_HYBRID_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--lead-portfolio", default=DEFAULT_LEAD_PORTFOLIO)
    parser.add_argument("--hybrid-portfolio", default=DEFAULT_HYBRID_PORTFOLIO)
    args = parser.parse_args(argv)

    result = build_phase4ac_hybrid_vs_lead_path_compare(
        output_root=args.output_root,
        lead_curve_path=args.lead_curve_path,
        hybrid_positions_path=args.hybrid_positions_path,
        hybrid_strict_root=args.hybrid_strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        lead_portfolio=args.lead_portfolio,
        hybrid_portfolio=args.hybrid_portfolio,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
