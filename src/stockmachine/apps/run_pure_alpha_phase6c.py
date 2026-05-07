from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table
from stockmachine.apps.run_pure_alpha_phase6 import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PORTFOLIO,
    DEFAULT_STRICT_CURVE_PATH,
    _daily_path_metrics,
    _load_strict_curve,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase6c_beta_drift_mitigation_20260507"
DEFAULT_HEDGE_COST_BPS_PER_TRADED_NOTIONAL = 1.0


@dataclass(frozen=True, slots=True)
class HedgeConfig:
    label: str
    beta_window: int
    hedge_fraction: float
    cap_abs_hedge: float
    activation_threshold: float = 0.0
    blend_window: int | None = None
    blend_weight: float = 0.5


HEDGE_CONFIGS: tuple[HedgeConfig, ...] = (
    HedgeConfig("lag60_full_cap030", beta_window=60, hedge_fraction=1.0, cap_abs_hedge=0.30),
    HedgeConfig("lag60_half_cap020", beta_window=60, hedge_fraction=0.5, cap_abs_hedge=0.20),
    HedgeConfig("lag60_gate015_full_cap030", beta_window=60, hedge_fraction=1.0, cap_abs_hedge=0.30, activation_threshold=0.15),
    HedgeConfig("lag60_gate020_full_cap030", beta_window=60, hedge_fraction=1.0, cap_abs_hedge=0.30, activation_threshold=0.20),
    HedgeConfig("lag126_full_cap020", beta_window=126, hedge_fraction=1.0, cap_abs_hedge=0.20),
    HedgeConfig("lag_blend60_126_full_cap025", beta_window=60, blend_window=126, hedge_fraction=1.0, cap_abs_hedge=0.25),
)


def build_phase6c_beta_drift_mitigation_artifacts(
    *,
    strict_curve_path: str | Path = DEFAULT_STRICT_CURVE_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    portfolio: str = DEFAULT_PORTFOLIO,
    model_name: str = DEFAULT_MODEL_NAME,
    hedge_cost_bps: float = DEFAULT_HEDGE_COST_BPS_PER_TRADED_NOTIONAL,
    configs: Sequence[HedgeConfig] = HEDGE_CONFIGS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_strict_curve(strict_curve_path, portfolio=portfolio)
    records = _base_records(curve)
    variant_curves = _build_hedged_variants(records=records, configs=configs, hedge_cost_bps=hedge_cost_bps)
    metrics = _build_variant_metrics(variant_curves)
    window_table = _build_window_table(variant_curves)
    chosen = _choose_reference_variant(metrics)

    curves_path = output_dir / "phase6c_beta_hedge_curves.csv"
    metrics_path = output_dir / "phase6c_beta_hedge_metrics.csv"
    windows_path = output_dir / "phase6c_beta_hedge_windows.csv"
    plot_path = output_dir / "phase6c_beta_drift_mitigation_plot.png"
    memo_path = output_dir / "phase6c_beta_drift_mitigation_memo.md"
    rollup_path = output_dir / "phase6c_rollup.json"

    variant_curves.to_csv(curves_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    window_table.to_csv(windows_path, index=False)
    _plot_variants(variant_curves, metrics=metrics, chosen_variant=chosen, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            window_table=window_table,
            chosen_variant=chosen,
            hedge_cost_bps=hedge_cost_bps,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "portfolio": portfolio,
        "model_name": model_name,
        "strict_curve_path": Path(strict_curve_path).as_posix(),
        "hedge_cost_bps_per_traded_notional": float(hedge_cost_bps),
        "chosen_reference_variant": chosen,
        "curves_artifact": curves_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "windows_artifact": windows_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "lagged_realized_beta_spy_hedge_overlay_diagnostic",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is a diagnostic overlay on SOTA daily returns, not a changed stock optimizer.",
            "Hedge beta estimates are lagged by one session to avoid lookahead.",
            "SPY hedge costs are modeled as a simple bps-per-traded-notional haircut.",
            "Borrow, margin, and stock transaction costs are not recharged here.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _base_records(curve: pd.DataFrame) -> pd.DataFrame:
    frame = curve.sort_values("return_date").copy()
    frame["return_date"] = pd.to_datetime(frame["return_date"])
    out = pd.DataFrame(
        {
            "return_date": frame["return_date"],
            "strategy_return": frame["gross_return"].astype(float),
            "long_return": frame["long_gross_return"].astype(float),
            "short_return": frame["short_gross_return"].astype(float),
            "spy_return": frame["benchmark_oto_return"].astype(float),
        }
    )
    out["baseline"] = out["strategy_return"]
    return out.reset_index(drop=True)


def _build_hedged_variants(
    *,
    records: pd.DataFrame,
    configs: Sequence[HedgeConfig],
    hedge_cost_bps: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = records[["return_date", "strategy_return", "spy_return", "long_return", "short_return"]].copy()
    baseline["variant"] = "baseline_no_hedge"
    baseline["beta_estimate"] = 0.0
    baseline["hedge_weight"] = 0.0
    baseline["hedge_turnover"] = 0.0
    baseline["hedge_cost_return"] = 0.0
    baseline["net_return"] = baseline["strategy_return"]
    rows.append(baseline)

    beta_cache = {
        window: _lagged_rolling_beta(records["strategy_return"], records["spy_return"], window)
        for window in sorted({cfg.beta_window for cfg in configs} | {cfg.blend_window for cfg in configs if cfg.blend_window})
    }
    for cfg in configs:
        beta = beta_cache[cfg.beta_window].copy()
        if cfg.blend_window is not None:
            beta = cfg.blend_weight * beta + (1.0 - cfg.blend_weight) * beta_cache[cfg.blend_window]
        beta = beta.fillna(0.0)
        hedge_weight = -cfg.hedge_fraction * beta
        if cfg.activation_threshold > 0:
            hedge_weight = hedge_weight.where(beta.abs() >= cfg.activation_threshold, 0.0)
        hedge_weight = hedge_weight.clip(lower=-cfg.cap_abs_hedge, upper=cfg.cap_abs_hedge)
        hedge_turnover = hedge_weight.diff().abs().fillna(hedge_weight.abs())
        hedge_cost_return = hedge_turnover * float(hedge_cost_bps) / 10000.0

        out = records[["return_date", "strategy_return", "spy_return", "long_return", "short_return"]].copy()
        out["variant"] = cfg.label
        out["beta_estimate"] = beta
        out["hedge_weight"] = hedge_weight
        out["hedge_turnover"] = hedge_turnover
        out["hedge_cost_return"] = hedge_cost_return
        out["net_return"] = out["strategy_return"] + out["hedge_weight"] * out["spy_return"] - out["hedge_cost_return"]
        rows.append(out)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["variant", "return_date"]).reset_index(drop=True)
    result["equity"] = result.groupby("variant", group_keys=False)["net_return"].apply(lambda s: (1.0 + s).cumprod())
    result["drawdown"] = result.groupby("variant", group_keys=False)["equity"].apply(lambda s: s / s.cummax() - 1.0)
    beta_parts: list[pd.Series] = []
    for _, group in result.groupby("variant", sort=False):
        beta = _variant_rolling60_beta(group)
        beta.index = group.index
        beta_parts.append(beta)
    result["rolling60_beta"] = pd.concat(beta_parts).sort_index()
    return result


def _lagged_rolling_beta(strategy: pd.Series, spy: pd.Series, window: int) -> pd.Series:
    cov = strategy.astype(float).rolling(window).cov(spy.astype(float))
    var = spy.astype(float).rolling(window).var()
    return (cov / var).shift(1)


def _variant_rolling60_beta(group: pd.DataFrame) -> pd.Series:
    cov = group["net_return"].astype(float).rolling(60).cov(group["spy_return"].astype(float))
    var = group["spy_return"].astype(float).rolling(60).var()
    return cov / var


def _build_variant_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in curves.groupby("variant", sort=True):
        group = group.sort_values("return_date")
        metrics = _daily_path_metrics(group["net_return"], benchmark=group["spy_return"])
        rolling60 = group["rolling60_beta"].dropna()
        rows.append(
            {
                "variant": variant,
                "start": str(group["return_date"].min().date()),
                "end": str(group["return_date"].max().date()),
                "days": int(len(group)),
                **metrics,
                "mean_daily_bps": float(group["net_return"].mean() * 10000.0),
                "mean_abs_hedge_weight": float(group["hedge_weight"].abs().mean()),
                "p90_abs_hedge_weight": float(group["hedge_weight"].abs().quantile(0.90)),
                "max_abs_hedge_weight": float(group["hedge_weight"].abs().max()),
                "mean_hedge_turnover": float(group["hedge_turnover"].mean()),
                "mean_hedge_cost_bps": float(group["hedge_cost_return"].mean() * 10000.0),
                "mean_abs_rolling60_beta": float(rolling60.abs().mean()) if not rolling60.empty else np.nan,
                "p90_abs_rolling60_beta": float(rolling60.abs().quantile(0.90)) if not rolling60.empty else np.nan,
                "max_abs_rolling60_beta": float(rolling60.abs().max()) if not rolling60.empty else np.nan,
                "rolling60_beta_breach_020_rate": float((rolling60.abs() >= 0.20).mean()) if not rolling60.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["realized_beta_to_spy", "annualized_return"], key=_sort_key).reset_index(drop=True)


def _sort_key(series: pd.Series) -> pd.Series:
    if series.name == "realized_beta_to_spy":
        return series.abs()
    return -series


WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_drawdown", "2015-06-04", "2015-08-21"),
    ("2016_beta_cluster", "2016-07-06", "2016-09-20"),
    ("2018_q4_stress", "2018-09-28", "2018-10-10"),
    ("2019_jun_jul_beta_cluster", "2019-06-05", "2019-07-15"),
    ("2019_aug_stress", "2019-08-01", "2019-08-31"),
)


def _build_window_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in curves.groupby("variant", sort=True):
        group = group.sort_values("return_date")
        for window, start, end in WINDOWS:
            sub = group[(group["return_date"] >= pd.Timestamp(start)) & (group["return_date"] <= pd.Timestamp(end))]
            if sub.empty:
                continue
            metrics = _daily_path_metrics(sub["net_return"], benchmark=sub["spy_return"])
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    "start": start,
                    "end": end,
                    "days": int(len(sub)),
                    "compound_return": metrics["total_return"],
                    "spy_compound_return": float((1.0 + sub["spy_return"].astype(float)).prod() - 1.0),
                    "realized_beta_to_spy": metrics["realized_beta_to_spy"],
                    "corr_to_spy": metrics["corr_to_spy"],
                    "mean_abs_hedge_weight": float(sub["hedge_weight"].abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def _choose_reference_variant(metrics: pd.DataFrame) -> str:
    candidates = metrics[metrics["variant"].ne("baseline_no_hedge")].copy()
    if candidates.empty:
        return "baseline_no_hedge"
    candidates["beta_penalty"] = candidates["realized_beta_to_spy"].abs()
    candidates["alpha_tax"] = float(
        metrics.loc[metrics["variant"].eq("baseline_no_hedge"), "annualized_return"].iloc[0]
    ) - candidates["annualized_return"].astype(float)
    eligible = candidates[
        (candidates["beta_penalty"] <= 0.05)
        & (candidates["alpha_tax"] <= 0.02)
        & (candidates["p90_abs_rolling60_beta"] <= 0.20)
    ].copy()
    if eligible.empty:
        eligible = candidates.copy()
    return str(
        eligible.sort_values(
            ["alpha_tax", "beta_penalty", "p90_abs_rolling60_beta"],
            ascending=[True, True, True],
        ).iloc[0]["variant"]
    )


def _plot_variants(
    curves: pd.DataFrame,
    *,
    metrics: pd.DataFrame,
    chosen_variant: str,
    output_path: Path,
) -> None:
    baseline = curves[curves["variant"].eq("baseline_no_hedge")].copy()
    chosen = curves[curves["variant"].eq(chosen_variant)].copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(baseline["return_date"], baseline["equity"], label="baseline", color="#0f766e", linewidth=2.0)
    if chosen_variant != "baseline_no_hedge":
        axes[0].plot(chosen["return_date"], chosen["equity"], label=chosen_variant, color="#2563eb", linewidth=1.8)
    spy = (1.0 + baseline["spy_return"].astype(float)).cumprod()
    axes[0].plot(baseline["return_date"], spy, label="SPY raw", color="#6b7280", alpha=0.8)
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Growth")
    axes[1].plot(baseline["return_date"], baseline["drawdown"] * 100.0, label="baseline", color="#0f766e")
    if chosen_variant != "baseline_no_hedge":
        axes[1].plot(chosen["return_date"], chosen["drawdown"] * 100.0, label=chosen_variant, color="#2563eb")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].legend(loc="lower left")
    axes[1].set_ylabel("Drawdown %")
    axes[2].plot(baseline["return_date"], baseline["rolling60_beta"], label="baseline 60d beta", color="#dc2626")
    if chosen_variant != "baseline_no_hedge":
        axes[2].plot(chosen["return_date"], chosen["rolling60_beta"], label=f"{chosen_variant} 60d beta", color="#2563eb")
    for y in (0.0, 0.05, -0.05, 0.20, -0.20):
        axes[2].axhline(y, color="gray" if y else "black", linestyle=":" if y else "--", linewidth=0.8)
    axes[2].legend(loc="upper left")
    axes[2].set_ylabel("60d beta")
    axes[2].set_xlabel("Date")
    fig.suptitle("Phase6C Beta Drift Mitigation Overlay (Validation Only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    window_table: pd.DataFrame,
    chosen_variant: str,
    hedge_cost_bps: float,
    plot_path: Path,
) -> str:
    display = metrics.copy()
    for column in ("annualized_return", "annualized_vol", "max_drawdown", "corr_to_spy"):
        display[column] = display[column].map(_fmt_pct_like if column != "corr_to_spy" else _fmt_float_like)
    for column in (
        "sharpe_no_rf",
        "realized_beta_to_spy",
        "mean_daily_bps",
        "mean_abs_hedge_weight",
        "p90_abs_hedge_weight",
        "max_abs_hedge_weight",
        "mean_hedge_turnover",
        "mean_hedge_cost_bps",
        "mean_abs_rolling60_beta",
        "p90_abs_rolling60_beta",
        "max_abs_rolling60_beta",
        "rolling60_beta_breach_020_rate",
    ):
        display[column] = display[column].map(_fmt_float_like)
    display = display[
        [
            "variant",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "max_drawdown",
            "corr_to_spy",
            "realized_beta_to_spy",
            "p90_abs_rolling60_beta",
            "mean_abs_hedge_weight",
            "mean_hedge_turnover",
            "mean_hedge_cost_bps",
        ]
    ]

    windows = window_table[window_table["variant"].isin(["baseline_no_hedge", chosen_variant])].copy()
    for column in ("compound_return", "spy_compound_return"):
        windows[column] = windows[column].map(_fmt_pct_like)
    for column in ("realized_beta_to_spy", "corr_to_spy", "mean_abs_hedge_weight"):
        windows[column] = windows[column].map(_fmt_float_like)

    chosen = metrics[metrics["variant"].eq(chosen_variant)].iloc[0]
    baseline = metrics[metrics["variant"].eq("baseline_no_hedge")].iloc[0]
    alpha_tax = float(baseline["annualized_return"] - chosen["annualized_return"])
    beta_reduction = abs(float(baseline["realized_beta_to_spy"])) - abs(float(chosen["realized_beta_to_spy"]))

    lines = [
        "# Phase6C Beta Drift Mitigation Study",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "- Candidate: current SOTA / Phase5F `short_overlay_only`.",
        "- Test: lagged realized-beta SPY hedge overlay.",
        "- No test-window performance is used.",
        f"- Hedge transaction cost assumption: `{hedge_cost_bps:.2f}` bps per traded SPY notional.",
        "",
        "## Variant Metrics",
        "",
        _text_table(display),
        "",
        "## Reference Variant",
        "",
        f"- Chosen diagnostic reference: `{chosen_variant}`.",
        f"- Annualized alpha tax versus baseline: `{alpha_tax * 100.0:.2f}%`.",
        f"- Absolute realized beta reduction versus baseline: `{beta_reduction:.4f}`.",
        "",
        "## Stress Windows",
        "",
        _text_table(windows),
        "",
        "## Plot",
        "",
        f"![Phase6C beta drift mitigation]({plot_path.as_posix()})",
        "",
        "## First Reading",
        "",
        "- A lagged SPY hedge can reduce realized beta, but the correct product lens is not maximum beta reduction; it is minimum alpha tax while clearing the beta gate.",
        "- The gated lag60 variants are therefore more interesting than always-on full hedging because they preserve most of the SOTA economics.",
        "- If we adopt an overlay, it should be frozen as a transparent risk-control rule with its own turnover/cost governance, not as another alpha selector.",
    ]
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test lagged realized-beta SPY hedge overlays for pure-alpha SOTA.")
    parser.add_argument("--strict-curve-path", default=str(DEFAULT_STRICT_CURVE_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--hedge-cost-bps", type=float, default=DEFAULT_HEDGE_COST_BPS_PER_TRADED_NOTIONAL)
    args = parser.parse_args(argv)
    result = build_phase6c_beta_drift_mitigation_artifacts(
        strict_curve_path=args.strict_curve_path,
        output_root=args.output_root,
        portfolio=args.portfolio,
        model_name=args.model_name,
        hedge_cost_bps=args.hedge_cost_bps,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
