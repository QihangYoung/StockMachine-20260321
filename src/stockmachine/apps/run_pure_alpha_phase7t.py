"""Phase7T universe underwriting for the locked-turnover shared-core portfolio.

This phase treats universe choice as a structural prior, not a free hyperparameter.
It evaluates a short, pre-declared set of explainable universe candidates using the
same Phase7K strict daily path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase7b import DEFAULT_SIZE_PANEL
from stockmachine.apps.run_pure_alpha_phase7i import PortfolioConfig
from stockmachine.apps.run_pure_alpha_phase7k import build_phase7k_locked_turnover_shared_capital


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7t_universe_underwriting_20260518_tb0p15"
DEFAULT_TURNOVER_BUDGET = 0.15
DEFAULT_VARIANTS: tuple[str, ...] = (
    "top500_clean_core_beta_full",
    "top1000_clean_core_beta_full",
    "adv20m_clean_core_beta_full",
    "adv30m_clean_core_beta_full",
    "adv50m_clean_core_beta_full",
)
PHASE4H_SUMMARY = RESEARCH_ROOT / "phase4h_universe_prior_strength_20260418" / "phase4h_universe_prior_summary_validation.csv"
PHASE4K_ASYMMETRIC = RESEARCH_ROOT / "phase4k_asymmetric_universe_constructor_20260418" / "phase4k_asymmetric_summary_validation.csv"


@dataclass(frozen=True)
class UniverseSpec:
    variant: str
    reason: str


UNIVERSE_SPECS: tuple[UniverseSpec, ...] = (
    UniverseSpec("top500_clean_core_beta_full", "large/liquid core; lower capacity breadth but cleaner implementation"),
    UniverseSpec("top1000_clean_core_beta_full", "current mainline scope; broadest supported current top1000-like pool"),
    UniverseSpec("adv10m_clean_core_beta_full", "widest supported ADV-threshold pool; keeps breadth while imposing a minimum liquidity prior"),
    UniverseSpec("adv20m_clean_core_beta_full", "liquidity-threshold expansion; broader than ADV30/50"),
    UniverseSpec("adv30m_clean_core_beta_full", "short-side plausible liquidity floor candidate"),
    UniverseSpec("adv50m_clean_core_beta_full", "most liquid core among ADV candidates; capacity/borrow-friendly prior"),
)


SHARED_CORE = PortfolioConfig(
    "shared_core_lambda_0p005",
    {
        "reversal_score": 0.25,
        "momentum_score": 0.10,
        "small_size_score": 0.30,
        "low_beta_score": 0.20,
        "cash_quality_score": 0.15,
    },
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7T universe underwriting grid.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--turnover-budget", type=float, default=DEFAULT_TURNOVER_BUDGET)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = tuple(item.strip() for item in str(args.variants).split(",") if item.strip())

    rollup = run_phase7t_universe_underwriting(
        output_root=output_root,
        variants=variants,
        turnover_budget=float(args.turnover_budget),
        size_panel_path=Path(args.size_panel_path),
        reuse_existing=bool(args.reuse_existing),
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def run_phase7t_universe_underwriting(
    *,
    output_root: Path,
    variants: Sequence[str],
    turnover_budget: float,
    size_panel_path: Path,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = output_root / variant
        metrics_path = variant_dir / "phase7k_strict_daily_metrics.csv"
        if not (reuse_existing and metrics_path.exists()):
            build_phase7k_locked_turnover_shared_capital(
                output_root=variant_dir,
                variant=variant,
                size_panel_path=size_panel_path,
                turnover_budget_grid=(turnover_budget,),
                portfolios=(SHARED_CORE,),
            )
        run_rows.append(_summarize_variant(variant_dir=variant_dir, variant=variant))

    summary = pd.DataFrame(run_rows).sort_values("sharpe_no_rf", ascending=False).reset_index(drop=True)
    summary.to_csv(output_root / "phase7t_universe_underwriting_summary.csv", index=False)

    scorecard = _score_universes(summary)
    scorecard.to_csv(output_root / "phase7t_universe_underwriting_scorecard.csv", index=False)

    phase4h = _read_existing_subset(PHASE4H_SUMMARY, variants=variants)
    phase4k = _read_asymmetric_prior(PHASE4K_ASYMMETRIC)
    if not phase4h.empty:
        phase4h.to_csv(output_root / "phase7t_prior_universe_summary_from_phase4h.csv", index=False)
    if not phase4k.empty:
        phase4k.to_csv(output_root / "phase7t_asymmetric_prior_from_phase4k.csv", index=False)

    memo = _memo(
        output_root=output_root,
        variants=variants,
        turnover_budget=turnover_budget,
        size_panel_path=size_panel_path,
        summary=summary,
        scorecard=scorecard,
        phase4h=phase4h,
        phase4k=phase4k,
    )
    (output_root / "phase7t_universe_underwriting_memo.md").write_text(memo, encoding="utf-8")

    return {
        "output_root": str(output_root),
        "variants": list(variants),
        "turnover_budget": turnover_budget,
        "size_panel_path": str(size_panel_path),
        "summary_path": str(output_root / "phase7t_universe_underwriting_summary.csv"),
        "scorecard_path": str(output_root / "phase7t_universe_underwriting_scorecard.csv"),
        "memo_path": str(output_root / "phase7t_universe_underwriting_memo.md"),
    }


def _summarize_variant(*, variant_dir: Path, variant: str) -> dict[str, Any]:
    metrics = pd.read_csv(variant_dir / "phase7k_strict_daily_metrics.csv")
    net = metrics[metrics["return_kind"].eq("net")].iloc[0].to_dict()
    curve = pd.read_csv(variant_dir / "phase7k_strict_daily_curve.csv", parse_dates=["return_date"])
    curve = curve[curve["portfolio"].eq(net["portfolio"])].sort_values("return_date").copy()
    diagnostics = pd.read_csv(variant_dir / "phase7k_daily_diagnostics.csv")
    diagnostics = diagnostics[diagnostics["portfolio"].eq(net["portfolio"])].copy()
    score_coverage = pd.read_csv(variant_dir / "phase7k_score_coverage.csv")

    net_returns = curve["net_return"].astype(float)
    benchmark = curve["benchmark_oto_return"].astype(float)
    net_bps = net_returns * 10000.0
    beta, corr, r2 = _lin_beta(net_returns, benchmark)
    rolling = _rolling_market_dependence(net_returns, benchmark)
    roll5 = (1.0 + net_returns).rolling(5).apply(np.prod, raw=True) - 1.0
    roll20 = (1.0 + net_returns).rolling(20).apply(np.prod, raw=True) - 1.0
    bench_q10 = benchmark.quantile(0.10)

    row: dict[str, Any] = {
        "variant": variant,
        "reason": _reason_for_variant(variant),
        "portfolio": net["portfolio"],
        "start": net.get("start"),
        "end": net.get("end"),
        "daily_rows": int(net.get("daily_rows", len(curve))),
        "final_equity": float(net["final_equity"]),
        "annualized_return": float(net["annualized_return"]),
        "annualized_vol": float(net["annualized_vol"]),
        "sharpe_no_rf": float(net["sharpe_no_rf"]),
        "max_drawdown": float(net["max_drawdown"]),
        "mean_daily_return_bps": float(net["mean_daily_return_bps"]),
        "hit_rate_daily": float(net["hit_rate_daily"]),
        "rolling_60_positive_rate": float(net["rolling_60_positive_rate"]),
        "mean_turnover": float(net["mean_turnover"]),
        "mean_cost_bps": float(net["mean_cost_bps"]),
        "mean_positions": float(net["mean_positions"]),
        "realized_beta_net": float(net["realized_beta_net"]),
        "corr_to_spy_net": float(net["corr_to_spy_net"]),
        "recomputed_beta": beta,
        "recomputed_corr": corr,
        "recomputed_r2": r2,
        "rolling_beta_60_p90": rolling.get("beta_60_p90"),
        "rolling_beta_60_max": rolling.get("beta_60_max"),
        "rolling_beta_252_p90": rolling.get("beta_252_p90"),
        "rolling_beta_252_max": rolling.get("beta_252_max"),
        "worst_1d_bps": float(net_bps.min()),
        "q05_1d_bps": float(net_bps.quantile(0.05)),
        "cvar05_1d_bps": float(net_bps[net_bps <= net_bps.quantile(0.05)].mean()),
        "worst_5d": float(roll5.min()),
        "worst_20d": float(roll20.min()),
        "benchmark_down_mean_bps": float(net_bps[benchmark < 0].mean()),
        "benchmark_bottom_decile_mean_bps": float(net_bps[benchmark <= bench_q10].mean()),
        "score_coverage_min": float(score_coverage["raw_coverage"].min()),
        "score_coverage_mean": float(score_coverage["raw_coverage"].mean()),
        "mean_abs_net_beta_constructed": float(diagnostics["net_beta"].abs().mean()),
        "p95_abs_net_beta_constructed": float(diagnostics["net_beta"].abs().quantile(0.95)),
        "mean_sector_abs_exposure_sum": float(diagnostics["sector_abs_exposure_sum"].mean()),
        "p95_sector_abs_exposure_sum": float(diagnostics["sector_abs_exposure_sum"].quantile(0.95)),
    }
    return row


def _rolling_market_dependence(net_returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    rows: dict[str, float] = {}
    for window in (60, 126, 252):
        betas: list[float] = []
        for idx in range(window - 1, len(net_returns)):
            beta, _, _ = _lin_beta(net_returns.iloc[idx - window + 1 : idx + 1], benchmark.iloc[idx - window + 1 : idx + 1])
            if np.isfinite(beta):
                betas.append(float(beta))
        if betas:
            series = pd.Series(betas)
            rows[f"beta_{window}_p90"] = float(series.quantile(0.90))
            rows[f"beta_{window}_max"] = float(series.max())
    return rows


def _lin_beta(y: pd.Series, x: pd.Series) -> tuple[float, float, float]:
    frame = pd.concat([pd.Series(y, name="y"), pd.Series(x, name="x")], axis=1).dropna()
    if len(frame) < 3 or float(frame["x"].var(ddof=1)) == 0.0:
        return np.nan, np.nan, np.nan
    corr = float(frame["y"].corr(frame["x"]))
    beta = float(frame["y"].cov(frame["x"]) / frame["x"].var(ddof=1))
    return beta, corr, float(corr * corr)


def _score_universes(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["rank_sharpe"] = frame["sharpe_no_rf"].rank(ascending=False, method="min")
    frame["rank_return"] = frame["annualized_return"].rank(ascending=False, method="min")
    frame["rank_drawdown"] = frame["max_drawdown"].abs().rank(ascending=True, method="min")
    frame["rank_beta"] = frame["realized_beta_net"].abs().rank(ascending=True, method="min")
    frame["rank_tail20"] = frame["worst_20d"].abs().rank(ascending=True, method="min")
    frame["rank_coverage"] = frame["score_coverage_min"].rank(ascending=False, method="min")
    frame["rank_cost"] = frame["mean_cost_bps"].rank(ascending=True, method="min")
    frame["underwriting_rank_sum"] = frame[
        ["rank_sharpe", "rank_return", "rank_drawdown", "rank_beta", "rank_tail20", "rank_coverage", "rank_cost"]
    ].sum(axis=1)
    return frame.sort_values("underwriting_rank_sum", ascending=True).reset_index(drop=True)


def _read_existing_subset(path: Path, *, variants: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "variant" in frame.columns:
        return frame[frame["variant"].isin(set(variants))].copy()
    return frame


def _read_asymmetric_prior(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    columns = [
        "pair",
        "long_variant",
        "short_variant",
        "constructed_sessions",
        "construction_rate",
        "mean_spread_return",
        "spread_hit_rate",
        "mean_spread_beta_residual",
        "mean_long_return",
        "mean_short_return_contribution",
        "mean_abs_net_beta",
        "median_long_count",
        "median_short_count",
    ]
    frame = pd.read_csv(path)
    return frame[[column for column in columns if column in frame.columns]].copy()


def _memo(
    *,
    output_root: Path,
    variants: Sequence[str],
    turnover_budget: float,
    size_panel_path: Path,
    summary: pd.DataFrame,
    scorecard: pd.DataFrame,
    phase4h: pd.DataFrame,
    phase4k: pd.DataFrame,
) -> str:
    metric_columns = [
        "variant",
        "sharpe_no_rf",
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "realized_beta_net",
        "rolling_beta_252_p90",
        "worst_20d",
        "mean_turnover",
        "score_coverage_min",
    ]
    rank_columns = [
        "variant",
        "underwriting_rank_sum",
        "rank_sharpe",
        "rank_return",
        "rank_drawdown",
        "rank_beta",
        "rank_tail20",
        "rank_coverage",
        "rank_cost",
    ]
    lines = [
        "# Phase7T Universe Underwriting",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- turnover budget: `{turnover_budget}`",
        f"- size panel: `{size_panel_path}`",
        f"- variants: `{', '.join(variants)}`",
        "- portfolio: `shared_core_lambda_0p005` only",
        "- interpretation: universe choice is treated as a structural prior; the scorecard is for underwriting, not automatic model selection.",
        "",
        "## Strict Path Summary",
        "",
        _markdown_table(summary[[column for column in metric_columns if column in summary.columns]]),
        "",
        "## Underwriting Rank",
        "",
        _markdown_table(scorecard[[column for column in rank_columns if column in scorecard.columns]]),
        "",
        "## Candidate Priors",
        "",
        _markdown_table(pd.DataFrame([spec.__dict__ for spec in UNIVERSE_SPECS if spec.variant in set(variants)])),
    ]
    if not phase4h.empty:
        phase4h_columns = [
            "variant",
            "mean_daily_names",
            "median_trailing_dollar_volume_20",
            "median_beta",
            "target_trimmed_mean_10_90",
            "negative_residual_share",
            "daily_residual_mean_positive_share",
        ]
        lines.extend(["", "## Existing Phase4H Universe Prior Evidence", "", _markdown_table(phase4h[[column for column in phase4h_columns if column in phase4h.columns]])])
    if not phase4k.empty:
        lines.extend(["", "## Existing Phase4K Asymmetric Prior Evidence", "", _markdown_table(phase4k)])
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- A higher Sharpe universe is not automatically selected; it must also pass liquidity, coverage, beta leakage, drawdown, and implementation checks.",
            "- Asymmetric long/short pools are not fully strict-tested here yet; Phase4K is included only as prior evidence.",
            "- If a candidate wins only through a narrower, more crowded, or structurally biased pool, it should be discounted rather than promoted.",
            "- The next step is a strict asymmetric Phase7T-B run if the symmetric grid supports the prior that short-side universe choice matters.",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in view.columns) + " |")
    return "\n".join(lines)


def _reason_for_variant(variant: str) -> str:
    for spec in UNIVERSE_SPECS:
        if spec.variant == variant:
            return spec.reason
    return "custom candidate"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
