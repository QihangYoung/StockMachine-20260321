"""Phase7V cross-asset insurance overlays for the shared-core strategy.

This runner treats external assets as insurance candidates for Phase7K's
shared_core path. It is intentionally diagnostic: fixed overlay weights are
tested as structural priors instead of optimized in-sample.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase7b import _markdown_table


DEFAULT_SHARED_CORE_CURVE = (
    RESEARCH_ROOT
    / "phase7k_locked_turnover_shared_capital_20260518_tb0p15"
    / "phase7k_strict_daily_curve.csv"
)
DEFAULT_DAILY_BAR_PATHS = (
    Path("data/silver/daily_bar/fmf_validation_etf_bootstrap.jsonl"),
)
DEFAULT_ADJ_FACTOR_PATHS = (
    Path("data/silver/adj_factor/fmf_validation_etf_bootstrap.jsonl"),
)
DEFAULT_BENCHMARK_INDEX_PATHS = (
    Path("data/silver/benchmark_index/fmf_validation_etf_bootstrap.jsonl"),
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7v_cross_asset_insurance_overlay_20260518"
DEFAULT_SHARED_PORTFOLIO = "shared_core_lambda_0p005_tb0p15"
DEFAULT_WEIGHTS = (0.05, 0.10, 0.15, 0.20)
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class Candidate:
    name: str
    components: Mapping[str, float]
    role: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("BIL_cash", {"BIL": 1.0}, "cash reserve"),
    Candidate("IEF_duration", {"IEF": 1.0}, "rate-cut/growth-scare hedge"),
    Candidate("LQD_credit", {"LQD": 1.0}, "credit-duration carry hedge"),
    Candidate("GLD_gold", {"GLD": 1.0}, "fiat/inflation/geopolitical hedge"),
    Candidate("FMF_trend_proxy", {"FMF": 1.0}, "managed-futures trend proxy"),
    Candidate("VXUS_ex_us_equity", {"VXUS": 1.0}, "global equity comparator"),
    Candidate("SPY_us_equity", {"SPY": 1.0}, "risk-on comparator"),
    Candidate(
        "equal_defensive",
        {"BIL": 0.20, "IEF": 0.20, "LQD": 0.20, "GLD": 0.20, "FMF": 0.20},
        "equal defensive basket",
    ),
    Candidate(
        "duration_gold_trend",
        {"IEF": 1.0 / 3.0, "GLD": 1.0 / 3.0, "FMF": 1.0 / 3.0},
        "three-way convexity basket",
    ),
    Candidate(
        "cash_duration_gold_trend",
        {"BIL": 0.25, "IEF": 0.25, "GLD": 0.25, "FMF": 0.25},
        "cash plus defensive convexity basket",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7V cross-asset insurance overlays.")
    parser.add_argument("--shared-core-curve", default=str(DEFAULT_SHARED_CORE_CURVE))
    parser.add_argument("--shared-portfolio", default=DEFAULT_SHARED_PORTFOLIO)
    parser.add_argument("--daily-bar-path", action="append", dest="daily_bar_paths")
    parser.add_argument("--adj-factor-path", action="append", dest="adj_factor_paths")
    parser.add_argument("--benchmark-index-path", action="append", dest="benchmark_index_paths")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--weights", default=",".join(f"{value:.2f}" for value in DEFAULT_WEIGHTS))
    parser.add_argument("--cash-symbol", default="BIL")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    daily_bar_paths = _paths_or_default(args.daily_bar_paths, DEFAULT_DAILY_BAR_PATHS)
    adj_factor_paths = _paths_or_default(args.adj_factor_paths, DEFAULT_ADJ_FACTOR_PATHS)
    benchmark_index_paths = _paths_or_default(args.benchmark_index_paths, DEFAULT_BENCHMARK_INDEX_PATHS)
    weights = _parse_weights(args.weights)

    rollup = build_phase7v_cross_asset_insurance_overlay(
        shared_core_curve=Path(args.shared_core_curve),
        shared_portfolio=str(args.shared_portfolio),
        daily_bar_paths=daily_bar_paths,
        adj_factor_paths=adj_factor_paths,
        benchmark_index_paths=benchmark_index_paths,
        output_root=output_root,
        weights=weights,
        cash_symbol=str(args.cash_symbol),
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=False))
    return 0


def build_phase7v_cross_asset_insurance_overlay(
    *,
    shared_core_curve: Path,
    shared_portfolio: str,
    daily_bar_paths: Sequence[Path],
    adj_factor_paths: Sequence[Path],
    benchmark_index_paths: Sequence[Path],
    output_root: Path,
    weights: Sequence[float],
    cash_symbol: str,
) -> dict[str, object]:
    shared = _load_shared_core_returns(shared_core_curve, shared_portfolio=shared_portfolio)
    required_symbols = tuple(
        sorted({symbol for candidate in CANDIDATES for symbol in candidate.components})
    )
    external = _load_external_returns(
        daily_bar_paths=daily_bar_paths,
        adj_factor_paths=adj_factor_paths,
        benchmark_index_paths=benchmark_index_paths,
        symbols=required_symbols,
    )
    candidate_returns = _build_candidate_returns(external, CANDIDATES)

    panel = shared.join(candidate_returns, how="inner")
    if panel.empty:
        raise ValueError("No common dates remain between shared_core and external assets.")
    panel = panel.sort_index()
    panel.index.name = "date"

    base = panel["shared_core"]
    base_metrics = _performance_metrics(base, name="shared_core_100")
    asset_diagnostics = _asset_diagnostics(panel, base=base, candidates=CANDIDATES)
    funded_summary, funded_daily = _overlay_summary(
        panel,
        base=base,
        candidates=CANDIDATES,
        weights=weights,
        overlay_kind="funded",
        cash_symbol=cash_symbol,
    )
    financed_summary, financed_daily = _overlay_summary(
        panel,
        base=base,
        candidates=CANDIDATES,
        weights=weights,
        overlay_kind="financed",
        cash_symbol=cash_symbol,
    )
    tail_windows = _tail_window_summary(
        panel,
        base=base,
        candidate_returns=candidate_returns,
        funded_daily=funded_daily,
        financed_daily=financed_daily,
    )

    daily_panel_path = output_root / "phase7v_daily_panel.csv"
    asset_path = output_root / "phase7v_asset_insurance_diagnostics.csv"
    funded_path = output_root / "phase7v_funded_overlay_summary.csv"
    financed_path = output_root / "phase7v_financed_overlay_summary.csv"
    tail_path = output_root / "phase7v_tail_window_summary.csv"
    rollup_path = output_root / "phase7v_rollup.json"
    memo_path = output_root / "phase7v_cross_asset_insurance_overlay_memo.md"

    panel.reset_index().to_csv(daily_panel_path, index=False)
    asset_diagnostics.to_csv(asset_path, index=False)
    funded_summary.to_csv(funded_path, index=False)
    financed_summary.to_csv(financed_path, index=False)
    tail_windows.to_csv(tail_path, index=False)

    rollup: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_core_curve": str(shared_core_curve),
        "shared_portfolio": shared_portfolio,
        "daily_bar_paths": [str(path) for path in daily_bar_paths],
        "adj_factor_paths": [str(path) for path in adj_factor_paths],
        "benchmark_index_paths": [str(path) for path in benchmark_index_paths],
        "output_root": str(output_root),
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "days": int(len(panel)),
        "weights": [float(value) for value in weights],
        "cash_symbol": cash_symbol,
        "base_metrics": base_metrics,
        "best_funded_by_sharpe": _best_row_payload(funded_summary, "sharpe"),
        "best_funded_by_drawdown": _best_row_payload(funded_summary, "max_dd_reduction_vs_base"),
        "best_financed_by_sharpe": _best_row_payload(financed_summary, "sharpe"),
        "best_financed_by_drawdown": _best_row_payload(financed_summary, "max_dd_reduction_vs_base"),
        "paths": {
            "daily_panel": str(daily_panel_path),
            "asset_diagnostics": str(asset_path),
            "funded_overlay_summary": str(funded_path),
            "financed_overlay_summary": str(financed_path),
            "tail_window_summary": str(tail_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_memo(
        memo_path=memo_path,
        rollup=rollup,
        asset_diagnostics=asset_diagnostics,
        funded_summary=funded_summary,
        financed_summary=financed_summary,
        tail_windows=tail_windows,
    )
    return rollup


def _load_shared_core_returns(path: Path, *, shared_portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "portfolio" in frame.columns:
        frame = frame.loc[frame["portfolio"].astype(str).eq(shared_portfolio)].copy()
    date_column = "return_date" if "return_date" in frame.columns else "date"
    if date_column not in frame.columns:
        raise KeyError(f"{path} is missing a return date column.")
    if "net_return" not in frame.columns:
        raise KeyError(f"{path} is missing net_return.")
    out = frame.loc[:, [date_column, "net_return"]].copy()
    out[date_column] = pd.to_datetime(out[date_column]).dt.normalize()
    out["shared_core"] = pd.to_numeric(out["net_return"], errors="coerce")
    out = out.dropna(subset=[date_column, "shared_core"]).drop_duplicates(
        subset=[date_column],
        keep="last",
    )
    return out.set_index(date_column).loc[:, ["shared_core"]].sort_index()


def _load_external_returns(
    *,
    daily_bar_paths: Sequence[Path],
    adj_factor_paths: Sequence[Path],
    benchmark_index_paths: Sequence[Path],
    symbols: Sequence[str],
) -> pd.DataFrame:
    bars = _read_jsonl_many([*daily_bar_paths, *benchmark_index_paths])
    if bars.empty:
        raise ValueError("No external daily bars were loaded.")
    factors = _read_jsonl_many(adj_factor_paths)
    symbols = tuple(dict.fromkeys(str(symbol) for symbol in symbols))

    bars = bars.loc[bars["symbol"].astype(str).isin(symbols)].copy()
    if bars.empty:
        raise ValueError(f"External daily bars are missing all required symbols: {symbols}.")
    bars["session_date"] = pd.to_datetime(bars["session_date"]).dt.normalize()
    bars["symbol"] = bars["symbol"].astype(str)
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["session_date", "symbol", "close"])
    bars = bars.drop_duplicates(subset=["session_date", "symbol"], keep="last")

    if not factors.empty:
        factors = factors.loc[factors["symbol"].astype(str).isin(symbols)].copy()
        factors["session_date"] = pd.to_datetime(factors["session_date"]).dt.normalize()
        factors["symbol"] = factors["symbol"].astype(str)
        factors["price_adjust_factor"] = pd.to_numeric(
            factors["price_adjust_factor"],
            errors="coerce",
        )
        factors = factors.drop_duplicates(subset=["session_date", "symbol"], keep="last")
        bars = bars.merge(
            factors.loc[:, ["session_date", "symbol", "price_adjust_factor"]],
            how="left",
            on=["session_date", "symbol"],
            validate="one_to_one",
        )
    else:
        bars["price_adjust_factor"] = np.nan

    bars["price_adjust_factor"] = bars["price_adjust_factor"].fillna(1.0)
    bars["adjusted_close"] = bars["close"] * bars["price_adjust_factor"]
    prices = (
        bars.pivot(index="session_date", columns="symbol", values="adjusted_close")
        .sort_index()
        .sort_index(axis=1)
    )
    returns = prices.pct_change(fill_method=None).iloc[1:].dropna(how="all")
    missing = sorted(set(symbols).difference(returns.columns))
    if missing:
        raise ValueError(f"External returns are missing required symbols: {missing}.")
    return returns.loc[:, list(symbols)].astype(float)


def _build_candidate_returns(external: pd.DataFrame, candidates: Sequence[Candidate]) -> pd.DataFrame:
    out = pd.DataFrame(index=external.index)
    for candidate in candidates:
        weights = pd.Series(candidate.components, dtype=float)
        weights = weights / weights.sum()
        out[candidate.name] = external.loc[:, list(weights.index)].mul(weights, axis=1).sum(axis=1)
    return out


def _asset_diagnostics(
    panel: pd.DataFrame,
    *,
    base: pd.Series,
    candidates: Sequence[Candidate],
) -> pd.DataFrame:
    worst_decile_mask = base <= base.quantile(0.10)
    worst_quintile_mask = base <= base.quantile(0.20)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        series = panel[candidate.name]
        metrics = _performance_metrics(series, name=candidate.name)
        row: dict[str, object] = {
            "asset": candidate.name,
            "role": candidate.role,
            "components": _format_components(candidate.components),
            "overlap_days": int(series.dropna().shape[0]),
            "corr_to_shared_core": float(series.corr(base)),
            "mean_return_on_shared_core_worst_decile_bps": float(
                series.loc[worst_decile_mask].mean() * 10000.0
            ),
            "hit_rate_on_shared_core_worst_decile": float((series.loc[worst_decile_mask] > 0.0).mean()),
            "mean_return_on_shared_core_worst_quintile_bps": float(
                series.loc[worst_quintile_mask].mean() * 10000.0
            ),
            "hit_rate_on_shared_core_worst_quintile": float((series.loc[worst_quintile_mask] > 0.0).mean()),
        }
        row.update({f"asset_{key}": value for key, value in metrics.items() if key != "name"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_return_on_shared_core_worst_decile_bps", "corr_to_shared_core"],
        ascending=[False, True],
    )


def _overlay_summary(
    panel: pd.DataFrame,
    *,
    base: pd.Series,
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    overlay_kind: str,
    cash_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if overlay_kind not in {"funded", "financed"}:
        raise ValueError(f"Unsupported overlay_kind: {overlay_kind}")

    base_metrics = _performance_metrics(base, name="shared_core_100")
    cash_column = f"{cash_symbol}_cash"
    cash_returns = panel[cash_column] if cash_column in panel.columns else pd.Series(0.0, index=panel.index)
    worst_decile_mask = base <= base.quantile(0.10)
    base_worst_decile_mean = float(base.loc[worst_decile_mask].mean() * 10000.0)
    daily = pd.DataFrame(index=panel.index)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        asset = panel[candidate.name]
        for weight in weights:
            if overlay_kind == "funded":
                returns = (1.0 - float(weight)) * base + float(weight) * asset
                gross_nav_exposure = 1.0
            else:
                returns = base + float(weight) * (asset - cash_returns)
                gross_nav_exposure = 1.0 + float(weight)
            name = f"{overlay_kind}_{candidate.name}_{int(round(weight * 100)):02d}pct"
            daily[name] = returns
            metrics = _performance_metrics(returns, name=name)
            row: dict[str, object] = {
                "overlay": name,
                "overlay_kind": overlay_kind,
                "asset": candidate.name,
                "role": candidate.role,
                "weight": float(weight),
                "components": _format_components(candidate.components),
                "gross_nav_exposure": gross_nav_exposure,
                "corr_to_shared_core": float(returns.corr(base)),
                "ann_return_drag_vs_base": float(base_metrics["ann_return"] - metrics["ann_return"]),
                "vol_reduction_vs_base": float(base_metrics["vol"] - metrics["vol"]),
                "max_dd_reduction_vs_base": float(metrics["max_dd"] - base_metrics["max_dd"]),
                "cvar05_reduction_vs_base_bps": float(
                    metrics["cvar05_1d_bps"] - base_metrics["cvar05_1d_bps"]
                ),
                "worst_20d_reduction_vs_base": float(
                    metrics["worst_20d"] - base_metrics["worst_20d"]
                ),
                "mean_return_on_base_worst_decile_bps": float(
                    returns.loc[worst_decile_mask].mean() * 10000.0
                ),
                "base_worst_decile_mean_bps": base_worst_decile_mean,
            }
            row["bad_day_improvement_bps"] = (
                row["mean_return_on_base_worst_decile_bps"] - base_worst_decile_mean
            )
            row.update({key: value for key, value in metrics.items() if key != "name"})
            rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["max_dd_reduction_vs_base", "ann_return_drag_vs_base"],
        ascending=[False, True],
    ), daily


def _tail_window_summary(
    panel: pd.DataFrame,
    *,
    base: pd.Series,
    candidate_returns: pd.DataFrame,
    funded_daily: pd.DataFrame,
    financed_daily: pd.DataFrame,
) -> pd.DataFrame:
    windows = {
        "base_worst_20d": _worst_window_dates(base, 20),
        "base_worst_60d": _worst_window_dates(base, 60),
        "base_worst_decile_days": base.index[base <= base.quantile(0.10)],
    }
    comparison_columns = [
        "shared_core",
        *candidate_returns.columns,
        *_pick_named_overlay_columns(funded_daily),
        *_pick_named_overlay_columns(financed_daily),
    ]
    comparison = pd.concat([panel.loc[:, ["shared_core"]], candidate_returns, funded_daily, financed_daily], axis=1)
    rows: list[dict[str, object]] = []
    for window_name, dates in windows.items():
        dates = pd.Index(dates).intersection(comparison.index)
        for column in comparison_columns:
            if column not in comparison.columns:
                continue
            series = comparison.loc[dates, column].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "window": window_name,
                    "series": column,
                    "days": int(series.shape[0]),
                    "cumulative_return": float((1.0 + series).prod() - 1.0),
                    "mean_daily_bps": float(series.mean() * 10000.0),
                    "hit_rate": float((series > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _pick_named_overlay_columns(frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for token in ("BIL_cash_10pct", "IEF_duration_10pct", "GLD_gold_10pct", "FMF_trend_proxy_10pct", "equal_defensive_20pct", "duration_gold_trend_20pct"):
        matches = [column for column in frame.columns if column.endswith(token)]
        selected.extend(matches)
    return selected


def _performance_metrics(returns: pd.Series, *, name: str) -> dict[str, float | str | int]:
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError(f"No returns available for {name}.")
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    ann_return = float(equity.iloc[-1] ** (TRADING_DAYS / len(clean)) - 1.0)
    vol = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(clean) > 1 else 0.0
    sharpe = float(ann_return / vol) if vol > 0.0 else np.nan
    rolling20 = (1.0 + clean).rolling(20).apply(np.prod, raw=True) - 1.0
    cvar05 = clean.loc[clean <= clean.quantile(0.05)].mean()
    return {
        "name": name,
        "days": int(len(clean)),
        "final_equity": float(equity.iloc[-1]),
        "ann_return": ann_return,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(drawdown.min()),
        "mean_daily_bps": float(clean.mean() * 10000.0),
        "hit_rate": float((clean > 0.0).mean()),
        "worst_1d_bps": float(clean.min() * 10000.0),
        "q05_1d_bps": float(clean.quantile(0.05) * 10000.0),
        "cvar05_1d_bps": float(cvar05 * 10000.0),
        "worst_20d": float(rolling20.min()),
        "max_underwater_days": int(_max_underwater_days(drawdown)),
    }


def _max_underwater_days(drawdown: pd.Series) -> int:
    max_run = 0
    current = 0
    for value in drawdown:
        if value < 0.0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _worst_window_dates(returns: pd.Series, window: int) -> pd.Index:
    rolling = (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
    if rolling.dropna().empty:
        return returns.index[:0]
    end_date = rolling.idxmin()
    end_position = returns.index.get_loc(end_date)
    start_position = max(0, int(end_position) - window + 1)
    return returns.index[start_position : int(end_position) + 1]


def _best_row_payload(frame: pd.DataFrame, column: str) -> dict[str, object]:
    if frame.empty:
        return {}
    row = frame.loc[frame[column].idxmax()]
    return row.to_dict()


def _write_memo(
    *,
    memo_path: Path,
    rollup: Mapping[str, object],
    asset_diagnostics: pd.DataFrame,
    funded_summary: pd.DataFrame,
    financed_summary: pd.DataFrame,
    tail_windows: pd.DataFrame,
) -> None:
    base = rollup["base_metrics"]
    assert isinstance(base, Mapping)
    funded_top = funded_summary.sort_values("max_dd_reduction_vs_base", ascending=False).head(10)
    financed_top = financed_summary.sort_values("max_dd_reduction_vs_base", ascending=False).head(10)
    asset_top = asset_diagnostics.head(10)
    tail_focus = tail_windows.loc[
        tail_windows["series"].isin(
            [
                "shared_core",
                "BIL_cash",
                "IEF_duration",
                "GLD_gold",
                "FMF_trend_proxy",
                "funded_GLD_gold_10pct",
                "funded_duration_gold_trend_20pct",
                "financed_GLD_gold_10pct",
                "financed_duration_gold_trend_20pct",
            ]
        )
    ].copy()
    lines = [
        "# Phase7V Cross-Asset Insurance Overlay",
        "",
        "Validation-only. No test lockbox is used.",
        "",
        "## Setup",
        "",
        f"- shared_core curve: `{rollup['shared_core_curve']}`",
        f"- external data: `{', '.join(str(path) for path in rollup['daily_bar_paths'])}`",
        f"- overlap: `{rollup['start']}` to `{rollup['end']}`, `{rollup['days']}` days",
        "- overlay weights: fixed structural grid, not in-sample optimized",
        "- funded overlay: `(1-w) * shared_core + w * asset`",
        "- financed overlay: `shared_core + w * (asset - BIL)`",
        "",
        "## Base Shared Core",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "days": base["days"],
                        "ann_return": base["ann_return"],
                        "vol": base["vol"],
                        "sharpe": base["sharpe"],
                        "max_dd": base["max_dd"],
                        "worst_20d": base["worst_20d"],
                        "cvar05_1d_bps": base["cvar05_1d_bps"],
                    }
                ]
            )
        ),
        "",
        "## Asset Insurance Diagnostics",
        "",
        _markdown_table(
            asset_top.loc[
                :,
                [
                    "asset",
                    "role",
                    "corr_to_shared_core",
                    "asset_ann_return",
                    "asset_vol",
                    "asset_sharpe",
                    "asset_max_dd",
                    "mean_return_on_shared_core_worst_decile_bps",
                    "hit_rate_on_shared_core_worst_decile",
                ],
            ]
        ),
        "",
        "## Best Funded Overlays By Drawdown Reduction",
        "",
        _markdown_table(
            funded_top.loc[
                :,
                [
                    "overlay",
                    "weight",
                    "ann_return",
                    "vol",
                    "sharpe",
                    "max_dd",
                    "ann_return_drag_vs_base",
                    "max_dd_reduction_vs_base",
                    "bad_day_improvement_bps",
                ],
            ]
        ),
        "",
        "## Best Financed Overlays By Drawdown Reduction",
        "",
        _markdown_table(
            financed_top.loc[
                :,
                [
                    "overlay",
                    "weight",
                    "ann_return",
                    "vol",
                    "sharpe",
                    "max_dd",
                    "ann_return_drag_vs_base",
                    "max_dd_reduction_vs_base",
                    "bad_day_improvement_bps",
                ],
            ]
        ),
        "",
        "## Tail Window Focus",
        "",
        _markdown_table(tail_focus),
        "",
        "## Interpretation",
        "",
        "- External assets are evaluated as insurance only if they help in shared_core bad states, not merely because they have standalone Sharpe.",
        "- Funded overlays mostly buy drawdown relief by diluting shared_core exposure; that is valid insurance, but it has an explicit return drag.",
        "- Financed overlays preserve shared_core notional exposure but add gross/NAV exposure and should be sized by margin and liquidity constraints.",
        "- FMF is a longer-window managed-futures proxy; it is useful for underwriting, but not identical to a live CTA sleeve.",
    ]
    memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_components(components: Mapping[str, float]) -> str:
    return ",".join(f"{symbol}:{weight:.4g}" for symbol, weight in components.items())


def _read_jsonl_many(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frames.append(pd.read_json(path, lines=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_weights(value: str) -> tuple[float, ...]:
    weights = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not weights:
        raise ValueError("At least one overlay weight is required.")
    if any(weight <= 0.0 or weight >= 1.0 for weight in weights):
        raise ValueError("Overlay weights must be between 0 and 1.")
    return weights


def _paths_or_default(values: Sequence[str] | None, defaults: Sequence[Path]) -> tuple[Path, ...]:
    if values:
        return tuple(Path(value) for value in values)
    return tuple(defaults)


if __name__ == "__main__":
    raise SystemExit(main())
