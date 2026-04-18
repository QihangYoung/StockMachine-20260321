from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_PHASE1_MEMBERSHIP = (
    RESEARCH_ROOT
    / "phase1_universe_builder_20260418"
    / "phase1_candidate_universe_membership_validation.csv.gz"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase2_beta_panel_20260418"
DEFAULT_DAILY_GLOBS = (
    "data/silver/daily_bar/phase0_top1000_yahoo_gap_20130805_20151231_chunk*.jsonl",
    "data/silver/daily_bar/phase0_top1000_sip_raw_20160104_20260416_chunk*.jsonl",
)
DEFAULT_ADJ_FACTOR_GLOBS = (
    "data/silver/adj_factor/phase0_top1000_yahoo_gap_20130805_20151231_adjfactor_chunk*.jsonl",
    "data/silver/adj_factor/phase0_top1000_sip_adjfactor_20160104_20260416_chunk*.jsonl",
)
DEFAULT_BENCHMARK_DAILY_GLOBS = ("data/silver/benchmark_index/fmf_validation_etf_bootstrap.jsonl",)
DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS = ("data/silver/adj_factor/fmf_validation_etf_bootstrap.jsonl",)


def build_phase2_beta_artifacts(
    *,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = "SPY",
    lookback_sessions: int = 252,
    min_observations: int = 126,
    shrinkage_target: float = 1.0,
    shrinkage_strength: float = 0.10,
    beta_clip_low: float | None = 0.0,
    beta_clip_high: float | None = 3.0,
    asof_lag_sessions: int = 1,
) -> dict[str, Any]:
    """Build validation-only lagged beta panel and coverage diagnostics."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(membership_path)
    if membership.empty:
        raise ValueError("Phase 1 membership artifact is empty.")
    _validate_membership(membership)
    membership["session_date"] = pd.to_datetime(membership["session_date"]).dt.date.astype(str)

    symbols = tuple(sorted(membership["symbol"].astype(str).unique()))
    validation_end = str(membership["session_date"].max())
    stock_prices = _load_adjusted_close(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=validation_end,
    )
    benchmark_prices = _load_adjusted_close(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=validation_end,
    )
    benchmark_returns = _daily_returns(benchmark_prices).rename(
        columns={"symbol_return": "benchmark_return"}
    )[["session_date", "benchmark_return"]]
    stock_returns = _daily_returns(stock_prices)
    return_pairs = stock_returns.merge(benchmark_returns, on="session_date", how="inner")
    beta_estimates = _lagged_rolling_beta(
        return_pairs,
        lookback_sessions=lookback_sessions,
        min_observations=min_observations,
        shrinkage_target=shrinkage_target,
        shrinkage_strength=shrinkage_strength,
        beta_clip_low=beta_clip_low,
        beta_clip_high=beta_clip_high,
        asof_lag_sessions=asof_lag_sessions,
    )

    beta_panel = _membership_beta_panel(membership, beta_estimates, min_observations)
    coverage = _variant_coverage(membership, beta_panel)
    summary = _variant_summary(coverage)
    stability = _symbol_stability(beta_panel)

    beta_panel_path = output_dir / "phase2_beta_panel_validation.csv.gz"
    coverage_path = output_dir / "phase2_variant_beta_coverage_validation.csv"
    summary_path = output_dir / "phase2_beta_coverage_summary_validation.csv"
    stability_path = output_dir / "phase2_symbol_beta_stability_validation.csv"
    memo_path = output_dir / "phase2_beta_panel_memo.md"
    rollup_path = output_dir / "phase2_beta_panel_rollup.json"

    beta_panel.to_csv(beta_panel_path, index=False, compression="gzip")
    coverage.to_csv(coverage_path, index=False)
    summary.to_csv(summary_path, index=False)
    stability.to_csv(stability_path, index=False)
    memo_path.write_text(
        _beta_memo(
            summary,
            lookback_sessions=lookback_sessions,
            min_observations=min_observations,
            shrinkage_target=shrinkage_target,
            shrinkage_strength=shrinkage_strength,
            beta_clip_low=beta_clip_low,
            beta_clip_high=beta_clip_high,
            asof_lag_sessions=asof_lag_sessions,
            benchmark_symbol=benchmark_symbol,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "membership_path": Path(membership_path).as_posix(),
        "benchmark_symbol": benchmark_symbol,
        "lookback_sessions": int(lookback_sessions),
        "min_observations": int(min_observations),
        "shrinkage_target": float(shrinkage_target),
        "shrinkage_strength": float(shrinkage_strength),
        "beta_clip_low": beta_clip_low,
        "beta_clip_high": beta_clip_high,
        "asof_lag_sessions": int(asof_lag_sessions),
        "validation_start": str(membership["session_date"].min()),
        "validation_end": validation_end,
        "membership_rows": int(len(membership)),
        "membership_symbols": int(len(symbols)),
        "stock_price_rows_loaded": int(len(stock_prices)),
        "benchmark_price_rows_loaded": int(len(benchmark_prices)),
        "return_pair_rows": int(len(return_pairs)),
        "beta_panel_rows": int(len(beta_panel)),
        "beta_available_rows": int(beta_panel["beta_available"].sum()),
        "variants": sorted(membership["variant"].astype(str).unique()),
        "beta_panel_artifact": beta_panel_path.as_posix(),
        "variant_coverage_artifact": coverage_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "symbol_stability_artifact": stability_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "Uses Phase 1 validation membership only; no test-window performance is computed.",
            "Beta is lagged by asof_lag_sessions and does not use same-session returns by default.",
            "Benchmark is SPY from adjusted close-to-close returns.",
            "Corporate-action quality remains provisional until the primary vendor is selected.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_membership(membership: pd.DataFrame) -> None:
    required = {"session_date", "symbol", "variant"}
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise ValueError(f"Phase 1 membership artifact is missing required columns: {missing}")


def _load_adjusted_close(
    *,
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    symbols: Sequence[str],
    end_date: str,
) -> pd.DataFrame:
    symbol_set = set(symbols)
    daily = _load_jsonl_frames(
        daily_globs,
        columns=("session_date", "symbol", "close"),
        symbols=symbol_set,
        end_date=end_date,
    )
    if daily.empty:
        raise ValueError(f"No daily bars found for symbols: {sorted(symbol_set)[:5]}")
    adj = _load_jsonl_frames(
        adj_factor_globs,
        columns=("session_date", "symbol", "price_adjust_factor"),
        symbols=symbol_set,
        end_date=end_date,
    )
    frame = daily.merge(adj, on=["session_date", "symbol"], how="left")
    frame["price_adjust_factor"] = pd.to_numeric(
        frame["price_adjust_factor"], errors="coerce"
    ).fillna(1.0)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adjusted_close"] = frame["close"] * frame["price_adjust_factor"]
    frame = frame[(frame["adjusted_close"].notna()) & (frame["adjusted_close"] > 0)]
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["symbol", "session_date"]).drop_duplicates(
        ["symbol", "session_date"],
        keep="last",
    )


def _load_jsonl_frames(
    patterns: Sequence[str | Path],
    *,
    columns: Sequence[str],
    symbols: set[str],
    end_date: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _expand_globs(patterns):
        frame = pd.read_json(path, lines=True)
        if frame.empty:
            continue
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.loc[:, list(columns)]
        frame = frame[frame["symbol"].astype(str).isin(symbols)]
        frame = frame[frame["session_date"].astype(str) <= end_date]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "session_date"]).copy()
    frame["symbol_return"] = frame.groupby("symbol")["adjusted_close"].pct_change()
    frame = frame.replace([float("inf"), float("-inf")], pd.NA)
    return frame[["session_date", "symbol", "symbol_return"]].dropna()


def _lagged_rolling_beta(
    return_pairs: pd.DataFrame,
    *,
    lookback_sessions: int,
    min_observations: int,
    shrinkage_target: float,
    shrinkage_strength: float,
    beta_clip_low: float | None,
    beta_clip_high: float | None,
    asof_lag_sessions: int,
) -> pd.DataFrame:
    if return_pairs.empty:
        return pd.DataFrame(columns=_beta_estimate_columns())
    frame = return_pairs.sort_values(["symbol", "session_date"]).copy()
    pieces = []
    for symbol, group in frame.groupby("symbol", sort=False):
        pieces.append(
            _lagged_rolling_beta_one_symbol(
                symbol,
                group,
                lookback_sessions=lookback_sessions,
                min_observations=min_observations,
                shrinkage_target=shrinkage_target,
                shrinkage_strength=shrinkage_strength,
                beta_clip_low=beta_clip_low,
                beta_clip_high=beta_clip_high,
                asof_lag_sessions=asof_lag_sessions,
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=_beta_estimate_columns())


def _lagged_rolling_beta_one_symbol(
    symbol: str,
    group: pd.DataFrame,
    *,
    lookback_sessions: int,
    min_observations: int,
    shrinkage_target: float,
    shrinkage_strength: float,
    beta_clip_low: float | None,
    beta_clip_high: float | None,
    asof_lag_sessions: int,
) -> pd.DataFrame:
    ordered = group.sort_values("session_date").copy()
    x = ordered["benchmark_return"].shift(asof_lag_sessions)
    y = ordered["symbol_return"].shift(asof_lag_sessions)
    xy = x * y
    x2 = x * x
    rolling = {
        "obs": x.rolling(lookback_sessions, min_periods=min_observations).count(),
        "sum_x": x.rolling(lookback_sessions, min_periods=min_observations).sum(),
        "sum_y": y.rolling(lookback_sessions, min_periods=min_observations).sum(),
        "sum_xy": xy.rolling(lookback_sessions, min_periods=min_observations).sum(),
        "sum_x2": x2.rolling(lookback_sessions, min_periods=min_observations).sum(),
    }
    obs = rolling["obs"]
    cov_num = rolling["sum_xy"] - (rolling["sum_x"] * rolling["sum_y"] / obs)
    var_num = rolling["sum_x2"] - (rolling["sum_x"] * rolling["sum_x"] / obs)
    beta_raw = cov_num / var_num
    beta_raw = beta_raw.where((obs >= min_observations) & (var_num > 0))
    beta_shrunk = (1.0 - shrinkage_strength) * beta_raw + shrinkage_strength * shrinkage_target
    beta = beta_shrunk.copy()
    if beta_clip_low is not None or beta_clip_high is not None:
        beta = beta.clip(lower=beta_clip_low, upper=beta_clip_high)
    result = pd.DataFrame(
        {
            "session_date": ordered["session_date"].to_numpy(),
            "symbol": symbol,
            "beta_raw": beta_raw,
            "beta_shrunk": beta_shrunk,
            "beta": beta,
            "beta_obs": obs.fillna(0).astype(int),
            "benchmark_variance_num": var_num,
        }
    )
    return result


def _membership_beta_panel(
    membership: pd.DataFrame,
    beta_estimates: pd.DataFrame,
    min_observations: int,
) -> pd.DataFrame:
    requested = membership[["session_date", "symbol"]].drop_duplicates().copy()
    panel = requested.merge(beta_estimates, on=["session_date", "symbol"], how="left")
    panel["beta_available"] = panel["beta"].notna() & (panel["beta_obs"].fillna(0) >= min_observations)
    panel["beta_missing_reason"] = "available"
    panel.loc[~panel["beta_available"], "beta_missing_reason"] = "insufficient_lagged_return_observations"
    panel["beta_obs"] = panel["beta_obs"].fillna(0).astype(int)
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _variant_coverage(membership: pd.DataFrame, beta_panel: pd.DataFrame) -> pd.DataFrame:
    merged = membership.merge(
        beta_panel[["session_date", "symbol", "beta", "beta_obs", "beta_available"]],
        on=["session_date", "symbol"],
        how="left",
    )
    coverage = (
        merged.groupby(["session_date", "variant"])
        .agg(
            members=("symbol", "nunique"),
            beta_available_members=("beta_available", "sum"),
            median_beta=("beta", "median"),
            p10_beta=("beta", lambda s: s.quantile(0.10)),
            p90_beta=("beta", lambda s: s.quantile(0.90)),
            median_beta_obs=("beta_obs", "median"),
        )
        .reset_index()
        .sort_values(["variant", "session_date"])
    )
    coverage["members"] = coverage["members"].astype(int)
    coverage["beta_available_members"] = coverage["beta_available_members"].astype(int)
    coverage["beta_coverage"] = coverage["beta_available_members"] / coverage["members"]
    return coverage


def _variant_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, frame in coverage.groupby("variant", sort=False):
        ge95 = frame["beta_coverage"] >= 0.95
        rows.append(
            {
                "variant": variant,
                "sessions": int(len(frame)),
                "first_session": _first_or_none(frame["session_date"]),
                "first_session_beta_coverage_ge_95pct": _first_or_none(frame.loc[ge95, "session_date"]),
                "share_sessions_beta_coverage_ge_95pct": float(ge95.mean()) if len(ge95) else 0.0,
                "median_beta_coverage": float(frame["beta_coverage"].median()) if len(frame) else 0.0,
                "min_beta_coverage": float(frame["beta_coverage"].min()) if len(frame) else 0.0,
                "median_members": float(frame["members"].median()) if len(frame) else 0.0,
                "median_beta_available_members": float(frame["beta_available_members"].median())
                if len(frame)
                else 0.0,
                "median_beta": float(frame["median_beta"].median()) if len(frame) else 0.0,
                "median_p10_beta": float(frame["p10_beta"].median()) if len(frame) else 0.0,
                "median_p90_beta": float(frame["p90_beta"].median()) if len(frame) else 0.0,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _symbol_stability(beta_panel: pd.DataFrame) -> pd.DataFrame:
    available = beta_panel[beta_panel["beta_available"]].copy()
    if available.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "beta_days",
                "first_beta_date",
                "last_beta_date",
                "mean_beta",
                "std_beta",
                "min_beta",
                "max_beta",
            ]
        )
    return (
        available.groupby("symbol")
        .agg(
            beta_days=("session_date", "nunique"),
            first_beta_date=("session_date", "min"),
            last_beta_date=("session_date", "max"),
            mean_beta=("beta", "mean"),
            std_beta=("beta", "std"),
            min_beta=("beta", "min"),
            max_beta=("beta", "max"),
        )
        .reset_index()
        .sort_values(["beta_days", "symbol"], ascending=[False, True])
    )


def _beta_memo(
    summary: pd.DataFrame,
    *,
    lookback_sessions: int,
    min_observations: int,
    shrinkage_target: float,
    shrinkage_strength: float,
    beta_clip_low: float | None,
    beta_clip_high: float | None,
    asof_lag_sessions: int,
    benchmark_symbol: str,
) -> str:
    lines = [
        "# Pure Alpha Phase 2 Beta Panel Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only lagged beta-estimation artifact. It does not compute "
        "alpha signals, portfolio returns, model selection, or test-window performance.",
        "",
        "## Specification",
        "",
        f"- benchmark: `{benchmark_symbol}`",
        f"- return input: adjusted close-to-close returns",
        f"- lookback sessions: `{lookback_sessions}`",
        f"- minimum observations: `{min_observations}`",
        f"- as-of lag sessions: `{asof_lag_sessions}`",
        f"- shrinkage target: `{shrinkage_target}`",
        f"- shrinkage strength: `{shrinkage_strength}`",
        f"- beta clip: `[{beta_clip_low}, {beta_clip_high}]`",
        "",
        "## Variant Coverage",
        "",
        "| Variant | Median Coverage | Min Coverage | First >=95% Coverage | Median Beta |",
        "|---|---:|---:|---|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['median_beta_coverage']} | {row['min_beta_coverage']} | "
            f"{row['first_session_beta_coverage_ge_95pct']} | {row['median_beta']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Beta at a decision session uses only prior-session returns by default.",
            "- The panel is keyed by `session_date` and `symbol` for later beta-matched construction.",
            "- Missing beta rows are explicit rather than silently filled.",
            "- Corporate-action quality remains provisional until the primary vendor gate is closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _beta_estimate_columns() -> list[str]:
    return [
        "session_date",
        "symbol",
        "beta_raw",
        "beta_shrunk",
        "beta",
        "beta_obs",
        "benchmark_variance_num",
    ]


def _expand_globs(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(dict.fromkeys(paths))


def _first_or_none(series: pd.Series | None) -> str | None:
    if series is None or len(series) == 0:
        return None
    return str(series.iloc[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 2 beta panel builder.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument(
        "--benchmark-adj-factor-glob",
        action="append",
        dest="benchmark_adj_factor_globs",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--lookback-sessions", type=int, default=252)
    parser.add_argument("--min-observations", type=int, default=126)
    parser.add_argument("--shrinkage-target", type=float, default=1.0)
    parser.add_argument("--shrinkage-strength", type=float, default=0.10)
    parser.add_argument("--beta-clip-low", type=float, default=0.0)
    parser.add_argument("--beta-clip-high", type=float, default=3.0)
    parser.add_argument("--asof-lag-sessions", type=int, default=1)
    args = parser.parse_args(argv)

    result = build_phase2_beta_artifacts(
        membership_path=args.membership_path,
        daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
        adj_factor_globs=tuple(args.adj_factor_globs or DEFAULT_ADJ_FACTOR_GLOBS),
        benchmark_daily_globs=tuple(args.benchmark_daily_globs or DEFAULT_BENCHMARK_DAILY_GLOBS),
        benchmark_adj_factor_globs=tuple(
            args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS
        ),
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        lookback_sessions=args.lookback_sessions,
        min_observations=args.min_observations,
        shrinkage_target=args.shrinkage_target,
        shrinkage_strength=args.shrinkage_strength,
        beta_clip_low=args.beta_clip_low,
        beta_clip_high=args.beta_clip_high,
        asof_lag_sessions=args.asof_lag_sessions,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
