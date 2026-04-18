from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_PHASE1_MEMBERSHIP = (
    RESEARCH_ROOT
    / "phase1_universe_builder_20260418"
    / "phase1_candidate_universe_membership_validation.csv.gz"
)
DEFAULT_PHASE2_BETA_PANEL = (
    RESEARCH_ROOT / "phase2_beta_panel_20260418" / "phase2_beta_panel_validation.csv.gz"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase3_baseline_signals_20260418"
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
SIGNAL_COLUMNS = (
    "random_control",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "liquidity_rank_score",
    "transparent_composite",
)
TARGET_COLUMN = "forward_beta_residual_return_5d"


def build_phase3_signal_artifacts(
    *,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP,
    beta_panel_path: str | Path = DEFAULT_PHASE2_BETA_PANEL,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = "SPY",
    holding_period_sessions: int = 5,
    min_cross_section: int = 20,
    top_bottom_quantile: float = 0.20,
) -> dict[str, Any]:
    """Build validation-only transparent baseline signal diagnostics."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(membership_path)
    beta_panel = pd.read_csv(beta_panel_path)
    _validate_inputs(membership, beta_panel)
    membership["session_date"] = pd.to_datetime(membership["session_date"]).dt.date.astype(str)
    beta_panel["session_date"] = pd.to_datetime(beta_panel["session_date"]).dt.date.astype(str)

    symbols = tuple(sorted(membership["symbol"].astype(str).unique()))
    validation_end = str(membership["session_date"].max())
    stock_prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=validation_end,
    )
    benchmark_prices = _load_adjusted_prices(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=validation_end,
    )
    base_features = _base_feature_panel(
        stock_prices,
        benchmark_prices,
        holding_period_sessions=holding_period_sessions,
        benchmark_symbol=benchmark_symbol,
    )
    signal_panel = _membership_signal_panel(membership, beta_panel, base_features)
    signal_panel = _add_cross_sectional_scores(signal_panel)
    signal_panel = signal_panel.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    daily_diagnostics = _daily_signal_diagnostics(
        signal_panel,
        min_cross_section=min_cross_section,
        top_bottom_quantile=top_bottom_quantile,
    )
    leaderboard = _signal_leaderboard(daily_diagnostics)
    correlation = _signal_correlation(signal_panel)

    signal_panel_path = output_dir / "phase3_baseline_signal_panel_validation.csv.gz"
    daily_path = output_dir / "phase3_baseline_signal_daily_diagnostics_validation.csv"
    leaderboard_path = output_dir / "phase3_baseline_signal_leaderboard_validation.csv"
    correlation_path = output_dir / "phase3_signal_correlation_validation.csv"
    memo_path = output_dir / "phase3_baseline_signal_memo.md"
    rollup_path = output_dir / "phase3_baseline_signal_rollup.json"

    signal_panel.to_csv(signal_panel_path, index=False, compression="gzip")
    daily_diagnostics.to_csv(daily_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    correlation.to_csv(correlation_path, index=False)
    memo_path.write_text(
        _signal_memo(
            leaderboard,
            holding_period_sessions=holding_period_sessions,
            min_cross_section=min_cross_section,
            top_bottom_quantile=top_bottom_quantile,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "membership_path": Path(membership_path).as_posix(),
        "beta_panel_path": Path(beta_panel_path).as_posix(),
        "benchmark_symbol": benchmark_symbol,
        "holding_period_sessions": int(holding_period_sessions),
        "min_cross_section": int(min_cross_section),
        "top_bottom_quantile": float(top_bottom_quantile),
        "validation_start": str(signal_panel["session_date"].min()) if not signal_panel.empty else None,
        "validation_end": str(signal_panel["session_date"].max()) if not signal_panel.empty else None,
        "signal_panel_rows": int(len(signal_panel)),
        "daily_diagnostic_rows": int(len(daily_diagnostics)),
        "leaderboard_rows": int(len(leaderboard)),
        "variants": sorted(signal_panel["variant"].astype(str).unique()) if not signal_panel.empty else [],
        "signals": list(SIGNAL_COLUMNS),
        "primary_target": TARGET_COLUMN,
        "signal_panel_artifact": signal_panel_path.as_posix(),
        "daily_diagnostics_artifact": daily_path.as_posix(),
        "leaderboard_artifact": leaderboard_path.as_posix(),
        "correlation_artifact": correlation_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is signal diagnostics only, not a beta-matched long-short portfolio.",
            "Forward labels are validation-only adjusted open-to-open returns.",
            "The random-control signal is deterministic and used only as a sanity check.",
            "No test-window performance, model selection, or production claim is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_inputs(membership: pd.DataFrame, beta_panel: pd.DataFrame) -> None:
    required_membership = {"session_date", "symbol", "variant", "liquidity_rank"}
    missing_membership = sorted(required_membership.difference(membership.columns))
    if missing_membership:
        raise ValueError(f"Phase 1 membership is missing required columns: {missing_membership}")
    required_beta = {"session_date", "symbol", "beta", "beta_available"}
    missing_beta = sorted(required_beta.difference(beta_panel.columns))
    if missing_beta:
        raise ValueError(f"Phase 2 beta panel is missing required columns: {missing_beta}")


def _load_adjusted_prices(
    *,
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    symbols: Sequence[str],
    end_date: str,
) -> pd.DataFrame:
    symbol_set = set(symbols)
    daily = _load_jsonl_frames(
        daily_globs,
        columns=("session_date", "symbol", "open", "close"),
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
        frame["price_adjust_factor"],
        errors="coerce",
    ).fillna(1.0)
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["adjusted_open"] = frame["open"] * frame["price_adjust_factor"]
    frame["adjusted_close"] = frame["close"] * frame["price_adjust_factor"]
    frame = frame[
        frame["adjusted_open"].notna()
        & frame["adjusted_close"].notna()
        & (frame["adjusted_open"] > 0)
        & (frame["adjusted_close"] > 0)
    ]
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


def _base_feature_panel(
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    holding_period_sessions: int,
    benchmark_symbol: str,
) -> pd.DataFrame:
    stock = _stock_feature_frame(stock_prices, holding_period_sessions=holding_period_sessions)
    benchmark = _benchmark_feature_frame(
        benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        holding_period_sessions=holding_period_sessions,
    )
    frame = stock.merge(benchmark, on="session_date", how="left")
    frame["forward_market_relative_return_5d"] = (
        frame["forward_return_5d"] - frame["benchmark_forward_return_5d"]
    )
    return frame


def _stock_feature_frame(prices: pd.DataFrame, *, holding_period_sessions: int) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "session_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["return_1d"] = grouped["adjusted_close"].pct_change()
    frame["return_5d"] = grouped["adjusted_close"].pct_change(5)
    frame["return_20d"] = grouped["adjusted_close"].pct_change(20)
    frame["return_60d"] = grouped["adjusted_close"].pct_change(60)
    frame["volatility_20d"] = grouped["return_1d"].transform(
        lambda series: series.rolling(20, min_periods=10).std()
    )
    frame["next_adjusted_open"] = grouped["adjusted_open"].shift(-1)
    frame["exit_adjusted_open"] = grouped["adjusted_open"].shift(-(holding_period_sessions + 1))
    frame["forward_return_5d"] = frame["exit_adjusted_open"] / frame["next_adjusted_open"] - 1.0
    frame["reversal_5d"] = -frame["return_5d"]
    frame["momentum_20d"] = frame["return_20d"]
    frame["momentum_60d"] = frame["return_60d"]
    frame["vol_adjusted_momentum_20d"] = frame["return_20d"] / frame["volatility_20d"]
    frame.loc[~np.isfinite(frame["vol_adjusted_momentum_20d"]), "vol_adjusted_momentum_20d"] = np.nan
    return frame[
        [
            "session_date",
            "symbol",
            "return_1d",
            "return_5d",
            "return_20d",
            "return_60d",
            "volatility_20d",
            "reversal_5d",
            "momentum_20d",
            "momentum_60d",
            "vol_adjusted_momentum_20d",
            "forward_return_5d",
        ]
    ]


def _benchmark_feature_frame(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str,
    holding_period_sessions: int,
) -> pd.DataFrame:
    frame = prices[prices["symbol"].astype(str) == benchmark_symbol].sort_values("session_date").copy()
    frame["benchmark_return_20d"] = frame["adjusted_close"].pct_change(20)
    frame["benchmark_next_adjusted_open"] = frame["adjusted_open"].shift(-1)
    frame["benchmark_exit_adjusted_open"] = frame["adjusted_open"].shift(
        -(holding_period_sessions + 1)
    )
    frame["benchmark_forward_return_5d"] = (
        frame["benchmark_exit_adjusted_open"] / frame["benchmark_next_adjusted_open"] - 1.0
    )
    return frame[["session_date", "benchmark_return_20d", "benchmark_forward_return_5d"]]


def _membership_signal_panel(
    membership: pd.DataFrame,
    beta_panel: pd.DataFrame,
    base_features: pd.DataFrame,
) -> pd.DataFrame:
    beta = beta_panel[["session_date", "symbol", "beta", "beta_available"]].copy()
    beta["beta_available"] = beta["beta_available"].map(_is_true)
    base = base_features.merge(beta, on=["session_date", "symbol"], how="left")
    base["beta_residual_momentum_20d"] = base["return_20d"] - (
        base["beta"] * base["benchmark_return_20d"]
    )
    base["forward_beta_residual_return_5d"] = base["forward_return_5d"] - (
        base["beta"] * base["benchmark_forward_return_5d"]
    )
    cols = [
        "session_date",
        "symbol",
        "beta",
        "beta_available",
        "reversal_5d",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        "forward_market_relative_return_5d",
        "forward_beta_residual_return_5d",
    ]
    panel = membership.merge(base[cols], on=["session_date", "symbol"], how="left")
    panel = panel[panel["beta_available"].map(_is_true)].copy()
    panel["random_control"] = [
        _deterministic_unit_random(session_date, symbol)
        for session_date, symbol in zip(panel["session_date"], panel["symbol"])
    ]
    panel["liquidity_rank_score"] = -pd.to_numeric(panel["liquidity_rank"], errors="coerce")
    return panel


def _add_cross_sectional_scores(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frame = group.copy()
        z_columns = []
        for column in (
            "reversal_5d",
            "momentum_20d",
            "beta_residual_momentum_20d",
            "vol_adjusted_momentum_20d",
        ):
            z_column = f"{column}_z"
            frame[z_column] = _zscore(frame[column])
            z_columns.append(z_column)
        frame["transparent_composite"] = frame[z_columns].mean(axis=1, skipna=True)
        for column in SIGNAL_COLUMNS:
            frame[f"{column}_percentile"] = frame[column].rank(pct=True)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else panel


def _daily_signal_diagnostics(
    panel: pd.DataFrame,
    *,
    min_cross_section: int,
    top_bottom_quantile: float,
) -> pd.DataFrame:
    rows = []
    for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=False):
        target = group[TARGET_COLUMN]
        for signal in SIGNAL_COLUMNS:
            subset = group[[signal, TARGET_COLUMN]].dropna()
            names = int(len(subset))
            if names < min_cross_section:
                continue
            top, bottom = _top_bottom_means(group, signal, target_col=TARGET_COLUMN, quantile=top_bottom_quantile)
            rows.append(
                {
                    "session_date": session_date,
                    "variant": variant,
                    "signal": signal,
                    "names": names,
                    "ic": float(subset[signal].corr(subset[TARGET_COLUMN], method="pearson")),
                    "rank_ic": float(subset[signal].corr(subset[TARGET_COLUMN], method="spearman")),
                    "top_mean_forward_beta_residual_return": top,
                    "bottom_mean_forward_beta_residual_return": bottom,
                    "top_minus_bottom_spread": top - bottom,
                    "long_leg_return": top,
                    "short_leg_return": -bottom,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _top_bottom_means(
    group: pd.DataFrame,
    signal: str,
    *,
    target_col: str,
    quantile: float,
) -> tuple[float, float]:
    ranked = group[[signal, target_col]].dropna().sort_values(signal, ascending=False)
    if ranked.empty:
        return float("nan"), float("nan")
    count = max(1, int(np.ceil(len(ranked) * quantile)))
    top = float(ranked.head(count)[target_col].mean())
    bottom = float(ranked.tail(count)[target_col].mean())
    return top, bottom


def _signal_leaderboard(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["variant", "signal"])
    rows = []
    for (variant, signal), group in daily.groupby(["variant", "signal"], sort=False):
        rows.append(
            {
                "variant": variant,
                "signal": signal,
                "sessions": int(len(group)),
                "mean_ic": float(group["ic"].mean()),
                "ic_ir": _information_ratio(group["ic"]),
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "rank_ic_ir": _information_ratio(group["rank_ic"]),
                "mean_top_minus_bottom_spread": float(group["top_minus_bottom_spread"].mean()),
                "spread_hit_rate": float((group["top_minus_bottom_spread"] > 0).mean()),
                "mean_long_leg_return": float(group["long_leg_return"].mean()),
                "mean_short_leg_return": float(group["short_leg_return"].mean()),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["variant", "mean_top_minus_bottom_spread", "mean_rank_ic"],
        ascending=[True, False, False],
    )


def _signal_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=False):
        corr = group[list(SIGNAL_COLUMNS)].corr(method="spearman")
        for left in SIGNAL_COLUMNS:
            for right in SIGNAL_COLUMNS:
                rows.append(
                    {
                        "variant": variant,
                        "signal_left": left,
                        "signal_right": right,
                        "spearman_corr": float(corr.loc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def _signal_memo(
    leaderboard: pd.DataFrame,
    *,
    holding_period_sessions: int,
    min_cross_section: int,
    top_bottom_quantile: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 3 Baseline Signal Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only baseline signal diagnostic packet. It does not construct "
        "a beta-matched portfolio, select a final model, or inspect test-window performance.",
        "",
        "## Label Contract",
        "",
        "- decision session: `T`",
        "- feature cut: available by `T` close",
        "- entry: adjusted open at `T+1`",
        f"- exit: adjusted open after `{holding_period_sessions}` holding sessions",
        "- primary diagnostic target: `forward_beta_residual_return_5d`",
        "",
        "## Diagnostic Settings",
        "",
        f"- minimum cross-section: `{min_cross_section}`",
        f"- top/bottom quantile: `{top_bottom_quantile}`",
        "",
        "## Leaderboard Snapshot",
        "",
        "| Variant | Signal | Sessions | Mean Rank IC | Mean Spread | Hit Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in leaderboard.head(24).iterrows():
        lines.append(
            f"| {row['variant']} | {row['signal']} | {row['sessions']} | "
            f"{row['mean_rank_ic']} | {row['mean_top_minus_bottom_spread']} | "
            f"{row['spread_hit_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Random control is included to detect accidental leakage or overfitting.",
            "- Signal ranking is diagnostic only; Phase 4 must still beta-match the long and short books.",
            "- Positive validation diagnostics are not production claims.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _information_ratio(series: pd.Series) -> float:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float(series.mean() / std)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / std


def _deterministic_unit_random(session_date: str, symbol: str) -> float:
    digest = hashlib.blake2b(f"{session_date}|{symbol}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _expand_globs(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(dict.fromkeys(paths))


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
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 3 baseline signals.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--beta-panel-path", default=str(DEFAULT_PHASE2_BETA_PANEL))
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
    parser.add_argument("--holding-period-sessions", type=int, default=5)
    parser.add_argument("--min-cross-section", type=int, default=20)
    parser.add_argument("--top-bottom-quantile", type=float, default=0.20)
    args = parser.parse_args(argv)

    result = build_phase3_signal_artifacts(
        membership_path=args.membership_path,
        beta_panel_path=args.beta_panel_path,
        daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
        adj_factor_globs=tuple(args.adj_factor_globs or DEFAULT_ADJ_FACTOR_GLOBS),
        benchmark_daily_globs=tuple(args.benchmark_daily_globs or DEFAULT_BENCHMARK_DAILY_GLOBS),
        benchmark_adj_factor_globs=tuple(
            args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS
        ),
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        holding_period_sessions=args.holding_period_sessions,
        min_cross_section=args.min_cross_section,
        top_bottom_quantile=args.top_bottom_quantile,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
