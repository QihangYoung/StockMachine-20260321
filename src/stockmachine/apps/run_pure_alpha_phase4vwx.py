from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN, _add_model_features
from stockmachine.apps.run_pure_alpha_phase4j import _add_selector_scores
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_SELECTOR,
    DEFAULT_SHORT_VARIANT,
    BASE_COLUMNS,
    _add_residual_targets,
    _group_exposure_matrix,
    _group_l1,
    _group_max_abs,
    _load_sec_sic_map,
    _zscore,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4vwx_multibeta_state_neutral_20260419"
DEFAULT_BENCHMARK_SYMBOL = "SPY"
DEFAULT_VALIDATION_PRICE_END = "2019-12-31"
MULTI_BETA_FEATURES = (
    "oto_beta_blend",
    "oto_beta_252d",
    "oto_beta_126d",
    "oto_beta_63d",
    "oto_downside_beta_252d",
    "oto_upside_beta_252d",
    "oto_stress_beta_252d",
    "oto_rebound_beta_252d",
)


def build_phase4vwx_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    long_score: str = DEFAULT_LONG_SCORE,
    short_selector: str = DEFAULT_SHORT_SELECTOR,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    sic2_soft_group_penalty: float = 25.0,
    multi_beta_penalty: float = 10.0,
    max_nmi_rows: int = 250_000,
    random_state: int = 260321,
) -> dict[str, Any]:
    """Run Phase4V/W/X multi-beta and state-conditional neutrality research."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_memo_path = output_dir / "phase4vwx_multibeta_state_neutral_plan_memo.md"
    phase_memo_path.write_text(_plan_memo(), encoding="utf-8")

    signal_keys = _load_signal_keys(signal_panel_path, (long_variant, short_variant))
    symbols = tuple(sorted(signal_keys["symbol"].unique()))
    price_bundle = _load_open_return_pairs(
        symbols=symbols,
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        benchmark_daily_globs=benchmark_daily_globs,
        benchmark_adj_factor_globs=benchmark_adj_factor_globs,
        benchmark_symbol=benchmark_symbol,
        end_date=validation_price_end,
    )
    multi_beta_panel = _build_multibeta_panel(signal_keys, price_bundle["return_pairs"])
    future_beta = _build_future_realized_beta_labels(signal_keys, price_bundle["return_pairs"])
    multibeta_with_labels = multi_beta_panel.merge(
        future_beta,
        on=["session_date", "symbol"],
        how="left",
    )
    phase4v_summary = _phase4v_coverage_summary(multi_beta_panel, signal_keys)

    phase4w_feature_diagnostics = _phase4w_feature_diagnostics(
        multibeta_with_labels,
        max_rows=max_nmi_rows,
        random_state=random_state,
    )
    phase4w_lead_exposures = _phase4w_existing_lead_exposures(
        multi_beta_panel,
        positions_path=RESEARCH_ROOT
        / "phase4s_sector_industry_mf_residual_v2_20260419"
        / "phase4s_positions_validation.csv.gz",
    )

    phase4v_panel_path = output_dir / "phase4v_multibeta_panel_validation.csv.gz"
    phase4v_summary_path = output_dir / "phase4v_multibeta_coverage_summary_validation.csv"
    phase4w_feature_path = output_dir / "phase4w_multibeta_future_realized_diagnostics.csv"
    phase4w_exposure_path = output_dir / "phase4w_existing_lead_multibeta_exposures.csv"
    multi_beta_panel.to_csv(phase4v_panel_path, index=False, compression="gzip")
    phase4v_summary.to_csv(phase4v_summary_path, index=False)
    phase4w_feature_diagnostics.to_csv(phase4w_feature_path, index=False)
    phase4w_lead_exposures.to_csv(phase4w_exposure_path, index=False)

    phase4x = _build_phase4x_portfolio(
        signal_panel_path=signal_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        multi_beta_panel=multi_beta_panel,
        output_dir=output_dir,
        long_variant=long_variant,
        short_variant=short_variant,
        long_score=long_score,
        short_selector=short_selector,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        sic2_soft_group_penalty=sic2_soft_group_penalty,
        multi_beta_penalty=multi_beta_penalty,
    )

    phase4vwx_memo_path = output_dir / "phase4vwx_multibeta_state_neutral_results_memo.md"
    rollup_path = output_dir / "phase4vwx_rollup.json"

    phase4vwx_memo_path.write_text(
        _results_memo(
            phase4v_summary=phase4v_summary,
            phase4w_feature_diagnostics=phase4w_feature_diagnostics,
            phase4w_lead_exposures=phase4w_lead_exposures,
            phase4x_summary=phase4x["summary"],
            phase4x_exposure_summary=phase4x["exposure_summary"],
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "benchmark_symbol": benchmark_symbol,
        "validation_price_end": validation_price_end,
        "long_variant": long_variant,
        "short_variant": short_variant,
        "multi_beta_features": list(MULTI_BETA_FEATURES),
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "sic2_soft_group_penalty": float(sic2_soft_group_penalty),
        "multi_beta_penalty": float(multi_beta_penalty),
        "signal_key_rows": int(len(signal_keys)),
        "symbols": int(len(symbols)),
        "return_pair_rows": int(len(price_bundle["return_pairs"])),
        "phase4v_panel_rows": int(len(multi_beta_panel)),
        "phase4x_constructed_sessions": int(len(phase4x["daily"])),
        "phase4x_skipped_sessions": int(len(phase4x["skipped"])),
        "plan_memo_artifact": phase_memo_path.as_posix(),
        "phase4v_panel_artifact": phase4v_panel_path.as_posix(),
        "phase4v_summary_artifact": phase4v_summary_path.as_posix(),
        "phase4w_feature_diagnostics_artifact": phase4w_feature_path.as_posix(),
        "phase4w_existing_lead_exposures_artifact": phase4w_exposure_path.as_posix(),
        "phase4x_positions_artifact": phase4x["positions_path"].as_posix(),
        "phase4x_daily_artifact": phase4x["daily_path"].as_posix(),
        "phase4x_summary_artifact": phase4x["summary_path"].as_posix(),
        "phase4x_exposure_summary_artifact": phase4x["exposure_summary_path"].as_posix(),
        "results_memo_artifact": phase4vwx_memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "multi_beta_estimation_future_realized_beta_diagnostics_and_multibeta_soft_neutral_lp",
        "limitations": [
            "Open-to-open multi-beta estimates are lagged and validation-only.",
            "Future realized beta labels are diagnostics only and are not fed into Phase4X.",
            "Phase4X evaluation is h10 forward-return diagnostics, not a refreshed daily multi-sleeve path.",
            "SIC sector labels are SEC SIC proxies, not GICS.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_signal_keys(signal_panel_path: str | Path, variants: Sequence[str]) -> pd.DataFrame:
    keys = pd.read_csv(
        signal_panel_path,
        usecols=["session_date", "variant", "symbol"],
        low_memory=False,
    )
    keys["session_date"] = pd.to_datetime(keys["session_date"]).dt.date.astype(str)
    keys["variant"] = keys["variant"].astype(str)
    keys["symbol"] = keys["symbol"].astype(str)
    keys = keys[keys["variant"].isin(set(variants))].copy()
    return keys[["session_date", "symbol"]].drop_duplicates().sort_values(
        ["symbol", "session_date"]
    ).reset_index(drop=True)


def _load_open_return_pairs(
    *,
    symbols: Sequence[str],
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    benchmark_daily_globs: Sequence[str | Path],
    benchmark_adj_factor_globs: Sequence[str | Path],
    benchmark_symbol: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    stock_prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=end_date,
    )
    benchmark_prices = _load_adjusted_prices(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=end_date,
    )
    stock_returns = _open_to_open_returns(stock_prices, "symbol_return")
    benchmark_returns = _open_to_open_returns(benchmark_prices, "benchmark_return")[
        ["session_date", "benchmark_return"]
    ]
    return_pairs = stock_returns.merge(benchmark_returns, on="session_date", how="inner")
    return {
        "stock_prices": stock_prices,
        "benchmark_prices": benchmark_prices,
        "return_pairs": return_pairs.sort_values(["symbol", "session_date"]).reset_index(drop=True),
    }


def _open_to_open_returns(prices: pd.DataFrame, return_column: str) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "session_date"]).copy()
    frame[return_column] = frame.groupby("symbol")["adjusted_open"].pct_change()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame[["session_date", "symbol", return_column]].dropna()


def _build_multibeta_panel(signal_keys: pd.DataFrame, return_pairs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, group in return_pairs.groupby("symbol", sort=False):
        frames.append(_multibeta_for_symbol(symbol, group))
    estimates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    panel = signal_keys.merge(estimates, on=["session_date", "symbol"], how="left")
    panel["oto_beta_blend"] = (
        0.50 * panel["oto_beta_252d"]
        + 0.30 * panel["oto_beta_126d"]
        + 0.20 * panel["oto_beta_63d"]
    )
    panel["oto_beta_instability"] = (panel["oto_beta_63d"] - panel["oto_beta_252d"]).abs()
    panel["oto_down_up_beta_gap"] = panel["oto_downside_beta_252d"] - panel["oto_upside_beta_252d"]
    panel["multibeta_available"] = panel[list(MULTI_BETA_FEATURES)].notna().all(axis=1)
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _multibeta_for_symbol(symbol: str, group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("session_date").reset_index(drop=True)
    x = pd.to_numeric(ordered["benchmark_return"], errors="coerce").shift(1).to_numpy(dtype=float)
    y = pd.to_numeric(ordered["symbol_return"], errors="coerce").shift(1).to_numpy(dtype=float)
    rows = {
        "session_date": ordered["session_date"].to_numpy(),
        "symbol": symbol,
    }
    for window, min_obs in ((252, 126), (126, 63), (63, 42)):
        beta, obs = _rolling_beta_arrays(x, y, window=window, min_obs=min_obs)
        rows[f"oto_beta_{window}d"] = beta
        rows[f"oto_beta_{window}d_obs"] = obs
    for label, state, min_obs in (
        ("downside", "downside", 42),
        ("upside", "upside", 42),
        ("stress", "stress", 25),
        ("rebound", "rebound", 25),
    ):
        beta, obs = _rolling_state_beta_arrays(
            x,
            y,
            window=252,
            min_obs=min_obs,
            state=state,
        )
        rows[f"oto_{label}_beta_252d"] = beta
        rows[f"oto_{label}_beta_252d_obs"] = obs
    return pd.DataFrame(rows)


def _rolling_beta_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.full(len(x), np.nan, dtype=float)
    obs_out = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        xw = x[start : i + 1]
        yw = y[start : i + 1]
        value, obs = _beta_from_arrays(xw, yw, min_obs=min_obs)
        beta[i] = value
        obs_out[i] = obs
    return beta, obs_out


def _rolling_state_beta_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int,
    min_obs: int,
    state: str,
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.full(len(x), np.nan, dtype=float)
    obs_out = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        xw = x[start : i + 1]
        yw = y[start : i + 1]
        valid = np.isfinite(xw) & np.isfinite(yw)
        xv = xw[valid]
        yv = yw[valid]
        if len(xv) == 0:
            continue
        if state == "downside":
            mask = xv < 0.0
        elif state == "upside":
            mask = xv > 0.0
        elif state == "stress":
            mask = xv <= np.nanquantile(xv, 0.20)
        elif state == "rebound":
            mask = xv >= np.nanquantile(xv, 0.80)
        else:
            raise ValueError(f"Unknown state: {state}")
        value, obs = _beta_from_arrays(xv[mask], yv[mask], min_obs=min_obs)
        beta[i] = value
        obs_out[i] = obs
    return beta, obs_out


def _beta_from_arrays(x: np.ndarray, y: np.ndarray, *, min_obs: int) -> tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]
    obs = int(len(xv))
    if obs < min_obs:
        return np.nan, obs
    var = float(np.var(xv, ddof=0))
    if var <= 0:
        return np.nan, obs
    cov = float(np.mean((xv - xv.mean()) * (yv - yv.mean())))
    raw = cov / var
    shrunk = 0.90 * raw + 0.10 * 1.0
    return float(np.clip(shrunk, 0.0, 3.0)), obs


def _build_future_realized_beta_labels(
    signal_keys: pd.DataFrame,
    return_pairs: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for symbol, group in return_pairs.groupby("symbol", sort=False):
        frames.append(_future_realized_for_symbol(symbol, group))
    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return signal_keys.merge(labels, on=["session_date", "symbol"], how="left")[
        [
            "session_date",
            "symbol",
            "future_20d_realized_beta",
            "future_60d_realized_beta",
            "future_20d_spy_return",
            "future_60d_spy_return",
        ]
    ]


def _future_realized_for_symbol(symbol: str, group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("session_date").reset_index(drop=True)
    x = pd.to_numeric(ordered["benchmark_return"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(ordered["symbol_return"], errors="coerce").to_numpy(dtype=float)
    rows: dict[str, Any] = {
        "session_date": ordered["session_date"].to_numpy(),
        "symbol": symbol,
    }
    for horizon in (20, 60):
        beta = np.full(len(ordered), np.nan, dtype=float)
        spy_return = np.full(len(ordered), np.nan, dtype=float)
        for i in range(len(ordered)):
            start = i + 2
            stop = min(len(ordered), start + horizon)
            if stop - start < max(10, horizon // 2):
                continue
            xw = x[start:stop]
            yw = y[start:stop]
            value, obs = _beta_from_arrays(xw, yw, min_obs=max(10, horizon // 2))
            if obs:
                beta[i] = value
                spy_return[i] = float(np.prod(1.0 + xw[np.isfinite(xw)]) - 1.0)
        rows[f"future_{horizon}d_realized_beta"] = beta
        rows[f"future_{horizon}d_spy_return"] = spy_return
    return pd.DataFrame(rows)


def _phase4v_coverage_summary(
    multi_beta_panel: pd.DataFrame,
    signal_keys: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    available = multi_beta_panel["multibeta_available"]
    rows.append(
        {
            "scope": "all_signal_keys",
            "rows": int(len(multi_beta_panel)),
            "symbols": int(signal_keys["symbol"].nunique()),
            "start": str(multi_beta_panel["session_date"].min()),
            "end": str(multi_beta_panel["session_date"].max()),
            "multibeta_available_rows": int(available.sum()),
            "multibeta_coverage": float(available.mean()) if len(available) else 0.0,
            "median_oto_beta_blend": float(multi_beta_panel["oto_beta_blend"].median()),
            "median_oto_beta_instability": float(
                multi_beta_panel["oto_beta_instability"].median()
            ),
            "test_window_used": False,
        }
    )
    return pd.DataFrame(rows)


def _phase4w_feature_diagnostics(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    columns = [
        *MULTI_BETA_FEATURES,
        "oto_beta_instability",
        "oto_down_up_beta_gap",
        "future_20d_realized_beta",
        "future_60d_realized_beta",
        "future_20d_spy_return",
        "future_60d_spy_return",
    ]
    sample = frame[["session_date", "symbol", *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_state)
    rows = []
    for target, state_col in (
        ("future_20d_realized_beta", "future_20d_spy_return"),
        ("future_60d_realized_beta", "future_60d_spy_return"),
    ):
        state_masks = {
            "all": pd.Series(True, index=sample.index),
            "future_up": sample[state_col] > 0,
            "future_down": sample[state_col] < 0,
            "future_stress": sample[state_col] <= sample[state_col].quantile(0.20),
            "future_rebound": sample[state_col] >= sample[state_col].quantile(0.80),
        }
        for state, mask in state_masks.items():
            state_sample = sample[mask]
            if state_sample.empty:
                continue
            for feature in (
                *MULTI_BETA_FEATURES,
                "oto_beta_instability",
                "oto_down_up_beta_gap",
            ):
                pair = state_sample[[feature, target]].dropna()
                if len(pair) < 100:
                    continue
                err = pair[feature] - pair[target]
                rows.append(
                    {
                        "target": target,
                        "future_state": state,
                        "feature": feature,
                        "rows": int(len(pair)),
                        "pearson_corr": float(pair[feature].corr(pair[target], method="pearson")),
                        "spearman_corr": float(pair[feature].corr(pair[target], method="spearman")),
                        "mean_error": float(err.mean()),
                        "mean_abs_error": float(err.abs().mean()),
                        "rmse": float(np.sqrt(np.mean(np.square(err)))),
                        "target_mean": float(pair[target].mean()),
                        "feature_mean": float(pair[feature].mean()),
                        "test_window_used": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["target", "future_state", "mean_abs_error", "feature"],
        ascending=[True, True, True, True],
    )


def _phase4w_existing_lead_exposures(
    multi_beta_panel: pd.DataFrame,
    *,
    positions_path: str | Path,
) -> pd.DataFrame:
    if not Path(positions_path).exists():
        return pd.DataFrame()
    positions = pd.read_csv(positions_path, low_memory=False)
    positions = positions[positions["portfolio"].astype(str) == "sic2_soft_neutral"].copy()
    positions["session_date"] = pd.to_datetime(positions["session_date"]).dt.date.astype(str)
    merged = positions.merge(multi_beta_panel, on=["session_date", "symbol"], how="left")
    windows = _diagnostic_windows()
    rows = []
    for name, start, end in windows:
        window = merged[(merged["session_date"] >= start) & (merged["session_date"] <= end)]
        if window.empty:
            continue
        row: dict[str, Any] = {
            "window": name,
            "start": start,
            "end": end,
            "position_rows": int(len(window)),
            "signal_sessions": int(window["session_date"].nunique()),
            "test_window_used": False,
        }
        for feature in MULTI_BETA_FEATURES:
            row[f"net_{feature}"] = float(
                (window["signed_weight"] * window[feature]).sum()
                / max(1, window["session_date"].nunique())
            )
            for side in ("long", "short"):
                side_frame = window[window["side"].astype(str) == side]
                row[f"{side}_{feature}"] = float(
                    (side_frame["side_weight"] * side_frame[feature]).sum()
                    / max(1, side_frame["session_date"].nunique())
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_phase4x_portfolio(
    *,
    signal_panel_path: str | Path,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    multi_beta_panel: pd.DataFrame,
    output_dir: Path,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    sic2_soft_group_penalty: float,
    multi_beta_penalty: float,
) -> dict[str, Any]:
    panel = _load_phase4x_panel(
        signal_panel_path=signal_panel_path,
        long_variant=long_variant,
        short_variant=short_variant,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        multi_beta_panel=multi_beta_panel,
    )
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for session_date in sessions:
        long_group = groups.get((long_variant, session_date))
        short_group = groups.get((short_variant, session_date))
        if long_group is None or short_group is None:
            skipped.append(_phase4x_skip(session_date, "missing_long_or_short_group", 0, 0))
            continue
        book, diagnostic, skip = _construct_phase4x_one_session(
            long_group,
            short_group,
            session_date=session_date,
            long_variant=long_variant,
            short_variant=short_variant,
            long_score=long_score,
            short_selector=short_selector,
            candidate_pool_per_side=candidate_pool_per_side,
            max_single_name_side_weight=max_single_name_side_weight,
            min_nonzero_names=min_nonzero_names,
            score_weight=score_weight,
            sic2_soft_group_penalty=sic2_soft_group_penalty,
            multi_beta_penalty=multi_beta_penalty,
        )
        positions.extend(book)
        if diagnostic is not None:
            daily_rows.append(diagnostic)
        if skip is not None:
            skipped.append(skip)
    positions_df = pd.DataFrame(positions)
    daily = pd.DataFrame(daily_rows)
    skipped_df = pd.DataFrame(skipped)
    summary = _phase4x_summary(daily, skipped_df)
    exposure_summary = _phase4x_window_exposure_summary(daily)

    positions_path = output_dir / "phase4x_multibeta_positions_validation.csv.gz"
    daily_path = output_dir / "phase4x_multibeta_daily_validation.csv"
    skipped_path = output_dir / "phase4x_multibeta_skipped_validation.csv"
    summary_path = output_dir / "phase4x_multibeta_summary_validation.csv"
    exposure_summary_path = output_dir / "phase4x_multibeta_window_exposure_summary.csv"
    positions_df.to_csv(positions_path, index=False, compression="gzip")
    daily.to_csv(daily_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    summary.to_csv(summary_path, index=False)
    exposure_summary.to_csv(exposure_summary_path, index=False)
    return {
        "positions": positions_df,
        "daily": daily,
        "skipped": skipped_df,
        "summary": summary,
        "exposure_summary": exposure_summary,
        "positions_path": positions_path,
        "daily_path": daily_path,
        "skipped_path": skipped_path,
        "summary_path": summary_path,
        "exposure_summary_path": exposure_summary_path,
    }


def _load_phase4x_panel(
    *,
    signal_panel_path: str | Path,
    long_variant: str,
    short_variant: str,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    multi_beta_panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = pd.read_csv(signal_panel_path, usecols=list(BASE_COLUMNS), low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[panel["variant"].isin({long_variant, short_variant})].copy()
    for column in BASE_COLUMNS:
        if column not in ("session_date", "variant", "symbol"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["beta", "forward_return_5d", TARGET_COLUMN])
    panel = _add_model_features(panel)
    scored = [_add_selector_scores(group) for _, group in panel.groupby(["variant", "session_date"])]
    panel = pd.concat(scored, ignore_index=True) if scored else panel
    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_residual_targets(panel, min_regression_rows=80, min_dummy_count=5)
    beta_cols = ["session_date", "symbol", *MULTI_BETA_FEATURES, "oto_beta_instability"]
    panel = panel.merge(multi_beta_panel[beta_cols], on=["session_date", "symbol"], how="left")
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _construct_phase4x_one_session(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    sic2_soft_group_penalty: float,
    multi_beta_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    needed = [
        "symbol",
        "beta",
        long_score,
        short_selector,
        "forward_return_5d",
        TARGET_COLUMN,
        "sic2_sector",
        "sic4_industry",
        "sic_description",
        "sic2_neutral_beta_residual",
        "sic4_neutral_beta_residual",
        "risk_factor_sic2_residual",
        "style_factor_sic2_residual",
        *MULTI_BETA_FEATURES,
    ]
    longs = long_group.dropna(subset=[c for c in needed if c in long_group.columns]).drop_duplicates("symbol")
    shorts = short_group.dropna(subset=[c for c in needed if c in short_group.columns]).drop_duplicates("symbol")
    if len(longs) < min_nonzero_names or len(shorts) < min_nonzero_names:
        return [], None, _phase4x_skip(session_date, "insufficient_candidates", len(longs), len(shorts))
    longs = longs.sort_values([long_score, "symbol"], ascending=[False, True]).head(candidate_pool_per_side)
    short_pool = shorts.loc[~shorts["symbol"].isin(set(longs["symbol"]))]
    shorts = short_pool.sort_values([short_selector, "symbol"], ascending=[False, True]).head(
        candidate_pool_per_side
    )
    if len(longs) * max_single_name_side_weight < 1.0 or len(shorts) * max_single_name_side_weight < 1.0:
        return [], None, _phase4x_skip(session_date, "insufficient_weight_capacity", len(longs), len(shorts))
    try:
        long_weights, short_weights = _optimize_phase4x_weights(
            longs,
            shorts,
            long_score=long_score,
            short_selector=short_selector,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            sic2_soft_group_penalty=sic2_soft_group_penalty,
            multi_beta_penalty=multi_beta_penalty,
        )
    except ValueError as exc:
        return [], None, _phase4x_skip(session_date, f"solver_failed:{exc}", len(longs), len(shorts))
    if (long_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _phase4x_skip(session_date, "insufficient_long_nonzero_names", len(longs), len(shorts))
    if (short_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _phase4x_skip(session_date, "insufficient_short_nonzero_names", len(longs), len(shorts))
    book = [
        *_phase4x_position_rows(
            longs,
            long_weights,
            session_date=session_date,
            long_variant=long_variant,
            short_variant=short_variant,
            side="long",
            score_column=long_score,
        ),
        *_phase4x_position_rows(
            shorts,
            short_weights,
            session_date=session_date,
            long_variant=long_variant,
            short_variant=short_variant,
            side="short",
            score_column=short_selector,
        ),
    ]
    diagnostic = _phase4x_diagnostic_row(longs, shorts, long_weights, short_weights, session_date)
    return book, diagnostic, None


def _optimize_phase4x_weights(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    long_score: str,
    short_selector: str,
    max_weight: float,
    score_weight: float,
    sic2_soft_group_penalty: float,
    multi_beta_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_long = len(longs)
    n_short = len(shorts)
    n = n_long + n_short
    score = np.concatenate(
        [
            _zscore(longs[long_score].to_numpy(dtype=float)),
            _zscore(shorts[short_selector].to_numpy(dtype=float)),
        ]
    )
    aeq = []
    beq = []
    row = np.zeros(n)
    row[:n_long] = 1.0
    aeq.append(row)
    beq.append(1.0)
    row = np.zeros(n)
    row[n_long:] = 1.0
    aeq.append(row)
    beq.append(1.0)
    row = np.zeros(n)
    row[:n_long] = longs["beta"].to_numpy(dtype=float)
    row[n_long:] = -shorts["beta"].to_numpy(dtype=float)
    aeq.append(row)
    beq.append(0.0)
    a_eq = np.vstack(aeq)
    b_eq = np.array(beq)
    soft_blocks = []
    penalties = []
    sic2_matrix = _group_exposure_matrix(longs, shorts, "sic2_sector")
    if len(sic2_matrix):
        soft_blocks.append(sic2_matrix)
        penalties.extend([sic2_soft_group_penalty] * sic2_matrix.shape[0])
    for feature in MULTI_BETA_FEATURES:
        row = np.zeros(n)
        row[:n_long] = longs[feature].to_numpy(dtype=float)
        row[n_long:] = -shorts[feature].to_numpy(dtype=float)
        soft_blocks.append(row.reshape(1, -1))
        penalties.append(multi_beta_penalty)
    soft_matrix = np.vstack(soft_blocks) if soft_blocks else np.zeros((0, n))
    c = -score_weight * score
    bounds: list[tuple[float, float | None]] = [(0.0, max_weight)] * n
    a_ub = None
    b_ub = None
    if len(soft_matrix):
        m = soft_matrix.shape[0]
        c = np.concatenate([c, np.asarray(penalties, dtype=float)])
        bounds.extend([(0.0, None)] * m)
        a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], m))])
        top = np.column_stack([soft_matrix, -np.eye(m)])
        bottom = np.column_stack([-soft_matrix, -np.eye(m)])
        a_ub = np.vstack([top, bottom])
        b_ub = np.zeros(2 * m)
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(str(result.message).replace(" ", "_"))
    x = np.clip(result.x[:n], 0.0, max_weight)
    long_weights = x[:n_long]
    short_weights = x[n_long:]
    if abs(long_weights.sum() - 1.0) > 1e-6 or abs(short_weights.sum() - 1.0) > 1e-6:
        raise ValueError("side_sum_constraint_breach")
    net_beta = float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"]))
    if abs(net_beta) > 1e-5:
        raise ValueError("beta_constraint_breach")
    return long_weights, short_weights


def _phase4x_diagnostic_row(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    session_date: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "session_date": session_date,
        "portfolio": "phase4x_sic2_multibeta_soft_neutral",
        "long_count": int((long_weights > 1e-12).sum()),
        "short_count": int((short_weights > 1e-12).sum()),
        "long_beta": float(np.dot(long_weights, longs["beta"])),
        "short_beta": float(np.dot(short_weights, shorts["beta"])),
        "net_beta": float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"])),
        "sic2_l1_exposure": _group_l1(longs, shorts, long_weights, short_weights, "sic2_sector"),
        "sic2_max_abs_exposure": _group_max_abs(
            longs, shorts, long_weights, short_weights, "sic2_sector"
        ),
        "sic4_l1_exposure": _group_l1(longs, shorts, long_weights, short_weights, "sic4_industry"),
        "test_window_used": False,
    }
    for feature in MULTI_BETA_FEATURES:
        long_value = float(np.dot(long_weights, longs[feature]))
        short_value = float(np.dot(short_weights, shorts[feature]))
        out[f"long_{feature}"] = long_value
        out[f"short_{feature}"] = short_value
        out[f"net_{feature}"] = long_value - short_value
    for label, column in (
        ("raw", "forward_return_5d"),
        ("beta_residual", TARGET_COLUMN),
        ("sic2_residual", "sic2_neutral_beta_residual"),
        ("sic4_residual", "sic4_neutral_beta_residual"),
        ("risk_factor_sic2_residual", "risk_factor_sic2_residual"),
        ("style_factor_sic2_residual", "style_factor_sic2_residual"),
    ):
        long_value = float(np.dot(long_weights, longs[column]))
        short_value = float(-np.dot(short_weights, shorts[column]))
        out[f"long_{label}"] = long_value
        out[f"short_{label}_contribution"] = short_value
        out[f"spread_{label}"] = long_value + short_value
    return out


def _phase4x_position_rows(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    *,
    session_date: str,
    long_variant: str,
    short_variant: str,
    side: str,
    score_column: str,
) -> list[dict[str, Any]]:
    sign = 1.0 if side == "long" else -1.0
    rows = []
    for (_, row), weight in zip(candidates.iterrows(), weights):
        if weight <= 1e-12:
            continue
        out = {
            "session_date": session_date,
            "portfolio": "phase4x_sic2_multibeta_soft_neutral",
            "long_variant": long_variant,
            "short_variant": short_variant,
            "side": side,
            "symbol": row["symbol"],
            "score_column": score_column,
            "score": float(row[score_column]),
            "beta": float(row["beta"]),
            "sic2_sector": row["sic2_sector"],
            "sic4_industry": row["sic4_industry"],
            "sic_description": row["sic_description"],
            "side_weight": float(weight),
            "signed_weight": float(sign * weight),
            "forward_return_5d": float(row["forward_return_5d"]),
            "forward_beta_residual_return_5d": float(row[TARGET_COLUMN]),
            "test_window_used": False,
        }
        for feature in MULTI_BETA_FEATURES:
            out[feature] = float(row[feature])
        rows.append(out)
    return rows


def _phase4x_summary(daily: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    requested = len(daily) + len(skipped)
    if daily.empty:
        return pd.DataFrame()
    row: dict[str, Any] = {
        "portfolio": "phase4x_sic2_multibeta_soft_neutral",
        "requested_sessions": int(requested),
        "constructed_sessions": int(len(daily)),
        "construction_rate": float(len(daily) / requested) if requested else np.nan,
        "mean_abs_net_beta": float(daily["net_beta"].abs().mean()),
        "mean_sic2_l1_exposure": float(daily["sic2_l1_exposure"].mean()),
        "mean_sic2_max_abs_exposure": float(daily["sic2_max_abs_exposure"].mean()),
        "mean_sic4_l1_exposure": float(daily["sic4_l1_exposure"].mean()),
        "median_long_count": float(daily["long_count"].median()),
        "median_short_count": float(daily["short_count"].median()),
        "test_window_used": False,
    }
    for feature in MULTI_BETA_FEATURES:
        row[f"mean_abs_net_{feature}"] = float(daily[f"net_{feature}"].abs().mean())
    for label in (
        "raw",
        "beta_residual",
        "sic2_residual",
        "sic4_residual",
        "risk_factor_sic2_residual",
        "style_factor_sic2_residual",
    ):
        row[f"mean_spread_{label}"] = float(daily[f"spread_{label}"].mean())
        row[f"hit_rate_{label}"] = float((daily[f"spread_{label}"] > 0).mean())
        row[f"mean_long_{label}"] = float(daily[f"long_{label}"].mean())
        row[f"mean_short_{label}_contribution"] = float(
            daily[f"short_{label}_contribution"].mean()
        )
    return pd.DataFrame([row])


def _phase4x_window_exposure_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for name, start, end in _diagnostic_windows():
        window = daily[(daily["session_date"] >= start) & (daily["session_date"] <= end)]
        if window.empty:
            continue
        row: dict[str, Any] = {
            "window": name,
            "start": start,
            "end": end,
            "sessions": int(len(window)),
            "mean_spread_raw": float(window["spread_raw"].mean()),
            "mean_abs_net_beta": float(window["net_beta"].abs().mean()),
            "mean_sic2_l1_exposure": float(window["sic2_l1_exposure"].mean()),
            "test_window_used": False,
        }
        for feature in MULTI_BETA_FEATURES:
            row[f"mean_abs_net_{feature}"] = float(window[f"net_{feature}"].abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _phase4x_skip(session_date: str, reason: str, long_candidates: int, short_candidates: int) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "portfolio": "phase4x_sic2_multibeta_soft_neutral",
        "long_candidates": int(long_candidates),
        "short_candidates": int(short_candidates),
        "reason": reason,
        "test_window_used": False,
    }


def _diagnostic_windows() -> list[tuple[str, str, str]]:
    return [
        ("full", "2014-08-05", "2019-12-31"),
        ("2018H1", "2018-01-01", "2018-06-30"),
        ("2018Q1_stress", "2018-01-26", "2018-04-02"),
        ("2018Q2_rebound", "2018-04-01", "2018-06-30"),
        ("2019_mid", "2019-05-01", "2019-08-31"),
        ("2019_may_drawdown", "2019-05-01", "2019-06-07"),
        ("2019_jun_jul_rebound", "2019-06-07", "2019-07-31"),
        ("2019_aug_drawdown", "2019-08-01", "2019-08-31"),
    ]


def _plan_memo() -> str:
    return f"""# Phase4V/W/X Multi-Beta and Market-State Neutralization Plan

Generated: {_utc_now()}

## Motivation

The Phase4U comovement diagnosis showed that the lead `sic2_soft_neutral` portfolio is ex-ante beta neutral but can still realize positive market beta in specific windows, especially around 2019 May/August. The likely failure mode is not a broken beta constraint; it is that a single historical beta is too thin a description of how long and short books behave across market states.

## Phase4V: Multi-Beta Estimation

Build lagged open-to-open beta estimates that are closer to the daily PnL path than the current close-to-close beta:

- 252d, 126d, and 63d ordinary beta;
- 252d downside beta, using trailing SPY-down days;
- 252d upside beta, using trailing SPY-up days;
- 252d stress beta, using the trailing bottom 20% SPY-return days;
- 252d rebound beta, using the trailing top 20% SPY-return days;
- beta blend and beta instability.

All estimates are lagged by one session and use validation-only price history. No future labels enter portfolio construction.

## Phase4W: Future Realized Beta Diagnostics

Use future 20/60-session realized beta only as a diagnostic label. Measure which ex-ante beta estimate best predicts realized beta overall and in future up/down/stress/rebound states. Also measure the existing Phase4S lead's net multi-beta exposures.

## Phase4X: Multi-Beta Soft-Neutral Construction

Keep the current Phase4S structure:

- long universe: `top1000_clean_core_beta_full`;
- short universe: `adv30m_clean_core_beta_full`;
- long score: `reversal_5d`;
- short score: `short_core_plus_overextension`;
- exact original beta neutrality;
- SIC2 soft neutrality.

Add soft penalties for net exposures to the multi-beta features. This does not require predicting tomorrow's market state; it tries to make the book robust across several possible states at the same time.

## Guardrails

- Test lockbox remains closed.
- Future realized beta labels are diagnostics only.
- This first pass evaluates Phase4X on h10 forward labels; a refreshed daily multi-sleeve path can be added after we decide whether the h10 tradeoff is promising.
"""


def _results_memo(
    *,
    phase4v_summary: pd.DataFrame,
    phase4w_feature_diagnostics: pd.DataFrame,
    phase4w_lead_exposures: pd.DataFrame,
    phase4x_summary: pd.DataFrame,
    phase4x_exposure_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Phase4V/W/X Multi-Beta and Market-State Neutralization Results",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only results. No test-window performance is computed.",
        "",
        "## Phase4V Coverage",
        "",
        _markdown_table(phase4v_summary),
        "",
        "## Phase4W Best Future Realized-Beta Proxies",
        "",
        _markdown_table(
            phase4w_feature_diagnostics.sort_values(
                ["target", "future_state", "mean_abs_error"]
            ).groupby(["target", "future_state"]).head(3)
        ),
        "",
        "## Existing Lead Multi-Beta Exposures",
        "",
        _markdown_table(phase4w_lead_exposures),
        "",
        "## Phase4X H10 Summary",
        "",
        _markdown_table(phase4x_summary),
        "",
        "## Phase4X Window Exposures",
        "",
        _markdown_table(phase4x_exposure_summary),
        "",
        "## Interpretation",
        "",
        "Phase4X should be judged first by whether it reduces state-conditional beta exposures without destroying h10 alpha. If it passes this first gate, the next step is a refreshed daily multi-sleeve PnL comparison against Phase4S lead and SPY.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "No rows."
    frame = df.head(max_rows).copy()
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase4V/W/X multi-beta and state-neutralization research."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--long-variant", default=DEFAULT_LONG_VARIANT)
    parser.add_argument("--short-variant", default=DEFAULT_SHORT_VARIANT)
    parser.add_argument("--long-score", default=DEFAULT_LONG_SCORE)
    parser.add_argument("--short-selector", default=DEFAULT_SHORT_SELECTOR)
    parser.add_argument("--candidate-pool-per-side", type=int, default=80)
    parser.add_argument("--max-single-name-side-weight", type=float, default=1.0 / 30.0)
    parser.add_argument("--min-nonzero-names", type=int, default=20)
    parser.add_argument("--score-weight", type=float, default=0.01)
    parser.add_argument("--sic2-soft-group-penalty", type=float, default=25.0)
    parser.add_argument("--multi-beta-penalty", type=float, default=10.0)
    args = parser.parse_args(argv)
    result = build_phase4vwx_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        validation_price_end=args.validation_price_end,
        long_variant=args.long_variant,
        short_variant=args.short_variant,
        long_score=args.long_score,
        short_selector=args.short_selector,
        candidate_pool_per_side=args.candidate_pool_per_side,
        max_single_name_side_weight=args.max_single_name_side_weight,
        min_nonzero_names=args.min_nonzero_names,
        score_weight=args.score_weight,
        sic2_soft_group_penalty=args.sic2_soft_group_penalty,
        multi_beta_penalty=args.multi_beta_penalty,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
