from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from stockmachine.apps.run_pure_alpha_phase4d import (
    RESEARCH_ROOT,
    TARGET_COLUMN,
    _add_model_features,
)
from stockmachine.apps.run_pure_alpha_phase4f import DEFAULT_PHASE3_SIGNAL_PANEL
from stockmachine.apps.run_pure_alpha_phase4j import _add_selector_scores


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4s_sector_industry_mf_residual_v2_20260419"
DEFAULT_H10_SIGNAL_PANEL = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_h10_probe_20260418"
    / "phase3_baseline_signal_panel_validation.csv.gz"
)
DEFAULT_CIK_MAPPING = (
    RESEARCH_ROOT
    / "non_price_phase0_two_factor_data_prep_20260419"
    / "sec_universe_cik_mapping.csv"
)
DEFAULT_SEC_SUBMISSIONS_DIR = Path("data/raw/sec/submissions")
DEFAULT_LONG_VARIANT = "top1000_clean_core_beta_full"
DEFAULT_SHORT_VARIANT = "adv30m_clean_core_beta_full"
DEFAULT_LONG_SCORE = "reversal_5d"
DEFAULT_SHORT_SELECTOR = "short_core_plus_overextension"

BASE_COLUMNS = (
    "session_date",
    "variant",
    "symbol",
    "beta",
    "lagged_close",
    "trailing_median_dollar_volume_20",
    "liquidity_rank",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "forward_return_5d",
    "benchmark_forward_return_5d",
    TARGET_COLUMN,
)
NUMERIC_RISK_ONLY_FEATURES = (
    "beta",
    "lagged_close_log",
    "trailing_median_dollar_volume_20_log",
    "liquidity_rank",
)
NUMERIC_STYLE_RISK_FEATURES = (
    *NUMERIC_RISK_ONLY_FEATURES,
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
)
RESIDUAL_TARGETS = (
    "sic2_neutral_beta_residual",
    "sic4_neutral_beta_residual",
    "risk_factor_sic2_residual",
    "risk_factor_sic4_residual",
    "style_factor_sic2_residual",
    "style_factor_sic4_residual",
)
PORTFOLIO_SPECS = (
    {"portfolio": "beta_only_top120", "hard_group": None, "soft_group": None},
    {"portfolio": "sic2_soft_neutral", "hard_group": None, "soft_group": "sic2_sector"},
    {"portfolio": "sic4_soft_neutral", "hard_group": None, "soft_group": "sic4_industry"},
    {"portfolio": "sic2_hard_neutral", "hard_group": "sic2_sector", "soft_group": None},
    {"portfolio": "sic2_hard_sic4_soft", "hard_group": "sic2_sector", "soft_group": "sic4_industry"},
)


def build_phase4s_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    long_score: str = DEFAULT_LONG_SCORE,
    short_selector: str = DEFAULT_SHORT_SELECTOR,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
) -> dict[str, Any]:
    """Run SIC-based neutralization and multi-factor residual diagnostics."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    coverage = _coverage_table(panel)

    positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    for spec in PORTFOLIO_SPECS:
        portfolio = str(spec["portfolio"])
        for session_date in sessions:
            long_group = groups.get((long_variant, session_date))
            short_group = groups.get((short_variant, session_date))
            if long_group is None or short_group is None:
                skipped.append(
                    _skip_row(session_date, portfolio, 0, 0, "missing_long_or_short_group")
                )
                continue
            book, diagnostic, skip = _construct_one_session(
                long_group,
                short_group,
                session_date=session_date,
                portfolio=portfolio,
                long_variant=long_variant,
                short_variant=short_variant,
                long_score=long_score,
                short_selector=short_selector,
                hard_group=spec["hard_group"],
                soft_group=spec["soft_group"],
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
            )
            positions.extend(book)
            if diagnostic is not None:
                daily_rows.append(diagnostic)
            if skip is not None:
                skipped.append(skip)

    positions_df = pd.DataFrame(positions)
    daily = pd.DataFrame(daily_rows)
    skipped_df = pd.DataFrame(skipped)
    summary = _summary(daily, skipped_df)
    residual_prior = _residual_prior_table(panel)

    positions_path = output_dir / "phase4s_positions_validation.csv.gz"
    daily_path = output_dir / "phase4s_daily_validation.csv"
    skipped_path = output_dir / "phase4s_skipped_validation.csv"
    summary_path = output_dir / "phase4s_summary_validation.csv"
    coverage_path = output_dir / "phase4s_sic_coverage_validation.csv"
    residual_prior_path = output_dir / "phase4s_residual_prior_validation.csv"
    memo_path = output_dir / "phase4s_sector_industry_mf_residual_memo.md"
    rollup_path = output_dir / "phase4s_rollup.json"

    positions_df.to_csv(positions_path, index=False, compression="gzip")
    daily.to_csv(daily_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    residual_prior.to_csv(residual_prior_path, index=False)
    memo_path.write_text(_memo(summary, coverage, residual_prior), encoding="utf-8")

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "cik_mapping_path": Path(cik_mapping_path).as_posix(),
        "sec_submissions_dir": Path(sec_submissions_dir).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "long_score": long_score,
        "short_selector": short_selector,
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "min_nonzero_names": int(min_nonzero_names),
        "score_weight": float(score_weight),
        "soft_group_penalty": float(soft_group_penalty),
        "min_regression_rows": int(min_regression_rows),
        "min_dummy_count": int(min_dummy_count),
        "risk_only_features": list(NUMERIC_RISK_ONLY_FEATURES),
        "style_risk_features": list(NUMERIC_STYLE_RISK_FEATURES),
        "residual_targets": list(RESIDUAL_TARGETS),
        "portfolio_specs": list(PORTFOLIO_SPECS),
        "panel_rows_loaded": int(len(panel)),
        "positions_rows": int(len(positions_df)),
        "daily_rows": int(len(daily)),
        "skipped_rows": int(len(skipped_df)),
        "summary_rows": int(len(summary)),
        "positions_artifact": positions_path.as_posix(),
        "daily_artifact": daily_path.as_posix(),
        "skipped_artifact": skipped_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "coverage_artifact": coverage_path.as_posix(),
        "residual_prior_artifact": residual_prior_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "sic_proxy_sector_industry_neutralization_and_multi_factor_residuals",
        "limitations": [
            "SEC SIC is used as a sector/industry proxy, not GICS.",
            "SIC fields are sourced from cached SEC submissions and are not a fully point-in-time industry history.",
            "Industry-neutral construction uses exact SIC2 neutrality and soft SIC4 neutrality; hard SIC4 neutrality is usually too restrictive for 20+ names per side.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_panel(signal_panel_path: str | Path, long_variant: str, short_variant: str) -> pd.DataFrame:
    panel = pd.read_csv(signal_panel_path, usecols=list(BASE_COLUMNS), low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[panel["variant"].isin({long_variant, short_variant})].copy()
    for column in BASE_COLUMNS:
        if column not in ("session_date", "variant", "symbol"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(
        subset=[
            "beta",
            "forward_return_5d",
            "benchmark_forward_return_5d",
            TARGET_COLUMN,
        ]
    )
    panel = _add_model_features(panel)
    scored = [_add_selector_scores(group) for _, group in panel.groupby(["variant", "session_date"])]
    panel = pd.concat(scored, ignore_index=True) if scored else panel
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _load_sec_sic_map(
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
) -> pd.DataFrame:
    mapping = pd.read_csv(cik_mapping_path)
    mapping["symbol"] = mapping["symbol"].astype(str)
    rows = []
    submissions_dir = Path(sec_submissions_dir)
    for row in mapping.itertuples(index=False):
        cik = str(row.cik).zfill(10)
        path = submissions_dir / f"CIK{cik}.json"
        sic = None
        sic_description = None
        if path.exists():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                sic = obj.get("sic")
                sic_description = obj.get("sicDescription")
            except json.JSONDecodeError:
                sic = None
                sic_description = None
        sic_str = str(sic).zfill(4) if sic is not None and str(sic).strip() else None
        rows.append(
            {
                "symbol": str(row.symbol),
                "cik": cik,
                "sic4": sic_str,
                "sic2_sector": f"SIC{sic_str[:2]}" if sic_str else None,
                "sic4_industry": f"SIC{sic_str}" if sic_str else None,
                "sic_description": sic_description,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("symbol", keep="last")


def _add_residual_targets(
    panel: pd.DataFrame,
    *,
    min_regression_rows: int,
    min_dummy_count: int,
) -> pd.DataFrame:
    frame = panel.copy()
    frame["sic2_neutral_beta_residual"] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        numeric_features=(),
        categorical_features=("sic2_sector",),
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    frame["sic4_neutral_beta_residual"] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        numeric_features=(),
        categorical_features=("sic4_industry",),
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    frame["risk_factor_sic2_residual"] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        numeric_features=NUMERIC_RISK_ONLY_FEATURES,
        categorical_features=("sic2_sector",),
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    frame["risk_factor_sic4_residual"] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        numeric_features=NUMERIC_RISK_ONLY_FEATURES,
        categorical_features=("sic4_industry",),
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    frame["style_factor_sic2_residual"] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        numeric_features=NUMERIC_STYLE_RISK_FEATURES,
        categorical_features=("sic2_sector",),
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    frame["style_factor_sic4_residual"] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        numeric_features=NUMERIC_STYLE_RISK_FEATURES,
        categorical_features=("sic4_industry",),
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    return frame


def _daily_residualize(
    frame: pd.DataFrame,
    *,
    target_column: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    min_regression_rows: int,
    min_dummy_count: int,
) -> pd.Series:
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    columns = [target_column, *numeric_features, *categorical_features]
    for _, group in frame.groupby(["variant", "session_date"], sort=False):
        valid = group[columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
        valid = valid.dropna(subset=list(numeric_features))
        if len(valid) < min_regression_rows:
            continue
        y = valid[target_column].to_numpy(dtype=float)
        pieces = [pd.Series(1.0, index=valid.index, name="intercept")]
        if numeric_features:
            x_num = valid[list(numeric_features)].astype(float)
            x_num = (x_num - x_num.mean(axis=0)) / x_num.std(axis=0, ddof=0).replace(0.0, np.nan)
            x_num = x_num.dropna(axis=1)
            if not x_num.empty:
                pieces.append(x_num)
        for feature in categorical_features:
            cats = valid[feature].fillna("UNKNOWN").astype(str)
            counts = cats.value_counts()
            cats = cats.where(cats.map(counts) >= min_dummy_count, "OTHER")
            dummies = pd.get_dummies(cats, prefix=feature, dtype=float)
            if dummies.shape[1] > 1:
                pieces.append(dummies.iloc[:, 1:])
        design_df = pd.concat(pieces, axis=1)
        design = design_df.to_numpy(dtype=float)
        if len(valid) <= design.shape[1]:
            continue
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals.loc[valid.index] = y - design @ coefficients
    return residuals


def _construct_one_session(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    portfolio: str,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    hard_group: str | None,
    soft_group: str | None,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    needed = [
        long_score,
        short_selector,
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_COLUMN,
        *RESIDUAL_TARGETS,
        "sic2_sector",
        "sic4_industry",
    ]
    longs = long_group.dropna(subset=[long_score, *needed[2:]]).drop_duplicates("symbol").copy()
    shorts = short_group.dropna(subset=[short_selector, *needed[2:]]).drop_duplicates("symbol").copy()
    if len(longs) < min_nonzero_names or len(shorts) < min_nonzero_names:
        return [], None, _skip_row(
            session_date, portfolio, len(longs), len(shorts), "insufficient_candidates"
        )
    longs = longs.sort_values([long_score, "symbol"], ascending=[False, True]).head(
        candidate_pool_per_side
    )
    shorts = shorts.sort_values([short_selector, "symbol"], ascending=[False, True]).head(
        candidate_pool_per_side
    )
    try:
        long_weights, short_weights = _optimize_joint_weights(
            longs,
            shorts,
            long_score=long_score,
            short_selector=short_selector,
            hard_group=hard_group,
            soft_group=soft_group,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            soft_group_penalty=soft_group_penalty,
        )
    except ValueError as exc:
        return [], None, _skip_row(
            session_date, portfolio, len(longs), len(shorts), f"optimizer_failed:{exc}"
        )
    if (long_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(
            session_date, portfolio, len(longs), len(shorts), "insufficient_long_nonzero"
        )
    if (short_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(
            session_date, portfolio, len(longs), len(shorts), "insufficient_short_nonzero"
        )

    book = [
        *_position_rows(
            longs,
            long_weights,
            session_date=session_date,
            portfolio=portfolio,
            long_variant=long_variant,
            short_variant=short_variant,
            side="long",
            score_column=long_score,
        ),
        *_position_rows(
            shorts,
            short_weights,
            session_date=session_date,
            portfolio=portfolio,
            long_variant=long_variant,
            short_variant=short_variant,
            side="short",
            score_column=short_selector,
        ),
    ]
    diagnostic = _diagnostic_row(
        longs,
        shorts,
        long_weights,
        short_weights,
        session_date=session_date,
        portfolio=portfolio,
        long_variant=long_variant,
        short_variant=short_variant,
        hard_group=hard_group,
        soft_group=soft_group,
        long_score=long_score,
        short_selector=short_selector,
    )
    return book, diagnostic, None


def _optimize_joint_weights(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    long_score: str,
    short_selector: str,
    hard_group: str | None,
    soft_group: str | None,
    max_weight: float,
    score_weight: float,
    soft_group_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_long = len(longs)
    n_short = len(shorts)
    n = n_long + n_short
    if n_long * max_weight < 1.0 - 1e-12 or n_short * max_weight < 1.0 - 1e-12:
        raise ValueError("insufficient_weight_capacity")
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
    if hard_group is not None:
        group_matrix = _group_exposure_matrix(longs, shorts, hard_group)
        for row in group_matrix[:-1]:
            aeq.append(row)
            beq.append(0.0)
    a_eq = np.vstack(aeq)
    b_eq = np.array(beq)
    soft_matrix = (
        _group_exposure_matrix(longs, shorts, soft_group)
        if soft_group is not None
        else np.zeros((0, n))
    )
    c = -score_weight * score
    bounds: list[tuple[float, float | None]] = [(0.0, max_weight)] * n
    a_ub = None
    b_ub = None
    if len(soft_matrix):
        # Linearize abs(group_exposure) with auxiliary variables t:
        #   Gx - t <= 0 and -Gx - t <= 0, minimizing sum(t).
        m = soft_matrix.shape[0]
        c = np.concatenate([c, np.full(m, soft_group_penalty, dtype=float)])
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


def _group_exposure_matrix(longs: pd.DataFrame, shorts: pd.DataFrame, group_column: str) -> np.ndarray:
    groups = sorted(set(longs[group_column].astype(str)).union(set(shorts[group_column].astype(str))))
    n_long = len(longs)
    n_short = len(shorts)
    matrix = np.zeros((len(groups), n_long + n_short), dtype=float)
    long_values = longs[group_column].astype(str).to_numpy()
    short_values = shorts[group_column].astype(str).to_numpy()
    for i, group in enumerate(groups):
        matrix[i, :n_long] = (long_values == group).astype(float)
        matrix[i, n_long:] = -((short_values == group).astype(float))
    return matrix


def _diagnostic_row(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    *,
    session_date: str,
    portfolio: str,
    long_variant: str,
    short_variant: str,
    hard_group: str | None,
    soft_group: str | None,
    long_score: str,
    short_selector: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "session_date": session_date,
        "portfolio": portfolio,
        "long_variant": long_variant,
        "short_variant": short_variant,
        "hard_group": hard_group or "",
        "soft_group": soft_group or "",
        "long_score": long_score,
        "short_selector": short_selector,
        "eligible_long_names": int(len(longs)),
        "eligible_short_names": int(len(shorts)),
        "long_count": int((long_weights > 1e-10).sum()),
        "short_count": int((short_weights > 1e-10).sum()),
        "long_beta": float(np.dot(long_weights, longs["beta"])),
        "short_beta": float(np.dot(short_weights, shorts["beta"])),
        "net_beta": float(
            np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"])
        ),
        "max_abs_position_weight": float(max(long_weights.max(), short_weights.max())),
        "sic2_l1_exposure": _group_l1(longs, shorts, long_weights, short_weights, "sic2_sector"),
        "sic2_max_abs_exposure": _group_max_abs(
            longs, shorts, long_weights, short_weights, "sic2_sector"
        ),
        "sic4_l1_exposure": _group_l1(longs, shorts, long_weights, short_weights, "sic4_industry"),
        "sic4_max_abs_exposure": _group_max_abs(
            longs, shorts, long_weights, short_weights, "sic4_industry"
        ),
        "test_window_used": False,
    }
    for label, column in (
        ("raw", "forward_return_5d"),
        ("beta_residual", TARGET_COLUMN),
        ("sic2_residual", "sic2_neutral_beta_residual"),
        ("sic4_residual", "sic4_neutral_beta_residual"),
        ("risk_factor_sic2_residual", "risk_factor_sic2_residual"),
        ("risk_factor_sic4_residual", "risk_factor_sic4_residual"),
        ("style_factor_sic2_residual", "style_factor_sic2_residual"),
        ("style_factor_sic4_residual", "style_factor_sic4_residual"),
    ):
        long_value = float(np.dot(long_weights, longs[column].to_numpy(dtype=float)))
        short_value = float(-np.dot(short_weights, shorts[column].to_numpy(dtype=float)))
        out[f"long_{label}"] = long_value
        out[f"short_{label}_contribution"] = short_value
        out[f"spread_{label}"] = long_value + short_value
    return out


def _position_rows(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    *,
    session_date: str,
    portfolio: str,
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
        rows.append(
            {
                "session_date": session_date,
                "portfolio": portfolio,
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
                "sic2_neutral_beta_residual": float(row["sic2_neutral_beta_residual"]),
                "sic4_neutral_beta_residual": float(row["sic4_neutral_beta_residual"]),
                "risk_factor_sic2_residual": float(row["risk_factor_sic2_residual"]),
                "risk_factor_sic4_residual": float(row["risk_factor_sic4_residual"]),
                "style_factor_sic2_residual": float(row["style_factor_sic2_residual"]),
                "style_factor_sic4_residual": float(row["style_factor_sic4_residual"]),
                "test_window_used": False,
            }
        )
    return rows


def _group_exposures(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    group_column: str,
) -> pd.Series:
    long_exp = pd.Series(long_weights, index=longs[group_column].astype(str)).groupby(level=0).sum()
    short_exp = pd.Series(short_weights, index=shorts[group_column].astype(str)).groupby(level=0).sum()
    return long_exp.sub(short_exp, fill_value=0.0)


def _group_l1(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    group_column: str,
) -> float:
    return float(_group_exposures(longs, shorts, long_weights, short_weights, group_column).abs().sum())


def _group_max_abs(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    group_column: str,
) -> float:
    exposures = _group_exposures(longs, shorts, long_weights, short_weights, group_column)
    return float(exposures.abs().max()) if len(exposures) else 0.0


def _coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        rows.append(
            {
                "variant": variant,
                "rows": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "known_sic_rows": int((group["sic4_industry"] != "UNKNOWN").sum()),
                "known_sic_row_coverage": float((group["sic4_industry"] != "UNKNOWN").mean()),
                "sic2_groups": int(group["sic2_sector"].nunique()),
                "sic4_groups": int(group["sic4_industry"].nunique()),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _residual_prior_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = [TARGET_COLUMN, *RESIDUAL_TARGETS]
    for variant, group in panel.groupby("variant", sort=True):
        for target in targets:
            values = group[target].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "target": target,
                    "rows": int(len(values)),
                    "mean_bps": float(values.mean() * 10000.0),
                    "median_bps": float(values.median() * 10000.0),
                    "std_bps": float(values.std(ddof=1) * 10000.0),
                    "positive_rate": float((values > 0).mean()),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _summary(daily: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    if daily.empty and skipped.empty:
        return pd.DataFrame()
    frames = []
    if not daily.empty:
        frames.append(daily[["portfolio", "session_date"]])
    if not skipped.empty:
        frames.append(skipped[["portfolio", "session_date"]])
    requested = (
        pd.concat(frames, ignore_index=True)
        .groupby("portfolio", sort=True)
        .agg(requested_sessions=("session_date", "nunique"))
        .reset_index()
    )
    if daily.empty:
        return requested
    rows = []
    for portfolio, group in daily.groupby("portfolio", sort=True):
        row: dict[str, Any] = {
            "portfolio": portfolio,
            "constructed_sessions": int(len(group)),
            "construction_rate": np.nan,
            "mean_sic2_l1_exposure": float(group["sic2_l1_exposure"].mean()),
            "mean_sic2_max_abs_exposure": float(group["sic2_max_abs_exposure"].mean()),
            "mean_sic4_l1_exposure": float(group["sic4_l1_exposure"].mean()),
            "mean_sic4_max_abs_exposure": float(group["sic4_max_abs_exposure"].mean()),
            "mean_abs_net_beta": float(group["net_beta"].abs().mean()),
            "median_long_count": float(group["long_count"].median()),
            "median_short_count": float(group["short_count"].median()),
            "test_window_used": False,
        }
        for label in (
            "raw",
            "beta_residual",
            "sic2_residual",
            "sic4_residual",
            "risk_factor_sic2_residual",
            "risk_factor_sic4_residual",
            "style_factor_sic2_residual",
            "style_factor_sic4_residual",
        ):
            spread = group[f"spread_{label}"]
            row[f"mean_spread_{label}"] = float(spread.mean())
            row[f"hit_rate_{label}"] = float((spread > 0).mean())
            row[f"mean_long_{label}"] = float(group[f"long_{label}"].mean())
            row[f"mean_short_{label}_contribution"] = float(
                group[f"short_{label}_contribution"].mean()
            )
        rows.append(row)
    summary = requested.merge(pd.DataFrame(rows), on="portfolio", how="left")
    summary["constructed_sessions"] = summary["constructed_sessions"].fillna(0).astype(int)
    summary["construction_rate"] = (
        summary["constructed_sessions"] / summary["requested_sessions"].replace(0, np.nan)
    ).fillna(0.0)
    return summary


def _skip_row(
    session_date: str,
    portfolio: str,
    eligible_long_names: int,
    eligible_short_names: int,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "portfolio": portfolio,
        "eligible_long_names": int(eligible_long_names),
        "eligible_short_names": int(eligible_short_names),
        "skip_reason": skip_reason,
        "test_window_used": False,
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    std = float(np.nanstd(clean))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros_like(clean)
    return (clean - float(np.nanmean(clean))) / std


def _memo(summary: pd.DataFrame, coverage: pd.DataFrame, residual_prior: pd.DataFrame) -> str:
    lines = [
        "# Phase4S Sector/Industry Neutral and Multi-Factor Residual Memo",
        "",
        "Validation-only. No test-window performance is used.",
        "",
        "Sector/industry is proxied by cached SEC SIC metadata: SIC2 as sector proxy, SIC4 as industry proxy.",
        "",
        "## SIC Coverage",
        "",
        _markdown_table(coverage) if not coverage.empty else "No coverage rows.",
        "",
        "## Portfolio Summary",
        "",
        _markdown_table(summary) if not summary.empty else "No portfolio rows.",
        "",
        "## Residual Prior Summary",
        "",
        _markdown_table(residual_prior) if not residual_prior.empty else "No residual prior rows.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase4S SIC neutralization and multi-factor residual diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--candidate-pool-per-side", type=int, default=80)
    parser.add_argument("--soft-group-penalty", type=float, default=25.0)
    args = parser.parse_args(argv)
    rollup = build_phase4s_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        candidate_pool_per_side=args.candidate_pool_per_side,
        soft_group_penalty=args.soft_group_penalty,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
