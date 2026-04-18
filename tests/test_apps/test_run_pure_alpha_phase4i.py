from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_CS_DEMEANED,
    TARGET_STYLE_NEUTRAL,
    build_phase4i_residual_target_artifacts,
)


def test_phase4i_builds_alternative_residual_targets(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_target_rows())

    result = build_phase4i_residual_target_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4i",
        variant_names=("top1000_clean_core_beta_full",),
        features=("momentum_20d", "beta"),
        n_bins=4,
        max_independence_rows=1_000,
        permutation_count=2,
        min_regression_rows=8,
        random_state=11,
    )
    prior = pd.read_csv(
        tmp_path / "phase4i" / "phase4i_residual_target_prior_summary_validation.csv"
    )
    cs = prior[prior["residual_target"] == TARGET_CS_DEMEANED].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert abs(cs["target_mean"]) < 1e-12
    assert TARGET_STYLE_NEUTRAL in set(prior["residual_target"])
    assert prior["test_window_used"].eq(False).all()


def test_phase4i_writes_independence_and_robust_utility(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_target_rows())

    build_phase4i_residual_target_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4i",
        variant_names=("top1000_clean_core_beta_full",),
        features=("momentum_20d", "beta"),
        n_bins=4,
        max_independence_rows=1_000,
        permutation_count=2,
        min_regression_rows=8,
        random_state=11,
    )
    independence = pd.read_csv(
        tmp_path / "phase4i" / "phase4i_feature_independence_by_target_validation.csv"
    )
    utility = pd.read_csv(
        tmp_path / "phase4i" / "phase4i_robust_feature_utility_by_target_validation.csv"
    )
    daily = pd.read_csv(
        tmp_path / "phase4i" / "phase4i_residual_target_daily_prior_validation.csv"
    )

    assert {"residual_target", "feature"}.issubset(independence.columns)
    assert {"residual_target", "robust_usefulness_label"}.issubset(utility.columns)
    assert not daily.empty
    assert utility["test_window_used"].eq(False).all()


def _target_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2019, 1, 1)
    for day in range(16):
        session = (start + timedelta(days=day)).isoformat()
        benchmark_forward = 0.01 if day % 2 == 0 else -0.01
        for index in range(16):
            beta = 0.75 + index * 0.03
            high_momentum = index >= 12
            residual = 0.01 - (0.04 if high_momentum else 0.0) + 0.004 * beta
            forward_return = residual + beta * benchmark_forward
            rows.append(
                {
                    "session_date": session,
                    "variant": "top1000_clean_core_beta_full",
                    "symbol": f"SYM{index:03d}",
                    "lagged_close": 20.0 + index,
                    "trailing_median_dollar_volume_20": (
                        100_000_000.0 + index * 1_000_000.0
                    ),
                    "liquidity_rank": index + 1,
                    "beta": beta,
                    "reversal_5d": -0.01 * index,
                    "momentum_20d": 0.01 * index,
                    "momentum_60d": 0.02 * index,
                    "beta_residual_momentum_20d": 0.01 * index,
                    "vol_adjusted_momentum_20d": 0.02 * index,
                    "forward_return_5d": forward_return,
                    "benchmark_forward_return_5d": benchmark_forward,
                    "forward_market_relative_return_5d": forward_return
                    - benchmark_forward,
                    "forward_beta_residual_return_5d": residual,
                }
            )
    return rows


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
