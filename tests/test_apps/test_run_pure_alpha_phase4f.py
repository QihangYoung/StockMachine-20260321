from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4f import (
    build_phase4f_feature_posterior_artifacts,
)


def test_phase4f_beta_bins_translate_dependence_into_posterior_shape(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_posterior_rows())

    result = build_phase4f_feature_posterior_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4f",
        variant_names=("top1000_clean_core_beta_full",),
        features=("beta",),
        n_bins=5,
        loser_quantile=0.20,
    )
    extremes = pd.read_csv(
        tmp_path / "phase4f" / "phase4f_feature_posterior_extremes_validation.csv"
    )
    beta_all = extremes[
        (extremes["feature"] == "beta") & (extremes["market_regime"] == "all")
    ].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert beta_all["high_target_mean"] < beta_all["low_target_mean"]
    assert beta_all["high_minus_low_short_contribution"] > 0
    assert beta_all["high_bottom_loser_share"] > beta_all["low_bottom_loser_share"]
    assert extremes["test_window_used"].eq(False).all()


def test_phase4f_writes_regime_bins_and_manifest(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_posterior_rows())

    build_phase4f_feature_posterior_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4f",
        variant_names=("top1000_clean_core_beta_full",),
        features=("beta", "return_5d"),
        n_bins=5,
        loser_quantile=0.20,
    )
    bins = pd.read_csv(tmp_path / "phase4f" / "phase4f_feature_posterior_bins_validation.csv")
    manifest = pd.read_csv(
        tmp_path / "phase4f" / "phase4f_feature_posterior_sample_manifest_validation.csv"
    )

    assert {"all", "market_up_5d", "market_down_5d"}.issubset(
        set(bins["market_regime"])
    )
    assert set(bins["feature"]) == {"beta", "return_5d"}
    assert manifest.loc[0, "complete_case_rows"] > 0
    assert bins["test_window_used"].eq(False).all()


def _posterior_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(20):
        session = (start + timedelta(days=day)).isoformat()
        benchmark_forward = 0.02 if day % 2 == 0 else -0.02
        for index in range(20):
            beta = 0.70 + index * 0.03
            high_beta_loser = index >= 16
            residual = -0.08 if high_beta_loser else 0.03
            forward_return = residual + beta * benchmark_forward
            rows.append(
                {
                    "session_date": session,
                    "variant": "top1000_clean_core_beta_full",
                    "symbol": f"SYM{index:03d}",
                    "lagged_close": 20.0 + index,
                    "trailing_median_dollar_volume_20": 100_000_000.0 + index * 1_000_000.0,
                    "liquidity_rank": index + 1,
                    "beta": beta,
                    "reversal_5d": -0.02 + index * 0.001,
                    "momentum_20d": 0.01 * index,
                    "momentum_60d": 0.02 * index,
                    "beta_residual_momentum_20d": 0.015 * index,
                    "vol_adjusted_momentum_20d": 0.02 * index,
                    "forward_return_5d": forward_return,
                    "benchmark_forward_return_5d": benchmark_forward,
                    "forward_beta_residual_return_5d": residual,
                }
            )
    return rows


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
