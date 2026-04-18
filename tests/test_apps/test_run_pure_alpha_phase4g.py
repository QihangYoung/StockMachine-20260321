from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4g import (
    build_phase4g_robust_feature_utility_artifacts,
)


def test_phase4g_prefers_feature_with_robust_short_edge(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_robust_rows())

    result = build_phase4g_robust_feature_utility_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4g",
        variant_names=("top1000_clean_core_beta_full",),
        features=("beta", "momentum_20d"),
        n_bins=5,
        loser_quantile=0.20,
        winner_quantile=0.20,
    )
    summary = pd.read_csv(
        tmp_path / "phase4g" / "phase4g_robust_feature_summary_validation.csv"
    )
    beta = summary[summary["feature"] == "beta"].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert beta["all_short_contribution_median"] > 0
    assert beta["all_short_contribution_trimmed_mean_10_90"] > 0
    assert beta["all_tail_balance"] > 0
    assert str(beta["robust_usefulness_label"]).startswith("1_")
    assert summary["test_window_used"].eq(False).all()


def test_phase4g_writes_bins_baseline_and_manifest(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_robust_rows())

    build_phase4g_robust_feature_utility_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4g",
        variant_names=("top1000_clean_core_beta_full",),
        features=("beta", "momentum_20d"),
        n_bins=5,
        loser_quantile=0.20,
        winner_quantile=0.20,
    )
    bins = pd.read_csv(tmp_path / "phase4g" / "phase4g_robust_feature_bins_validation.csv")
    baseline = pd.read_csv(
        tmp_path / "phase4g" / "phase4g_robust_feature_baseline_validation.csv"
    )
    manifest = pd.read_csv(
        tmp_path / "phase4g" / "phase4g_robust_feature_sample_manifest_validation.csv"
    )

    assert {"all", "market_up_5d", "market_down_5d"}.issubset(
        set(bins["market_regime"])
    )
    assert {"target_median", "target_trimmed_mean_10_90", "top_winner_share"}.issubset(
        bins.columns
    )
    assert set(baseline["market_regime"]) == {"all", "market_up_5d", "market_down_5d"}
    assert manifest.loc[0, "complete_case_rows"] > 0


def _robust_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(20):
        session = (start + timedelta(days=day)).isoformat()
        benchmark_forward = 0.02 if day % 2 == 0 else -0.02
        for index in range(20):
            beta = 0.70 + index * 0.03
            high_beta_loser = index >= 16
            residual = -0.04 if high_beta_loser else 0.02
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
