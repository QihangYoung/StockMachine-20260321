from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4e import (
    build_phase4e_feature_independence_artifacts,
)


def test_phase4e_detects_nonlinear_feature_target_dependence(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_independence_rows())

    result = build_phase4e_feature_independence_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4e",
        variant_names=("top1000_clean_core_beta_full",),
        n_bins=4,
        max_rows=1_000,
        permutation_count=5,
        random_state=17,
    )
    target = pd.read_csv(
        tmp_path / "phase4e" / "phase4e_feature_target_independence_validation.csv"
    )
    momentum = target[target["feature"] == "momentum_20d"].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert momentum["excess_normalized_mutual_information"] > 0.2
    assert target["test_window_used"].eq(False).all()


def test_phase4e_writes_feature_pair_independence_and_clusters(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_independence_rows())

    build_phase4e_feature_independence_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4e",
        variant_names=("top1000_clean_core_beta_full",),
        n_bins=4,
        max_rows=1_000,
        permutation_count=5,
        cluster_threshold=0.05,
        random_state=17,
    )
    pairs = pd.read_csv(
        tmp_path / "phase4e" / "phase4e_feature_pair_independence_validation.csv"
    )
    clusters = pd.read_csv(
        tmp_path / "phase4e" / "phase4e_feature_dependence_clusters_validation.csv"
    )
    manifest = pd.read_csv(
        tmp_path
        / "phase4e"
        / "phase4e_feature_independence_sample_manifest_validation.csv"
    )
    momentum_pair = pairs[
        (
            (pairs["feature_left"] == "momentum_20d")
            & (pairs["feature_right"] == "momentum_20d_z")
        )
        | (
            (pairs["feature_left"] == "momentum_20d_z")
            & (pairs["feature_right"] == "momentum_20d")
        )
    ].iloc[0]

    assert momentum_pair["excess_normalized_mutual_information"] > 0.2
    assert not clusters.empty
    assert manifest.loc[0, "sampled_rows"] > 0
    assert pairs["test_window_used"].eq(False).all()


def _independence_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(20):
        session = (start + timedelta(days=day)).isoformat()
        for index in range(20):
            signal = (index + day) % 2 == 0
            rows.append(
                {
                    "session_date": session,
                    "variant": "top1000_clean_core_beta_full",
                    "symbol": f"SYM{index:03d}",
                    "lagged_close": 20.0 + ((index * 7 + day * 11) % 23),
                    "trailing_median_dollar_volume_20": (
                        100_000_000.0 + ((index * 13 + day * 17) % 29) * 1_000_000.0
                    ),
                    "liquidity_rank": ((index * 3 + day * 5) % 20) + 1,
                    "beta": 0.9 + ((index * 11 + day * 7) % 31) * 0.01,
                    "reversal_5d": -0.05 if signal else 0.05,
                    "momentum_20d": 1.0 if signal else -1.0,
                    "momentum_60d": 1.0 if signal else -1.0,
                    "beta_residual_momentum_20d": 0.8 if signal else -0.8,
                    "vol_adjusted_momentum_20d": 1.2 if signal else -1.2,
                    "forward_return_5d": -0.04 if signal else 0.04,
                    "forward_beta_residual_return_5d": -0.05 if signal else 0.05,
                }
            )
    return rows


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
