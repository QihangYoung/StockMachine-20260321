from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4h import (
    build_phase4h_universe_prior_artifacts,
)


def test_phase4h_compares_universe_priors_and_feature_utility(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_universe_rows())

    result = build_phase4h_universe_prior_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4h",
        variant_names=("top1000_clean_core_beta_full", "adv20m_clean_core_beta_full"),
        features=("momentum_20d",),
        n_bins=4,
    )
    summary = pd.read_csv(
        tmp_path / "phase4h" / "phase4h_universe_prior_summary_validation.csv"
    )
    utility = pd.read_csv(
        tmp_path / "phase4h" / "phase4h_feature_utility_by_universe_validation.csv"
    )

    top1000 = summary[summary["variant"] == "top1000_clean_core_beta_full"].iloc[0]
    adv20m = summary[summary["variant"] == "adv20m_clean_core_beta_full"].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert top1000["target_median"] > adv20m["target_median"]
    assert set(utility["variant"]) == {
        "top1000_clean_core_beta_full",
        "adv20m_clean_core_beta_full",
    }
    assert utility["test_window_used"].eq(False).all()


def test_phase4h_writes_year_regime_and_bucket_tables(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_universe_rows())

    build_phase4h_universe_prior_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4h",
        variant_names=("top1000_clean_core_beta_full", "adv20m_clean_core_beta_full"),
        features=("momentum_20d",),
        n_bins=4,
    )
    by_year = pd.read_csv(
        tmp_path / "phase4h" / "phase4h_universe_prior_by_year_validation.csv"
    )
    by_regime = pd.read_csv(
        tmp_path / "phase4h" / "phase4h_universe_prior_by_regime_validation.csv"
    )
    by_bucket = pd.read_csv(
        tmp_path / "phase4h" / "phase4h_universe_prior_by_bucket_validation.csv"
    )

    assert not by_year.empty
    assert {"all", "market_up_5d", "market_down_5d"}.issubset(
        set(by_regime["market_regime"])
    )
    assert {"liquidity_rank", "beta"}.issubset(set(by_bucket["bucket_feature"]))


def _universe_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2019, 1, 1)
    for variant, base_residual in (
        ("top1000_clean_core_beta_full", 0.02),
        ("adv20m_clean_core_beta_full", -0.01),
    ):
        for day in range(12):
            session = (start + timedelta(days=day)).isoformat()
            benchmark_forward = 0.01 if day % 2 == 0 else -0.01
            for index in range(12):
                beta = 0.8 + index * 0.03
                high_momentum = index >= 9
                residual = base_residual - (0.03 if high_momentum else 0.0)
                forward_return = residual + beta * benchmark_forward
                rows.append(
                    {
                        "session_date": session,
                        "variant": variant,
                        "symbol": f"{variant[:3]}{index:03d}",
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
