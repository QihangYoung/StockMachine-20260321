from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4k import (
    build_phase4k_asymmetric_universe_artifacts,
)


def test_phase4k_constructs_beta_matched_asymmetric_books(tmp_path) -> None:
    signal_panel_path = _write_panel(tmp_path, _rows())

    result = build_phase4k_asymmetric_universe_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4k",
        long_variants=("top1000_clean_core_beta_full",),
        short_variants=("adv20m_clean_core_beta_full",),
        min_regression_rows=10,
        beta_match_tolerance=0.01,
    )
    daily = pd.read_csv(tmp_path / "phase4k" / "phase4k_asymmetric_daily_validation.csv")
    positions = pd.read_csv(
        tmp_path / "phase4k" / "phase4k_asymmetric_positions_validation.csv.gz"
    )

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert len(daily) == 1
    assert abs(daily.loc[0, "net_beta"]) <= 0.01
    assert daily.loc[0, "short_cs_demeaned_residual_contribution"] > 0
    assert positions["test_window_used"].eq(False).all()
    assert set(positions.loc[positions["side"] == "long", "symbol"]).isdisjoint(
        set(positions.loc[positions["side"] == "short", "symbol"])
    )


def test_phase4k_records_skip_when_short_pool_is_missing(tmp_path) -> None:
    signal_panel_path = _write_panel(
        tmp_path,
        [row for row in _rows() if row["variant"] == "top1000_clean_core_beta_full"],
    )

    build_phase4k_asymmetric_universe_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4k",
        long_variants=("top1000_clean_core_beta_full",),
        short_variants=("adv20m_clean_core_beta_full",),
        min_regression_rows=10,
    )
    skipped = pd.read_csv(tmp_path / "phase4k" / "phase4k_asymmetric_skipped_validation.csv")

    assert skipped.loc[0, "skip_reason"] == "missing_session"
    assert skipped["test_window_used"].eq(False).all()


def _rows() -> list[dict[str, object]]:
    rows = []
    for variant, prefix in (
        ("top1000_clean_core_beta_full", "L"),
        ("adv20m_clean_core_beta_full", "S"),
    ):
        for index in range(60):
            is_top = index < 30
            score = 100.0 - index if variant.startswith("top1000") else float(index)
            reversal_score = score if variant.startswith("top1000") else -score
            forward_residual = (
                0.02
                if variant.startswith("top1000") and is_top
                else -0.02
                if variant.startswith("adv20m") and index >= 30
                else 0.002
            )
            rows.append(
                {
                    "session_date": "2020-01-02",
                    "variant": variant,
                    "symbol": f"{prefix}{index:03d}",
                    "lagged_close": 20.0 + index,
                    "trailing_median_dollar_volume_20": 1_000_000.0 + index * 1_000,
                    "liquidity_rank": index + 1,
                    "beta": 0.8 + (index % 30) * 0.01,
                    "reversal_5d": reversal_score,
                    "momentum_20d": score,
                    "momentum_60d": score / 2.0,
                    "beta_residual_momentum_20d": score,
                    "vol_adjusted_momentum_20d": score,
                    "transparent_composite": score,
                    "forward_return_5d": forward_residual + 0.002,
                    "benchmark_forward_return_5d": 0.002,
                    "forward_beta_residual_return_5d": forward_residual,
                }
            )
    return rows


def _write_panel(tmp_path, rows: list[dict[str, object]]):
    path = tmp_path / "phase3_signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
