from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4o import (
    build_phase4o_long_selector_refresh_artifacts,
)


def test_phase4o_refreshes_long_selectors_and_portfolios(tmp_path) -> None:
    signal_panel_path = _write_panel(tmp_path, _rows())

    result = build_phase4o_long_selector_refresh_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4o",
        long_variants=("top500_clean_core_beta_full", "top1000_clean_core_beta_full"),
        short_variants=("adv30m_clean_core_beta_full",),
        long_selectors=("reversal_5d", "long_dip_in_20d_winner"),
        candidate_count=3,
        min_names_per_side=2,
        target_names_per_side=3,
        max_names_per_side=5,
        max_single_name_side_weight=0.5,
        beta_match_tolerance=0.01,
        min_regression_rows=8,
    )
    selector_summary = pd.read_csv(
        tmp_path / "phase4o" / "phase4o_long_selector_summary_validation.csv"
    )
    portfolio_summary = pd.read_csv(
        tmp_path / "phase4o" / "phase4o_long_selector_portfolio_summary_validation.csv"
    )
    positions = pd.read_csv(
        tmp_path / "phase4o" / "phase4o_long_selector_portfolio_positions_validation.csv.gz"
    )

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert result["primary_evaluation"] == "standalone_long_selector_performance"
    assert {"reversal_5d", "long_dip_in_20d_winner"}.issubset(
        set(selector_summary["selector"])
    )
    assert selector_summary["test_window_used"].eq(False).all()
    assert not portfolio_summary.empty
    assert portfolio_summary["test_window_used"].eq(False).all()
    assert positions["test_window_used"].eq(False).all()


def _rows() -> list[dict[str, object]]:
    rows = []
    for variant, prefix in (
        ("top500_clean_core_beta_full", "A"),
        ("top1000_clean_core_beta_full", "L"),
        ("adv30m_clean_core_beta_full", "S"),
    ):
        for index in range(30):
            is_long = variant.startswith("top")
            is_top_long = is_long and index < 10
            is_short_loser = variant.startswith("adv30m") and index >= 20
            long_score = 100.0 - index
            short_score = float(index)
            forward_residual = (
                0.02
                if is_top_long
                else -0.02
                if is_short_loser
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
                    "beta": 0.8 + (index % 10) * 0.02,
                    "reversal_5d": long_score if is_long else -short_score,
                    "momentum_20d": long_score if is_long else short_score,
                    "momentum_60d": long_score / 2.0 if is_long else short_score / 2.0,
                    "beta_residual_momentum_20d": long_score if is_long else short_score,
                    "vol_adjusted_momentum_20d": long_score if is_long else short_score,
                    "transparent_composite": long_score if is_long else short_score,
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
