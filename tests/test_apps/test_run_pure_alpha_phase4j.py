from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4i import TARGET_CS_DEMEANED
from stockmachine.apps.run_pure_alpha_phase4j import build_phase4j_short_selector_artifacts


def test_phase4j_scores_short_selectors_across_residual_targets(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_selector_rows())

    result = build_phase4j_short_selector_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4j",
        variant_names=("top1000_clean_core_beta_full",),
        residual_targets=("beta_residual", TARGET_CS_DEMEANED),
        candidate_count=3,
        min_regression_rows=10,
    )
    summary = pd.read_csv(tmp_path / "phase4j" / "phase4j_short_selector_summary_validation.csv")
    core = summary[
        (summary["residual_target"] == TARGET_CS_DEMEANED)
        & (summary["selector"] == "short_core_plus_overextension")
    ].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert core["mean_short_contribution_target"] > 0
    assert core["target_hit_rate"] > 0
    assert summary["test_window_used"].eq(False).all()


def test_phase4j_writes_daily_selector_diagnostics(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_selector_rows())

    build_phase4j_short_selector_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4j",
        variant_names=("top1000_clean_core_beta_full",),
        residual_targets=("beta_residual", TARGET_CS_DEMEANED),
        candidate_count=3,
        min_regression_rows=10,
    )
    daily = pd.read_csv(tmp_path / "phase4j" / "phase4j_short_selector_daily_validation.csv")

    assert {
        "short_contribution_beta_residual",
        "short_contribution_cs_demeaned_residual",
        "loser_quantile_capture_rate",
    }.issubset(daily.columns)
    assert daily["selected_names"].eq(3).all()


def _selector_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(12):
        session = (start + timedelta(days=day)).isoformat()
        benchmark_forward = 0.01 if day % 2 == 0 else -0.01
        for index in range(18):
            high_extension = index >= 15
            beta = 0.8 + index * 0.02
            residual = 0.01 - (0.06 if high_extension else 0.0)
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
                    "reversal_5d": -0.02 if high_extension else 0.01,
                    "momentum_20d": 0.30 if high_extension else 0.01 * index,
                    "momentum_60d": 0.20 if high_extension else 0.01 * index,
                    "beta_residual_momentum_20d": 0.25 if high_extension else 0.01 * index,
                    "vol_adjusted_momentum_20d": 3.0 if high_extension else 0.1 * index,
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
