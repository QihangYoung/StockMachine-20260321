from __future__ import annotations

import pytest

from stockmachine.alpha import (
    assert_supported_alpha_expert,
    get_alpha_expert,
    list_alpha_expert_names,
)


def test_alpha_registry_lists_supported_baselines() -> None:
    assert list_alpha_expert_names() == (
        "factor_baseline",
        "ridge",
        "hist_gbm",
        "ensemble_hist_gbm_ridge_mean",
        "ensemble_hist_gbm_ridge_rank",
    )
    assert get_alpha_expert("hist_gbm").family == "tree"
    assert get_alpha_expert("ridge").task == "cross_sectional_regression"
    assert get_alpha_expert("ensemble_hist_gbm_ridge_mean").components == ("hist_gbm", "ridge")
    assert get_alpha_expert("ensemble_hist_gbm_ridge_rank").combine_method == "rank_average"


def test_alpha_registry_rejects_unknown_expert() -> None:
    with pytest.raises(ValueError, match="Unsupported alpha expert"):
        assert_supported_alpha_expert("unknown_model")
