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
        "huber_regression",
        "elastic_net",
        "hist_gbm",
        "extra_trees",
        "random_forest",
        "lightgbm_regressor",
        "lightgbm_ranker",
        "catboost_regressor",
        "xgboost_regressor",
        "lstm_regressor",
        "transformer_regressor",
        "ensemble_hist_gbm_ridge_mean",
        "ensemble_hist_gbm_ridge_rank",
        "ensemble_hist_gbm_random_forest_mean",
        "ensemble_hist_gbm_random_forest_rank",
        "ensemble_hist_gbm_lightgbm_regressor_mean",
        "ensemble_hist_gbm_lightgbm_regressor_rank",
        "ensemble_random_forest_lightgbm_regressor_mean",
        "ensemble_random_forest_lightgbm_regressor_rank",
        "ensemble_hist_gbm_ridge_random_forest_mean",
        "ensemble_hist_gbm_ridge_random_forest_rank",
        "ensemble_hist_gbm_random_forest_lightgbm_regressor_mean",
        "ensemble_hist_gbm_random_forest_lightgbm_regressor_rank",
        "ensemble_extra_trees_hist_gbm_rank",
        "ensemble_extra_trees_lightgbm_ranker_rank",
        "ensemble_extra_trees_hist_gbm_lightgbm_ranker_rank",
    )
    assert get_alpha_expert("hist_gbm").family == "tree"
    assert get_alpha_expert("ridge").task == "cross_sectional_regression"
    assert get_alpha_expert("huber_regression").family == "linear"
    assert get_alpha_expert("extra_trees").family == "tree"
    assert get_alpha_expert("lightgbm_ranker").family == "ranker"
    assert get_alpha_expert("catboost_regressor").family == "tree"
    assert get_alpha_expert("lstm_regressor").family == "sequence"
    assert get_alpha_expert("transformer_regressor").family == "sequence"
    assert get_alpha_expert("ensemble_hist_gbm_ridge_mean").components == ("hist_gbm", "ridge")
    assert get_alpha_expert("ensemble_hist_gbm_ridge_rank").combine_method == "rank_average"
    assert get_alpha_expert("ensemble_hist_gbm_random_forest_mean").components == (
        "hist_gbm",
        "random_forest",
    )
    assert get_alpha_expert("ensemble_hist_gbm_lightgbm_regressor_rank").components == (
        "hist_gbm",
        "lightgbm_regressor",
    )
    assert get_alpha_expert("ensemble_random_forest_lightgbm_regressor_mean").components == (
        "random_forest",
        "lightgbm_regressor",
    )
    assert get_alpha_expert("ensemble_hist_gbm_ridge_random_forest_rank").components == (
        "hist_gbm",
        "ridge",
        "random_forest",
    )
    assert get_alpha_expert("ensemble_hist_gbm_random_forest_lightgbm_regressor_rank").components == (
        "hist_gbm",
        "random_forest",
        "lightgbm_regressor",
    )
    assert get_alpha_expert("ensemble_extra_trees_hist_gbm_rank").components == (
        "extra_trees",
        "hist_gbm",
    )
    assert get_alpha_expert("ensemble_extra_trees_lightgbm_ranker_rank").components == (
        "extra_trees",
        "lightgbm_ranker",
    )
    assert get_alpha_expert("ensemble_extra_trees_hist_gbm_lightgbm_ranker_rank").components == (
        "extra_trees",
        "hist_gbm",
        "lightgbm_ranker",
    )


def test_alpha_registry_rejects_unknown_expert() -> None:
    with pytest.raises(ValueError, match="Unsupported alpha expert"):
        assert_supported_alpha_expert("unknown_model")
