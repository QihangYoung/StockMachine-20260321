from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class AlphaExpertSpec:
    """Static description for one supported alpha expert."""

    name: str
    family: str
    task: str
    description: str
    default_horizon: int = 5
    components: tuple[str, ...] = field(default_factory=tuple)
    combine_method: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "task": self.task,
            "description": self.description,
            "default_horizon": self.default_horizon,
            "components": list(self.components),
            "combine_method": self.combine_method,
        }


_ALPHA_EXPERT_SPECS: dict[str, AlphaExpertSpec] = {
    "factor_baseline": AlphaExpertSpec(
        name="factor_baseline",
        family="factor",
        task="cross_sectional_ranking",
        description="White-box factor blend built from momentum, volatility, and liquidity ranks.",
    ),
    "ridge": AlphaExpertSpec(
        name="ridge",
        family="linear",
        task="cross_sectional_regression",
        description="Median-imputed and standardized ridge regression baseline.",
    ),
    "huber_regression": AlphaExpertSpec(
        name="huber_regression",
        family="linear",
        task="cross_sectional_regression",
        description="Robust Huber regressor over the standardized baseline feature panel.",
    ),
    "elastic_net": AlphaExpertSpec(
        name="elastic_net",
        family="linear",
        task="cross_sectional_regression",
        description="ElasticNet baseline that adds sparse linear shrinkage over the feature panel.",
    ),
    "hist_gbm": AlphaExpertSpec(
        name="hist_gbm",
        family="tree",
        task="cross_sectional_regression",
        description="Histogram gradient boosting regressor over the baseline feature panel.",
    ),
    "extra_trees": AlphaExpertSpec(
        name="extra_trees",
        family="tree",
        task="cross_sectional_regression",
        description="ExtraTrees regressor over the baseline feature panel for noisy tabular ranking.",
    ),
    "random_forest": AlphaExpertSpec(
        name="random_forest",
        family="tree",
        task="cross_sectional_regression",
        description="Random forest regressor over the baseline feature panel as a bagging benchmark.",
    ),
    "lightgbm_regressor": AlphaExpertSpec(
        name="lightgbm_regressor",
        family="tree",
        task="cross_sectional_regression",
        description="LightGBM regressor over the baseline feature panel for boosted tabular alpha.",
    ),
    "lightgbm_ranker": AlphaExpertSpec(
        name="lightgbm_ranker",
        family="ranker",
        task="cross_sectional_ranking",
        description="LightGBM LambdaRank model trained on per-date cross-sectional ranking targets.",
    ),
    "catboost_regressor": AlphaExpertSpec(
        name="catboost_regressor",
        family="tree",
        task="cross_sectional_regression",
        description="CatBoost regressor over the baseline feature panel as a robust boosting expert.",
    ),
    "xgboost_regressor": AlphaExpertSpec(
        name="xgboost_regressor",
        family="tree",
        task="cross_sectional_regression",
        description="XGBoost regressor over the baseline feature panel as a gradient-boosted expert.",
    ),
    "lstm_regressor": AlphaExpertSpec(
        name="lstm_regressor",
        family="sequence",
        task="cross_sectional_regression",
        description="PyTorch LSTM regressor over rolling windows of the baseline feature panel.",
    ),
    "transformer_regressor": AlphaExpertSpec(
        name="transformer_regressor",
        family="sequence",
        task="cross_sectional_regression",
        description="Lightweight PyTorch Transformer regressor over rolling windows of the baseline feature panel.",
    ),
    "ensemble_hist_gbm_ridge_mean": AlphaExpertSpec(
        name="ensemble_hist_gbm_ridge_mean",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Mean blend of normalized hist_gbm and ridge expert scores.",
        components=("hist_gbm", "ridge"),
        combine_method="mean_score",
    ),
    "ensemble_hist_gbm_ridge_rank": AlphaExpertSpec(
        name="ensemble_hist_gbm_ridge_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of hist_gbm and ridge expert predictions.",
        components=("hist_gbm", "ridge"),
        combine_method="rank_average",
    ),
    "ensemble_hist_gbm_random_forest_mean": AlphaExpertSpec(
        name="ensemble_hist_gbm_random_forest_mean",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Mean blend of hist_gbm and random_forest expert scores.",
        components=("hist_gbm", "random_forest"),
        combine_method="mean_score",
    ),
    "ensemble_hist_gbm_random_forest_rank": AlphaExpertSpec(
        name="ensemble_hist_gbm_random_forest_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of hist_gbm and random_forest expert predictions.",
        components=("hist_gbm", "random_forest"),
        combine_method="rank_average",
    ),
    "ensemble_hist_gbm_lightgbm_regressor_mean": AlphaExpertSpec(
        name="ensemble_hist_gbm_lightgbm_regressor_mean",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Mean blend of hist_gbm and lightgbm_regressor expert scores.",
        components=("hist_gbm", "lightgbm_regressor"),
        combine_method="mean_score",
    ),
    "ensemble_hist_gbm_lightgbm_regressor_rank": AlphaExpertSpec(
        name="ensemble_hist_gbm_lightgbm_regressor_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of hist_gbm and lightgbm_regressor expert predictions.",
        components=("hist_gbm", "lightgbm_regressor"),
        combine_method="rank_average",
    ),
    "ensemble_random_forest_lightgbm_regressor_mean": AlphaExpertSpec(
        name="ensemble_random_forest_lightgbm_regressor_mean",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Mean blend of random_forest and lightgbm_regressor expert scores.",
        components=("random_forest", "lightgbm_regressor"),
        combine_method="mean_score",
    ),
    "ensemble_random_forest_lightgbm_regressor_rank": AlphaExpertSpec(
        name="ensemble_random_forest_lightgbm_regressor_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of random_forest and lightgbm_regressor expert predictions.",
        components=("random_forest", "lightgbm_regressor"),
        combine_method="rank_average",
    ),
    "ensemble_hist_gbm_ridge_random_forest_mean": AlphaExpertSpec(
        name="ensemble_hist_gbm_ridge_random_forest_mean",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Mean blend of hist_gbm, ridge, and random_forest expert scores.",
        components=("hist_gbm", "ridge", "random_forest"),
        combine_method="mean_score",
    ),
    "ensemble_hist_gbm_ridge_random_forest_rank": AlphaExpertSpec(
        name="ensemble_hist_gbm_ridge_random_forest_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of hist_gbm, ridge, and random_forest expert predictions.",
        components=("hist_gbm", "ridge", "random_forest"),
        combine_method="rank_average",
    ),
    "ensemble_hist_gbm_random_forest_lightgbm_regressor_mean": AlphaExpertSpec(
        name="ensemble_hist_gbm_random_forest_lightgbm_regressor_mean",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Mean blend of hist_gbm, random_forest, and lightgbm_regressor expert scores.",
        components=("hist_gbm", "random_forest", "lightgbm_regressor"),
        combine_method="mean_score",
    ),
    "ensemble_hist_gbm_random_forest_lightgbm_regressor_rank": AlphaExpertSpec(
        name="ensemble_hist_gbm_random_forest_lightgbm_regressor_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of hist_gbm, random_forest, and lightgbm_regressor expert predictions.",
        components=("hist_gbm", "random_forest", "lightgbm_regressor"),
        combine_method="rank_average",
    ),
    "ensemble_extra_trees_hist_gbm_rank": AlphaExpertSpec(
        name="ensemble_extra_trees_hist_gbm_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of extra_trees and hist_gbm expert predictions.",
        components=("extra_trees", "hist_gbm"),
        combine_method="rank_average",
    ),
    "ensemble_extra_trees_lightgbm_ranker_rank": AlphaExpertSpec(
        name="ensemble_extra_trees_lightgbm_ranker_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of extra_trees and lightgbm_ranker expert predictions.",
        components=("extra_trees", "lightgbm_ranker"),
        combine_method="rank_average",
    ),
    "ensemble_extra_trees_hist_gbm_lightgbm_ranker_rank": AlphaExpertSpec(
        name="ensemble_extra_trees_hist_gbm_lightgbm_ranker_rank",
        family="ensemble",
        task="cross_sectional_ranking",
        description="Rank-average blend of extra_trees, hist_gbm, and lightgbm_ranker expert predictions.",
        components=("extra_trees", "hist_gbm", "lightgbm_ranker"),
        combine_method="rank_average",
    ),
}


def list_alpha_experts() -> tuple[AlphaExpertSpec, ...]:
    """Return all supported alpha experts in a stable order."""

    ordered_names = (
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
    return tuple(_ALPHA_EXPERT_SPECS[name] for name in ordered_names)


def list_alpha_expert_names() -> tuple[str, ...]:
    """Return all supported alpha expert names."""

    return tuple(spec.name for spec in list_alpha_experts())


def get_alpha_expert(name: str) -> AlphaExpertSpec:
    """Resolve one alpha expert spec or raise a user-facing error."""

    normalized_name = str(name).strip()
    try:
        return _ALPHA_EXPERT_SPECS[normalized_name]
    except KeyError as exc:  # pragma: no cover - trivial branch
        supported = ", ".join(list_alpha_expert_names())
        raise ValueError(f"Unsupported alpha expert '{name}'. Supported experts: {supported}.") from exc


def assert_supported_alpha_expert(name: str) -> str:
    """Validate one expert name and return the normalized identifier."""

    return get_alpha_expert(name).name
