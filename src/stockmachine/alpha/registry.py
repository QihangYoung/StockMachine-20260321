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
    "hist_gbm": AlphaExpertSpec(
        name="hist_gbm",
        family="tree",
        task="cross_sectional_regression",
        description="Histogram gradient boosting regressor over the baseline feature panel.",
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
}


def list_alpha_experts() -> tuple[AlphaExpertSpec, ...]:
    """Return all supported alpha experts in a stable order."""

    ordered_names = (
        "factor_baseline",
        "ridge",
        "hist_gbm",
        "ensemble_hist_gbm_ridge_mean",
        "ensemble_hist_gbm_ridge_rank",
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
