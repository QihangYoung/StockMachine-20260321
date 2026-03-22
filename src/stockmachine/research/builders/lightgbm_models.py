from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from stockmachine.research.builders.common import FEATURE_COLUMNS, build_tree_model_pipeline, prepare_model_frame

LIGHTGBM_IMPORT_ERROR = (
    "LightGBM is required for the P1 model builders. Install the `lightgbm` package "
    "or choose a different builder."
)


def _load_lightgbm() -> object:
    """Import LightGBM lazily so missing dependency errors are user-friendly."""

    try:
        return importlib.import_module("lightgbm")
    except ImportError as exc:  # pragma: no cover - exercised via explicit regression test
        raise ImportError(LIGHTGBM_IMPORT_ERROR) from exc


def build_lightgbm_regressor() -> Pipeline:
    """Construct a LightGBM regressor over the baseline feature panel."""

    lightgbm = _load_lightgbm()
    return build_tree_model_pipeline(
        lightgbm.LGBMRegressor(
            objective="regression",
            learning_rate=0.03,
            n_estimators=400,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=7,
            n_jobs=-1,
        )
    )


@dataclass(slots=True)
class LightGBMRankerModel:
    """Small adapter that handles imputation and group-aware fitting."""

    feature_columns: tuple[str, ...] = field(default_factory=lambda: FEATURE_COLUMNS)
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    model: object | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        *,
        group: Sequence[int],
    ) -> "LightGBMRankerModel":
        frame = prepare_model_frame(X)
        features = frame[list(self.feature_columns)]
        model = self._ensure_model()
        rank_labels = self._rank_groupwise_labels(y, group)
        model.fit(self.imputer.fit_transform(features), rank_labels, group=list(group))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = prepare_model_frame(X)
        features = frame[list(self.feature_columns)]
        model = self._ensure_model()
        transformed = self.imputer.transform(features)
        return np.asarray(model.predict(transformed))

    def _ensure_model(self) -> object:
        if self.model is None:
            lightgbm = _load_lightgbm()
            self.model = lightgbm.LGBMRanker(
                objective="lambdarank",
                learning_rate=0.03,
                n_estimators=400,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=7,
                n_jobs=-1,
            )
        return self.model

    @staticmethod
    def _rank_groupwise_labels(y: Sequence[float], group: Sequence[int]) -> np.ndarray:
        values = pd.Series(list(y), dtype=float)
        group_sizes = list(group)
        if values.empty:
            return np.asarray([], dtype=int)
        if sum(group_sizes) != len(values):
            raise ValueError("group sizes must sum to the number of training rows")

        group_ids: list[int] = []
        for group_index, group_size in enumerate(group_sizes):
            group_ids.extend([group_index] * group_size)

        # LightGBM's default label_gain supports relevance labels in [0, 30].
        # We map each date's continuous return labels to percentile-based buckets
        # so larger future returns still receive larger relevance grades.
        percentile_ranks = values.groupby(group_ids, sort=False).rank(
            method="first",
            ascending=True,
            pct=True,
        )
        scaled_ranks = np.floor(percentile_ranks.to_numpy() * 30.0).astype(int)
        return np.clip(scaled_ranks, 0, 30)


def build_lightgbm_ranker(*, feature_columns: Iterable[str] = FEATURE_COLUMNS) -> LightGBMRankerModel:
    """Construct a LightGBM ranker for cross-sectional selection."""

    return LightGBMRankerModel(feature_columns=tuple(feature_columns))


def get_lightgbm_model_builders() -> dict[str, object]:
    """Return the built-in LightGBM model builders."""

    return {
        "lightgbm_regressor": build_lightgbm_regressor,
        "lightgbm_ranker": build_lightgbm_ranker,
    }
