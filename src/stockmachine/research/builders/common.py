from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE12_FEATURE_COLUMNS: tuple[str, ...] = (
    "gap_1",
    "ret_1d",
    "mom_5",
    "mom_10",
    "mom_20",
    "mom_60",
    "vol_20",
    "vol_60",
    "range_1d",
    "volume_ratio_20",
    "rel_mom_20",
    "rel_mom_60",
)

SHORT_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "gap_z_20",
    "intraday_return",
    "ret_2d",
    "mom_3",
    "vol_5",
    "vol_10",
    "range_5",
    "volume_ratio_5",
)

TREND_POSITION_FEATURE_COLUMNS: tuple[str, ...] = (
    "close_ma5_gap",
    "close_ma20_gap",
    "price_position_20d",
)

RELATIVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rel_ret_1d",
    "rel_ret_5d",
    "sector_rel_ret_1d",
)


def combine_feature_columns(*feature_groups: Iterable[str]) -> tuple[str, ...]:
    """Combine feature groups while preserving order and removing duplicates."""

    ordered: list[str] = []
    for group in feature_groups:
        for feature in group:
            if feature not in ordered:
                ordered.append(feature)
    return tuple(ordered)


BASELINE12_PLUS_SHORT_AND_RELATIVE_FEATURE_COLUMNS: tuple[str, ...] = combine_feature_columns(
    BASELINE12_FEATURE_COLUMNS,
    SHORT_STATE_FEATURE_COLUMNS,
    RELATIVE_FEATURE_COLUMNS,
)

FEATURE_COLUMNS: tuple[str, ...] = combine_feature_columns(
    BASELINE12_FEATURE_COLUMNS,
    SHORT_STATE_FEATURE_COLUMNS,
    TREND_POSITION_FEATURE_COLUMNS,
    RELATIVE_FEATURE_COLUMNS,
)


def build_numeric_linear_preprocessor(
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> ColumnTransformer:
    """Construct the shared preprocessing stack for linear baseline models."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(feature_columns),
            )
        ],
        remainder="drop",
    )


def build_linear_model_pipeline(
    model: object,
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Wrap one linear model with the shared preprocessing pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_numeric_linear_preprocessor(feature_columns=feature_columns)),
            ("model", model),
        ]
    )


def build_tree_model_pipeline(model: object) -> Pipeline:
    """Wrap one tree model with median imputation for the tabular feature panel."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort model frames consistently before fitting rankers or regressors."""

    sort_columns = [column for column in ("date", "symbol") if column in frame.columns]
    if not sort_columns:
        return frame.copy()
    return frame.sort_values(sort_columns).reset_index(drop=True)


def build_query_group_sizes(frame: pd.DataFrame) -> list[int]:
    """Build ranker query group sizes from one per-date panel frame."""

    prepared = prepare_model_frame(frame)
    if prepared.empty:
        return []
    return prepared.groupby("date", sort=False).size().tolist()
