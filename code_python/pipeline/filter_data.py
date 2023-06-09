import json
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline_utils import unrolling_factory

HERE = Path(__file__).parent

with open(HERE / "filters.json", "r") as f:
    FILTERS: dict[str, list[str]] = json.load(f)

with open(HERE / "subsets.json", "r") as f:
    SUBSETS: dict[str, list[str]] = json.load(f)

with open(HERE / "hyperopt_space.json", "r") as f:
    HYPEROPT_SPACE: dict[str, dict[str, list]] = json.load(f)


def get_feature_ids(feat_filter: str) -> list[str]:
    """
    Get a subset of columns from a DataFrame.

    Args:
        feat_filter: List of subsets with which to filter the DataFrame.

    Returns:
        List of feature names.
    """
    filters: list[str] = FILTERS[feat_filter]
    feature_ids: list[str] = []
    for subset_id in filters:
        feature_ids.extend(SUBSETS[subset_id])
    return feature_ids


def filter_features(df: pd.DataFrame, feat_filter: str, **kwargs) -> pd.DataFrame:
    """
    Get a subset of columns from a DataFrame.

    Args:
        df: DataFrame to filter.
        feat_filter: List of subsets with which to filter the DataFrame.
        **kwargs: Keyword arguments to pass to get_subset.

    Returns:
        DataFrame with subset of columns.
    """
    feature_ids = get_feature_ids(feat_filter)
    return get_subset(df, feature_ids, **kwargs)


def get_subset(df: pd.DataFrame, feature_ids: list[str]) -> pd.DataFrame:
    """
    Get a subset of columns from a DataFrame.

    Args:
        df: DataFrame to filter.
        feature_ids: List of columns to get (from a subsets.json file).
        dropna: Whether to drop rows with NaN values.

    Returns:
        DataFrame with subset of columns.
    """
    subset_df: pd.DataFrame = df[feature_ids]
    return subset_df


def filter_dataset(raw_dataset: pd.DataFrame,
                   structure_feats: list[str],
                   scalar_feats: list[str],
                   target_feats: list[str],
                   dropna: bool = True,
                   unroll: Optional[dict] = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the dataset.

    Args:
        raw_dataset: Raw dataset.
        structure_feats: Structure features.
        scalar_feats: Scalar features.
        target_feats: Target features.

    Returns:
        Input features and targets.
    """
    dataset: pd.DataFrame = raw_dataset[structure_feats + scalar_feats + target_feats]
    if dropna:
        dataset: pd.DataFrame = dataset.dropna()

    if unroll:  # TODO: This gets complicated if unrolling multiple columns?
        structure_features: pd.DataFrame = unrolling_factory[unroll["representation"]](dataset[structure_feats], **unroll)
    else:
        structure_features: pd.DataFrame = dataset[structure_feats]

    scalar_features: pd.DataFrame = dataset[scalar_feats]

    scalars_available: bool = not scalar_features.columns.empty
    struct_available: bool = not structure_features.columns.empty
    if struct_available and not scalars_available:
        training_features: pd.DataFrame = structure_features
    elif scalars_available and not struct_available:
        training_features: pd.DataFrame = scalar_features
    else:
        scalar_features: pd.DataFrame = scalar_features.reset_index(drop=True)
        training_features: pd.DataFrame = pd.concat([structure_features, scalar_features], axis=1)

    targets: pd.DataFrame = dataset[target_feats]

    return training_features, targets
