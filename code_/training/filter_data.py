import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from pipeline_utils import unrolling_factory

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"

with open(HERE / "filters.json", "r") as f:
    FILTERS: dict[str, list[str]] = json.load(f)

with open(HERE / "subsets.json", "r") as f:
    SUBSETS: dict[str, list[str]] = json.load(f)


def get_appropriate_dataset(model: str, imputer: Optional[str] = None) -> pd.DataFrame:
    if model == "HGB" or imputer:
        dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
        print("Using dataset with NaNs")
    else:
        dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset.pkl"

    opv_dataset: pd.DataFrame = pd.read_pickle(dataset).reset_index(drop=True)
    return opv_dataset


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


# def filter_features(df: pd.DataFrame, feat_filter: str, **kwargs) -> pd.DataFrame:
#     """
#     Get a subset of columns from a DataFrame.
#
#     Args:
#         df: DataFrame to filter.
#         feat_filter: List of subsets with which to filter the DataFrame.
#         **kwargs: Keyword arguments to pass to get_subset.
#
#     Returns:
#         DataFrame with subset of columns.
#     """
#     feature_ids = get_feature_ids(feat_filter)
#     return get_subset(df, feature_ids, **kwargs)


# def get_subset(df: pd.DataFrame, feature_ids: list[str]) -> pd.DataFrame:
#     """
#     Get a subset of columns from a DataFrame.
#
#     Args:
#         df: DataFrame to filter.
#         feature_ids: List of columns to get (from a subsets.json file).
#         dropna: Whether to drop rows with NaN values.
#
#     Returns:
#         DataFrame with subset of columns.
#     """
#     subset_df: pd.DataFrame = df[feature_ids]
#     return subset_df


def sanitize_dataset(training_features: pd.DataFrame,
                     targets: pd.DataFrame,
                     dropna: bool,
                     **kwargs
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanitize the training features and targets in case the unrolled training features contain NaN values.

    Args:
        training_features: Training features.
        targets: Targets.
        dropna: Whether to drop NaN values.
        **kwargs: Keyword arguments to pass to filter_dataset.

    Returns:
        Sanitized training features and targets.
    """
    if dropna:
        training_features: pd.DataFrame = training_features.dropna()
        targets: pd.DataFrame = targets.loc[training_features.index]
        return training_features, targets
    else:
        return training_features, targets


def filter_dataset(raw_dataset: pd.DataFrame,
                   structure_feats: list[str],
                   scalar_feats: list[str],
                   target_feats: list[str],
                   dropna: bool = True,
                   unroll: Union[dict, list, None] = None,
                   **kwargs
                   ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    # Add multiple lists together as long as they are not NoneType
    all_feats: list[str] = [feat for feat_list in [structure_feats, scalar_feats, target_feats] if feat_list
                            for feat in feat_list]
    dataset: pd.DataFrame = raw_dataset[all_feats]

    if dropna:
        dataset: pd.DataFrame = dataset.dropna()

    if unroll:
        if isinstance(unroll, dict):
            structure_features: pd.DataFrame = unrolling_factory[unroll["representation"]](dataset[structure_feats],
                                                                                           **unroll)
        elif isinstance(unroll, list):
            multiple_unrolled_structure_feats: list[pd.DataFrame] = []
            for unroll_dict in unroll:
                single_structure_feat: pd.DataFrame = filter_dataset(dataset,
                                                                     structure_feats=unroll_dict["columns"],
                                                                     scalar_feats=[],
                                                                     target_feats=[],
                                                                     dropna=dropna,
                                                                     unroll=unroll_dict)[0]
                multiple_unrolled_structure_feats.append(single_structure_feat)
            structure_features: pd.DataFrame = pd.concat(multiple_unrolled_structure_feats, axis=1)
        else:
            raise ValueError(f"Unroll must be a dict or list, not {type(unroll)}")
    elif structure_feats:
        structure_features: pd.DataFrame = dataset[structure_feats]
    else:
        structure_features: pd.DataFrame = dataset[[]]

    scalar_features: pd.DataFrame = dataset[scalar_feats]

    # scalars_available, struct_available = not scalar_features.columns.empty, not structure_features.columns.empty
    # if struct_available and not scalars_available:
    #     training_features: pd.DataFrame = structure_features
    # elif scalars_available and not struct_available:
    #     training_features: pd.DataFrame = scalar_features
    # else:
    #     # scalar_features: pd.DataFrame = scalar_features.reset_index(drop=True)
    #     training_features: pd.DataFrame = pd.concat([structure_features, scalar_features], axis=1)
    # print(structure_features.loc[[369, 370, 371, 372, 379, 380, 381], :])
    training_features: pd.DataFrame = pd.concat([structure_features, scalar_features], axis=1)

    targets: pd.DataFrame = dataset[target_feats]

    training_features, targets = sanitize_dataset(training_features, targets, dropna=dropna, **kwargs)

    # if not (scalars_available and struct_available):
    new_struct_feats: list[str] = structure_features.columns.tolist()
    return training_features, targets, new_struct_feats
