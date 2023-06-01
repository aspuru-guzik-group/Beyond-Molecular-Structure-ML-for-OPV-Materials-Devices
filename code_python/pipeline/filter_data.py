import pandas as pd

from code_python.pipeline import FILTERS, SUBSETS


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


def get_subset(df: pd.DataFrame, feature_ids: list[str], dropna: bool = False) -> pd.DataFrame:
    """
    Get a subset of columns from a DataFrame.

    Args:
        df: DataFrame to filter.
        feature_ids: List of columns to get (from a subsets.json file).
        dropna: Whether to drop rows with NaN values.

    Returns:
        DataFrame with subset of columns.
    """
    if dropna:
        subset_df: pd.DataFrame = df[feature_ids].dropna()
    else:
        subset_df: pd.DataFrame = df[feature_ids]
    return subset_df


def filter_dataset(raw_dataset: pd.DataFrame, structure_feats: list[str], scalar_feats: list[str],
                   target_feats: list[str], **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    structure_features: pd.DataFrame = raw_dataset[structure_feats]
    scalar_features: pd.DataFrame = get_subset(raw_dataset, scalar_feats, **kwargs)
    training_features: pd.DataFrame = pd.concat([structure_features, scalar_features], axis=1)

    targets: pd.DataFrame = raw_dataset[target_feats]

    return training_features, targets
