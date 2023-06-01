from typing import Optional, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def do_scaling(df: pd.DataFrame,
               scaler: Optional[MinMaxScaler, StandardScaler],
               scalar_features: Optional[list[str]] = None
               ) -> tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    """
    Scale features using a scaler.

    Args:
        df: DataFrame of all features.
        scalar_features: Features to scale.
        scaler: Instance of scaler to use.

    Returns:
        Scaled features and the instance of the scaler.
    """
    if scalar_features is None:
        all_features = df

    else:
        # Split scalar and structural features
        scalar_feats: pd.DataFrame = df[scalar_features]
        structural_feats: pd.DataFrame = df.drop(scalar_features, axis=1)
        # Scale scalar features
        scaled_features: pd.DataFrame = pd.DataFrame(scaler.fit_transform(scalar_feats), columns=scalar_feats.columns)
        # Rejoin features
        all_features: pd.DataFrame = pd.concat([structural_feats, scaled_features], axis=1)

    return all_features, scaler


def scale_features(train: pd.DataFrame,
                   test: pd.DataFrame,
                   scaler: Union[MinMaxScaler, StandardScaler],
                   scalar_features: Optional[list[str]] = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using a scaler.

    Args:
        train: Training features.
        test: Testing features.
        scaler: Instance of scaler to use.
        scalar_features: Features to scale.

    Returns:
        Scaled train and test features.
    """
    train_scaled, scaler = do_scaling(train, scaler, scalar_features)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled


def scale_targets(train: pd.DataFrame,
                  test: pd.DataFrame,
                  scaler: Union[MinMaxScaler, StandardScaler]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale targets using a scaler.

    Args:
        train: Training targets.
        test: Testing targets.
        scaler: Instance of scaler to use.

    Returns:
        Scaled train and test targets.
    """
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
    return train_scaled, test_scaled
