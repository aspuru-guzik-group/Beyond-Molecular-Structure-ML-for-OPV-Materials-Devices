import pandas as pd

from sklearn.model_selection import KFold
from typing import Optional

from skopt import BayesSearchCV

from code_python import DATASETS
from code_python.pipeline import HYPEROPT_SPACE
from code_python.pipeline.scaling import scale_features, scale_targets
from code_python.training import SEEDS
from code_python.pipeline.pipeline_utils import evaluate_model, regressor_factory, scaler_factory
from code_python.pipeline.filter_data import filter_dataset, get_feature_ids


def split_k_folds(x: pd.DataFrame,
                  y: pd.DataFrame,
                  seed: int,
                  scaler: Optional[str],
                  scalar_features: Optional[list[str]] = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into k folds and scales the features and targets with scaler.

    Args:
        x: Input features.
        y: Targets.
        seed: Random seed.
        scaler: Scaler to use.
        scalar_features: Features to scale.

    Returns:
        Scaled train and test features and targets.
    """
    kfold: KFold = KFold(n_splits=5, shuffle=True, random_state=seed)

    scaler = scaler_factory[scaler]() if scaler else None

    for train_index, test_index in kfold.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if scaler:
            # Scale train and test features independently
            x_train_scaled, x_test_scaled = scale_features(x_train, x_test, scaler, scalar_features)
            # Scale train and test targets independently
            y_train_scaled, y_test_scaled = scale_targets(y_train, y_test, scaler)

        else:
            x_train_scaled, x_test_scaled = x_train, x_test
            y_train_scaled, y_test_scaled = y_train, y_test

        yield x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled


def optimize_hyperparameters(x_train, y_train, regressor, seed: int) -> dict[str, float]:
    """
    Optimize hyperparameters.

    Args:
        dataset: Dataset to use.

    Returns:
        Dictionary of hyperparameters.
    """
    # Define the hyperparameter optimization function
    bayes_opt = BayesSearchCV(regressor(random_state=seed),
                              HYPEROPT_SPACE,
                              # TODO: Figure this out
                              cv=kfold,
                              # n_points=5,
                              # n_iter=10,
                              n_jobs=-1,
                              random_state=42)

    # Fit the hyperparameter optimization function on the training set
    bayes_opt.fit(x_train, y_train)

    # Get the optimal hyperparameters
    best_params = bayes_opt.best_params_
    return best_params


def run(dataset: pd.DataFrame,
        structural_features: list[str],
        scalar_filter: Optional[str],
        scaler: str,
        regressor: str,
        hyperparameter_optimization: bool = False,
        **kwargs) -> None:
    """
    Run the model.

    Args:
        dataset: Dataset to use.
        structural_features: Structural features to use.
        scalar_filter: Scalar features to use.
        scaler: Scaler to use.
        regressor: Regressor to use.
        hyperparameter_optimization: Whether to optimize hyperparameters.
        **kwargs: Keyword arguments.

    Returns:
        None.
    """
    # Select features to use in the model
    scalar_features: list[str] = get_feature_ids(scalar_filter)

    # Filter dataset
    x, y = filter_dataset(dataset, structure_feats=structural_features, scalar_feats=scalar_features, **kwargs)

    # Get splits
    for seed in SEEDS:
        for x_train, x_test, y_train, y_test in split_k_folds(x, y, seed, scaler, scalar_features):

            if hyperparameter_optimization:
                # Find best hyperparameters
                best_params = optimize_hyperparameters(x_train, y_train, regressor, seed)
                # Train model with optimal hyperparameters
                regressor = regressor_factory[regressor](random_state=seed, **best_params)
            else:
                # Train model
                regressor = regressor_factory[regressor](random_state=seed, **kwargs)
                regressor.fit(x_train, y_train)

            # Evaluate model
            # "fold": [], "r": [], "r2": [], "rmse": [], "mae": []
            y_pred = regressor.predict(x_test)
            scores: dict[str, float] = evaluate_model(y_test, y_pred)


def main():
    dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
    opv_dataset: pd.DataFrame = pd.read_pickle(dataset)

    radius = 6
    n_bits = 4096
    structural_features: list[str] = [f"Donor ECFP{2 * radius}_{n_bits}", f"Acceptor ECFP{2 * radius}_{n_bits}"]

    run(opv_dataset, structural_features)
