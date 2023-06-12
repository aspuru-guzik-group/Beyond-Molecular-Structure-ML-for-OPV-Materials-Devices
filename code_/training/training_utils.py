import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from skopt import BayesSearchCV

from data_handling import save_results, target_abbrev
from filter_data import filter_dataset, get_feature_ids
from models import model_dropna, regressor_factory, regressor_search_space
from pipeline_utils import get_feature_pipelines
from scoring import cross_validate_regressor, process_scores

# from pipeline_utils import representation_scaling_factory

HERE: Path = Path(__file__).resolve().parent

# Seeds for generating random states
with open("seeds.json", "r") as f:
    SEEDS: list[int] = json.load(f)
    # SEEDS: list[int] = SEEDS[:1]  # ATTN: Testing only

# Number of folds for cross-validation
N_FOLDS: int = 5
# N_FOLDS: int = 2  # ATTN: Testing only

# Number of iterations for Bayesian optimization
BO_ITER: int = 50


# BO_ITER: int = 1  # ATTN: Testing only


# def preprocess_properties_and_processing(scalar_features: list[str],
#                                          scaler_type: str,
#                                          ) -> ColumnTransformer:
#     transformer = Pipeline(steps=[("transformer", scaler_factory[scaler_type]())])
#     preprocessor: ColumnTransformer = ColumnTransformer(
#         remainder="passthrough",
#         transformers=[
#             ("stdscaler", transformer, scalar_features),
#         ])
#     return preprocessor


# def iterate_and_remove(obj):
#     acceptable_types = (str, int, float, np.ndarray, list, dict, tuple)
#     if isinstance(obj, list):
#         obj = [iterate_and_remove(element) for element in obj if isinstance(element, acceptable_types)]
#         # for element in obj:
#         #     iterate_and_remove(element)
#     elif isinstance(obj, tuple):
#         obj = list(obj)
#         obj = [iterate_and_remove(element) for element in obj if isinstance(element, acceptable_types)]
#         # for element in obj:
#         #     iterate_and_remove(element)
#     elif isinstance(obj, dict):
#         obj = {key: iterate_and_remove(value) for key, value in obj.items() if isinstance(value, acceptable_types)}
#         # keys_to_remove = []
#         # for key, value in obj.items():
#         #     if not isinstance(value, acceptable_types):
#         #         keys_to_remove.append(key)
#         #     else:
#         #         iterate_and_remove(value)
#         # for key in keys_to_remove:
#         #     del obj[key]
#     # elif isinstance(obj, (list, dict, tuple, np.ndarray)):
#     #     obj = obj
#     else:
#         # If obj is not a list or a dictionary, remove it (optional)
#         obj = None
#
#     return obj

# def convert_estimator_params(estimator_params: Union[list, dict]) -> Union[list, dict]:
#     if isinstance(estimator_params, list):
#         for obj in estimator_params:
#             obj = convert_estimator_params(obj)
#     elif isinstance(estimator_params, dict):
#         for key, value in estimator_params.items():
#             estimator_params[key] = convert_estimator_params(value)
#     else:
#
#
#
# if isinstance(value, np.int64):
#     estimator_params[key] = int(value)
# elif isinstance(value, np.float64):
#     estimator_params[key] = float(value)
# elif isinstance(value, np.ndarray):
#     estimator_params[key] = value.tolist()


# def run_structure_only(dataset: pd.DataFrame,
#                        representation: str,
#                        structural_features: list[str],
#                        target_features: list[str],
#                        regressor_type: str,
#                        hyperparameter_optimization: bool = False,
#                        unroll: Union[dict, list, None] = None,
#                        **kwargs) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
#     """
#             Run the model.
#
#             Args:
#                 dataset: Dataset to use.
#                 structural_features: Structural features to use.
#                 scaler: Scaler to use.
#                 regressor: Regressor to use.
#                 hyperparameter_optimization: Whether to optimize hyperparameters.
#                 **kwargs: Keyword arguments.
#
#             Returns:
#                 None.
#             """
#     # Filter dataset
#     X, y = filter_dataset(dataset,
#                           structure_feats=structural_features,
#                           scalar_feats=[],
#                           target_feats=target_features,
#                           unroll=unroll,
#                           dropna=model_dropna(regressor_type)
#                           )
#
#     # Establish preprocessing and training pipeline
#     preprocessor: Pipeline = Pipeline(steps=[(representation_scaling_factory[representation]["type"],
#                                               representation_scaling_factory[representation]["callable"]())])
#
#     return _run(X, y,
#                 preprocessor=preprocessor,
#                 regressor_type=regressor_type,
#                 hyperparameter_optimization=hyperparameter_optimization,
#                 **kwargs)


def train_regressor(dataset: pd.DataFrame,
                    representation: str,
                    structural_features: list[str],
                    unroll: Union[dict[str, str], list[dict[str, str]], None],
                    scalar_filter: Optional[str],
                    subspace_filter: Optional[str],
                    regressor_type: str,
                    target_features: list[str],
                    hyperparameter_optimization: bool,
                    ) -> None:
    scores, predictions = _prepare_data(dataset=dataset,
                                        representation=representation,
                                        structural_features=structural_features,
                                        unroll=unroll,
                                        scalar_filter=scalar_filter,
                                        subspace_filter=subspace_filter,
                                        target_features=target_features,
                                        regressor_type=regressor_type,
                                        hyperparameter_optimization=hyperparameter_optimization,
                                        )

    scores = process_scores(scores)

    targets_dir: str = "-".join([target_abbrev[target] for target in target_features])
    features_dir: str = "-".join([representation])
    results_dir: Path = HERE.parent.parent / "results" / f"target_{targets_dir}" / f"features_{features_dir}"
    save_results(scores, predictions,
                 results_dir=results_dir,
                 regressor_type=regressor_type,
                 hyperparameter_optimization=hyperparameter_optimization,
                 )


def _prepare_data(dataset: pd.DataFrame,
                  representation: str,
                  structural_features: list[str],
                  scalar_filter: Optional[str],
                  subspace_filter: Optional[str],
                  target_features: list[str],
                  regressor_type: str,
                  unroll: Union[dict, list, None] = None,
                  hyperparameter_optimization: bool = False,
                  **kwargs
                  ) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
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
    if scalar_filter:
        scalar_features: list[str] = get_feature_ids(scalar_filter)
        if subspace_filter:
            scalar_features: list[str] = get_feature_ids(subspace_filter)
    else:
        scalar_features: list[str] = []

    # Filter dataset
    X, y, unrolled_feats = filter_dataset(dataset,
                                          structure_feats=structural_features,
                                          scalar_feats=scalar_features,
                                          target_feats=target_features,
                                          unroll=unroll,
                                          dropna=model_dropna(regressor_type)
                                          )

    transformers: list[tuple[str, Pipeline, list[str]]] = get_feature_pipelines(unrolled_features=unrolled_feats,
                                                                                representation=representation,
                                                                                numeric_features=scalar_features)

    preprocessor: ColumnTransformer = ColumnTransformer(
        # remainder="passthrough",
        transformers=[*transformers,
                      # (representation_scaling_factory[representation]["type"],
                      # representation_scaling_factory[representation]["callable"](), new_struct_feats),
                      # ()
                      # ("power", PowerTransformer(), power_numeric_feats),
                      # ("minmax", MinMaxScaler(), minmax_numeric_feats)
                      ])

    return _run(X, y,
                preprocessor=preprocessor,
                regressor_type=regressor_type,
                hyperparameter_optimization=hyperparameter_optimization,
                **kwargs)


def _run(X, y,
         preprocessor: Union[ColumnTransformer, Pipeline],
         regressor_type: str,
         hyperparameter_optimization: bool = False,
         **kwargs) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
    # Get seeds for initializing random state of splitting and training
    seed_scores: dict[int, dict[str, float]] = {}
    seed_predictions: dict[int, np.ndarray] = {}
    for seed in SEEDS:

        # Splitting for model cross-validation
        cv_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        # MinMax scale everything if model is a neural network
        if regressor_type == "NN":
            y_transform = Pipeline(
                steps=[("power", PowerTransformer()),
                       ("minmax", MinMaxScaler())])
            preprocessor = Pipeline(
                steps=[("preprocessor", preprocessor),
                       ("minmax", MinMaxScaler())])
        else:
            y_transform = PowerTransformer()

        y_transform_regressor: TransformedTargetRegressor = TransformedTargetRegressor(
            regressor=regressor_factory[regressor_type](**kwargs),
            transformer=y_transform
        )

        # # Set random state for preprocessing
        # if isinstance(preprocessor, ColumnTransformer) and "power" in preprocessor.transformers:
        #     preprocessor.set_params(**{"power": {"random_state": seed}})

        regressor = Pipeline(
            steps=[("preprocessor", preprocessor),
                   # ("regressor", regressor_factory[regressor_type](random_state=seed, **kwargs))]
                   ("regressor", y_transform_regressor)]
        )

        if hyperparameter_optimization:
            # Hyperparameter optimization
            best_estimator = _optimize_hyperparams(X, y, cv_outer=cv_outer, seed=seed,
                                                   regressor_type=regressor_type, regressor=regressor)

            scores, predictions = cross_validate_regressor(best_estimator, X, y, cv_outer)

        else:
            scores, predictions = cross_validate_regressor(regressor, X, y, cv_outer)

        seed_scores[seed] = scores
        seed_predictions[seed] = predictions.flatten()

    seed_predictions: pd.DataFrame = pd.DataFrame.from_dict(seed_predictions, orient="columns")
    return seed_scores, seed_predictions


def _optimize_hyperparams(X, y,
                          cv_outer: KFold,
                          seed: int,
                          regressor_type: str,
                          regressor: Pipeline) -> Pipeline:
    # Splitting for outer cross-validation loop
    estimators: list[BayesSearchCV] = []
    for train_index, test_index in cv_outer.split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # Splitting for inner hyperparameter optimization loop
        cv_inner = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        # Bayesian hyperparameter optimization
        bayes = BayesSearchCV(regressor,
                              regressor_search_space[regressor_type],
                              n_iter=BO_ITER,
                              cv=cv_inner,
                              n_jobs=-1,
                              random_state=seed,
                              refit=True,
                              scoring="r2",
                              return_train_score=True,
                              )
        # bayes.fit(X_train, y_train)
        bayes.fit(X_train, y_train)
        print(f"Best parameters: {bayes.best_params_}")
        estimators.append(bayes)

    # Extract the best estimator from hyperparameter optimization
    best_idx: int = np.argmax([est.best_score_ for est in estimators])
    best_estimator: Pipeline = estimators[best_idx].best_estimator_
    return best_estimator
