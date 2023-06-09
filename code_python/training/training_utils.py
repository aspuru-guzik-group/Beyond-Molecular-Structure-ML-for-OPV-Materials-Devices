import json
from pathlib import Path
from typing import Optional, Union
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics._scorer import make_scorer, r2_scorer
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV

sys.path.append("../pipeline/")
from filter_data import filter_dataset, get_feature_ids
from pipeline_utils import representation_scaling_factory, scaler_factory
from models import regressor_factory, regressor_search_space


# Seeds for generating random states
with open("seeds.json", "r") as f:
    SEEDS: list[int] = json.load(f)
    # SEEDS: list[int] = SEEDS[:1]  # ATTN: Testing only

# Number of folds for cross-validation
N_FOLDS: int = 5
# N_FOLDS: int = 2  # ATTN: Testing only

# Number of iterations for Bayesian optimization
BO_ITER: int = 100
# BO_ITER: int = 1  # ATTN: Testing only


def rmse_score(y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the root mean squared error.

    Args:
        y_test: Test targets.
        y_pred: Predicted targets.

    Returns:
        Root mean squared error.
    """
    return mean_squared_error(y_test, y_pred, squared=False)


def np_r(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient.

    Args:
        y_true: Test targets.
        y_pred: Predicted targets.

    Returns:
        Pearson correlation coefficient.
    """
    y_true = y_true.to_numpy().flatten()
    # y_pred = y_pred.tolist()
    r = np.corrcoef(y_true, y_pred, rowvar=False)[0, 1]
    return r


def pearson(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = y_true.to_numpy().flatten()
    y_pred = y_pred.flatten()
    r = pearsonr(y_true, y_pred).statistic
    return r


# r_scorer = make_scorer(r_regression, greater_is_better=True)
# r_scorer = make_scorer(np_r, greater_is_better=True)
r_scorer = make_scorer(pearson, greater_is_better=True)
rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def preprocess_properties_and_processing(scalar_features: list[str],
                                         scaler_type: str,
                                         ) -> ColumnTransformer:
    transformer = Pipeline(steps=[("transformer", scaler_factory[scaler_type]())])
    preprocessor: ColumnTransformer = ColumnTransformer(
        remainder="passthrough",
        transformers=[
            ("stdscaler", transformer, scalar_features),
        ])
    return preprocessor


# def get_scores(regressor, X, y, cv) -> dict[str, float]:
#     scores: dict[str, float] = cross_validate(regressor, X, y,
#                                               cv=cv,
#                                               scoring={"r":    r_scorer,
#                                                        "r2":   r2_scorer,
#                                                        "rmse": rmse_scorer,
#                                                        "mae":  mae_scorer},
#                                               return_estimator=True,
#                                               n_jobs=-1,
#                                               )
#     return scores


def cross_validate_regressor(regressor, X, y, cv) -> tuple[dict[str, float], np.ndarray]:
    # Training and scoring on each fold
    scores: dict[str, float] = cross_validate(regressor, X, y,
                                              cv=cv,
                                              scoring={"r":    r_scorer,
                                                       "r2":   r2_scorer,
                                                       "rmse": rmse_scorer,
                                                       "mae":  mae_scorer},
                                              # return_estimator=True,
                                              n_jobs=-1,
                                              )

    # ATTN: Not exactly the same as getting predictions for each fold
    predictions: np.ndarray = cross_val_predict(regressor, X, y,
                                                cv=cv,
                                                n_jobs=-1,
                                                )
    return scores, predictions


def run_structure_only(dataset: pd.DataFrame,
                       representation: str,
                       structural_features: list[str],
                       target_features: list[str],
                       regressor_type: str,
                       hyperparameter_optimization: bool = False,
                       unroll: Optional[dict] = None,
                       **kwargs) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
    """
            Run the model.

            Args:
                dataset: Dataset to use.
                structural_features: Structural features to use.
                scaler: Scaler to use.
                regressor: Regressor to use.
                hyperparameter_optimization: Whether to optimize hyperparameters.
                **kwargs: Keyword arguments.

            Returns:
                None.
            """
    # Filter dataset
    X, y = filter_dataset(dataset,
                          structure_feats=structural_features,
                          scalar_feats=[],
                          target_feats=target_features,
                          unroll=unroll, )

    # TODO: Transformer for structural or categorical features?
    preprocessor: Pipeline = Pipeline(steps=[(representation_scaling_factory[representation]["type"],
                                              representation_scaling_factory[representation]["callable"]())])

    return run_(X, y,
                preprocessor=preprocessor,
                regressor_type=regressor_type,
                hyperparameter_optimization=hyperparameter_optimization,
                **kwargs)


def run_structure_and_scalar(dataset: pd.DataFrame,
                             structural_features: list[str],
                             scalar_filter: str,
                             target_features: list[str],
                             scaler_type: str,
                             regressor_type: str,
                             hyperparameter_optimization: bool = False,
                             unroll: Optional[dict] = None,
                             **kwargs) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
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
    X, y = filter_dataset(dataset,
                          structure_feats=structural_features,
                          scalar_feats=scalar_features,
                          target_feats=target_features,
                          unroll=unroll, )

    # Establish preprocessing and training pipeline
    numeric_transformer = Pipeline(steps=[
        (scaler_type, scaler_factory[scaler_type]()), ]
    )

    # TODO: Transformer for structural or categorical features?

    preprocessor: ColumnTransformer = ColumnTransformer(
        remainder="passthrough",
        transformers=[
            ("numeric", numeric_transformer, scalar_features),
        ])

    return run_(X, y,
                preprocessor=preprocessor,
                regressor_type=regressor_type,
                hyperparameter_optimization=hyperparameter_optimization,
                **kwargs)


def run_hyperparam_opt(X, y,
                       cv_outer: KFold,
                       n_folds: int,
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
        bayes.fit(X_train, y_train)
        print(f"Best parameters: {bayes.best_params_}")
        estimators.append(bayes)

    # Extract the best estimator from hyperparameter optimization
    best_idx: int = np.argmax([est.best_score_ for est in estimators])
    best_estimator: Pipeline = estimators[best_idx].best_estimator_
    return best_estimator


def run_(X, y,
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

        regressor = Pipeline(
            steps=[("preprocessor", preprocessor),
                   # ("regressor", regressor_factory[regressor_type](random_state=seed, **kwargs))]
                   ("regressor", regressor_factory[regressor_type](**kwargs))]
        )

        if hyperparameter_optimization:
            # Hyperparameter optimization
            best_estimator = run_hyperparam_opt(X, y, cv_outer=cv_outer, n_folds=N_FOLDS, seed=seed,
                                                regressor_type=regressor_type, regressor=regressor)

            scores, predictions = cross_validate_regressor(best_estimator, X, y, cv_outer)

        else:
            scores, predictions = cross_validate_regressor(regressor, X, y, cv_outer)

        seed_scores[seed] = scores
        seed_predictions[seed] = predictions.flatten()

    seed_predictions: pd.DataFrame = pd.DataFrame.from_dict(seed_predictions, orient="columns")
    return seed_scores, seed_predictions


def process_scores(scores: dict[int, dict[str, float]]) -> dict[Union[int, str], dict[str, float]]:
    avg_r = round(np.mean([seed["test_r"] for seed in scores.values()]), 2)
    stdev_r = round(np.std([seed["test_r"] for seed in scores.values()]), 2)
    avg_r2 = round(np.mean([seed["test_r2"] for seed in scores.values()]), 2)
    stdev_r2 = round(np.std([seed["test_r2"] for seed in scores.values()]), 2)
    print("Average scores:\t", f"r: {avg_r}±{stdev_r}\t", f"r2: {avg_r2}±{stdev_r2}")

    score_types: list[str] = ["r", "r2", "rmse", "mae"]
    avgs: list[float] = [np.mean([seed[f"test_{score}"] for seed in scores.values()]) for score in score_types]
    stdevs: list[float] = [np.std([seed[f"test_{score}"] for seed in scores.values()]) for score in score_types]
    for score, avg, stdev in zip(score_types, avgs, stdevs):
        scores[f"{score}_avg"] = avg
        scores[f"{score}_stdev"] = stdev

    return scores


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def save_results(scores: dict[int, dict[str, float]],
                 predictions: pd.DataFrame,
                 results_dir: Path,
                 subdir_ids: list[str],
                 regressor_type: str,
                 ) -> None:
    sub_dir_name: str = "_".join([str(id) for id in subdir_ids])
    sub_dir: Path = results_dir / sub_dir_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    fname_root: str = f"{regressor_type}_{sub_dir_name}"

    scores_file: Path = sub_dir / f"{fname_root}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)

    predictions_file: Path = sub_dir / f"{fname_root}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    print("Saved results to:")
    print(scores_file)
    print(predictions_file)
