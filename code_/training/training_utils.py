import json
import platform
from pathlib import Path
from typing import Callable, Optional, Union
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from skopt import BayesSearchCV
from skorch.regressor import NeuralNetRegressor

import torch
import torch.nn as nn
from pytorch_models import GNNPredictor, GPRegressor, NNModel
from torch.utils.data import DataLoader
# from _ml_for_opvs.ML_models.pytorch.data.data_utils import PolymerDataset
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR
import time

from data_handling import remove_unserializable_keys, save_results
from filter_data import filter_dataset, get_feature_ids
from models import (
    ecfp_only_kernels,
    get_ecfp_only_kernel,
    hyperopt_by_default,
    model_dropna,
    regressor_factory,
    regressor_search_space,
    get_skorch_nn,
)
from pipeline_utils import (
    generate_feature_pipeline,
    get_feature_pipelines,
    imputer_factory,
)
from scoring import (
    cross_validate_multioutput_regressor,
    cross_validate_regressor,
    process_scores,
    multi_scorer,
    score_lookup,
)
from scipy.stats import pearsonr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# from pipeline_utils import representation_scaling_factory

HERE: Path = Path(__file__).resolve().parent

os_type: str = platform.system().lower()
TEST: bool = False if os_type == "linux" else True

# Seeds for generating random states
with open(HERE / "seeds.json", "r") as f:
    SEEDS: list[int] = json.load(f)
    SEEDS: list[int] = SEEDS if not TEST else SEEDS[:1]

# Number of folds for cross-validation
N_FOLDS: int = 5 if not TEST else 2

# Number of iterations for Bayesian optimization
BO_ITER: int = 42 if not TEST else 1

# Path to config for Pytorch model
CONFIG_PATH: Path = HERE / "ANN_config.json"

# Set seed for PyTorch model
torch.manual_seed(0)


def train_regressor(
    dataset: pd.DataFrame,
    representation: str,
    structural_features: Optional[list[str]],
    unroll: Union[dict[str, str], list[dict[str, str]], None],
    scalar_filter: Optional[str],
    subspace_filter: Optional[str],
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    imputer: Optional[str] = None,
    output_dir_name: str = "results",
) -> None:
    # try:
    scores, predictions = _prepare_data(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll,
        scalar_filter=scalar_filter,
        subspace_filter=subspace_filter,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        imputer=imputer,
        hyperparameter_optimization=hyperparameter_optimization,
    )
    scores = process_scores(scores)
    save_results(
        scores,
        predictions,
        representation=representation,
        scalar_filter=scalar_filter,
        subspace_filter=subspace_filter,
        target_features=target_features,
        regressor_type=regressor_type,
        imputer=imputer,
        hyperparameter_optimization=hyperparameter_optimization,
        output_dir_name=output_dir_name,
    )


# except Exception as e:
#     print(f"\n\nEXCEPTION ENCOUNTERED. Failed to train {regressor_type} on {representation}...\n\n", e)


def get_hgb_features(filter: str, regressor_type: str) -> str:
    if regressor_type == "HGB" and filter != "material properties":
        return filter + " all"
    else:
        return filter


def _prepare_data(
    dataset: pd.DataFrame,
    representation: str,
    structural_features: list[str],
    scalar_filter: Optional[str],
    subspace_filter: Optional[str],
    target_features: list[str],
    regressor_type: str,
    unroll: Union[dict, list, None] = None,
    transform_type: str = "Standard",
    hyperparameter_optimization: bool = False,
    imputer: Optional[str] = None,
    **kwargs,
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
        scalar_filter = get_hgb_features(scalar_filter, regressor_type)
        scalar_features: list[str] = get_feature_ids(scalar_filter)
        print("n numeric features:", len(scalar_features))
        if subspace_filter:
            subspace_filter = get_hgb_features(subspace_filter, regressor_type)
            scalar_features: list[str] = get_feature_ids(subspace_filter)
            print("n numeric features in subspace:", len(scalar_features))
    else:
        scalar_features: list[str] = []

    # Filter dataset
    X, y, unrolled_feats = filter_dataset(
        dataset,
        structure_feats=structural_features,
        scalar_feats=scalar_features,
        target_feats=target_features,
        unroll=unroll,
        dropna=model_dropna(regressor_type),
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = get_feature_pipelines(
        unrolled_features=unrolled_feats,
        representation=representation,
        numeric_features=scalar_features,
    )

    if imputer:
        transformers.append(
            (f"{imputer} impute", imputer_factory[imputer], scalar_features)
        )
        print("Using imputer:", imputer)

    preprocessor: ColumnTransformer = ColumnTransformer(transformers=[*transformers])
    if regressor_type in ecfp_only_kernels:
        kernel: Union[str, Callable] = get_ecfp_only_kernel(
            representation, scalar_filter, regressor_type
        )
        print("Using kernel:", kernel)
        # kernel: str = "tanimoto" if representation == "ECFP" else "rbf"
        return _run(
            X,
            y,
            preprocessor=preprocessor,
            regressor_type=regressor_type,
            transform_type=transform_type,
            hyperparameter_optimization=hyperparameter_optimization,
            kernel=kernel,
            **kwargs,
        )
    else:
        return _run(
            X,
            y,
            preprocessor=preprocessor,
            regressor_type=regressor_type,
            transform_type=transform_type,
            hyperparameter_optimization=hyperparameter_optimization,
            **kwargs,
        )


def _run(
    X,
    y,
    preprocessor: Union[ColumnTransformer, Pipeline],
    regressor_type: str,
    transform_type: str,
    hyperparameter_optimization: bool = False,
    **kwargs,
) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
    # Get seeds for initializing random state of splitting and training
    seed_scores: dict[int, dict[str, float]] = {}
    seed_predictions: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        # Splitting for model cross-validation
        cv_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        # MinMax scale everything if model is a neural network
        if regressor_type in ["NN", "ANN"]:
            y_transform = Pipeline(
                steps=[
                    *[
                        (step[0], step[1])
                        for step in generate_feature_pipeline(transform_type).steps
                    ],
                    ("MinMax NN", MinMaxScaler()),
                ]
            )
            preprocessor = Pipeline(
                steps=[("preprocessor", preprocessor), ("MinMax NN", MinMaxScaler())]
            )
        else:
            y_transform: Pipeline = generate_feature_pipeline(transform_type)

        # Handling non-native multi-output regressors and the skorch-based ANN
        if y.shape[1] > 1 and regressor_type not in ["RF", "ANN"]:
            y_transform_regressor: TransformedTargetRegressor = TransformedTargetRegressor(
                regressor=MultiOutputRegressor(
                    estimator=regressor_factory[regressor_type]()
                ),
                # regressor_factory[regressor_type](),
                transformer=y_transform,
            )
        # elif regressor_type == "ANN":
        #     y_dims = y.shape[1]
        #     y = _pd_to_np(y)
        #     y = y.reshape(-1, y_dims)
        #     X = _pd_to_np(X)
        #     # convert to numpy array with float 32
        #     X = X.astype(np.float32)
        #     y = y.astype(np.float32)
        #
        #     # load NNModel parameters from config file
        #     with open(CONFIG_PATH, "r") as f:
        #         config = json.load(f)

            # scores, predictions = run_pytorch(X, y, cv_outer, config)

        else:
            y_transform_regressor: TransformedTargetRegressor = TransformedTargetRegressor(
                # regressor=regressor_factory[regressor_type](**kwargs),
                regressor=regressor_factory[regressor_type](),
                transformer=y_transform,
            )

        if regressor_type != "ANN":
            regressor = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", y_transform_regressor),
                ]
            )
            regressor.set_output(transform="pandas")

        # ATTN: Hyperparameter optimization is only automatically implemented for the following regressors KNN, NN.
        if hyperparameter_optimization or (regressor_type in hyperopt_by_default):
            best_estimator, regressor_params = _optimize_hyperparams(
                X,
                y,
                cv_outer=cv_outer,
                seed=seed,
                regressor_type=regressor_type,
                regressor=regressor,
            )
            scores, predictions = cross_validate_regressor(
                best_estimator, X, y, cv_outer
            )
            scores["best_params"] = regressor_params
        elif regressor_type == "ANN":
            pass
        else:
            scores, predictions = cross_validate_regressor(regressor, X, y, cv_outer)
        seed_scores[seed] = scores
        seed_predictions[seed] = predictions.flatten()
        print(f"{scores=}", f"{predictions=}")

    seed_predictions: pd.DataFrame = pd.DataFrame.from_dict(
        seed_predictions, orient="columns"
    )
    return seed_scores, seed_predictions


def _pd_to_np(data):
    if isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError("Data must be either a pandas DataFrame or a numpy array.")


def _optimize_hyperparams(
    X, y, cv_outer: KFold, seed: int, regressor_type: str, regressor: Pipeline
) -> Pipeline:
    # Splitting for outer cross-validation loop
    estimators: list[BayesSearchCV] = []
    for train_index, test_index in cv_outer.split(X, y):
        X_train = split_for_training(X, train_index)
        y_train = split_for_training(y, train_index)
        # if isinstance([X, y], pd.DataFrame):
        #     X_train = X.iloc[train_index]
        #     y_train = y.iloc[train_index]
        # elif isinstance([X, y], np.ndarray):
        #     X_train = X[train_index]
        #     y_train = y[train_index]

        # Splitting for inner hyperparameter optimization loop
        cv_inner = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        print("\n\n")
        print(
            "OPTIMIZING HYPERPARAMETERS FOR REGRESSOR", regressor_type, "\tSEED:", seed
        )
        # Bayesian hyperparameter optimization
        bayes = BayesSearchCV(
            regressor,
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
        print(f"\n\nBest parameters: {bayes.best_params_}\n\n")
        estimators.append(bayes)

    # Extract the best estimator from hyperparameter optimization
    best_idx: int = np.argmax([est.best_score_ for est in estimators])
    best_estimator: Pipeline = estimators[best_idx].best_estimator_
    try:
        regressor_params: dict = best_estimator.named_steps.regressor.get_params()
        regressor_params = remove_unserializable_keys(regressor_params)
    except:
        regressor_params = {"bad params": "couldn't get them"}

    return best_estimator, regressor_params


def split_for_training(
    data: Union[pd.DataFrame, np.ndarray], indices: np.ndarray
) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        split_data = data.iloc[indices]
    elif isinstance(data, np.ndarray):
        split_data = data[indices]
    else:
        raise ValueError("Data must be either a pandas DataFrame or a numpy array.")
    return split_data


def run_graphs_only(
    dataset: pd.DataFrame,
    structural_features: list[str],
    target_features: list[str],
    regressor_type: str,
    hyperparameter_optimization: bool = False,
    unroll: Optional[dict] = None,
    **kwargs,
) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
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
    X, y, new_struct_feats = filter_dataset(
        dataset,
        structure_feats=structural_features,
        scalar_feats=[],
        target_feats=target_features,
        unroll=unroll,
        dropna=model_dropna(regressor_type),
    )

    return _run_graphs(
        X,
        y,
        regressor_type=regressor_type,
        hyperparameter_optimization=hyperparameter_optimization,
        **kwargs,
    )


def _run_graphs(
    X, y, regressor_type: str, hyperparameter_optimization: bool = False, **kwargs
) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:
    # Get seeds for initializing random state of splitting and training
    seed_scores: dict[int, dict[str, float]] = {}
    seed_predictions: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        # Splitting for model cross-validation
        cv_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        y_transform = QuantileTransformer(
            output_distribution="normal", random_state=seed
        )

        y_transform_regressor: TransformedTargetRegressor = TransformedTargetRegressor(
            regressor=regressor_factory[regressor_type](**kwargs),
            transformer=y_transform,
        )

        regressor = Pipeline(
            steps=[  # ("preprocessor", preprocessor),
                # ("regressor", regressor_factory[regressor_type](random_state=seed, **kwargs))]
                ("regressor", y_transform_regressor)
            ]
        )

        if hyperparameter_optimization:
            # Hyperparameter optimization
            best_estimator = _optimize_hyperparams(
                X,
                y,
                cv_outer=cv_outer,
                seed=seed,
                regressor_type=regressor_type,
                regressor=regressor,
            )

            scores, predictions = cross_validate_regressor(
                best_estimator, X, y, cv_outer
            )

        else:
            scores, predictions = cross_validate_regressor(regressor, X, y, cv_outer)

        seed_scores[seed] = scores
        seed_predictions[seed] = predictions.flatten()

    seed_predictions: pd.DataFrame = pd.DataFrame.from_dict(
        seed_predictions, orient="columns"
    )
    return seed_scores, seed_predictions


# def run_pytorch(X, y, cv_outer, config):
#     """Runs the pytorch model given a fold of the data.
#
#     Args:
#         X (torch.tensor): One fold of the input data from cross-validation
#         y (torch.tensor): One fold of the output data from cross-validation
#         model (nn.Model): NNModel with specific configurations
#         config (dict): configuration for the training and test parameters
#
#     Returns:
#         dict: dictionary of scores from the cross-validation
#     """
#     # Get input size and y_dims
#     input_size: int = X.shape[1]
#     y_dims: int = y.shape[1]
#     # Perform cross-validation and train on each fold
#     # TODO: handle multioutput
#     # NOTE: This assumes the order of columns in y is [PCE, VOC, JSC, FF].
#     if y_dims > 1:
#         scores: dict[str, list] = {
#             f"test_{score}_{output}": []
#             for score, output in product(
#                 ["r", "r2", "rmse", "mae"], ["PCE", "Voc", "Jsc", "FF", "PCE_eqn"]
#             )
#         }
#         scores["fit_time"] = []
#         scores["score_time"] = []
#         scores["test_r"] = []
#         scores["test_r2"] = []
#         scores["test_rmse"] = []
#         scores["test_mae"] = []
#     else:
#         scores: dict = {
#             "fit_time": [],
#             "score_time": [],
#             "test_r": [],
#             "test_r2": [],
#             "test_rmse": [],
#             "test_mae": [],
#         }
#     predictions = []
#     for i, (train_index, test_index) in enumerate(cv_outer.split(X, y)):
#         train_x = X[train_index]
#         train_y = y[train_index]
#         test_x = X[test_index]
#         test_y = y[test_index]
#         print(len(train_x), len(test_x))
#         # Scale X and Y data (standard and then min max)
#         standard_x = StandardScaler()
#         standard_y = StandardScaler()
#         min_max_x = MinMaxScaler()
#         min_max_y = MinMaxScaler()
#         train_x = standard_x.fit_transform(train_x)
#         train_x = min_max_x.fit_transform(train_x)
#         train_y = standard_y.fit_transform(train_y)
#         train_y = min_max_y.fit_transform(train_y)
#         test_x = standard_x.transform(test_x)
#         test_x = min_max_x.transform(test_x)
#         test_y = standard_y.transform(test_y)
#         test_y = min_max_y.transform(test_y)
#         # convert to torch tensors
#         train_x = torch.from_numpy(train_x)
#         train_y = torch.from_numpy(train_y)
#         test_x = torch.from_numpy(test_x)
#         test_y = torch.from_numpy(test_y)
#         # Initiate model
#         model = NNModel(
#             input_size=input_size,
#             output_size=y_dims,
#             embedding_size=config["embedding_size"],
#             hidden_size=config["hidden_size"],
#             n_layers=config["n_layers"],
#         )
#         train_set = PolymerDataset(train_x, train_y)
#         test_set = PolymerDataset(test_x, test_y)
#         # ATTN: DataLoader defines batch size and whether to shuffle data
#         train_dataloader = DataLoader(
#             train_set, batch_size=config["train_batch_size"], shuffle=True
#         )
#         test_dataloader = DataLoader(
#             test_set, batch_size=config["test_batch_size"], shuffle=False
#         )
#         # NOTE: Setting loss and optimizer
#         # Choose Loss Function
#         if config["loss"] == "MSE":
#             loss_fn = nn.MSELoss()
#         elif config["loss"] == "CrossEntropy":
#             loss_fn = nn.CrossEntropyLoss()
#
#         # Choose PyTorch Optimizer
#         if config["optimizer"] == "Adam":
#             optimizer = optim.Adam(
#                 model.parameters(),
#                 lr=config["init_lr"],
#             )
#
#         # train model
#         device: torch.device = torch.device("cuda:0")
#         model.to(device)
#         running_loss = 0
#         n_examples = 0
#         n_iter = 0
#         running_valid_loss = 0
#         n_valid_iter = 0
#         # print training configs
#         print(config)
#         # print model summary
#         print(model)
#         pytorch_total_params = sum(
#             p.numel() for p in model.parameters() if p.requires_grad
#         )
#         print("MODEL_PARAMETERS: {}".format(pytorch_total_params))
#         # Early stopping
#         # last_loss = 100
#         # patience = 10
#         # trigger_times = 0
#         # early_stop = False
#
#         # TODO: Setup logger and writer with Tensorflow!
#         # NOTE: paths are hard-coded because it's not backward compatible ...
#         # train_log: Path = (
#         #     HERE.parent.parent
#         #     / "results"
#         #     / "target_PCE"
#         #     / "features_ECFP"
#         #     / "ANN_log"
#         #     / "train"
#         # )
#         # train_writer: SummaryWriter = SummaryWriter(log_dir=train_log)
#         # valid_log: Path = (
#         #     HERE.parent.parent
#         #     / "results"
#         #     / "target_PCE"
#         #     / "features_ECFP"
#         #     / "ANN_log"
#         #     / "valid"
#         # )
#         # valid_writer: SummaryWriter = SummaryWriter(log_dir=valid_log)
#
#         # NOTE: Start training (boilerplate)
#         # start time
#         start_time = time.time()
#         for epoch in range(config["num_of_epochs"]):
#             ## TRAINING LOOP
#             ## Make sure gradient tracking is on
#             model.train(True)
#             ## LOOP for 1 EPOCH
#             for i, data in enumerate(train_dataloader):
#                 inputs, targets = data  # [batch_size, input_size]
#                 # convert to cuda
#                 inputs, targets = inputs.to(device="cuda"), targets.to(device="cuda")
#                 # convert to float
#                 inputs, targets = inputs.float(), targets.float()
#                 # Zero your gradients for every batch!
#                 optimizer.zero_grad()
#                 # Make predictions for this batch
#                 outputs = model(inputs)
#                 # Compute the loss and its gradients
#                 loss = loss_fn(outputs, targets)
#                 # backpropagation
#                 loss.backward()
#                 # Adjust learning weights
#                 optimizer.step()
#                 # Gather data and report
#                 running_loss += float(loss.item())
#                 # Gather number of examples trained
#                 n_examples += len(inputs)
#                 # Gather number of iterations (batches) trained
#                 n_iter += 1
#                 # Stop training after max iterations
#             # train_writer.add_scalar("loss_batch", loss, n_examples)
#             # train_writer.add_scalar("loss_avg", running_loss / n_iter, n_examples)
#             # print progress report
#             print("EPOCH: {}, N_EXAMPLES: {}, LOSS: {}".format(epoch, n_examples, loss))
#             # VALIDATION LOOP
#             model.train(False)
#             n_valid_examples = 0
#             valid_loss_batch = 0
#             for i_test, valid_data in enumerate(test_dataloader):
#                 valid_inputs, valid_targets = valid_data
#                 valid_inputs, valid_targets = valid_inputs.to(
#                     device="cuda"
#                 ), valid_targets.to(device="cuda")
#                 # convert to float
#                 valid_inputs, valid_targets = (
#                     valid_inputs.float(),
#                     valid_targets.float(),
#                 )
#                 # Make predictions for this batch
#                 valid_outputs = model(valid_inputs)
#                 # gather number of examples in test set
#                 n_valid_examples += len(valid_inputs)
#                 # Compute the loss
#                 valid_loss = loss_fn(valid_outputs, valid_targets)
#                 valid_loss_batch += valid_loss
#                 # Log test loss
#             # valid_writer.add_scalar("loss_batch", valid_loss, n_examples)
#
#         # end time
#         end_time = time.time()
#         # print training time
#         train_time = end_time - start_time
#         print("Training time: {}".format(end_time - start_time))
#         # test model
#         # Inference
#         # start time
#         start_score_time = time.time()
#         test_prediction = []
#         ground_truth = []
#         n_test_examples = 0
#         model.train(False)
#         test_loss_batch = 0
#         for i_test, test_data in enumerate(test_dataloader):
#             test_inputs, test_targets = test_data
#             test_inputs, test_targets = test_inputs.to(device="cuda"), test_targets.to(
#                 device="cuda"
#             )
#             # convert to float
#             test_inputs, test_targets = (
#                 test_inputs.float(),
#                 test_targets.float(),
#             )
#             # Make predictions for this batch
#             test_outputs = model(test_inputs)
#             # gather number of examples in test set
#             n_test_examples += len(test_inputs)
#             # Compute the loss
#             test_loss = loss_fn(test_outputs, test_targets)
#             test_loss_batch += test_loss
#             # gather predictions and ground truth for result summary
#             test_prediction.extend(test_outputs.tolist())
#             ground_truth.extend(test_targets.tolist())
#         # end time
#         end_score_time = time.time()
#         score_time = end_score_time - start_score_time
#         # close SummaryWriter
#         # train_writer.close()
#         # valid_writer.close()
#         # reverse min-max scaling
#         test_prediction: np.ndarray = min_max_y.inverse_transform(test_prediction)
#         test_prediction: np.ndarray = standard_y.inverse_transform(test_prediction)
#         ground_truth: np.ndarray = min_max_y.inverse_transform(ground_truth)
#         ground_truth: np.ndarray = standard_y.inverse_transform(ground_truth)
#         # compute scores
#         # TODO: check how the multioutput models are being trained, tested, and how the scores are getting combined/computed!
#         if y_dims > 1:
#             # compute scoring metrics for each column, and then the pce_eqn
#             targets: list = ["PCE", "Voc", "Jsc", "FF", "PCE_eqn"]
#             for target in targets:
#                 for score in ["r", "r2", "rmse", "mae"]:
#                     scores[f"test_{score}_{target}"].append(
#                         score_lookup[score][target](test_prediction, ground_truth)
#                     )
#         scores["fit_time"].append(train_time)
#         scores["score_time"].append(score_time)
#         scores["test_r"].append(
#             np.corrcoef(test_prediction.flatten(), ground_truth.flatten())[0, 1]
#         )
#         scores["test_r2"].append(r2_score(test_prediction, ground_truth))
#         scores["test_rmse"].append(
#             np.sqrt(mean_squared_error(test_prediction, ground_truth))
#         )
#         scores["test_mae"].append(mean_absolute_error(test_prediction, ground_truth))
#         # add predictions
#         test_prediction: list = test_prediction.tolist()
#         predictions.extend(test_prediction)
#     predictions: np.ndarray = np.array(predictions)
#
#     return scores, predictions
