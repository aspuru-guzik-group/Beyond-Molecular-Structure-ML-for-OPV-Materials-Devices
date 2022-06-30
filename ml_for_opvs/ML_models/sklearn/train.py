"""Output needs to:
1. Create model file that saves model (usable for inference), arguments, and config files.
2. Output of prediction files must have consistent column names. (Ex. predicted_value, ground_truth_value)
3. Summary file that contains R, R2, RMSE, MAE of all folds.
"""
from ctypes import Union
import os
from typing import Tuple
import pandas as pd
import pickle  # for saving scikit-learn models
import numpy as np
import json
from pathlib import Path
from numpy import mean, std
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import xgboost

from ml_for_opvs.ML_models.sklearn.pipeline import (
    process_features,
    process_target,
    get_space_dict,
)


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


# create scoring function
score_func = make_scorer(custom_scorer, greater_is_better=False)


def dataset_find(result_path: str):
    """Finds the dataset name for the given path from the known datasets we have.

    Args:
        result_path (str): filepath to results
    Returns:
        dataset_name (str): dataset name
    """
    result_path_list: list = result_path.split("/")
    datasets: list = ["CO2_Soleimani", "PV_Wang", "OPV_Min", "Swelling_Xu"]
    for dataset_name in datasets:
        if dataset_name in result_path_list:
            return dataset_name


def main(config: dict):
    """Runs training and calls from pipeline to perform preprocessing.

    Args:
        config (dict): Configuration parameters.
    """
    # process training parameters
    # with open(config["train_params_path"]) as train_param_path:
    #     train_param: dict = json.load(train_param_path)
    # for param in train_param.keys():
    #     config[param] = train_param[param]

    # process multiple data files
    train_paths: str = config["train_paths"]
    validation_paths: str = config["validation_paths"]

    # if multiple train and validation paths, X-Fold Cross-Validation occurs here.
    fold: int = 0
    outer_r: list = []
    outer_r2: list = []
    outer_rmse: list = []
    outer_mae: list = []
    progress_dict: dict = {"fold": [], "r": [], "r2": [], "rmse": [], "mae": []}
    for train_path, validation_path in zip(train_paths, validation_paths):
        train_df: pd.DataFrame = pd.read_csv(train_path)
        val_df: pd.DataFrame = pd.read_csv(validation_path)
        # process SMILES vs. Fragments vs. Fingerprints. How to handle that? handle this and tokenization in pipeline
        (
            input_train_array,
            input_val_array,
        ) = process_features(  # additional features are added at the end of array
            train_df[config["feature_names"].split(",")],
            val_df[config["feature_names"].split(",")],
        )
        # process target values
        (
            target_train_array,
            target_val_array,
            target_max,
            target_min,
        ) = process_target(
            train_df[config["target_name"].split(",")],
            val_df[config["target_name"].split(",")],
        )

        # choose model
        # TODO: factory pattern
        # setup model with default parameters
        if config["model_type"] == "RF":
            model = RandomForestRegressor(
                criterion="squared_error",
                max_features=1.0,
                random_state=config["random_state"],
                bootstrap=True,
                n_jobs=-1,
            )
        elif config["model_type"] == "BRT":
            model = xgboost.XGBRegressor(
                objective="reg:squarederror",
                alpha=0.9,
                random_state=config["random_state"],
                n_jobs=-1,
                learning_rate=0.2,
                n_estimators=100,
                max_depth=10,
                subsample=1,
            )
        # KRR and LR do not require HPO, they do not have space parameters
        # MUST be paired with hyperparameter_optimization == False
        elif config["model_type"] == "KRR":
            assert (
                config["hyperparameter_optimization"] == False
            ), "KRR cannot be paired with HPO"
            kernel = PairwiseKernel(gamma=1, gamma_bounds="fixed", metric="laplacian")
            model = KernelRidge(alpha=0.05, kernel=kernel, gamma=1)
        elif config["model_type"] == "LR" and not config["hyperparameter_optimization"]:
            assert (
                config["hyperparameter_optimization"] == False
            ), "LR cannot be paired with HPO"
            model = LinearRegression()
        elif config["model_type"] == "SVM":
            model = SVR(kernel="rbf", degree="3")
        else:
            raise NameError("Model not found. Please use RF, BRT, LR, KRR")

        # run hyperparameter optimization
        if config["hyperparameter_optimization"]:
            # setup HPO space
            space = get_space_dict(
                config["hyperparameter_space_path"], config["model_type"]
            )
            # define search
            search = BayesSearchCV(
                estimator=model,
                search_spaces=space,
                scoring=score_func,
                cv=KFold(n_splits=5, shuffle=False),
                refit=True,
                n_jobs=-1,
                verbose=0,
                n_iter=25,
            )
            # train
            # execute search
            result = search.fit(input_train_array, target_train_array)
            # save best hyperparams for the best model from each fold
            best_params: dict = result.best_params_
            # get the best performing model fit on the whole training set
            model = result.best_estimator_
            # inference on hold out set
            yhat: np.ndarray = model.predict(input_val_array)
        else:
            # train
            model.fit(input_train_array, target_train_array)
            # inference on hold out set
            yhat: np.ndarray = model.predict(input_val_array)

        # reverse min-max scaling
        yhat: np.ndarray = (yhat * (target_max - target_min)) + target_min
        y_test: np.ndarray = (target_val_array * (target_max - target_min)) + target_min

        # make new files
        # save model, outputs, generates new directory based on training/dataset/model/features/target
        results_path: Path = Path(os.path.abspath(config["results_path"]))
        model_dir_path: Path = results_path / "{}".format(config["model_type"])
        feature_dir_path: Path = model_dir_path / "{}".format(config["feature_names"])
        target_dir_path: Path = feature_dir_path / "{}".format(config["target_name"])
        # create folders if not present
        try:
            target_dir_path.mkdir(parents=True, exist_ok=True)
        except:
            print("Folder already exists.")
        # save model
        model_path: Path = target_dir_path / "model_{}.sav".format(fold)
        pickle.dump(model, open(model_path, "wb"))  # difficult to maintain
        # save best hyperparams for the best model from each fold
        if config["hyperparameter_optimization"]:
            hyperparam_path: Path = target_dir_path / "hyperparameter_optimization_{}.csv".format(fold)
            hyperparam_df: pd.DataFrame = pd.DataFrame.from_dict(best_params, orient="index")
            hyperparam_df = hyperparam_df.transpose()
            hyperparam_df.to_csv(hyperparam_path, index=False)
        # save outputs
        prediction_path: Path = target_dir_path / "prediction_{}.csv".format(fold)
        # export predictions
        yhat_df: pd.DataFrame = pd.DataFrame(
            yhat, columns=["predicted_{}".format(config["target_name"])]
        )
        for feature in list(config["feature_names"].split(",")):
            yhat_df[feature] = val_df[feature]
        yhat_df.to_csv(prediction_path, index=False)
        fold += 1

        # evaluate the model
        r: float = np.corrcoef(y_test, yhat)[0, 1]
        r2: float = (r) ** 2
        rmse: float = np.sqrt(mean_squared_error(y_test, yhat))
        mae: float = mean_absolute_error(y_test, yhat)
        # report progress (best training score)
        print(">r=%.3f, r2=%.3f, rmse=%.3f, mae=%.3f" % (r, r2, rmse, mae))
        progress_dict["fold"].append(fold)
        progress_dict["r"].append(r)
        progress_dict["r2"].append(r2)
        progress_dict["rmse"].append(rmse)
        progress_dict["mae"].append(mae)
        # append to outer list
        outer_r.append(r)
        outer_r2.append(r2)
        outer_rmse.append(rmse)
        outer_mae.append(mae)

    # make new file
    # summarize results
    progress_path: Path = target_dir_path / "progress_report.csv"
    progress_df: pd.DataFrame = pd.DataFrame.from_dict(progress_dict, orient="index")
    progress_df = progress_df.transpose()
    progress_df.to_csv(progress_path, index=False)
    summary_path: Path = target_dir_path / "summary.csv"
    summary_dict: dict = {
        "Dataset": dataset_find(config["results_path"]),
        "num_of_folds": fold,
        "Features": config["feature_names"],
        "Targets": config["target_name"],
        "Model": config["model_type"],
        "r_mean": mean(outer_r),
        "r_std": std(outer_r),
        "r2_mean": mean(outer_r2),
        "r2_std": std(outer_r2),
        "rmse_mean": mean(outer_rmse),
        "rmse_std": std(outer_rmse),
        "mae_mean": mean(outer_mae),
        "mae_std": std(outer_mae),
        "num_of_data": len(input_train_array) + len(input_val_array),
    }
    summary_df: pd.DataFrame = pd.DataFrame.from_dict(summary_dict, orient="index")
    summary_df = summary_df.transpose()
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--train_paths",
        type=str,
        nargs="+",
        help="Path to training data. If multiple training data: format is 'train_0.csv, train_1.csv, train_2.csv', required that multiple validation paths are provided.",
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        nargs="+",
        help="Path to validation data. If multiple validation data: format is 'val_0.csv, val_1.csv, val_2.csv', required that multiple training paths are provided.",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="Choose input features. Format is: ex. SMILES, T_K, P_Mpa - Always put representation at the front.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        help="Choose target value. Format is ex. a_separation_factor",
    )
    parser.add_argument("--model_type", type=str, help="Choose model type. (RF, BRT)")
    parser.add_argument(
        "--hyperparameter_optimization",
        type=bool,
        help="Enable hyperparameter optimization. BayesSearchCV over a default space.",
    )
    parser.add_argument(
        "--model_config_path", type=str, help="Filepath of model config JSON"
    )
    parser.add_argument(
        "--hyperparameter_space_path",
        type=str,
        help="Filepath of hyperparameter space optimization JSON",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Filepath to location of result summaries and predictions up to the dataset is sufficient.",
    )
    parser.add_argument(
        "--random_state", type=int, default=22, help="Integer number for random seed."
    )

    args = parser.parse_args()
    config = vars(args)
    main(config)
