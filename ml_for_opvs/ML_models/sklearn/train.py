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
from copy import deepcopy
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import mean, std
from skopt import BayesSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost
import shap

from ml_for_opvs.ML_models.sklearn.pipeline import (
    process_features,
    process_target,
    get_space_dict,
    get_space_multi_dict,
)


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


# create scoring function
score_func = make_scorer(custom_scorer, greater_is_better=False)


# NOTE: Old
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

    # TODO: Replace all these filtered files with a single file that has all the data.
    # Paths to data files for the training and test sets based on type of input representation and filters?
    # process multiple data files
    train_paths: str = config["train_paths"]
    test_paths: str = config["test_paths"]

    # NOTE: ATTENTION! Folds have different number of training / validation folds. YUCK!
    # WTF: ???
    # calculate minimum number of examples, and cut it short
    min_val_len: list = []
    for train_path, test_path in zip(train_paths, test_paths):
        train_df: pd.DataFrame = pd.read_csv(train_path)
        val_df: pd.DataFrame = pd.read_csv(test_path)
        min_val_len.append(len(val_df))

    min_val: int = min(min_val_len)

    # NOTE: Ability to run subset from another filter type (?)
    # column names
    column_names = config["feature_names"].split(",")

    # if multiple train and test paths, X-Fold Cross-test occurs here.
    fold: int = 0
    outer_r: list = []
    outer_r2: list = []
    outer_rmse: list = []
    outer_mae: list = []
    progress_dict: dict = {"fold": [], "r": [], "r2": [], "rmse": [], "mae": []}
    # Stores shapley values across folds
    shapley_total: list = []
    for train_path, test_path in zip(train_paths, test_paths):
        # print training config
        # print(config)
        train_df: pd.DataFrame = pd.read_csv(train_path)
        val_df: pd.DataFrame = pd.read_csv(test_path)
        val_df: pd.DataFrame = val_df[0:min_val]
        # process SMILES vs. Fragments vs. Fingerprints. How to handle that? handle this and tokenization in pipeline
        # No input representation, only features
        if config["input_representation"] == "None":
            input_rep_bool = False
        else:
            input_rep_bool = True
        (
            input_train_array,
            input_val_array,
        ) = process_features(  # additional features are added at the end of array
            train_df[column_names], val_df[column_names], input_rep_bool
        )
        # process target values
        target_df_columns = config["target_name"].split(",")
        # try:
        #     target_df_columns.extend(config["feature_names"].split(","))
        # except:
        #     print("No Additional Features")
        if config["input_representation"] == "None":
            input_rep_bool = False
        else:
            input_rep_bool = True

        (
            target_train_array,
            target_val_array,
            target_max,
            target_min,
        ) = process_target(
            train_df[target_df_columns],
            val_df[target_df_columns],
            train_df[column_names],
            input_rep_bool,
        )
        # target_train_array: [num_of_train, num_of_targets]
        # TODO: factory pattern!
        # setup model with default parameters
        if config["multi_output_type"] == "ensemble":
            ensemble_results: list = []
            for i in range(1, len(target_df_columns)):
                target_train_select: np.ndarray = target_train_array[:, i]
                target_val_select: np.ndarray = target_val_array[:, i]
                target_max_select: float = target_max[i]
                target_min_select: float = target_min[i]
                # print(f"{target_train_select.shape=}")
                # print(f"{target_val_select.shape=}")
                # print(f"{target_max_select=}")
                # print(f"{target_min_select=}")
                if config["model_type"] == "RF":
                    model = RandomForestRegressor(
                        criterion="squared_error",
                        max_features=1.0,
                        random_state=config["random_state"],
                        bootstrap=True,
                        n_jobs=-1,
                    )
                elif config["model_type"] == "XGBoost":
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
                # KRR and MLR do not require HPO, they do not have space parameters
                # MUST be paired with hyperparameter_optimization == False
                elif config["model_type"] == "KRR":
                    kernel = PairwiseKernel(
                        gamma=1, gamma_bounds="fixed", metric="laplacian"
                    )
                    model = KernelRidge(alpha=0.05, kernel=kernel, gamma=1)
                elif config["model_type"] == "MLR":
                    model = LinearRegression()
                elif config["model_type"] == "SVM":
                    model = SVR(kernel="rbf", degree=3, cache_size=16000)
                elif config["model_type"] == "KNN":
                    model = KNeighborsRegressor()
                elif config["model_type"] == "Lasso":
                    model = Lasso(alpha=1.0)
                else:
                    raise NameError(
                        "Model not found. Please use RF, XGBoost, SVM, KRR, MLR, KNN, Lasso"
                    )

                # NOTE: Only hyperparameter optimization for RF and XGBoost
                # run hyperparameter optimization
                if config["hyperparameter_optimization"] == "True":
                    print("running hyperparameter optimization...")
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
                        n_iter=10,
                    )
                    # train
                    # execute search
                    result = search.fit(input_train_array, target_train_select)
                    # save best hyperparams for the best model from each fold
                    best_params: dict = result.best_params_
                    # get the best performing model fit on the subset of training set
                    model = result.best_estimator_
                    # re-train on whole training set using best hyperparameters
                    model.fit(input_train_array, target_train_select)
                    # inference on hold out set
                    yhat: np.ndarray = model.predict(input_val_array)
                else:
                    # train
                    model.fit(input_train_array, target_train_select)
                    # inference on hold out set
                    yhat: np.ndarray = model.predict(input_val_array)

                # Feature Importance
                # explainer = shap.TreeExplainer(model)
                # shap_values = explainer.shap_values(input_val_array)
                # print(f"{shap_values.shape=}")
                # shapley_total.append(list(shap_values))
                # print(f"{yhat.shape=}")
                # reverse min-max scaling
                # NOTE: Stanley changed to MinMaxScaler
                yhat: np.ndarray = (
                    yhat * (target_max_select - target_min_select)
                ) + target_min_select
                # print(f"{yhat.shape=}")
                # print(f"{yhat=}")
                ensemble_results.append(yhat)
            yhat: np.ndarray = np.array(ensemble_results)
            yhat: np.ndarray = yhat.T
        else:
            if config["model_type"] == "RF":
                model = RandomForestRegressor(
                    criterion="squared_error",
                    max_features=1.0,
                    random_state=config["random_state"],
                    bootstrap=True,
                    n_jobs=-1,
                )
            elif config["model_type"] == "XGBoost":
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
            # KRR and MLR do not require HPO, they do not have space parameters
            # MUST be paired with hyperparameter_optimization == False
            elif config["model_type"] == "KRR":
                kernel = PairwiseKernel(
                    gamma=1, gamma_bounds="fixed", metric="laplacian"
                )
                model = KernelRidge(alpha=0.05, kernel=kernel, gamma=1)
            elif config["model_type"] == "MLR":
                model = LinearRegression()
            elif config["model_type"] == "SVM":
                model = SVR(kernel="rbf", degree=3, cache_size=16000)
            elif config["model_type"] == "KNN":
                model = KNeighborsRegressor()
            elif config["model_type"] == "Lasso":
                model = Lasso(alpha=1.0)
            else:
                raise NameError(
                    "Model not found. Please use RF, XGBoost, SVM, KRR, MLR, KNN, Lasso"
                )

            # Multi-output configuration
            if config["multi_output_type"] == "multi":
                model = MultiOutputRegressor(estimator=model, n_jobs=-1)
                target_train_select: np.ndarray = target_train_array[:, 1:]
                target_val_select: np.ndarray = target_val_array[:, 1:]
                target_max_select: np.ndarray = target_max[1:]
                target_min_select: np.ndarray = target_min[1:]
            elif config["multi_output_type"] == "chain":
                # NOTE: Not suitable because there is data leakage.
                pass
            else:
                target_train_select: np.ndarray = target_train_array.squeeze()
                target_val_select: np.ndarray = target_val_array.squeeze()
                target_max_select: np.ndarray = target_max.squeeze()
                target_min_select: np.ndarray = target_min.squeeze()

            # run hyperparameter optimization
            if config["hyperparameter_optimization"] == "True":
                print("running hyperparameter optimization...")
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
                    n_iter=10,
                )
                # train
                # execute search
                result = search.fit(input_train_array, target_train_select)
                # save best hyperparams for the best model from each fold
                best_params: dict = result.best_params_
                # get the best performing model fit on the subset of training set
                model = result.best_estimator_
                # re-train on whole training set using best hyperparameters
                model.fit(input_train_array, target_train_select)
                # inference on hold out set
                yhat: np.ndarray = model.predict(input_val_array)
            else:
                # train
                model.fit(input_train_array, target_train_select)
                # inference on hold out set
                yhat: np.ndarray = model.predict(input_val_array)

            # Feature Importance
            # explainer = shap.TreeExplainer(model)
            # shap_values = explainer.shap_values(input_val_array)
            # print(f"{shap_values.shape=}")
            # shapley_total.append(list(shap_values))

            # reverse min-max scaling
            yhat: np.ndarray = (
                yhat * (target_max_select - target_min_select)
            ) + target_min_select
            y_test: np.ndarray = (
                target_val_select * (target_max_select - target_min_select)
            ) + target_min_select

        # Convert to PCE and test against that.
        # Multi-output configuration
        if (
            config["multi_output_type"] == "multi"
            or config["multi_output_type"] == "ensemble"
        ):
            yhat_multi: np.ndarray = deepcopy(yhat)
            # print(f"{yhat_multi=}")
            # print(f"{yhat_multi.shape=}")
            yhat: np.ndarray = yhat_multi[:, 0]  # first target
            # print(f"{yhat.shape=}")
            for i in range(1, yhat_multi.shape[1]):
                yhat: np.ndarray = yhat * yhat_multi[:, i]
                # print(f"{yhat.shape=}")
            yhat: np.ndarray = yhat / 100
            y_test: np.ndarray = (
                target_val_array[:, 0] * (target_max[0] - target_min[0])
            ) + target_min[
                0
            ]  # GET PCE from 1st column

        # make new files
        # save model, outputs, generates new directory based on training/dataset/model/features/target
        results_path: Path = Path(os.path.abspath(config["results_path"]))
        if (
            config["multi_output_type"] == "multi"
            or config["multi_output_type"] == "ensemble"
        ):
            model_dir_path: Path = results_path / "{}_{}".format(
                config["model_type"], config["multi_output_type"]
            )
        else:
            model_dir_path: Path = results_path / "{}".format(config["model_type"])
        feature_dir_path: Path = model_dir_path / "{}".format(config["feature_names"])
        feature_set_path: Path = feature_dir_path / "{}".format(config["feature_set"])
        target_dir_path: Path = feature_set_path / "{}".format(config["target_name"])
        # create folders if not present
        try:
            target_dir_path.mkdir(parents=True, exist_ok=True)
        except:
            print("Folder already exists.")

        # Feature Importance Plots
        # shap.summary_plot(shap_values, input_val_array, show=False)
        # plt.savefig(target_dir_path / "shapley_{}.png".format(fold))
        # plt.clf()

        # save model
        # model_path: Path = target_dir_path / "model_{}.pkl".format(fold)
        # pickle.dump(model, open(model_path, "wb"))  # difficult to maintain
        # save best hyperparams for the best model from each fold
        if config["hyperparameter_optimization"] == "True":
            hyperparam_path: Path = (
                target_dir_path / "hyperparameter_optimization_{}.csv".format(fold)
            )
            hyperparam_df: pd.DataFrame = pd.DataFrame.from_dict(
                best_params, orient="index"
            )
            hyperparam_df = hyperparam_df.transpose()
            hyperparam_df.to_csv(hyperparam_path, index=False)
        # save outputs
        prediction_path: Path = target_dir_path / "prediction_{}.csv".format(fold)
        # export predictions
        yhat_df: pd.DataFrame = pd.DataFrame(
            yhat, columns=["predicted_{}".format(config["target_name"])]
        )
        yhat_df[config["target_name"]] = y_test
        yhat_df.to_csv(prediction_path, index=False)
        fold += 1

        # evaluate_model the model
        r: float = np.corrcoef(y_test, yhat)[0, 1]
        r2: float = r2_score(y_test, yhat)
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

    # After Cross-Validation Training, Summarize Results!
    # Save Feature Importance Plots
    # shapley_total: np.ndarray = np.array(shapley_total)
    # print(f"{shapley_total.shape=}")
    # shapley_total_avg: np.ndarray = np.mean(shapley_total, axis=0)
    # print(f"{shapley_total_avg.shape=}")
    # shap.summary_plot(shapley_total_avg, input_val_array, show=False)
    # plt.savefig(target_dir_path / "shapley_avg.png")
    # assert False

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
        help="Path to training data. If multiple training data: format is 'train_0.csv, train_1.csv, train_2.csv', required that multiple test paths are provided.",
    )
    parser.add_argument(
        "--test_paths",
        type=str,
        nargs="+",
        help="Path to test data. If multiple test data: format is 'val_0.csv, val_1.csv, val_2.csv', required that multiple training paths are provided.",
    )
    parser.add_argument(
        "--input_representation",
        type=str,
        help="Choose input representation. Format is: ex. SMILES.",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        default=None,
        help="Choose input features. Format is: ex. D_A_ratio_m_m,solvent",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default=None,
        help="Choose input features. Format is: ex. fabrication_wo_solid,solvent_properties",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        help="Choose target value. Format is ex. FF_percent,Voc_V,Jsc_mA_cm_pow_neg2,calc_PCE_percent",
    )
    parser.add_argument(
        "--multi_output_type",
        type=str,
        help="Choose type of multi-output. Options are: multi, ensemble, chain.",
    )
    parser.add_argument(
        "--model_type", type=str, help="Choose model type. (RF, XGBoost, SVM, KRR, MLR)"
    )
    parser.add_argument(
        "--hyperparameter_optimization",
        type=str,
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
