import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

# from code_ import DATASETS
# from code_.pipeline import SUBSETS
from code_.training.pipeline_utils import unroll_lists_to_columns, unroll_solvent_descriptors

HERE = Path(__file__).parent
DATASETS = HERE.parent.parent / "datasets"
dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
solvents = DATASETS / "Min_2020_n558" / "raw" / "solvent properties_nan.csv"

with open("subsets.json", "r") as f:
    SUBSETS: dict[str, list[str]] = json.load(f)

# dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
opv_dataset = pd.read_pickle(dataset)
solvent_properties = pd.read_csv(solvents, index_col="Name")
with open(DATASETS / "Min_2020_n558" / "selected_properties.json", "r") as f:
    solvent_descriptors: list[str] = json.load(f)["solvent"]

with open("seeds.json", "r") as f:
    seeds: list[int] = json.load(f)

for rad, bit in zip([6, 5, 4, 3], [4096, 2048, 1024, 512]):
    print(rad, bit)
    radius = rad
    n_bits = bit

    subsets: dict[str, list[str]] = SUBSETS



    # Define columns in subset
    subset_columns: list[str] = [
        # Outputs
        "calculated PCE (%)",
        # Structural representations
        f"Donor ECFP{2 * radius}_{n_bits}",
        f"Acceptor ECFP{2 * radius}_{n_bits}",
        # Material properties
        "HOMO_A (eV)", "LUMO_A (eV)", "Ehl_A (eV)", "Eg_A (eV)",
        "HOMO_D (eV)", "LUMO_D (eV)", "Ehl_D (eV)", "Eg_D (eV)",
        # Fabrication features
        "D:A ratio (m/m)",
        "solvent",
        "solvent additive",
        "solvent additive conc. (% v/v)",
        "temperature of thermal annealing",
        "solvent descriptors",
        "solvent additive descriptors",
        # # Device features
        # "HTL energy level (eV)",
        # "ETL energy level (eV)",
        # # Fab features with low quantity
        # "total solids conc. (mg/mL)",
        # "annealing time (min)",
        # "Active layer spin coating speed (rpm)",
    ]

    # Remove rows containing NaNs
    print(len(opv_dataset))
    # filtered_opv_dataset = opv_dataset[subset_columns].dropna()
    # print(len(filtered_opv_dataset), len(filtered_opv_dataset) / len(opv_dataset))
    # opv_dataset = opv_dataset[subset_columns].dropna()

    # Unroll columns
    unroll_fp_cols: list[str] = [f"Donor ECFP{2 * radius}_{n_bits}", f"Acceptor ECFP{2 * radius}_{n_bits}"]
    new_fp_cols: list[str] = [*[f"D EFCP{2 * radius}_bit{i}" for i in range(n_bits)],
                              *[f"A ECFP{2 * radius}_bit{i}" for i in range(n_bits)]]
    opv_fp: pd.DataFrame = unroll_lists_to_columns(opv_dataset[unroll_fp_cols], unroll_fp_cols, new_fp_cols)

    opv_solv_desc: pd.DataFrame = unroll_solvent_descriptors(opv_dataset[["solvent descriptors", "solvent additive descriptors"]])

    opv_mp = opv_dataset[subsets["material properties"]]

    opv_mp_b = opv_dataset[subsets["material properties missing"]]

    opv_fab = opv_dataset[subsets["fabrication"]]

    opv_fab_b = opv_dataset[subsets["fabrication missing"]]

    opv_solv_token = opv_dataset[subsets["fabrication labels"]]

    # # Tokenize on the fly...
    # opv_fab["solvent"] = opv_solv_token["solvent"].map(solvent_properties["Token"])
    # opv_fab["solvent additive"] = opv_solv_token["solvent additive"].map(solvent_properties["Token"])

    opv_dev = opv_dataset[subsets["device architecture"]]
    opv_dev_b = opv_dataset[subsets["device architecture missing"]]

    opv_mob = opv_dataset[subsets["mobilities"]]
    opv_mob_log = opv_dataset[subsets["log mobilities"]]

    opv_everything = opv_dataset.drop(columns=["DOI",
                                               "Voc (V)", "Jsc (mA cm^-2)", "FF (%)", "PCE (%)", "calculated PCE (%)",
                                               "Donor SMILES", "Donor SELFIES", "Acceptor SMILES", "Acceptor SELFIES"])

    # Rejoin columns with unrolled features
    X = opv_fp  # FPs
    # X = pd.concat((opv_fp, opv_mp), axis=1)  # FPs + MPs
    # X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_token), axis=1)  # FPs + MPs + processing tokens
    # X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_desc), axis=1)  # FPs + MPs + processing descriptors
    # X = pd.concat((opv_fab, opv_solv_desc), axis=1)  # processing descriptors only
    # X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_desc, opv_dev), axis=1)  # FPs + MPs + processing descriptors + interlayers
    # X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_desc, opv_dev, opv_fab_b), axis=1)
    # X = pd.concat((opv_fp, opv_solv_desc, opv_mp, opv_mp_b, opv_fab, opv_fab_b, opv_dev, opv_dev_b, opv_mob), axis=1)  # Kitchen sink
    # X = pd.concat((opv_fp, opv_solv_desc, opv_mp, opv_mp_b, opv_fab, opv_fab_b, opv_dev, opv_dev_b, opv_mob_log), axis=1)  # Log sink
    # X = pd.concat((opv_fp, opv_solv_desc, opv_mp, opv_mp_b, opv_fab, opv_fab_b, opv_dev, opv_dev_b), axis=1)  # Kitchen sink without mobilities

    # enc = OrdinalEncoder()
    # enc.set_output(transform="pandas")
    # X = enc.fit_transform(opv_everything)  # EVERYTHING

    y = opv_dataset["calculated PCE (%)"]

    # Get max value in calculated PCE
    max_pce = ceil(max(y))


    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel("Actual PCE (%)")
        plt.xlim(0, max_pce)
        plt.ylim(0, max_pce)
        plt.plot([0, max_pce], [0, max_pce], color='black', linestyle='-', linewidth=1)
        plt.ylabel("Predicted PCE (%)")
        plt.show()


    def run_hgb(random_state: int) -> dict[str, list[float]]:
        kf_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # Define hyperparameter search space

        outer_scores = {"r2": [], "rmse": [], "mae": []}
        for train_index, test_index in kf_outer.split(X, y):
            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

            # TODO: Scaling somewhere here?

            # Define inner cross-validation
            # kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)

            # Define the hyperparameter optimization function
            # opt_rf = BayesSearchCV(RandomForestRegressor(random_state=42),
            #                        param_space,
            #                        cv=kf_inner,
            #                        # n_points=5,
            #                        # n_iter=10,
            #                        n_jobs=-1,
            #                        random_state=42)
            #
            # # Fit the hyperparameter optimization function on the training set
            # opt_rf.fit(X_train_outer, y_train_outer)

            # Get the optimal hyperparameters for this fold
            # best_params = opt_rf.best_params_
            # print(best_params)

            # Train the final model for this fold using the optimal hyperparameters
            hgbr = HistGradientBoostingRegressor(
                # **best_params,
                # categorical_features=["Donor", "Acceptor", "solvent", "solvent additive", "HTL", "ETL"],
                random_state=random_state)
            hgbr.fit(X_train_outer, y_train_outer)

            # Evaluate the model on the test set and get the r score
            predicted = hgbr.predict(X_test_outer)
            # r_score = scipy.stats.pearsonr(y_test_outer, predicted)[0]
            r2_score = sklearn.metrics.r2_score(y_test_outer, predicted)
            rmse_score = sklearn.metrics.mean_squared_error(y_test_outer, predicted, squared=False)
            mae_score = sklearn.metrics.mean_absolute_error(y_test_outer, predicted)

            # plot_predictions(y_test_outer, predicted, f"R = {round(r_score, 2)} R2 = {round(r2_score, 2)}")

            # Print the r score for this fold
            # print(r_score)

            # explainer = shap.TreeExplainer(hgbr, X_test_outer)
            # shap_values = explainer(X_test_outer, check_additivity=False)
            # shap.plots.beeswarm(shap_values, max_display=20, show=False)
            # plt.tight_layout()
            # plt.show()

            # Append the r score to the list of outer r scores
            outer_scores["r2"].append(r2_score)
            outer_scores["rmse"].append(rmse_score)
            outer_scores["mae"].append(mae_score)

        # Print the average r score and standard deviation
        # print(outer_r_scores)
        # print("mean:\t", np.mean(outer_r_scores), "\tstdev:\t", np.std(outer_r_scores))

        return outer_scores


    scores = {"r2": [], "rmse": [], "mae": []}
    for state in seeds:
        results = run_hgb(state)
        scores["r2"].extend(results["r2"])
        scores["rmse"].extend(results["rmse"])
        scores["mae"].extend(results["mae"])
    print("R2 mean:\t", np.mean(scores["r2"]), "\tR2 stderr:\t", np.std(scores["r2"])/np.sqrt(len(seeds)))
    print("RMSE mean:\t", np.mean(scores["rmse"]), "\tRMSE stderr:\t", np.std(scores["rmse"])/np.sqrt(len(seeds)))
    print("MAE mean:\t", np.mean(scores["mae"]), "\tMAE stderr:\t", np.std(scores["mae"])/np.sqrt(len(seeds)))

# for state in seeds:
#     print("Training on the whole dataset now...")
#     rf_outer = HistGradientBoostingRegressor(random_state=seeds[0])
#     rf_outer.fit(X, y)
#     # predicted = rf_outer.predict(X)
#
#
#     explainer = shap.TreeExplainer(rf_outer, X)
#     shap_values = explainer(X, check_additivity=False)
#     # shap.plots.bar(shap_values, show=False)
#     # plt.tight_layout()
#     # plt.show()
#     shap.plots.beeswarm(shap_values, max_display=13, show=False)
#     plt.tight_layout()
#     plt.show()
