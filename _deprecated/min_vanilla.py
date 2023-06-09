import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap as shap
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from code_python import DATASETS

# saeki = pd.read_pickle("../../datasets/Saeki_2022_n1318/saeki_corrected_512.pkl")
# dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset.pkl"
dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
opv_dataset = pd.read_pickle(dataset)
solvent_properties = pd.read_csv(DATASETS / "Min_2020_n558" / "raw" / "solvent properties.csv", index_col="Name")
with open(DATASETS / "Min_2020_n558" / "selected_properties.json", "r") as f:
    solvent_descriptors: list[str] = json.load(f)["solvent"]
radius = 5
n_bits = 4096
random_state = 69

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
    "HTL energy level (eV)",
    "ETL energy level (eV)",
    # # Fab features with low quantity
    # "total solids conc. (mg/mL)",
    # "annealing time (min)",
    # "Active layer spin coating speed (rpm)",
]

# Remove rows containing NaNs
print(len(opv_dataset))
filtered_opv_dataset = opv_dataset[subset_columns].dropna()
print(len(filtered_opv_dataset), len(filtered_opv_dataset) / len(opv_dataset))

# Unroll columns
fp_cols: pd.DataFrame = filtered_opv_dataset[
    [f"Donor ECFP{2 * radius}_{n_bits}", f"Acceptor ECFP{2 * radius}_{n_bits}"]]
opv_fp: pd.DataFrame = pd.concat([fp_cols[col].apply(pd.Series) for col in fp_cols.columns], axis=1)
# print(len(opv_fp.columns))
opv_fp.columns = [*[f"D EFCP{2 * radius}_bit{i}" for i in range(n_bits)],
                  *[f"A ECFP{2 * radius}_bit{i}" for i in range(n_bits)]]


opv_solv_cols: pd.DataFrame = filtered_opv_dataset[["solvent descriptors", "solvent additive descriptors"]]
opv_solv_desc: pd.DataFrame = pd.concat([opv_solv_cols[col].apply(pd.Series) for col in opv_solv_cols.columns], axis=1)
opv_solv_desc.columns = [*[f"solvent {d}" for d in solvent_descriptors],
                         *[f"additive {d}" for d in solvent_descriptors]]

# TODO: Scaling somewhere here?
opv_mp = filtered_opv_dataset[["HOMO_A (eV)", "LUMO_A (eV)", "Ehl_A (eV)", "Eg_A (eV)",
                               "HOMO_D (eV)", "LUMO_D (eV)", "Ehl_D (eV)", "Eg_D (eV)"
                               ]]

opv_fab = filtered_opv_dataset[["D:A ratio (m/m)",
                                "solvent additive conc. (% v/v)",
                                "temperature of thermal annealing", ]]

opv_solv_token = filtered_opv_dataset[["solvent",
                                "solvent additive",]]

# Tokenize on the fly...
opv_fab["solvent"] = opv_solv_token["solvent"].map(solvent_properties["Token"])
opv_fab["solvent additive"] = opv_solv_token["solvent additive"].map(solvent_properties["Token"])

# opv_fab_b = filtered_opv_dataset[["total solids conc. (mg/mL)", "annealing time (min)", "Active layer spin coating speed (rpm)"]]

opv_dev = filtered_opv_dataset[["HTL energy level (eV)", "ETL energy level (eV)"]]

# Rejoin columns with unrolled features
# X = opv_fp  # FPs
# X = pd.concat((opv_fp, opv_mp), axis=1)  # FPs + MPs
# X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_token), axis=1)  # FPs + MPs + processing tokens
X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_desc), axis=1)  # FPs + MPs + processing descriptors
# X = pd.concat((opv_fab, opv_solv_desc), axis=1)  # processing descriptors only
# X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_desc, opv_dev), axis=1)  # FPs + MPs + processing descriptors + interlayers
# X = pd.concat((opv_fp, opv_mp, opv_fab, opv_solv_desc, opv_dev, opv_fab_b), axis=1)

y = filtered_opv_dataset["calculated PCE (%)"]

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


def run_rf(random_state: int) -> list[float]:
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Define hyperparameter search space
    param_space = {'n_estimators':      Integer(50, 1000),
                   'max_depth':         Integer(2, 2000),
                   'min_samples_split': Integer(2, 1318),
                   'min_samples_leaf':  Integer(1, 1318),
                   # 'max_features': Real(0.01, 1.0, prior='uniform'),
                   # 'max_leaf_nodes': Integer(2, 1000),
                   # 'min_impurity_decrease': Real(0.0, 1.0, prior='uniform'),
                   }

    outer_r_scores = []
    for train_index, test_index in kf_outer.split(X, y):
        X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

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
        rf = RandomForestRegressor(
            # **best_params,
            random_state=random_state)
        rf.fit(X_train_outer, y_train_outer)

        # Evaluate the model on the test set and get the r score
        predicted = rf.predict(X_test_outer)
        r_score = scipy.stats.pearsonr(y_test_outer, predicted)[0]
        r2_score = sklearn.metrics.r2_score(y_test_outer, predicted)

        # plot_predictions(y_test_outer, predicted, f"R = {round(r_score, 2)} R2 = {round(r2_score, 2)}")

        # Print the r score for this fold
        # print(r_score)

        # explainer = shap.TreeExplainer(rf, X_test_outer)
        # shap_values = explainer(X_test_outer, check_additivity=False)
        # shap.plots.beeswarm(shap_values, max_display=20, show=False)
        # plt.tight_layout()
        # plt.show()

        # Append the r score to the list of outer r scores
        outer_r_scores.append(r_score)

    # Print the average r score and standard deviation
    # print(outer_r_scores)
    print("mean:\t", np.mean(outer_r_scores), "\tstdev:\t", np.std(outer_r_scores))

    return outer_r_scores


# print("Training on the whole dataset now...")

# Define the hyperparameter optimization function
# opt_full_rf = BayesSearchCV(RandomForestRegressor(random_state=42),
#                             param_space,
#                             cv=kf_outer,
#                             n_points=5,
#                             # n_iter=100,
#                             n_jobs=-1,
#                             random_state=42)

# Fit the hyperparameter optimization function on the training set
# opt_full_rf.fit(X, y)

# Get the optimal hyperparameters for this fold
# best_params = opt_full_rf.best_params_
# print(best_params)

# Train the final model on the entire dataset using the optimal hyperparameters
# rf = RandomForestRegressor(**best_params,
#                            random_state=42)
# rf.fit(X, y)




r_scores = []
for state in [6, 13, 42, 69, 420, 1234567890, 2 ** 32 - 4]:
    results = run_rf(state)
    r_scores.extend(results)
print("overall mean:\t", np.mean(r_scores), "\toverall stdev:\t", np.std(r_scores))

rf_outer = RandomForestRegressor(random_state=6)
rf_outer.fit(X, y)
# predicted = rf_outer.predict(X)

explainer = shap.TreeExplainer(rf_outer, X)
shap_values = explainer(X, check_additivity=False)
# shap.plots.bar(shap_values, show=False)
# plt.tight_layout()
# plt.show()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.show()
