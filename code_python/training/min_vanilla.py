from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import shap as shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from code_python import DATASETS

# saeki = pd.read_pickle("../../datasets/Saeki_2022_n1318/saeki_corrected_512.pkl")
dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset.pkl"
opv_dataset = pd.read_pickle(dataset)
radius = 3
n_bits = 512
random_state = 69

fp_cols: pd.DataFrame = opv_dataset[[f"Donor ECFP{2*radius}_{n_bits}", f"Acceptor ECFP{2*radius}_{n_bits}"]]
opv_fp: pd.DataFrame = pd.concat([fp_cols[col].apply(pd.Series) for col in fp_cols.columns], axis=1)
# print(len(opv_fp.columns))
opv_fp.columns = [*[f"D EFCP{2*radius}_{i}" for i in range(n_bits)], *[f"A ECFP{2*radius}_bit{i}" for i in range(n_bits)]]

opv_mp = opv_dataset[["HOMO_A (eV)", "LUMO_A (eV)", "Ehl_A (eV)", "Eg_A (eV)",
                      "HOMO_D (eV)", "LUMO_D (eV)", "Ehl_D (eV)", "Eg_D (eV)"
                      ]]

X = pd.concat((opv_fp, opv_mp), axis=1)
# X = opv_fp
y = opv_dataset["calculated PCE (%)"]


def run_rf(random_state: int) -> None:
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Define hyperparameter search space
    param_space = {'n_estimators': Integer(50, 1000),
                   'max_depth': Integer(2, 2000),
                   'min_samples_split': Integer(2, 1318),
                   'min_samples_leaf': Integer(1, 1318),
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
        r_score = scipy.stats.pearsonr(y_test_outer, rf.predict(X_test_outer))[0]

        # Print the r score for this fold
        # print(r_score)

        # Append the r score to the list of outer r scores
        outer_r_scores.append(r_score)

    # Print the average r score and standard deviation
    # print(outer_r_scores)
    print("mean:\t", np.mean(outer_r_scores), "\tstdev:\t", np.std(outer_r_scores))

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

# explainer = shap.TreeExplainer(rf, X)
# shap_values = explainer(X)
# shap.plots.bar(shap_values)
# shap.plots.beeswarm(shap_values)


for state in [6, 13, 42, 69, 1234567890, 2**32-4]:
    run_rf(state)
