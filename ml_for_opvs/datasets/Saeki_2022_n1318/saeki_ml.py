import numpy as np
import pandas as pd
import scipy
import shap as shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

saeki = pd.read_pickle("saeki_corrected.pkl")

"""
Using scikit-learn, use a stratified KFold cross-validation to split the dataset into training and test sets.
Then train a random forest model on the training set and evaluate the model on the test set. Print the r score for each fold.
Then print the average r score and standard deviation.
"""
np_fp = pd.DataFrame([saeki["n,p(FP)"][i] for i in range(len(saeki))],
                     columns=[*[f"n_EFCP6_bit{i}" for i in range(1024)], *[f"p_ECFP6_bit{i}" for i in range(1024)]]
                     )
np_mp = saeki[["'-HOMO_n(eV)", "'-LUMO_n(eV)", "Eg_n(eV)", "M (g/mol)",
               "'-HOMO_p(eV)", "'-LUMO_p(eV)", "Eg_p(eV)", "Mw (kg/mol)", "Mn(kg/mol)", "PDI"
               ]]
X = pd.concat((np_fp, np_mp), axis=1)
y = saeki["PCE_ave(%)"]

kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter search space
param_space = {'n_estimators': Integer(50, 500),
               'max_depth': Integer(2, 20)}

outer_r_scores = []
for train_index, test_index in kf_outer.split(X, y):
    X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
    y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

    # Define inner cross-validation
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define the hyperparameter optimization function
    opt_rf = BayesSearchCV(RandomForestRegressor(random_state=42),
                           param_space,
                           cv=kf_inner,
                           n_iter=10,
                           n_jobs=-1,
                           random_state=42)

    # Fit the hyperparameter optimization function on the training set
    opt_rf.fit(X_train_outer, y_train_outer)

    # Get the optimal hyperparameters for this fold
    best_params = opt_rf.best_params_
    print(best_params)

    # Train the final model for this fold using the optimal hyperparameters
    rf = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                               max_depth=best_params['max_depth'],
                               random_state=42)
    rf.fit(X_train_outer, y_train_outer)

    # Evaluate the model on the test set and get the r score
    r_score = scipy.stats.pearsonr(y_test_outer, rf.predict(X_test_outer))[0]

    # Print the r score for this fold
    print(r_score)

    # Append the r score to the list of outer r scores
    outer_r_scores.append(r_score)

# Print the average r score and standard deviation
print(outer_r_scores)
print(np.mean(outer_r_scores), np.std(outer_r_scores))

# Train the final model on the entire dataset using the optimal hyperparameters
rf = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                           max_depth=best_params['max_depth'],
                           random_state=42)
rf.fit(X, y)

explainer = shap.TreeExplainer(rf, X)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar")
shap.plots.beeswarm(shap_values)
