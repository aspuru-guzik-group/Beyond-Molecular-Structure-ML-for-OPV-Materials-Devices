from scipy.sparse.construct import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pkg_resources
import numpy as np
import pandas as pd
import copy as copy
from numpy import mean
from numpy import std
import matplotlib
import matplotlib.pyplot as plt
from opv_ml.ML_models.sklearn.data.OPV_Min.data import Dataset
from rdkit import Chem
from collections import deque
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV

from opv_ml.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer

TRAIN_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/hw_frag/train_frag_master.csv"
)

AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/augmentation/train_aug_master15.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)


def custom_scorer(y, yhat):
    mse = mean_squared_error(y, yhat)
    return mse


def augment_smi_in_loop(x, y, num_of_augment, da_pair):
    """
    Function that creates augmented DA and AD pairs with X number of augmented SMILES
    Uses doRandom=True for augmentation

    Returns
    ---------
    aug_smi_array: tokenized array of augmented smile
    aug_pce_array: array of PCE
    """
    aug_smi_list = []
    aug_pce_list = []
    period_idx = x.index(".")
    donor_smi = x[0:period_idx]
    acceptor_smi = x[period_idx + 1 :]
    # keep track of unique donors and acceptors
    unique_donor = [donor_smi]
    unique_acceptor = [acceptor_smi]
    donor_mol = Chem.MolFromSmiles(donor_smi)
    acceptor_mol = Chem.MolFromSmiles(acceptor_smi)
    augmented = 0
    while augmented < num_of_augment:
        donor_aug_smi = Chem.MolToSmiles(donor_mol, doRandom=True)
        acceptor_aug_smi = Chem.MolToSmiles(acceptor_mol, doRandom=True)
        if (
            donor_aug_smi not in unique_donor
            and acceptor_aug_smi not in unique_acceptor
        ):
            unique_donor.append(donor_aug_smi)
            unique_acceptor.append(acceptor_aug_smi)
            aug_smi_list.append(donor_aug_smi + "." + acceptor_aug_smi)
            aug_pce_list.append(y)
            aug_smi_list.append(acceptor_aug_smi + "." + donor_aug_smi)
            aug_pce_list.append(y)
            augmented += 1

    aug_pce_array = np.asarray(aug_pce_list)

    return aug_smi_list, aug_pce_array


def augment_donor_frags_in_loop(x, y: float):
    """
    Function that augments donor frags by swapping D.A -> A.D, and D1D2D3 -> D2D3D1 -> D3D1D2
    Assumes that input (x) is DA_tokenized.
    Returns 2 arrays, one of lists with augmented DA and AD pairs
    and another with associated PCE (y)
    """
    # assuming 1 = ".", first part is donor!
    x = list(x)
    period_idx = x.index(1)
    if 0 in x:
        last_zero_idx = len(x) - 1 - x[::-1].index(0)
        donor_frag_to_aug = x[last_zero_idx + 1 : period_idx]
    else:
        donor_frag_to_aug = x[:period_idx]
    donor_frag_deque = deque(donor_frag_to_aug)
    aug_donor_list = []
    aug_pce_list = []
    for i in range(len(donor_frag_to_aug)):
        donor_frag_deque_rotate = copy.copy(donor_frag_deque)
        donor_frag_deque_rotate.rotate(i)
        rotated_donor_frag_list = list(donor_frag_deque_rotate)
        # replace original frags with rotated donor frags
        if 0 in x:
            rotated_donor_frag_list = (
                x[: last_zero_idx + 1] + rotated_donor_frag_list + x[period_idx:]
            )
        else:
            rotated_donor_frag_list = rotated_donor_frag_list + x[period_idx:]
        # NOTE: do not keep original
        if rotated_donor_frag_list != x:
            aug_donor_list.append(rotated_donor_frag_list)
            aug_pce_list.append(y)

    return aug_donor_list, aug_pce_list


# create scoring function
r_score = make_scorer(custom_scorer, greater_is_better=True)

# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "bigsmiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "hw_frag": 0,
    "aug_hw_frag": 0,
    "brics": 0,
    "manual": 0,
    "aug_manual": 0,
    "fingerprint": 1,
}

if unique_datatype["fingerprint"] == 1:
    radius = 3
    nbits = 512

shuffled = False
if unique_datatype["smiles"] == 1:
    dataset = Dataset(TRAIN_MASTER_DATA, 0, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup()
    datatype = "SMILES"
elif unique_datatype["bigsmiles"] == 1:
    dataset = Dataset(MANUAL_MASTER_DATA, 1, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup()
    datatype = "BigSMILES"
elif unique_datatype["selfies"] == 1:
    dataset = Dataset(TRAIN_MASTER_DATA, 2, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup()
    datatype = "SELFIES"
elif unique_datatype["aug_smiles"] == 1:
    dataset = Dataset(TRAIN_MASTER_DATA, 0, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup_aug_smi()
    datatype = "AUG_SMILES"
elif unique_datatype["hw_frag"] == 1:
    dataset = Dataset(TRAIN_MASTER_DATA, 0, shuffled)
    x, y = dataset.setup_cv()
    datatype = "HW_FRAG"
elif unique_datatype["aug_hw_frag"] == 1:
    dataset = Dataset(TRAIN_MASTER_DATA, 0, shuffled)
    x, y = dataset.setup_cv()
    datatype = "AUG_HW_FRAG"
elif unique_datatype["brics"] == 1:
    dataset = Dataset(BRICS_MASTER_DATA, 0, shuffled)
    x, y = dataset.setup_frag_BRICS()
    datatype = "BRICS"
elif unique_datatype["manual"] == 1:
    dataset = Dataset(MANUAL_MASTER_DATA, 0, shuffled)
    x, y = dataset.setup_manual_frag()
    datatype = "MANUAL"
elif unique_datatype["aug_manual"] == 1:
    dataset = Dataset(MANUAL_MASTER_DATA, 0, shuffled)
    x, y = dataset.setup_manual_frag()
    datatype = "AUG_MANUAL"
elif unique_datatype["fingerprint"] == 1:
    dataset = Dataset(FP_MASTER_DATA, 0, shuffled)
    x, y = dataset.setup_fp(radius, nbits)
    datatype = "FINGERPRINT"
    print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))

if shuffled:
    datatype += "_SHUFFLED"

# outer cv gives different training and testing sets for inner cv
cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
outer_corr_coef = list()
outer_rmse = list()

for train_ix, test_ix in cv_outer.split(x):
    # split data
    x_train, x_test = x[train_ix], x[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]
    if unique_datatype["aug_manual"] == 1 or unique_datatype["aug_hw_frag"] == 1:
        # concatenate augmented data to x_train and y_train
        print("AUGMENTED")
        aug_x_train = list(copy.copy(x_train))
        aug_y_train = list(copy.copy(y_train))
        for x_, y_ in zip(x_train, y_train):
            x_aug, y_aug = augment_donor_frags_in_loop(x_, y_)
            aug_x_train.extend(x_aug)
            aug_y_train.extend(y_aug)

        x_train = np.array(aug_x_train)
        y_train = np.array(aug_y_train)
    # augment smiles data
    elif unique_datatype["aug_smiles"] == 1:
        aug_x_train = list(copy.copy(x_train))
        aug_y_train = list(copy.copy(y_train))
        for x_, y_ in zip(x_train, y_train):
            x_aug, y_aug = augment_smi_in_loop(x_, y_, 15, True)
            aug_x_train.extend(x_aug)
            aug_y_train.extend(y_aug)
        # tokenize Augmented SMILES
        (
            tokenized_input,
            max_seq_length,
            vocab_length,
            input_dict,  # returns dictionary of vocab
        ) = Tokenizer().tokenize_data(aug_x_train)
        tokenized_test = Tokenizer().tokenize_from_dict(
            x_test, max_seq_length, input_dict
        )
        x_test = np.array(tokenized_test)
        x_train = np.array(tokenized_input)
        y_train = np.array(aug_y_train)

    # configure the cross-validation procedure
    # inner cv allows for finding best model w/ best params
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
    # define the model
    model = RandomForestRegressor(
        criterion="squared_error",
        max_features="auto",
        random_state=0,
        bootstrap=True,
        n_jobs=-1,
    )
    # define search space
    space = dict()
    space["n_estimators"] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    space["min_samples_leaf"] = [1, 2, 3, 4, 5, 6]
    space["min_samples_split"] = [2, 3, 4]
    space["max_depth"] = (12, 20)

    # define search
    search = BayesSearchCV(
        estimator=model,
        search_spaces=space,
        scoring=r_score,
        cv=cv_inner,
        refit=True,
        n_jobs=-1,
        verbose=0,
        n_iter=25,
    )
    # define search
    # search = GridSearchCV(
    #     model, space, scoring=r_score, cv=cv_inner, refit=True, n_jobs=-1, verbose=1
    # )
    # execute search
    result = search.fit(x_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # get permutation importances of best performing model (overcomes bias toward high-cardinality (very unique) features)

    # get feature importances of best performing model
    # importances = best_model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    # forest_importances = pd.Series(importances)
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    # evaluate model on the hold out dataset
    yhat = best_model.predict(x_test)
    # evaluate the model
    corr_coef = np.corrcoef(y_test, yhat)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    # store the result
    outer_corr_coef.append(corr_coef)
    outer_rmse.append(rmse)
    # report progress (best training score)
    print(
        ">corr_coef=%.3f, est=%.3f, cfg=%s"
        % (corr_coef, result.best_score_, result.best_params_)
    )

# summarize the estimated performance of the model
print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))

# add R score from cross-validation results
# ablation_df = pd.read_csv(ABLATION_STUDY)
# results_list = [
#     "OPV",
#     "RF",
#     "sklearn",
#     "Manual Fragments",
#     mean(outer_results),
#     std(outer_results),
# ]
# ablation_df.loc[len(ablation_df.index) + 1] = results_list
# ablation_df.to_csv(ABLATION_STUDY, index=False)


def parity_plot(y_test, yhat):
    """Creates a parity plot for two column data (predicted data and ground truth data)"""
    # find slope and y-int of linear line of best fit
    m, b = np.polyfit(y_test, yhat, 1,)
    print(m, b)
    # find correlation coefficient
    corr_coef = np.corrcoef(y_test, yhat,)[0, 1]
    # find rmse
    rmse = np.sqrt(mean_squared_error(y_test, yhat,))

    fig, ax = plt.subplots()
    # ax.set_title("Predicted vs. Experimental PCE (%)", fontsize=22)
    ax.set_xlabel("Experimental_PCE_(%)", fontsize=18)
    ax.set_ylabel("Predicted_PCE_(%)", fontsize=18)
    ax.scatter(
        y_test, yhat, s=4, alpha=0.7, color="#0AB68B",
    )
    ax.plot(
        y_test,
        m * y_test + b,
        color="black",
        label=[
            "R: " + str(round(corr_coef, 3)) + "  " + "RMSE: " + str(round(rmse, 3))
        ],
    )
    ax.plot([0, 1], [0, 1], "--", color="blue", label="Perfect Correlation")
    ax.legend(loc="upper left", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    # add text box with slope and y-int
    # textstr = "slope: " + str(m) + "\n" + "y-int: " + str(b)
    # ax.text(0.5, 0.5, textstr, wrap=True, verticalalignment="top")
    plt.show()


parity_plot(y_test, yhat)
