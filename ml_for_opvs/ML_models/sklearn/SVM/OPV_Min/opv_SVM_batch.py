import copy
import math
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union
from collections import deque
from rdkit import Chem
import random

# for plotting
import pkg_resources
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from ml_for_opvs.ML_models.sklearn.data.OPV_Min.data import Dataset

# sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

from ml_for_opvs.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer

TRAIN_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/augmentation/train_aug_master4.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)

DATA_EVAL = pkg_resources.resource_filename(
    "ml_for_opvs", "data/ablation_study/poor_prediction_data.csv"
)

SUMMARY_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "ML_models/sklearn/SVM/OPV_Min/"
)

# For Manual Fragments!
MANUAL_DONOR_CSV = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/donor_frags.csv"
)

MANUAL_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/acceptor_frags.csv"
)

np.set_printoptions(precision=3)
SEED_VAL = 4


def custom_scorer(y, yhat):
    corr_coef = np.corrcoef(y, yhat)[0, 1]
    return corr_coef


def augment_smi_in_loop(x, y, num_of_augment, swap: bool):
    """
    Function that creates augmented DA with X number of augmented SMILES
    Uses doRandom=True for augmentation
    Ex. num_of_augment = 4 --> 5x amount of data, if swap = True --> 2x amount of data
    Result of both --> 10x amount of data.
    
    Arguments
    ----------
    num_of_augment: number of new random SMILES
    swap: whether to augmented frags by swapping D.A -> A.D

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
    # add canonical SMILES
    canonical_smi = Chem.CanonSmiles(donor_smi) + "." + Chem.CanonSmiles(acceptor_smi)
    aug_smi_list.append(canonical_smi)
    aug_pce_list.append(y)
    if swap:
        swap_canonical_smi = (
            Chem.CanonSmiles(acceptor_smi) + "." + Chem.CanonSmiles(donor_smi)
        )
        aug_smi_list.append(swap_canonical_smi)
        aug_pce_list.append(y)

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
            if swap:
                aug_smi_list.append(acceptor_aug_smi + "." + donor_aug_smi)
                aug_pce_list.append(y)
            augmented += 1

    aug_pce_array = np.asarray(aug_pce_list)

    return aug_smi_list, aug_pce_array


def augment_donor_frags_in_loop(x, y: float, device_idx, swap: bool):
    """
    Function that augments donor frags by swapping D.A -> A.D, and D1D2D3 -> D2D3D1 -> D3D1D2
    Assumes that input (x) is DA_tokenized.
    Returns 2 arrays, one of lists with augmented DA and AD pairs
    and another with associated PCE (y)

    Arguments
    ----------
    x: d-a pair to augment
    swap: whether to augmented frags by swapping D.A -> A.D
    """
    # assuming 1 = ".", first part is donor!
    x = list(x)
    period_idx = x.index(1)
    device_idx = len(x) - device_idx
    # for padding
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
        # for padding
        if 0 in x:
            aug_donor_frags = (
                x[: last_zero_idx + 1] + rotated_donor_frag_list + x[period_idx:]
            )
        else:
            aug_donor_frags = rotated_donor_frag_list + x[period_idx:]
        if swap:
            if 0 in x:
                swap_aug_donor_frags = (
                    x[: last_zero_idx + 1]
                    + x[period_idx + 1 : device_idx]
                    + [x[period_idx]]
                    + rotated_donor_frag_list
                    + x[device_idx:]
                )
            else:
                swap_aug_donor_frags = (
                    x[period_idx + 1 : device_idx]
                    + [x[period_idx]]
                    + rotated_donor_frag_list
                    + x[device_idx:]
                )
        # NOTE: augment original too
        aug_donor_list.append(aug_donor_frags)
        aug_pce_list.append(y)
        if swap:
            aug_donor_list.append(swap_aug_donor_frags)
            aug_pce_list.append(y)

    return aug_donor_list, aug_pce_list


# create scoring function
r_score = make_scorer(custom_scorer, greater_is_better=True)

# log results
summary_df = pd.DataFrame(
    columns=["Datatype", "R_mean", "R_std", "RMSE_mean", "RMSE_std", "num_of_data"]
)

# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "bigsmiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "manual": 0,
    "aug_manual": 0,
    "fingerprint": 0,
}
parameter_type = {
    "none": 0,
    "electronic": 1,
    "device": 0,
    "impt_device": 0,
}
for param in parameter_type:
    if parameter_type[param] == 1:
        dev_param = param
        if dev_param == "none":
            SUMMARY_DIR = SUMMARY_DIR + "none_opv_svm_results.csv"
        elif dev_param == "electronic":
            SUMMARY_DIR = SUMMARY_DIR + "electronic_opv_svm_results.csv"
        elif dev_param == "device":
            SUMMARY_DIR = SUMMARY_DIR + "device_opv_svm_results.csv"
        elif dev_param == "impt_device":
            SUMMARY_DIR = SUMMARY_DIR + "impt_device_opv_svm_results.csv"
for i in range(len(unique_datatype)):
    # reset conditions
    unique_datatype = {
        "smiles": 0,
        "bigsmiles": 0,
        "selfies": 0,
        "aug_smiles": 0,
        "brics": 0,
        "manual": 0,
        "aug_manual": 0,
        "fingerprint": 0,
    }

    index_list = list(np.zeros(len(unique_datatype) - 1))
    index_list.insert(i, 1)
    # set datatype with correct condition
    index = 0
    unique_var_keys = list(unique_datatype.keys())
    for j in index_list:
        unique_datatype[unique_var_keys[index]] = j
        index += 1

    if unique_datatype["fingerprint"] == 1:
        radius = 3
        nbits = 512

    shuffled = False
    if unique_datatype["smiles"] == 1:
        dataset = Dataset(TRAIN_MASTER_DATA, 0, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup(dev_param)
        datatype = "SMILES"
    elif unique_datatype["bigsmiles"] == 1:
        dataset = Dataset(MANUAL_MASTER_DATA, 1, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup(dev_param)
        datatype = "BigSMILES"
    elif unique_datatype["selfies"] == 1:
        dataset = Dataset(TRAIN_MASTER_DATA, 2, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup(dev_param)
        datatype = "SELFIES"
    elif unique_datatype["aug_smiles"] == 1:
        dataset = Dataset(TRAIN_MASTER_DATA, 0, shuffled)
        dataset.prepare_data()
        x, y, token_dict = dataset.setup_aug_smi(dev_param)
        num_of_augment = 4  # 1+4x amount of data
        datatype = "AUG_SMILES"
    elif unique_datatype["brics"] == 1:
        dataset = Dataset(BRICS_MASTER_DATA, 0, shuffled)
        x, y = dataset.setup_frag_BRICS(dev_param)
        datatype = "BRICS"
    elif unique_datatype["manual"] == 1:
        dataset = Dataset(MANUAL_MASTER_DATA, 0, shuffled)
        x, y, device_idx = dataset.setup_manual_frag(dev_param)
        datatype = "MANUAL"
    elif unique_datatype["aug_manual"] == 1:
        dataset = Dataset(MANUAL_MASTER_DATA, 0, shuffled)
        x, y, device_idx = dataset.setup_manual_frag(dev_param)
        datatype = "AUG_MANUAL"
    elif unique_datatype["fingerprint"] == 1:
        dataset = Dataset(FP_MASTER_DATA, 0, shuffled)
        x, y = dataset.setup_fp(radius, nbits, dev_param)
        datatype = "FINGERPRINT"
        print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))

    if shuffled:
        datatype += "_SHUFFLED"

    print(datatype)  # Ensures we know which model is running

    # outer cv gives different training and testing sets for inner cv
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
    outer_corr_coef = list()
    outer_rmse = list()

    for train_ix, test_ix in cv_outer.split(x):
        # split data
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if unique_datatype["aug_manual"] == 1:
            print("AUGMENTED")
            # concatenate augmented data to x_train and y_train
            aug_x_train = []
            aug_y_train = []
            for x_, y_ in zip(x_train, y_train):
                x_aug, y_aug = augment_donor_frags_in_loop(x_, y_, device_idx, True)
                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)

            x_train = np.array(aug_x_train)
            y_train = np.array(aug_y_train)
        elif unique_datatype["aug_smiles"] == 1:
            aug_x_train = []
            aug_y_train = []
            x_aug_dev_list = []
            for x_, y_ in zip(x_train, y_train):
                x_list = list(x_)
                x_aug, y_aug = augment_smi_in_loop(
                    str(x_list[0]), y_, num_of_augment, True
                )
                # add dev params for the number of augmented smiles
                for x_a in x_aug:
                    x_aug_dev = x_list[1:]
                    x_aug_dev_list.append(x_aug_dev)
                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)
            # tokenize Augmented SMILES
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,  # dictionary of vocab
            ) = Tokenizer().tokenize_data(aug_x_train)

            # preprocess x_test_array
            x_test_array = []
            x_test_dev_list = []
            for x_t in x_test:
                x_t_list = list(x_t)
                x_test_array.append(x_t_list[0])
                x_test_dev_list.append(x_t_list[1:])

            tokenized_test, test_max_seq_length = Tokenizer().tokenize_from_dict(
                x_test_array, max_seq_length, input_dict
            )

            # make sure test set max_seq_length is same as train set max_seq_length
            # NOTE: test set could have longer sequence because we separated the tokenization
            if test_max_seq_length > max_seq_length:
                tokenized_input, max_seq_length = Tokenizer().tokenize_from_dict(
                    aug_x_train, test_max_seq_length, input_dict
                )

            # add device parameters to token2idx
            token_idx = len(input_dict)
            for token in token_dict:
                input_dict[token] = token_idx
                token_idx += 1

            # tokenize device parameters
            tokenized_dev_input_list = []
            for dev in x_aug_dev_list:
                tokenized_dev_input = []
                for _d in dev:
                    if isinstance(_d, str):
                        tokenized_dev_input.append(input_dict[_d])
                    else:
                        tokenized_dev_input.append(_d)
                tokenized_dev_input_list.append(tokenized_dev_input)

            tokenized_dev_test_list = []
            for dev in x_test_dev_list:
                tokenized_dev_test = []
                for _d in dev:
                    if isinstance(_d, str):
                        tokenized_dev_test.append(input_dict[_d])
                    else:
                        tokenized_dev_test.append(_d)
                tokenized_dev_test_list.append(tokenized_dev_test)

            # add device parameters to data
            input_idx = 0
            while input_idx < len(tokenized_input):
                tokenized_input[input_idx].extend(tokenized_dev_input_list[input_idx])
                input_idx += 1

            test_input_idx = 0
            while test_input_idx < len(tokenized_test):
                tokenized_test[test_input_idx].extend(
                    tokenized_dev_test_list[test_input_idx]
                )
                test_input_idx += 1
            x_test = np.array(tokenized_test)
            x_train = np.array(tokenized_input)
            y_train = np.array(aug_y_train)
        # configure the cross-validation procedure
        # inner cv allows for finding best model w/ best params
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
        # define the model
        model = SVR()

        # define search space
        space = dict()
        space["kernel"] = ["linear", "poly", "rbf"]
        space["degree"] = (1, 10)
        # define search
        search = BayesSearchCV(
            estimator=model,
            search_spaces=space,
            scoring="neg_mean_squared_error",
            n_iter=25,
            cv=cv_inner,
            n_jobs=-1,
            verbose=0,
        )
        # execute search
        result = search.fit(x_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(x_test)
        print("Y_TEST: ", y_test)
        print("Y_HAT: ", yhat)
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
    summary_series = pd.DataFrame(
        {
            "Datatype": datatype,
            "R_mean": mean(outer_corr_coef),
            "R_std": std(outer_corr_coef),
            "RMSE_mean": mean(outer_rmse),
            "RMSE_std": std(outer_rmse),
            "num_of_data": len(x),
        },
        index=[0],
    )
    summary_df = pd.concat([summary_df, summary_series], ignore_index=True,)
summary_df.to_csv(SUMMARY_DIR, index=False)

# add R score from cross-validation results
# ablation_df = pd.read_csv(ABLATION_STUDY)
# results_list = [
#     "OPV",
#     "SVR",
#     "sklearn",
#     "Manual Fragments",
#     mean(outer_results),
#     std(outer_results),
# ]
# ablation_df.loc[len(ablation_df.index) + 1] = results_list
# ablation_df.to_csv(ABLATION_STUDY, index=False)
