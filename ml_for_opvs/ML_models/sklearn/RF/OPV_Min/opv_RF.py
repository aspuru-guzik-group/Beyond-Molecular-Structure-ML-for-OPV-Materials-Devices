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
import matplotlib.pyplot as plt
from ml_for_opvs.ML_models.sklearn.data.OPV_Min.data import Dataset
from rdkit import Chem
from collections import deque
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from sklearn import preprocessing

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
    "ml_for_opvs", "ML_models/sklearn/RF/OPV_Min/"
)

FEATURE_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "ML_models/sklearn/RF/OPV_Min/"
)


np.set_printoptions(precision=3)
SEED_VAL = 4


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


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
r_score = make_scorer(custom_scorer, greater_is_better=False)

# log results
summary_df = pd.DataFrame(
    columns=["Datatype", "R_mean", "R_std", "RMSE_mean", "RMSE_std", "num_of_data"]
)

# ALL THE PARAMETERS!!!
# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "bigsmiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "manual": 0,
    "aug_manual": 0,
    "fingerprint": 1,
}
parameter_type = {
    "none": 0,
    "electronic": 0,
    "electronic_only": 1,
    "device": 0,
    "device_only": 0,
    "device_solvent": 0,
    "device_solvent_only": 0,
    "fabrication": 0,
    "fabrication_only": 0,
    "fabrication_solvent": 0,
    "fabrication_solvent_only": 0,
}
target_type = {
    "PCE": 1,
    "FF": 0,
    "JSC": 0,
    "VOC": 0,
}
for target in target_type:
    if target_type[target] == 1:
        target_predict = target
        if target_predict == "PCE":
            SUMMARY_DIR = SUMMARY_DIR + "PCE_"
            FEATURE_DIR = FEATURE_DIR + "PCE_"
        elif target_predict == "FF":
            SUMMARY_DIR = SUMMARY_DIR + "FF_"
            FEATURE_DIR = FEATURE_DIR + "FF_"
        elif target_predict == "JSC":
            SUMMARY_DIR = SUMMARY_DIR + "JSC_"
            FEATURE_DIR = FEATURE_DIR + "JSC_"
        elif target_predict == "VOC":
            SUMMARY_DIR = SUMMARY_DIR + "VOC_"
            FEATURE_DIR = FEATURE_DIR + "VOC_"

for param in parameter_type:
    if parameter_type[param] == 1:
        dev_param = param
        if dev_param == "none":
            SUMMARY_DIR = SUMMARY_DIR + "none_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "none_opv_rf_feature_impt.csv"
            device_idx = 0
        elif dev_param == "electronic":
            SUMMARY_DIR = SUMMARY_DIR + "electronic_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "electronic_opv_rf_feature_impt.csv"
            device_idx = 4
        elif dev_param == "electronic_only":
            SUMMARY_DIR = SUMMARY_DIR + "electronic_only_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "electronic_only_opv_rf_feature_impt.csv"
            device_idx = 4
        elif dev_param == "device":
            SUMMARY_DIR = SUMMARY_DIR + "device_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "device_opv_rf_feature_impt.csv"
            device_idx = 11
        elif dev_param == "device_only":
            SUMMARY_DIR = SUMMARY_DIR + "device_only_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "device_only_opv_rf_feature_impt.csv"
            device_idx = 11
        elif dev_param == "device_solvent":
            SUMMARY_DIR = SUMMARY_DIR + "device_solv_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "device_solv_opv_rf_feature_impt.csv"
            # for device_solvent
            feature_idx = [
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
            ]
            device_idx = 19
        elif dev_param == "device_solvent_only":
            SUMMARY_DIR = SUMMARY_DIR + "device_solv_only_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "device_solv_only_opv_rf_feature_impt.csv"
            # for device_solvent
            feature_idx = [
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
            ]
            device_idx = 19
        elif dev_param == "fabrication":
            SUMMARY_DIR = SUMMARY_DIR + "fabrication_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "fabrication_opv_rf_feature_impt.csv"
            device_idx = 7
        elif dev_param == "fabrication_only":
            SUMMARY_DIR = SUMMARY_DIR + "fabrication_only_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "fabrication_only_opv_rf_feature_impt.csv"
            device_idx = 7
        elif dev_param == "fabrication_solvent":
            SUMMARY_DIR = SUMMARY_DIR + "fabrication_solv_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "fabrication_solv_opv_rf_feature_impt.csv"
            # for fabrication_solvent
            feature_idx = [
                13,
                15,
                16,
                17,
                18,
                19,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
            ]
            device_idx = 15
        elif dev_param == "fabrication_solvent_only":
            SUMMARY_DIR = SUMMARY_DIR + "fabrication_solv_only_opv_rf_results.csv"
            FEATURE_DIR = FEATURE_DIR + "fabrication_solv_only_opv_rf_feature_impt.csv"
            # for fabrication_solvent
            feature_idx = [
                13,
                15,
                16,
                17,
                18,
                19,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
            ]
            device_idx = 15

# feature = True
feature = False

feature_impt_df = pd.DataFrame()

if unique_datatype["smiles"] == 1:
    dataset = Dataset()
    dataset.prepare_data(TRAIN_MASTER_DATA, "smi")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "SMILES"
elif unique_datatype["bigsmiles"] == 1:
    dataset = Dataset()
    dataset.prepare_data(TRAIN_MASTER_DATA, "bigsmi")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "BigSMILES"
elif unique_datatype["selfies"] == 1:
    dataset = Dataset()
    dataset.prepare_data(TRAIN_MASTER_DATA, "selfies")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "SELFIES"
elif unique_datatype["aug_smiles"] == 1:
    dataset = Dataset()
    dataset.prepare_data(TRAIN_MASTER_DATA, "smi")
    x, y, max_target, min_target, token_dict = dataset.setup_aug_smi(
        dev_param, target_predict
    )
    num_of_augment = 4  # 1+4x amount of data
    datatype = "AUG_SMILES"
elif unique_datatype["brics"] == 1:
    dataset = Dataset()
    dataset.prepare_data(BRICS_MASTER_DATA, "brics")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "BRICS"
elif unique_datatype["manual"] == 1:
    dataset = Dataset()
    dataset.prepare_data(MANUAL_MASTER_DATA, "manual")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "MANUAL"
elif unique_datatype["aug_manual"] == 1:
    dataset = Dataset()
    dataset.prepare_data(MANUAL_MASTER_DATA, "manual")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "AUG_MANUAL"
elif unique_datatype["fingerprint"] == 1:
    dataset = Dataset()
    dataset.prepare_data(FP_MASTER_DATA, "fp")
    x, y, max_target, min_target = dataset.setup(dev_param, target_predict)
    datatype = "FINGERPRINT"

print(datatype)  # Ensures we know which model is running
print(dev_param)
print(target_predict)

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
            if dev_param == "none":
                x_aug, y_aug = augment_smi_in_loop(x_, y_, num_of_augment, True)
            else:
                x_list = list(x_)
                x_aug, y_aug = augment_smi_in_loop(
                    str(x_list[0]), y_, num_of_augment, True
                )
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

        if dev_param == "none":
            tokenized_test, max_test_seq_length = Tokenizer().tokenize_from_dict(
                x_test, max_seq_length, input_dict
            )

        else:
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
    print(x_train[0])
    # execute search
    result = search.fit(x_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # get permutation importances of best performing model (overcomes bias toward high-cardinality (very unique) features)

    # get feature importances of best performing model
    # NOTE: set cutoff threshold for importance OR set X most important features
    # NOTE: set labels for each feature!
    if feature:
        dataset_columns = list(dataset.data.columns)
        # col_idx = 0
        # dataset_columns_dict = {}
        # for col in dataset_columns:
        #     dataset_columns_dict[col] = col_idx
        #     col_idx += 1
        feature_columns = []
        for idx in feature_idx:
            feature_columns.append(dataset_columns[idx])
        importances = best_model.feature_importances_
        importances = importances[len(importances) - device_idx : len(importances)]
        forest_importances = pd.DataFrame(importances, index=feature_columns)
        feature_impt_df = pd.concat(
            [feature_impt_df, forest_importances], axis=1, ignore_index=False,
        )

    # evaluate model on the hold out dataset
    yhat = best_model.predict(x_test)
    # reverse min-max scaling
    yhat = (yhat * (max_target - min_target)) + min_target
    y_test = (y_test * (max_target - min_target)) + min_target

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

if feature:
    # feature importance summary
    if dev_param in [
        "device",
        "device_solvent",
        "fabrication",
        "fabrication_solvent",
        "device_only",
        "device_solvent_only",
        "fabrication_only",
        "fabrication_solvent_only",
    ]:
        feature_impt_df.to_csv(FEATURE_DIR)

