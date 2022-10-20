import ast
import json
from tokenize import Token
from typing import Tuple, Union
import pandas as pd
import numpy as np
from xgboost import train

from ml_for_opvs.ML_models.sklearn.tokenizer import Tokenizer

np.set_printoptions(suppress=True)


def tokenize_from_dict(token2idx: dict, input_value: Union[list, str]) -> list:
    """

    Args:
        token2idx (dict): dictionary of unique tokens with corresponding indices.
        input_value (list, str): input_value with tokens that match the token2idx.

    Returns:
        tokenized_list (list): list of tokenized inputs
    """
    tokenized_list: list = []
    for token in input_value:
        tokenized_list.append(token2idx[token])

    return tokenized_list


def pad_input(input_list_of_list, max_input_length) -> list:
    """Pad the input_value (pre-padding) with 0's until max_length is met.

    Args:
        input_list_of_list (list): list of inputs.
        max_length (int): max length of any input_value in the entire dataset.

    Returns:
        input_list_of_list (list): list of inputs with pre-padding.
    """
    for input_list in input_list_of_list:
        for i in range(max_input_length - len(input_list)):
            input_list.insert(0, 0)

    return input_list_of_list


def feature_scale(feature_series: pd.Series) -> np.array:
    """
    Min-max scaling of a feature.
    Args:
        feature_series: a pd.Series of a feature
    Returns:
        scaled_feature: a np.array (same index) of feature that is min-max scaled
        max_value: maximum value from the entire feature array
    """
    feature_array = feature_series.to_numpy().astype("float64")
    max_value = np.nanmax(feature_array)
    min_value = np.nanmin(feature_array)
    return max_value, min_value


def filter_nan(df_to_filter):
    """
    Args:
        df_to_filter (_type_): _description_

    Returns:
        filtered_df (df.Dataframe):
    """
    pass


def process_features(
    train_feature_df, test_feature_df, input_rep_bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Processes various types of features (str, float, list) and returns "training ready" arrays.

    Args:
        train_feature_df (pd.DataFrame): subset of train_df with selected features.
        test_feature_df (pd.DataFrame): subset of test_df with selected features.
        input_rep_bool (bool): True = presence of input_representation. False = absence of input_representation (only use features).

    Returns:
        input_train_array (np.array): tokenized, padded array ready for training
        input_test_array (np.array): tokenized, padded array ready for test
    """
    assert len(train_feature_df) > 1, train_feature_df
    assert len(test_feature_df) > 1, test_feature_df
    # First in column_headers will always be input_representation
    column_headers = train_feature_df.columns
    if input_rep_bool:
        input_representation = column_headers[0]
    else:
        input_representation = None

    # calculate feature scale dict
    feature_scale_dict: dict = {}
    concat_df = pd.concat([train_feature_df, test_feature_df], ignore_index=True)
    for column in column_headers:
        if any(
            [
                isinstance(concat_df[column][1], np.float64),
                isinstance(concat_df[column][1], float),
                isinstance(concat_df[column][1], np.int64),
                isinstance(concat_df[column][1], int),
            ]
        ):
            feature_max, feature_min = feature_scale(concat_df[column])
            feature_column_max = column + "_max"
            feature_column_min = column + "_min"
            feature_scale_dict[feature_column_max] = feature_max
            feature_scale_dict[feature_column_min] = feature_min

    # TOKENIZATION
    # must loop through entire dataframe for token2idx
    input_instance = None
    if input_rep_bool:
        try:
            input_value = ast.literal_eval(concat_df[input_representation][1])
            if isinstance(input_value[0], list):
                input_instance = "list_of_list"
                # print("input_value is a list of list")
            else:
                input_instance = "list"
                # print("input_value is a list which could be: 1) fragments or 2) SMILES")
        except:  # The input_value was not a list, so ast.literal_eval will raise valueError.
            input_instance = "str"
            input_value = concat_df[input_representation][1]
            # print("input_value is a string")

        if (
            input_instance == "list"
        ):  # could be list of fragments or list of (augmented) SMILES.
            # check if list of: 1) fragments or 2) SMILES or 3) fingerprints
            if "Augmented_SMILES" == input_representation:
                augmented_smi_list: list = []
                for index, row in concat_df.iterrows():
                    input_value = ast.literal_eval(row[input_representation])
                    for aug_value in input_value:
                        augmented_smi_list.append(aug_value)
                augmented_smi_series: pd.Series = pd.Series(augmented_smi_list)
                (
                    tokenized_array,
                    max_length,
                    vocab_length,
                    token2idx,
                ) = Tokenizer().tokenize_data(augmented_smi_series)
            else:  # fragments or fingerprints
                token2idx = {}
                token_idx = 0
                for index, row in concat_df.iterrows():
                    input_value = ast.literal_eval(row[input_representation])
                    for frag in input_value:
                        if frag not in list(token2idx.keys()):
                            token2idx[frag] = token_idx
                            token_idx += 1
        elif input_instance == "list_of_list":  # list of list of augmented fragments
            token2idx: dict = {}
            token_idx: int = 0
            for index, row in concat_df.iterrows():
                input_value = ast.literal_eval(row[input_representation])
                for aug_value in input_value:
                    for frag in aug_value:
                        if frag not in list(token2idx.keys()):
                            token2idx[frag] = token_idx
                            token_idx += 1
        elif input_instance == "str":
            if "SMILES" in input_representation:
                (
                    tokenized_array,
                    max_length,
                    vocab_length,
                    token2idx,
                ) = Tokenizer().tokenize_data(concat_df[input_representation])
            elif "SELFIES" in input_representation:
                token2idx, max_length = Tokenizer().tokenize_selfies(
                    concat_df[input_representation]
                )
        else:
            raise TypeError("input_value is neither str or list. Fix it!")

        # Tokenize string features
        for index, row in concat_df.iterrows():
            for column in column_headers:
                if column != input_representation and isinstance(row[column], str):
                    if row[column] not in list(token2idx.keys()):
                        token_idx = len(token2idx)
                        token2idx[row[column]] = token_idx

    max_input_length = 0  # for padding
    # processing training data
    input_train_list = []
    for index, row in train_feature_df.iterrows():
        # augmented data needs to be processed differently.
        if any(
            [
                "Augmented_SMILES" == input_representation,
                input_instance == "list_of_list",
            ]
        ):
            # get feature variables (not the input representation)
            feature_list = []
            for column in column_headers:
                try:
                    input_value = ast.literal_eval(row[column])
                except:
                    input_value = row[column]
                if any(
                    [
                        isinstance(input_value, np.float64),
                        isinstance(input_value, float),
                        isinstance(input_value, np.int64),
                        isinstance(input_value, int),
                    ]
                ):
                    # feature scaling (min-max)
                    input_value = row[column]
                    column_max = column + "_max"
                    column_min = column + "_min"
                    input_column_max = feature_scale_dict[column_max]
                    input_column_min = feature_scale_dict[column_min]
                    input_value = (input_value - input_column_min) / (
                        input_column_max - input_column_min
                    )
                    feature_list.append(input_value)
                elif isinstance(input_value, str):
                    str_feature = tokenize_from_dict(token2idx, input_value)
                    feature_list.extend(str_feature)

            # process augmented input representations
            input_value = ast.literal_eval(row[input_representation])
            for aug_value in input_value:
                tokenized_list = []
                if "Augmented_SMILES" == input_representation:
                    tokenized_list.extend(
                        Tokenizer().tokenize_from_dict(token2idx, aug_value)
                    )  # SMILES
                else:
                    tokenized_list.extend(
                        tokenize_from_dict(token2idx, aug_value)
                    )  # fragments
                tokenized_list.extend(feature_list)
                input_train_list.append(tokenized_list)
                if len(tokenized_list) > max_input_length:  # for padding
                    max_input_length = len(tokenized_list)

        else:
            tokenized_list = []
            feature_list = []
            for column in column_headers:
                # input_value type can be (list, str, float, int)
                try:
                    input_value = ast.literal_eval(row[column])
                except:
                    input_value = row[column]
                # tokenization
                if isinstance(input_value, list):
                    tokenized_list.extend(
                        tokenize_from_dict(token2idx, input_value)
                    )  # fragments
                elif (
                    isinstance(input_value, str) and column == input_representation
                ):  # input_representation
                    tokenized_list.extend(
                        Tokenizer().tokenize_from_dict(token2idx, input_value)
                    )  # SMILES
                elif (
                    isinstance(input_value, str) and column != input_representation
                ):  # string feature
                    feature_list.extend(tokenize_from_dict(token2idx, [input_value]))
                elif any(
                    [
                        isinstance(input_value, np.float64),
                        isinstance(input_value, float),
                        isinstance(input_value, np.int64),
                        isinstance(input_value, int),
                    ]
                ):
                    # feature scaling (min-max)
                    input_value = row[column]
                    column_max = column + "_max"
                    column_min = column + "_min"
                    input_column_max = feature_scale_dict[column_max]
                    input_column_min = feature_scale_dict[column_min]
                    input_value = (input_value - input_column_min) / (
                        input_column_max - input_column_min
                    )
                    feature_list.append(input_value)
                else:
                    print(type(input_value))
                    raise ValueError("Missing value. Cannot be null value in dataset!")
            if len(tokenized_list) > max_input_length:  # for padding
                max_input_length = len(tokenized_list)
            # add features
            tokenized_list.extend(feature_list)
            input_train_list.append(tokenized_list)

    # processing test data
    input_test_list = []
    for index, row in test_feature_df.iterrows():
        # augmented data needs to be processed differently.
        if any(
            [
                "Augmented_SMILES" == input_representation,
                input_instance == "list_of_list",
            ]
        ):
            # get feature variables (not the input representation)
            feature_list = []
            for column in column_headers:
                try:
                    input_value = ast.literal_eval(row[column])
                except:
                    input_value = row[column]
                if any(
                    [
                        isinstance(input_value, np.float64),
                        isinstance(input_value, float),
                        isinstance(input_value, np.int64),
                        isinstance(input_value, int),
                    ]
                ):
                    # feature scaling (min-max)
                    input_value = row[column]
                    column_max = column + "_max"
                    column_min = column + "_min"
                    input_column_max = feature_scale_dict[column_max]
                    input_column_min = feature_scale_dict[column_min]
                    input_value = (input_value - input_column_min) / (
                        input_column_max - input_column_min
                    )
                    feature_list.append(input_value)
                elif isinstance(input_value, str):
                    str_feature = tokenize_from_dict(token2idx, input_value)
                    feature_list.extend(str_feature)

            # process augmented input representations
            input_value = ast.literal_eval(row[input_representation])
            for aug_value in input_value:
                tokenized_list = []
                if "Augmented_SMILES" == input_representation:
                    tokenized_list.extend(
                        Tokenizer().tokenize_from_dict(token2idx, aug_value)
                    )  # SMILES
                else:
                    tokenized_list.extend(
                        tokenize_from_dict(token2idx, aug_value)
                    )  # fragments
                tokenized_list.extend(feature_list)
                input_test_list.append(tokenized_list)
                if len(tokenized_list) > max_input_length:  # for padding
                    max_input_length = len(tokenized_list)

        else:
            tokenized_list = []
            feature_list = []
            for column in column_headers:
                # input_value type can be (list, str, float, int)
                try:
                    input_value = ast.literal_eval(row[column])
                except:
                    input_value = row[column]
                # tokenization
                if isinstance(input_value, list):
                    tokenized_list.extend(
                        tokenize_from_dict(token2idx, input_value)
                    )  # fragments
                elif (
                    isinstance(input_value, str) and column == input_representation
                ):  # input_representation
                    tokenized_list.extend(
                        Tokenizer().tokenize_from_dict(token2idx, input_value)
                    )  # SMILES
                elif (
                    isinstance(input_value, str) and column != input_representation
                ):  # string feature
                    feature_list.extend(tokenize_from_dict(token2idx, [input_value]))
                elif any(
                    [
                        isinstance(input_value, np.float64),
                        isinstance(input_value, float),
                        isinstance(input_value, np.int64),
                        isinstance(input_value, int),
                    ]
                ):
                    # feature scaling (min-max)
                    input_value = row[column]
                    column_max = column + "_max"
                    column_min = column + "_min"
                    input_column_max = feature_scale_dict[column_max]
                    input_column_min = feature_scale_dict[column_min]
                    input_value = (input_value - input_column_min) / (
                        input_column_max - input_column_min
                    )
                    feature_list.append(input_value)
                else:
                    print(type(input_value))
                    raise ValueError("Missing value. Cannot be null value in dataset!")
            if len(tokenized_list) > max_input_length:  # for padding
                max_input_length = len(tokenized_list)
            # add features
            tokenized_list.extend(feature_list)
            input_test_list.append(tokenized_list)

    # padding
    input_train_list = pad_input(input_train_list, max_input_length)
    input_test_list = pad_input(input_test_list, max_input_length)
    input_train_array = np.array(input_train_list)
    input_test_array = np.array(input_test_list)
    assert type(input_train_array[0]) == np.ndarray, input_train_array
    assert type(input_test_array[0]) == np.ndarray, input_test_array

    return input_train_array, input_test_array


def process_target(
    train_target_df, test_target_df, train_df, input_rep_bool
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Processes one target value through the following steps:
    1) min-max scaling
    2) return as array

    Args:
        train_target_df (pd.DataFrame): target values for training dataframe
        test_target_df (pd.DataFrame): target values for test dataframe
        train_df (pd.DataFrame): training dataframe for recovering input_representation
        input_rep_bool (bool): True = presence of input_representation. False = absence of input_representation (only use features).
    Returns:
        target_train_array (np.array): array of training targets
        target_test_array (np.array): array of test targets
        target_max (float): maximum value in dataset
        target_min (float): minimum value in dataset
    """
    assert len(train_target_df) > 1, train_target_df
    assert len(test_target_df) > 1, test_target_df
    # first column will always be the target column
    # NOTE: Max and Min values are found from training set -> prevents data leakage
    target_max, target_min = feature_scale(train_target_df[train_target_df.columns[0]])

    # First in column_headers will always be input_representation
    column_headers = train_df.columns
    for column in column_headers:
        if type(train_df[column][1]) == str:
            input_representation = column

    # additional data points for targets if data is augmented
    input_instance = None
    try:
        input_value = ast.literal_eval(train_df[input_representation][1])
        if isinstance(input_value[0], list):
            input_instance = "list_of_list"
            # print("input_value is a list of list")
        else:
            input_instance = "list"
            # print("input_value is a list which could be: 1) fragments or 2) SMILES")
    except:  # The input_value was not a list, so ast.literal_eval will raise valueError.
        input_instance = "str"
        # print("input_value is a string")

    # duplicate number of target test with the number of augmented data points
    if input_rep_bool:
        if any(
            [
                "Augmented_SMILES" == input_representation,
                input_instance == "list_of_list",
            ]
        ):
            target_train_list = []
            for index, row in train_target_df.iterrows():
                input_value = ast.literal_eval(row[input_representation])
                for i in range(len(input_value)):
                    target_train_list.append(row[train_target_df.columns[0]])

            target_test_list = []
            for index, row in test_target_df.iterrows():
                input_value = ast.literal_eval(row[input_representation])
                for i in range(len(input_value)):
                    target_test_list.append(row[test_target_df.columns[0]])

            target_train_array = np.array(target_train_list)
            target_test_array = np.array(target_test_list)
        else:
            target_train_array = train_target_df[train_target_df.columns[0]].to_numpy()
            target_train_array = np.ravel(target_train_array)
            target_test_array = test_target_df[test_target_df.columns[0]].to_numpy()
            target_test_array = np.ravel(target_test_array)
    else:
        target_train_array = train_target_df[train_target_df.columns[0]].to_numpy()
        target_train_array = np.ravel(target_train_array)
        target_test_array = test_target_df[test_target_df.columns[0]].to_numpy()
        target_test_array = np.ravel(target_test_array)

    target_train_array = (target_train_array - target_min) / (target_max - target_min)
    target_test_array = (target_test_array - target_min) / (target_max - target_min)

    return target_train_array, target_test_array, target_max, target_min


def get_space_dict(space_json_path, model_type):
    """Opens json file and returns a dictionary of the space.

    Args:
        space_json_path (str): filepath to json containing search space of hyperparameters

    Returns:
        space (dict): dictionary of necessary hyperparameters
    """
    space = {}
    with open(space_json_path) as json_file:
        space_json = json.load(json_file)
    if model_type == "RF":
        space_keys = [
            "n_estimators",
            "min_samples_leaf",
            "min_samples_split",
            "max_depth",
        ]
    elif model_type == "XGBoost":
        space_keys = [
            "alpha",
            "n_estimators",
            "max_depth",
            "subsample",
            "min_child_weight",
        ]
    elif model_type == "KRR":
        space_keys = ["alpha"]
    elif model_type == "SVM":
        space_keys = ["kernel", "degree"]
    elif model_type == "MLR":
        space_keys = ["fit_intercept"]
    elif model_type == "KNN":
        space_keys = ["n_neighbors", "leaf_size"]
    elif model_type == "Lasso":
        space_keys = ["alpha"]

    for key in space_keys:
        assert key in space_json.keys(), key
        space[key] = space_json[key]

    return space


# l = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
# l2 = [[1, 2, 3], [3, 4, 5], [2, 3]]

# print(np.array(l)[0], type(np.array(l)[0]))
