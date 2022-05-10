# data.py for classical ML
from cmath import nan
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf
from sklearn import preprocessing

import torch
from torch.utils.data import random_split
import yaspin

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

# For Manual Fragments!
MANUAL_DONOR_CSV = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/donor_frags.csv"
)

MANUAL_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/acceptor_frags.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)

from ml_for_opvs.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer
from ml_for_opvs.data.postprocess.OPV_Min.BRICS.brics_frag import BRIC_FRAGS
from ml_for_opvs.data.postprocess.OPV_Min.manual_frag.manual_frag import manual_frag


class Dataset:
    """
    Class that contains functions to prepare the data into a 
    dataframe with the feature variables and the PCE, etc.
    """

    def __init__(self):
        pass

    def prepare_data(self, data_dir: str, input: int):
        """
        Function that concatenates donor-acceptor pair
        """
        self.data = pd.read_csv(data_dir)
        self.input = input
        self.data["DA_pair"] = " "
        # concatenate Donor and Acceptor Inputs
        if self.input == "smi":
            representation = "SMILES"
        elif self.input == "bigsmi":
            representation = "Big_SMILES"
        elif self.input == "selfies":
            representation = "SELFIES"

        if self.input == "smi" or self.input == "bigsmi" or self.input == "selfies":
            for index, row in self.data.iterrows():
                self.data.at[index, "DA_pair"] = (
                    row["Donor_{}".format(representation)]
                    + "."
                    + row["Acceptor_{}".format(representation)]
                )

    def add_device_params(
        self, parameter: str, tokenized_input: list, token_dict: dict
    ) -> np.array:
        """
        Function that adds specific parameters to data

        Args:
            parameter: type of parameters to include:
                - electronic: HOMO, LUMO
                - device: all device parameters
                - fabrication: all the fabrication parameters (D:A ratio - Annealing Temp.)
                - electronic_only: add HOMO, LUMO (should not incl. representation data)
            token_dict: dictionary of tokens with token2idx of chemical representation inputs

        Returns:
            tokenized_input: same input array but with added parameters
        """
        # add device parameters to the end of input
        index = 0
        while index < len(tokenized_input):
            if parameter == "electronic" or parameter == "electronic_only":
                homo_d = self.data["HOMO_D (eV)"].to_numpy().astype("float32")
                lumo_d = self.data["LUMO_D (eV)"].to_numpy().astype("float32")
                homo_a = self.data["HOMO_A (eV)"].to_numpy().astype("float32")
                lumo_a = self.data["LUMO_A (eV)"].to_numpy().astype("float32")
                tokenized_input[index].append(homo_d[index])
                tokenized_input[index].append(lumo_d[index])
                tokenized_input[index].append(homo_a[index])
                tokenized_input[index].append(lumo_a[index])
            elif parameter == "device":
                d_a_ratio = self.data["D:A ratio (m/m)"].to_numpy().astype("float32")
                total_solids_conc = (
                    self.data["total solids conc. (mg/mL)"].to_numpy().astype("float32")
                )
                solvent_add_conc = (
                    self.data["solvent additive conc. (%v/v)"]
                    .to_numpy()
                    .astype("float32")
                )
                active_layer_thickness = (
                    self.data["active layer thickness (nm)"]
                    .to_numpy()
                    .astype("float32")
                )
                annealing_temp = (
                    self.data["annealing temperature"].to_numpy().astype("float32")
                )
                hole_mobility_blend = (
                    self.data["hole mobility blend (cm^2 V^-1 s^-1)"]
                    .to_numpy()
                    .astype("float32")
                )
                electron_mobility_blend = (
                    self.data["electron mobility blend (cm^2 V^-1 s^-1)"]
                    .to_numpy()
                    .astype("float32")
                )

                # tokenize non-numerical variables
                # for str (non-numerical) variables
                dict_idx = len(token_dict)
                solvent = self.data["solvent"]
                for input in solvent:
                    # unique solvents
                    if input not in token_dict:
                        token_dict[input] = dict_idx
                        dict_idx += 1
                solvent_add = self.data["solvent additive"]
                for input in solvent_add:
                    # unique solvent additives
                    if input not in token_dict:
                        token_dict[input] = dict_idx
                        dict_idx += 1
                hole_contact_layer = self.data["hole contact layer"]
                for input in hole_contact_layer:
                    # unique hole contact layer
                    if input not in token_dict:
                        token_dict[input] = dict_idx
                        dict_idx += 1
                electron_contact_layer = self.data["electron contact layer"]
                for input in electron_contact_layer:
                    # unique electron contact layer
                    if input not in token_dict:
                        token_dict[input] = dict_idx
                        dict_idx += 1

                tokenized_input[index].append(d_a_ratio[index])
                tokenized_input[index].append(total_solids_conc[index])
                tokenized_input[index].append(solvent[index])
                tokenized_input[index].append(solvent_add[index])
                tokenized_input[index].append(solvent_add_conc[index])
                tokenized_input[index].append(active_layer_thickness[index])
                tokenized_input[index].append(annealing_temp[index])
                tokenized_input[index].append(hole_mobility_blend[index])
                tokenized_input[index].append(electron_mobility_blend[index])
                tokenized_input[index].append(hole_contact_layer[index])
                tokenized_input[index].append(electron_contact_layer[index])

            elif parameter == "fabrication":
                d_a_ratio = self.data["D:A ratio (m/m)"].to_numpy().astype("float32")
                total_solids_conc = (
                    self.data["total solids conc. (mg/mL)"].to_numpy().astype("float32")
                )
                solvent_add_conc = (
                    self.data["solvent additive conc. (%v/v)"]
                    .to_numpy()
                    .astype("float32")
                )
                active_layer_thickness = (
                    self.data["active layer thickness (nm)"]
                    .to_numpy()
                    .astype("float32")
                )
                annealing_temp = (
                    self.data["annealing temperature"].to_numpy().astype("float32")
                )

                # tokenize non-numerical variables
                # for str (non-numerical) variables
                dict_idx = len(token_dict)
                solvent = self.data["solvent"]
                for input in solvent:
                    # unique solvents
                    if input not in token_dict:
                        token_dict[input] = dict_idx
                        dict_idx += 1
                solvent_add = self.data["solvent additive"]
                for input in solvent_add:
                    # unique solvent additives
                    if input not in token_dict:
                        token_dict[input] = dict_idx
                        dict_idx += 1

                tokenized_input[index].append(d_a_ratio[index])
                tokenized_input[index].append(total_solids_conc[index])
                tokenized_input[index].append(solvent[index])
                tokenized_input[index].append(solvent_add[index])
                tokenized_input[index].append(solvent_add_conc[index])
                tokenized_input[index].append(active_layer_thickness[index])
                tokenized_input[index].append(annealing_temp[index])
            else:
                return np.asarray(tokenized_input)
            index += 1
        return tokenized_input, token_dict

    def tokenize_data(self, tokenized_input: list, token_dict: dict) -> np.array:
        """
         Function that tokenizes data considering all types of data

        Args:
            tokenized_input: input list with added parameters
            token_dict: dictionary of tokens with token2idx of chemical representation inputs

        Returns:
            tokenized_data: tokenized input array w/ added parameters
        """
        # tokenize data
        data_pt_idx = 0
        while data_pt_idx < len(tokenized_input):
            token_idx = 0
            while token_idx < len(tokenized_input[data_pt_idx]):
                token = tokenized_input[data_pt_idx][token_idx]
                if token == "nan":
                    tokenized_input[data_pt_idx][token_idx] = nan
                elif isinstance(token, str):
                    tokenized_input[data_pt_idx][token_idx] = token_dict[token]
                token_idx += 1
            data_pt_idx += 1

        return tokenized_input

    def filter_nan(self, tokenized_data: np.array, target_array: np.array) -> np.array:
        """
        Function that filters out "nan" values and target_array

        Args:
            tokenized_data: input array with added parameters
            target_array: array with target values (PCE, Jsc, FF, Voc)

        Returns:
            filtered_tokenized_data: filtered array of tokenized inputs
        """
        # filter out "nan" values
        filtered_tokenized_input = []
        filtered_target_array = []
        nan_idx = 0
        while nan_idx < len(tokenized_data):
            nan_bool = False
            for item in tokenized_data[nan_idx]:
                if str(item) == "nan":
                    nan_bool = True
            if not nan_bool:
                filtered_tokenized_input.append(tokenized_data[nan_idx])
                filtered_target_array.append(target_array[nan_idx])
            nan_idx += 1

        return filtered_tokenized_input, filtered_target_array

    def setup(self, parameter, target):
        """
        Function that sets up data ready for training 
        # NOTE: only run parameter_only on setup("electronic_only", target)

        Args:
            parameter: type of parameters to include:
                - electronic: HOMO, LUMO
                - device: all device parameters
                - fabrication: all the fabrication parameters (D:A ratio - Annealing Temp.)
            target: the target value we want to predict for (PCE, Jsc, Voc, FF)

        """
        if target == "PCE":
            target_array = self.data["calc_PCE (%)"].to_numpy().astype("float32")
        elif target == "JSC":
            target_array = self.data["Jsc (mA cm^-2)"].to_numpy().astype("float32")
        elif target == "VOC":
            target_array = self.data["Voc (V)"].to_numpy().astype("float32")
        elif target == "FF":
            target_array = self.data["FF (%)"].to_numpy().astype("float32")

        # minimize range of target between 0-1
        # find max of target_array
        self.max_target = target_array.max()
        target_array = target_array / self.max_target

        if self.input == "smi":
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == "bigsmi":
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == "selfies":
            # tokenize data using selfies
            tokenized_input = []
            token_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                self.data["DA_pair"]
            )
            for index, row in self.data.iterrows():
                tokenized_selfie = sf.selfies_to_encoding(
                    self.data.at[index, "DA_pair"],
                    token_dict,
                    pad_to_len=-1,
                    enc_type="label",
                )
                tokenized_input.append(tokenized_selfie)

            # tokenized_input = np.asarray(tokenized_input)
            tokenized_input = Tokenizer().pad_input(tokenized_input, max_selfie_length)
        elif self.input == "brics":
            tokenized_input = []
            for i in range(len(self.data["DA_tokenized_BRICS"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["DA_tokenized_BRICS"][i])
                tokenized_input.append(da_pair_list)
            # add device parameters to the end of input
            b_frag = BRIC_FRAGS(TRAIN_MASTER_DATA)
            token_dict = b_frag.bric_frag()
        elif self.input == "manual":
            tokenized_input = []
            for i in range(len(self.data["DA_manual_tokenized"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["DA_manual_tokenized"][i])
                tokenized_input.append(da_pair_list)
            # add device parameters to the end of input
            manual = manual_frag(
                TRAIN_MASTER_DATA, MANUAL_DONOR_CSV, MANUAL_ACCEPTOR_CSV
            )
            token_dict = manual.return_frag_dict()
        elif self.input == "fp":
            column_da_pair = "DA_FP_radius_3_nbits_512"
            tokenized_input = []
            for i in range(len(self.data[column_da_pair])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data[column_da_pair][i])
                tokenized_input.append(da_pair_list)
            token_dict = {0: 0, 1: 1}

        # add parameters
        if "only" in parameter:
            # create empty list with same dimensions as target_array
            empty_input = [[] for _ in range(len(target_array))]
            # add device parameters to the end of input
            tokenized_input, token_dict = self.add_device_params(
                parameter, empty_input, {}
            )
            # tokenize data
            tokenized_input = self.tokenize_data(tokenized_input, token_dict)

            # filter out "nan" values
            filtered_tokenized_input, filtered_target_array = self.filter_nan(
                tokenized_input, target_array
            )
            return (
                np.asarray(filtered_tokenized_input),
                np.asarray(filtered_target_array),
            )
        elif parameter != "none":
            # add device parameters to the end of input
            tokenized_input, token_dict = self.add_device_params(
                parameter, tokenized_input, token_dict
            )
            # tokenize data
            tokenized_input = self.tokenize_data(tokenized_input, token_dict)

            # filter out "nan" values
            filtered_tokenized_input, filtered_target_array = self.filter_nan(
                tokenized_input, target_array
            )
            print(token_dict)
            return (
                np.asarray(filtered_tokenized_input),
                np.asarray(filtered_target_array),
            )
        else:
            return np.asarray(tokenized_input), np.asarray(target_array)

    def setup_aug_smi(self, parameter, target):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        Args:
            parameter: type of parameters to include:
                - electronic: HOMO, LUMO
                - device: all device parameters
                - fabrication: all the fabrication parameters (D:A ratio - Annealing Temp.)
            target: the target value we want to predict for (PCE, Jsc, Voc, FF)
        """
        if target == "PCE":
            target_array = self.data["calc_PCE (%)"].to_numpy().astype("float32")
        elif target == "JSC":
            target_array = self.data["Jsc (mA cm^-2)"].to_numpy().astype("float32")
        elif target == "VOC":
            target_array = self.data["Voc (V)"].to_numpy().astype("float32")
        elif target == "FF":
            target_array = self.data["FF (%)"].to_numpy().astype("float32")

        # minimize range of target between 0-1
        # find max of target_array
        self.max_target = target_array.max()
        target_array = target_array / self.max_target

        # convert Series to list
        x = self.data["DA_pair"].to_list()
        # convert list to list of lists
        idx = 0
        for _x in x:
            x[idx] = [_x]
            idx += 1
        # FOR AUGMENTATION: device index (don't augment the device stuff)
        if parameter != "none":
            # add device parameters to the end of input
            tokenized_input, token_dict = self.add_device_params(parameter, x, {})
            # filter out "nan" values
            filtered_tokenized_input, filtered_target_array = self.filter_nan(
                tokenized_input, target_array
            )
            return (
                np.asarray(filtered_tokenized_input, dtype="object"),
                np.asarray(filtered_target_array, dtype="float32"),
                token_dict,
            )
        else:
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
            return np.asarray(x), np.asarray(target_array), token_dict


# dataset = Dataset()
# dataset.prepare_data(TRAIN_MASTER_DATA, "smi")
# x, y = dataset.setup("device", "JSC")
# print("1")
# print(x, y)
# dataset.prepare_data(TRAIN_MASTER_DATA, "bigsmi")
# x, y = dataset.setup("electronic", "JSC")
# print("2")
# print(x, y)
# dataset.prepare_data(TRAIN_MASTER_DATA, "selfies")
# x, y = dataset.setup("none", "VOC")
# print("3")
# print(x, y)
# dataset.prepare_data(TRAIN_MASTER_DATA, "smi")
# x, y, token_dict = dataset.setup_aug_smi("none", "PCE")
# print("4")
# print(x, y)
# dataset.prepare_data(BRICS_MASTER_DATA, "brics")
# x, y = dataset.setup("device", "JSC")
# print("5")
# print(x, y)
# dataset.prepare_data(MANUAL_MASTER_DATA, "manual")
# x, y = dataset.setup("device", "FF")
# print("6")
# print(x, y)
# dataset.prepare_data(FP_MASTER_DATA, "fp")
# x, y = dataset.setup("device", "PCE")
# print("7")
# print(x, y)

