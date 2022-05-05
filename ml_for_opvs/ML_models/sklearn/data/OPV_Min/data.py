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

    def __init__(self, data_dir, input: int, shuffled: bool):
        self.data = pd.read_csv(data_dir)
        self.input = input
        self.shuffled = shuffled

    def prepare_data(self):
        """
        Function that concatenates donor-acceptor pair
        """
        self.data["DA_pair"] = " "
        # concatenate Donor and Acceptor Inputs
        if self.input == 0:
            representation = "SMILES"
        elif self.input == 1:
            representation = "BigSMILES"
        elif self.input == 2:
            representation = "SELFIES"

        for index, row in self.data.iterrows():
            self.data.at[index, "DA_pair"] = (
                row["Donor_{}".format(representation)]
                + "."
                + row["Acceptor_{}".format(representation)]
            )

    def setup(self, parameter):
        """
        NOTE: for SMILES
        Function that sets up data ready for training 

        Args:
        parameter: type of parameters to include:
            - electronic: HOMO, LUMO
            - device: all device parameters
            - impt_device: all the important device parameters (D:A ratio - Annealing Temp.)
        """
        if self.input == 0:
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == 1:
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == 2:
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
        if self.shuffled:
            pce_array = self.data["PCE (%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE (%)"].to_numpy().astype("float32")

        # minimize range of pce between 0-1
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        # add device parameters to the end of input
        index = 0
        while index < len(tokenized_input):
            if parameter == "electronic":
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

            elif parameter == "impt_device":
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
                return np.asarray(tokenized_input), np.asarray(pce_array)
            index += 1

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

        # filter out "nan" values
        nan_array = np.isnan(tokenized_input)
        filtered_tokenized_input = []
        filtered_pce_array = []
        nan_idx = 0
        while nan_idx < len(nan_array):
            if True not in nan_array[nan_idx]:
                filtered_tokenized_input.append(tokenized_input[nan_idx])
                filtered_pce_array.append(pce_array[nan_idx])
            nan_idx += 1

        # split data into cv
        print(token_dict)
        # print(filtered_tokenized_input)
        print(len(filtered_tokenized_input), len(filtered_pce_array))
        return np.asarray(filtered_tokenized_input), np.asarray(filtered_pce_array)

    def setup_aug_smi(self, parameter):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        """
        if self.shuffled:
            pce_array = self.data["PCE (%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE (%)"].to_numpy().astype("float32")

        # minimize range of pce between 0-1
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        # convert Series to list
        x = self.data["DA_pair"].to_list()
        # convert list to list of lists
        idx = 0
        for _x in x:
            x[idx] = [_x]
            idx += 1
        # FOR AUGMENTATION: device index (don't augment the device stuff)
        index = 0
        while index < len(x):
            if parameter == "electronic":
                token_dict = []
                homo_d = self.data["HOMO_D (eV)"].to_numpy().astype("float32")
                lumo_d = self.data["LUMO_D (eV)"].to_numpy().astype("float32")
                homo_a = self.data["HOMO_A (eV)"].to_numpy().astype("float32")
                lumo_a = self.data["LUMO_A (eV)"].to_numpy().astype("float32")
                x[index].append(homo_d[index])
                x[index].append(lumo_d[index])
                x[index].append(homo_a[index])
                x[index].append(lumo_a[index])
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
                token_dict = []
                solvent = self.data["solvent"]
                for input in solvent:
                    # unique solvents
                    if input not in token_dict:
                        token_dict.append(input)
                solvent_add = self.data["solvent additive"]
                for input in solvent_add:
                    # unique solvent additives
                    if input not in token_dict:
                        token_dict.append(input)
                hole_contact_layer = self.data["hole contact layer"]
                for input in hole_contact_layer:
                    # unique hole contact layer
                    if input not in token_dict:
                        token_dict.append(input)
                electron_contact_layer = self.data["electron contact layer"]
                for input in electron_contact_layer:
                    # unique electron contact layer
                    if input not in token_dict:
                        token_dict.append(input)

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
                x[index].append(hole_mobility_blend[index])
                x[index].append(electron_mobility_blend[index])
                x[index].append(hole_contact_layer[index])
                x[index].append(electron_contact_layer[index])
            elif parameter == "impt_device":
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
                token_dict = []
                solvent = self.data["solvent"]
                for input in solvent:
                    # unique solvents
                    if input not in token_dict:
                        token_dict.append(input)
                solvent_add = self.data["solvent additive"]
                for input in solvent_add:
                    # unique solvent additives
                    if input not in token_dict:
                        token_dict.append(input)

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
            else:
                return (
                    np.asarray(self.data["DA_pair"]),
                    pce_array,
                    token_dict,
                )
            index += 1

        # filter out "nan" values
        filtered_x = []
        filtered_y = []
        nan_idx = 0
        while nan_idx < len(x):
            nan_bool = False
            for _x in x[nan_idx]:
                if str(_x) == "nan":
                    nan_bool = True
            if not nan_bool:
                filtered_x.append(x[nan_idx])
                filtered_y.append(pce_array[nan_idx])
            nan_idx += 1
        return (
            np.asarray(filtered_x, dtype="object"),
            np.asarray(filtered_y, dtype="float32"),
            token_dict,
        )

    def setup_frag_BRICS(self, parameter):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE (%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE (%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []
        for i in range(len(self.data["DA_tokenized_BRICS"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_tokenized_BRICS"][i])
            x.append(da_pair_list)
            y.append(pce_array[i])

        # add device parameters to the end of input
        b_frag = BRIC_FRAGS(TRAIN_MASTER_DATA)
        token_dict = b_frag.bric_frag()

        index = 0
        while index < len(x):
            if parameter == "electronic":
                homo_d = self.data["HOMO_D (eV)"].to_numpy().astype("float32")
                lumo_d = self.data["LUMO_D (eV)"].to_numpy().astype("float32")
                homo_a = self.data["HOMO_A (eV)"].to_numpy().astype("float32")
                lumo_a = self.data["LUMO_A (eV)"].to_numpy().astype("float32")
                x[index].append(homo_d[index])
                x[index].append(lumo_d[index])
                x[index].append(homo_a[index])
                x[index].append(lumo_a[index])
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

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
                x[index].append(hole_mobility_blend[index])
                x[index].append(electron_mobility_blend[index])
                x[index].append(hole_contact_layer[index])
                x[index].append(electron_contact_layer[index])

            elif parameter == "impt_device":
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

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
            else:
                return np.asarray(x), np.asarray(y)
            index += 1

        # tokenize data
        data_pt_idx = 0
        while data_pt_idx < len(x):
            token_idx = 0
            while token_idx < len(x[data_pt_idx]):
                token = x[data_pt_idx][token_idx]
                if token == "nan":
                    x[data_pt_idx][token_idx] = nan
                elif isinstance(token, str):
                    x[data_pt_idx][token_idx] = token_dict[token]
                token_idx += 1
            data_pt_idx += 1

        # filter out "nan" values
        nan_array = np.isnan(x)
        filtered_x = []
        filtered_y = []
        nan_idx = 0
        while nan_idx < len(nan_array):
            if True not in nan_array[nan_idx]:
                filtered_x.append(x[nan_idx])
                filtered_y.append(y[nan_idx])
            nan_idx += 1

        # split data into cv
        print(token_dict)
        # print(filtered_x)
        print(len(filtered_x), len(filtered_y))

        filtered_x = np.asarray(filtered_x)
        filtered_y = np.asarray(filtered_y)
        return filtered_x, filtered_y

    def setup_manual_frag(self, parameter):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE (%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE (%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []
        for i in range(len(self.data["DA_manual_tokenized"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_manual_tokenized"][i])
            x.append(da_pair_list)
            y.append(pce_array[i])

        # add device parameters to the end of input
        manual = manual_frag(TRAIN_MASTER_DATA, MANUAL_DONOR_CSV, MANUAL_ACCEPTOR_CSV)
        token_dict = manual.return_frag_dict()

        index = 0
        while index < len(x):
            if parameter == "electronic":
                homo_d = self.data["HOMO_D (eV)"].to_numpy().astype("float32")
                lumo_d = self.data["LUMO_D (eV)"].to_numpy().astype("float32")
                homo_a = self.data["HOMO_A (eV)"].to_numpy().astype("float32")
                lumo_a = self.data["LUMO_A (eV)"].to_numpy().astype("float32")
                x[index].append(homo_d[index])
                x[index].append(lumo_d[index])
                x[index].append(homo_a[index])
                x[index].append(lumo_a[index])
                device_idx = 4
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

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
                x[index].append(hole_mobility_blend[index])
                x[index].append(electron_mobility_blend[index])
                x[index].append(hole_contact_layer[index])
                x[index].append(electron_contact_layer[index])
                device_idx = 11
            elif parameter == "impt_device":
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

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
                device_idx = 7
            else:
                device_idx = 0
                return np.asarray(x), np.asarray(y), device_idx
            index += 1

        # tokenize data
        data_pt_idx = 0
        while data_pt_idx < len(x):
            token_idx = 0
            while token_idx < len(x[data_pt_idx]):
                token = x[data_pt_idx][token_idx]
                if token == "nan":
                    x[data_pt_idx][token_idx] = nan
                elif isinstance(token, str):
                    x[data_pt_idx][token_idx] = token_dict[token]
                token_idx += 1
            data_pt_idx += 1

        # filter out "nan" values
        nan_array = np.isnan(x)
        filtered_x = []
        filtered_y = []
        nan_idx = 0
        while nan_idx < len(nan_array):
            if True not in nan_array[nan_idx]:
                filtered_x.append(x[nan_idx])
                filtered_y.append(y[nan_idx])
            nan_idx += 1

        # split data into cv
        print(token_dict)
        # print(filtered_x)
        print(len(filtered_x), len(filtered_y))

        filtered_x = np.asarray(filtered_x)
        filtered_y = np.asarray(filtered_y)
        return filtered_x, filtered_y, device_idx

    def setup_fp(self, radius: int, nbits: int, parameter: str):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE (%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE (%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []

        column_da_pair = "DA_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        for i in range(len(self.data[column_da_pair])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data[column_da_pair][i])
            x.append(da_pair_list)
            y.append(pce_array[i])

        # add device parameters to the end of input
        token_dict = {0: 0, 1: 1}
        index = 0
        while index < len(x):
            if parameter == "electronic":
                homo_d = self.data["HOMO_D (eV)"].to_numpy().astype("float32")
                lumo_d = self.data["LUMO_D (eV)"].to_numpy().astype("float32")
                homo_a = self.data["HOMO_A (eV)"].to_numpy().astype("float32")
                lumo_a = self.data["LUMO_A (eV)"].to_numpy().astype("float32")
                x[index].append(homo_d[index])
                x[index].append(lumo_d[index])
                x[index].append(homo_a[index])
                x[index].append(lumo_a[index])
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

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
                x[index].append(hole_mobility_blend[index])
                x[index].append(electron_mobility_blend[index])
                x[index].append(hole_contact_layer[index])
                x[index].append(electron_contact_layer[index])

            elif parameter == "impt_device":
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

                x[index].append(d_a_ratio[index])
                x[index].append(total_solids_conc[index])
                x[index].append(solvent[index])
                x[index].append(solvent_add[index])
                x[index].append(solvent_add_conc[index])
                x[index].append(active_layer_thickness[index])
                x[index].append(annealing_temp[index])
            else:
                return np.asarray(x), np.asarray(y)
            index += 1

        # tokenize data
        data_pt_idx = 0
        while data_pt_idx < len(x):
            token_idx = 0
            while token_idx < len(x[data_pt_idx]):
                token = x[data_pt_idx][token_idx]
                if token == "nan":
                    x[data_pt_idx][token_idx] = nan
                elif isinstance(token, str):
                    x[data_pt_idx][token_idx] = token_dict[token]
                token_idx += 1
            data_pt_idx += 1

        # filter out "nan" values
        nan_array = np.isnan(x)
        filtered_x = []
        filtered_y = []
        nan_idx = 0
        while nan_idx < len(nan_array):
            if True not in nan_array[nan_idx]:
                filtered_x.append(x[nan_idx])
                filtered_y.append(y[nan_idx])
            nan_idx += 1

        # split data into cv
        print(token_dict)
        # print(filtered_x)
        print(len(filtered_x), len(filtered_y))

        filtered_x = np.asarray(filtered_x)
        filtered_y = np.asarray(filtered_y)
        return filtered_x, filtered_y


# dataset = Dataset(TRAIN_MASTER_DATA, 0, False)
# dataset = Dataset(BRICS_MASTER_DATA, 0, False)
# dataset = Dataset(MANUAL_MASTER_DATA, 0, False)
# dataset = Dataset(FP_MASTER_DATA, 0, False)
# dataset.prepare_data()
# x, y = dataset.setup("impt_device")
# scaler = preprocessing.MinMaxScaler().fit(x)
# x_scaled = scaler.transform(x)
# print(x_scaled[0])
# x, y, token_dict = dataset.setup_aug_smi("device")
# x, y = dataset.setup_frag_BRICS("device")
# x, y = dataset.setup_manual_frag("device")
# x, y = dataset.setup_fp(3, 512, "device")

