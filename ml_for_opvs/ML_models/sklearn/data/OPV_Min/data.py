# data.py for classical ML
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf

import torch
from torch.utils.data import random_split

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

from opv_ml.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer


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

    def setup(self):
        """
        NOTE: for SMILES
        Function that sets up data ready for training 
        """
        if self.input == 0:
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == 1:
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == 2:
            # tokenize data using selfies
            tokenized_input = []
            selfie_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                self.data["DA_pair"]
            )
            print(selfie_dict)
            for index, row in self.data.iterrows():
                tokenized_selfie = sf.selfies_to_encoding(
                    self.data.at[index, "DA_pair"],
                    selfie_dict,
                    pad_to_len=-1,
                    enc_type="label",
                )
                tokenized_input.append(tokenized_selfie)

            # tokenized_input = np.asarray(tokenized_input)
            tokenized_input = Tokenizer().pad_input(tokenized_input, max_selfie_length)
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        # minimize range of pce between 0-1
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        # split data into cv
        return np.asarray(tokenized_input), pce_array

    def setup_aug_smi(self):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        """
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        # minimize range of pce between 0-1
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce
        return np.asarray(self.data["DA_pair"]), pce_array

    def setup_frag(self):
        """
        Function that sets up data ready for training 
        (Tokenization, Normalizing, Splitting Data)
        """
        # create new training and testing dataframe
        self.train_df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        self.val_df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        self.test_df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])

        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")
        # minimize range of pce between 0-1 (applies to NN but not for ML)
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        # split data
        # train = 80%, test = 10%, val = 10%
        total_size = len(pce_array)
        test = round(total_size * 0.10)
        val = round(total_size * 0.10)
        train = total_size - test - val

        pce_train, pce_val, pce_test = random_split(
            self.data, [train, val, test], generator=torch.Generator().manual_seed(1)
        )
        row_index = 0
        for i in pce_train.indices:
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized"][i])
            self.train_df.at[row_index, "tokenized_input"] = da_pair_list
            self.train_df.at[row_index, "PCE"] = pce_array[i]
            row_index += 1

        row_index = 0
        for i in pce_val.indices:
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized"][i])
            self.val_df.at[row_index, "tokenized_input"] = da_pair_list
            self.val_df.at[row_index, "PCE"] = pce_array[i]
            row_index += 1

        row_index = 0
        for i in pce_test.indices:
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized"][i])
            self.test_df.at[row_index, "tokenized_input"] = da_pair_list
            self.test_df.at[row_index, "PCE"] = pce_array[i]
            row_index += 1

    def setup_frag_aug(self):
        # create new training and testing dataframe
        self.train_df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        self.val_df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        self.test_df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])

        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")
        # minimize range of pce between 0-1 (applies to NN but not for ML)
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        # split data
        # train = 80%, test = 10%, val = 10%
        total_size = len(pce_array)
        test = round(total_size * 0.10)
        val = round(total_size * 0.10)
        train = total_size - test - val

        pce_train, pce_val, pce_test = random_split(
            self.data, [train, val, test], generator=torch.Generator().manual_seed(1)
        )
        row_index = 0
        for i in pce_train.indices:
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized"][i])
            self.train_df.at[row_index, "tokenized_input"] = da_pair_list
            self.train_df.at[row_index, "PCE"] = pce_array[i]
            # add augmented data (ad_pair and duplicate PCEs)
            ad_pair_list = json.loads(self.data["AD_pair_tokenized"][i])
            self.train_df.at[row_index + 1, "tokenized_input"] = ad_pair_list
            self.train_df.at[row_index + 1, "PCE"] = pce_array[i]
            row_index += 2
        row_index = 0
        for i in pce_val.indices:
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized"][i])
            self.val_df.at[row_index, "tokenized_input"] = da_pair_list
            self.val_df.at[row_index, "PCE"] = pce_array[i]
            row_index += 1
        row_index = 0
        for i in pce_test.indices:
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized"][i])
            self.test_df.at[row_index, "tokenized_input"] = da_pair_list
            self.test_df.at[row_index, "PCE"] = pce_array[i]
            row_index += 1

    def setup_cv(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []
        for i in range(len(self.data["DA_pair_tokenized_aug"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized_aug"][i])
            # only add the first da_pair
            x.append(da_pair_list[0])
            y.append(pce_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_cv_aug(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []
        for i in range(len(self.data["DA_pair_tokenized_aug"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_pair_tokenized_aug"][i])
            for da_pair in da_pair_list:
                x.append(da_pair)
                y.append(pce_array[i])
            ad_pair_list = json.loads(self.data["AD_pair_tokenized_aug"][i])
            for ad_pair in ad_pair_list:
                x.append(ad_pair)
                y.append(pce_array[i])

        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_frag_BRICS(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []
        for i in range(len(self.data["DA_tokenized_BRICS"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_tokenized_BRICS"][i])
            x.append(da_pair_list)
            y.append(pce_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_manual_frag(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        x = []
        y = []
        for i in range(len(self.data["DA_manual_tokenized"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["DA_manual_tokenized"][i])
            x.append(da_pair_list)
            y.append(pce_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_fp(self, radius: int, nbits: int):
        self.df = pd.DataFrame(columns=["tokenized_input", "PCE"], index=[0])
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

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
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y


# dataset = Dataset(MANUAL_MASTER_DATA, 1, False)
# dataset.prepare_data()
# x, y = dataset.setup()
# x, y = dataset.setup_cv()
# x, y = dataset.setup_aug_smi(AUG_SMI_MASTER_DATA)
# x, y = dataset.setup_fp(2, 512)
# print(x[1], y[1])

