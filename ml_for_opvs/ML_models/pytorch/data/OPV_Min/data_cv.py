from typing import Dict, List, Optional, Union

import os
import numpy as np
import pandas as pd
import ast  # for str -> list conversion
import copy

# for plotting
import matplotlib.pyplot as plt

import pkg_resources
import pytorch_lightning as pl
from opv_ml.ML_models.pytorch.data.OPV_Min.tokenizer import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import selfies as sf

# for transformer
from transformers import AutoTokenizer

# for cross-validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/hw_frag/train_frag_master.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
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

CHEMBERT_TOKENIZER = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/Transformer/tokenizer_chembert/"
)

CHEMBERT = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/Transformer/chembert/"
)

TROUBLESHOOT = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/Transformer/"
)

SEED_VAL = 4

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dataset definition
class OPVDataset(Dataset):
    # load the dataset
    def __init__(self, input_representation, opv_data):
        self.x = input_representation
        self.y = opv_data  # PCE (9-25-2021)

    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_splits(self, seed_val, n_test=0.10, val_test=0.10):
        # train = 80%, val = 10%, test = 10%
        total_size = len(self.x)
        test = round(total_size * n_test)
        val = round(total_size * val_test)
        train = len(self.x) - test - val

        return random_split(
            self, [train, val, test], generator=torch.Generator().manual_seed(seed_val)
        )

    def get_splits_aug(self, aug_x_da, aug_x_ad, pce_array, seed_val):
        """Function that gets split of train/test/val and then adds augmented training set to train"""
        # splits original dataset into train,val,test
        train, val, test = self.get_splits(seed_val=seed_val)
        # adds new augmented training data to ONLY training set
        non_aug_size = len(self.x)  # add to index to avoid original dataset

        # get allowed augmented data from training data indices
        total_aug_x = []
        total_aug_y = []
        for idx in train.indices:
            aug_da_list = aug_x_da[idx]  # list of augmented data for each da pair
            aug_ad_list = aug_x_ad[idx]  # list of augmented data for each ad pair
            i = 0
            j = 0
            for aug_da in aug_da_list:
                if i != 0:
                    total_aug_x.append(aug_da)
                    total_aug_y.append(pce_array[idx])
                i += 1
            for aug_ad in aug_ad_list:
                if j != 0:
                    total_aug_x.append(aug_ad)
                    total_aug_y.append(pce_array[idx])
                j += 1
        # print(len(total_aug_x))  # 10680

        # combine original dataset with augmented dataset
        aug_x = list(self.x)
        aug_x.extend(total_aug_x)
        aug_y = list(self.y)
        aug_y.extend(total_aug_y)
        aug_x = np.array(aug_x)
        aug_y = np.array(aug_y)
        train.dataset = OPVDataset(aug_x, aug_y)  # train.indices is not modified

        # checking augmented dataset
        # print(aug_x[1000:1006], aug_y[1000:1006])

        # add augmented indices
        aug_idx_list = list(train.indices)  # original data indices
        idx = 0
        while idx < len(total_aug_x):
            aug_idx = idx + non_aug_size
            aug_idx_list.append(aug_idx)
            idx += 1
        train.indices = aug_idx_list

        return train, val, test

    def get_splits_cv(self, kth_fold, k_fold=5):
        # k_fold=5 --> train = 80%, test = 20%, but we will take validation set from test set
        # --> train = 80%, val = 10% (50% of 20%), test = 10%
        np.random.seed(SEED_VAL)
        cv_outer = KFold(n_splits=k_fold, shuffle=True, random_state=SEED_VAL)
        fold_count = 0
        val_proportion = 1 / 2
        for train_ix, test_ix in cv_outer.split(self.x):
            if kth_fold == fold_count:
                num_of_val = int(len(test_ix) * val_proportion)
                val_ix = np.random.choice(test_ix, num_of_val, replace=False)
                test_ix = np.setdiff1d(test_ix, val_ix)
                train = Subset(self, train_ix)
                val = Subset(self, val_ix)
                test = Subset(self, test_ix)
            fold_count += 1
        return train, val, test

    def get_splits_aug_cv(self, aug_x_da, aug_x_ad, pce_array, kth_fold):
        """Function that gets split of train/test/val and then adds augmented training set to train"""
        # splits original dataset into train,val,test
        train, val, test = self.get_splits_cv(kth_fold=kth_fold)
        # adds new augmented training data to ONLY training set
        non_aug_size = len(self.x)  # add to index to avoid original dataset

        # get allowed augmented data from training data indices
        total_aug_x = []
        total_aug_y = []
        for idx in train.indices:
            aug_da_list = aug_x_da[idx]  # list of augmented data for each da pair
            aug_ad_list = aug_x_ad[idx]  # list of augmented data for each ad pair
            i = 0
            j = 0
            for aug_da in aug_da_list:
                if i != 0:
                    total_aug_x.append(aug_da)
                    total_aug_y.append(pce_array[idx])
                i += 1
            for aug_ad in aug_ad_list:
                if j != 0:
                    total_aug_x.append(aug_ad)
                    total_aug_y.append(pce_array[idx])
                j += 1
        # print(len(total_aug_x))  # 10680

        # combine original dataset with augmented dataset
        aug_x = list(self.x)
        aug_x.extend(total_aug_x)
        aug_y = list(self.y)
        aug_y.extend(total_aug_y)
        aug_x = np.array(aug_x)
        aug_y = np.array(aug_y)
        train.dataset = OPVDataset(aug_x, aug_y)  # train.indices is not modified

        # checking augmented dataset
        # print(aug_x[1000:1006], aug_y[1000:1006])

        # add augmented indices
        aug_idx_list = list(train.indices)  # original data indices
        idx = 0
        while idx < len(total_aug_x):
            aug_idx = idx + non_aug_size
            aug_idx_list.append(aug_idx)
            idx += 1
        train.indices = aug_idx_list

        return train, val, test


""" input:
    0 - SMILES
    1 - Big_SMILES
    2 - SELFIES
"""


class OPVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        smiles: int,  # True - string representation, False - Fragments
        bigsmiles: int,
        selfies: int,
        aug_smiles: int,  # number of data augmented SMILES
        hw_frag: int,
        aug_hw_frag: int,
        brics: int,
        manual: int,
        aug_manual: int,
        fingerprint: int,
        fp_radius: int,
        fp_nbits: int,
        cv: int,
        pt_model: str,
        pt_tokenizer: str,
        shuffled: bool,
        seed_val: int,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.transform = None
        self.smiles = smiles
        self.bigsmiles = bigsmiles
        self.selfies = selfies
        self.aug_smiles = aug_smiles
        self.hw_frag = hw_frag
        self.aug_hw_frag = aug_hw_frag
        self.brics = brics
        self.manual = manual
        self.aug_manual = aug_manual
        self.fingerprint = fingerprint
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.cv = cv
        self.pt_model = pt_model
        self.pt_tokenizer = pt_tokenizer
        self.shuffled = shuffled
        self.max_length = 1
        self.seed_val = seed_val

    def setup(self) -> None:
        self.data = pd.read_csv(DATA_DIR)
        # concatenate Donor and Acceptor Inputs
        if self.smiles == 1 or self.aug_smiles == 1:
            representation = "SMILES"
            for index, row in self.data.iterrows():
                self.data.at[index, "DA_pair"] = (
                    row["Donor_{}".format(representation)]
                    + "."
                    + row["Acceptor_{}".format(representation)]
                )
        elif self.bigsmiles == 1:
            self.data = pd.read_csv(MANUAL_MASTER_DATA)
            representation = "BigSMILES"
            for index, row in self.data.iterrows():
                self.data.at[index, "DA_pair"] = (
                    row["Donor_{}".format(representation)]
                    + "."
                    + row["Acceptor_{}".format(representation)]
                )
        elif self.selfies == 1:
            representation = "SELFIES"
            for index, row in self.data.iterrows():
                self.data.at[index, "DA_pair"] = (
                    row["Donor_{}".format(representation)]
                    + "."
                    + row["Acceptor_{}".format(representation)]
                )

    def prepare_data(self):
        """
        Setup dataset with fragments that have been tokenized and augmented already in rdkit_frag_in_dataset.py
        """
        # convert other columns into numpy arrays
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")

        # minimize range of pce between 0-1
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        self.pce_array = pce_array

        self.data_size = len(pce_array)

        if self.pt_model != None:
            self.prepare_transformer()
        else:
            if self.smiles == 1 or self.bigsmiles == 1:
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_data(self.data["DA_pair"])
                self.max_seq_length = max_seq_length
                self.vocab_length = vocab_length
                da_pair_list = tokenized_input

            elif self.selfies == 1:
                # tokenize data using selfies
                tokenized_input = []
                selfie_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                    self.data["DA_pair"]
                )
                self.max_seq_length = max_selfie_length
                self.vocab_length = len(selfie_dict)
                print(selfie_dict)
                for index, row in self.data.iterrows():
                    print(self.data.at[index, "DA_pair"])
                    tokenized_selfie = sf.selfies_to_encoding(
                        self.data.at[index, "DA_pair"],
                        selfie_dict,
                        pad_to_len=-1,
                        enc_type="label",
                    )
                    tokenized_input.append(tokenized_selfie)

                tokenized_input = np.asarray(tokenized_input)
                tokenized_input = Tokenizer().pad_input(
                    tokenized_input, max_selfie_length
                )
                da_pair_list = tokenized_input

            # convert str to list for DA_pairs
            elif self.aug_smiles == 1:
                self.data_aug_smi = pd.read_csv(AUGMENT_SMILES_DATA)

                da_aug_list = []
                for i in range(len(self.data_aug_smi["DA_pair_tokenized_aug"])):
                    da_aug_list.append(
                        ast.literal_eval(self.data_aug_smi["DA_pair_tokenized_aug"][i])
                    )

                ad_aug_list = []
                for i in range(len(self.data_aug_smi["AD_pair_tokenized_aug"])):
                    ad_aug_list.append(
                        ast.literal_eval(self.data_aug_smi["AD_pair_tokenized_aug"][i])
                    )
                # original data comes from first augmented d-a / a-d pair from each pair
                da_pair_list = []
                for i in range(len(da_aug_list)):
                    da_pair_list.append(
                        da_aug_list[i][0]
                    )  # PROBLEM: different lengths, therefore cannot np.array nicely

                # extra code for vocab length
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_data(self.data["DA_pair"])
                self.max_seq_length = len(da_aug_list[0][0])
                print("max_length_aug_smi: ", self.max_seq_length)
                self.vocab_length = vocab_length
                print("LEN: ", len(da_pair_list))

            elif self.hw_frag == 1:
                self.data = pd.read_csv(FRAG_MASTER_DATA)
                da_pair_list = []
                for i in range(len(self.data["DA_pair_tokenized"])):
                    da_pair_list.append(
                        ast.literal_eval(self.data["DA_pair_tokenized"][i])
                    )
                self.vocab_length = 239
                # max_seq_length
                self.max_seq_length = len(da_pair_list[0])
                print(self.max_seq_length)

            elif self.aug_hw_frag == 1:
                self.data = pd.read_csv(FRAG_MASTER_DATA)
                da_aug_list = []
                for i in range(len(self.data["DA_pair_tokenized_aug"])):
                    da_aug_list.append(
                        ast.literal_eval(self.data["DA_pair_tokenized_aug"][i])
                    )
                ad_aug_list = []
                for i in range(len(self.data["AD_pair_tokenized_aug"])):
                    ad_aug_list.append(
                        ast.literal_eval(self.data["AD_pair_tokenized_aug"][i])
                    )
                self.vocab_length = 239
                # original data comes from first augmented d-a / a-d pair from each pair
                da_pair_list = []
                for i in range(len(da_aug_list)):
                    da_pair_list.append(da_aug_list[i][0])
                self.max_seq_length = len(da_pair_list[0])

            elif self.brics == 1:
                self.data = pd.read_csv(BRICS_MASTER_DATA)
                da_pair_list = []
                print("BRICS: ", len(self.data["DA_tokenized_BRICS"]))
                for i in range(len(self.data["DA_tokenized_BRICS"])):
                    da_pair_list.append(
                        ast.literal_eval(self.data["DA_tokenized_BRICS"][i])
                    )
                self.vocab_length = 191
                self.max_seq_length = len(da_pair_list[0])

            elif self.manual == 1:
                self.data = pd.read_csv(MANUAL_MASTER_DATA)
                da_pair_list = []
                print("MANUAL: ", len(self.data["DA_manual_tokenized"]))
                for i in range(len(self.data["DA_manual_tokenized"])):
                    da_pair_list.append(
                        ast.literal_eval(self.data["DA_manual_tokenized"][i])
                    )
                self.vocab_length = 337
                self.max_seq_length = len(da_pair_list[0])

            elif self.aug_manual == 1:
                self.data = pd.read_csv(MANUAL_MASTER_DATA)
                da_aug_list = []
                for i in range(len(self.data["DA_manual_tokenized_aug"])):
                    da_aug_list.append(
                        ast.literal_eval(self.data["DA_manual_tokenized_aug"][i])
                    )
                ad_aug_list = []
                for i in range(len(self.data["AD_manual_tokenized_aug"])):
                    ad_aug_list.append(
                        ast.literal_eval(self.data["AD_manual_tokenized_aug"][i])
                    )
                self.vocab_length = 337
                # original data comes from first augmented d-a / a-d pair from each pair
                da_pair_list = []
                for i in range(len(da_aug_list)):
                    da_pair_list.append(da_aug_list[i][0])
                self.max_seq_length = len(da_pair_list[0])

            elif self.fingerprint == 1:
                self.data = pd.read_csv(FP_MASTER_DATA)
                da_pair_list = []
                column_da_pair = (
                    "DA_FP"
                    + "_radius_"
                    + str(self.fp_radius)
                    + "_nbits_"
                    + str(self.fp_nbits)
                )
                print("Fingerprint: ", len(self.data[column_da_pair]))
                for i in range(len(self.data[column_da_pair])):
                    da_pair_list.append(ast.literal_eval(self.data[column_da_pair][i]))
                self.vocab_length = self.fp_nbits
                self.max_seq_length = len(da_pair_list[0])

                # Double-check the amount of augmented training data
                # total_aug_data = 0
                # for aug_list in da_aug_list:
                #     for aug in aug_list:
                #         total_aug_data += 1

                # for aug_list in ad_aug_list:
                #     for aug in aug_list:
                #         total_aug_data += 1

                # print("TOTAL NUM: ", total_aug_data)

                # for creating OPVDataset, must use first element of each augment array
                # replace da_pair_array
                # because we don't want to augment test set nor include any augmented test set in training set,
                # but also have the original dataset have the correct order (for polymers)
                # expected number of total training set: 2055 = (444*0.75) + (333*(number_of_augmented_frags)=1722)
                # expected number can change due to different d-a pairs having different number of augmentation frags
            da_pair_array = np.array(da_pair_list)
            pce_dataset = OPVDataset(da_pair_array, pce_array)
            if self.aug_hw_frag == 1 or self.aug_manual == 1 or self.aug_smiles == 1:
                if self.cv != None:
                    (
                        self.pce_train,
                        self.pce_val,
                        self.pce_test,
                    ) = pce_dataset.get_splits_aug_cv(
                        da_aug_list, ad_aug_list, pce_array, kth_fold=self.cv
                    )
                else:
                    (
                        self.pce_train,
                        self.pce_val,
                        self.pce_test,
                    ) = pce_dataset.get_splits_aug(
                        da_aug_list, ad_aug_list, pce_array, seed_val=self.seed_val
                    )
            else:
                if self.cv != None:
                    (
                        self.pce_train,
                        self.pce_val,
                        self.pce_test,
                    ) = pce_dataset.get_splits_cv(kth_fold=self.cv)
                else:
                    (
                        self.pce_train,
                        self.pce_val,
                        self.pce_test,
                    ) = pce_dataset.get_splits(seed_val=self.seed_val)
            print("LEN: ", len(da_pair_list))
        print("test_idx: ", self.pce_test.indices)

    def prepare_transformer(self):
        """Function that cleans raw data for prep by transformers"""
        # tokenize data with transformer
        tokenizer = AutoTokenizer.from_pretrained(self.pt_tokenizer)
        tokenizer.padding_side = "right"
        self.data = self.data.drop_duplicates()
        self.data = self.data.reset_index(drop=True)
        # convert other columns into numpy arrays
        if self.shuffled:
            pce_array = self.data["PCE(%)_shuffled"].to_numpy().astype("float32")
        else:
            pce_array = self.data["PCE(%)"].to_numpy().astype("float32")
        # minimize range of pce between 0-1
        # find max of pce_array
        self.max_pce = pce_array.max()
        pce_array = pce_array / self.max_pce

        # tokenize data
        tokenized_input = tokenizer(
            list(self.data["DA_pair"]), padding="longest", return_tensors="pt"
        )
        input_ids = tokenized_input["input_ids"]
        input_masks = tokenized_input["attention_mask"]
        max_length = len(input_ids[0])
        self.max_length = max_length
        if self.smiles and self.input == 0:
            pce_dataset = OPVDataset(input_ids, pce_array)
            self.pce_train, self.pce_val, self.pce_test = pce_dataset.get_splits(
                seed_val=self.seed_val
            )
            # train_df = pd.DataFrame(self.pce_train.dataset.x.numpy())
            # val_df = pd.DataFrame(self.pce_val.dataset.x.numpy())
            # test_df = pd.DataFrame(self.pce_test.dataset.x.numpy())
            # train_df.to_csv(TROUBLESHOOT + "train_data_x.csv", index=False)
            # val_df.to_csv(TROUBLESHOOT + "val_data_x.csv", index=False)
            # test_df.to_csv(TROUBLESHOOT + "test_data_x.csv", index=False)
        elif self.input == 2:
            pce_dataset = OPVDataset(input_ids, pce_array)
            self.pce_train, self.pce_val, self.pce_test = pce_dataset.get_splits(
                seed_val=self.seed_val
            )
        elif self.aug_smiles:
            data = pd.read_csv(AUGMENT_SMILES_DATA)
            pce_array = data["PCE(%)"].to_numpy().astype("float32")

            # minimize range of pce between 0-1
            # find max of pce_array
            self.max_pce = pce_array.max()
            pce_array = pce_array / self.max_pce

            self.data_size = len(pce_array)
            # print("num_of_pairs: ", self.data_size)

            # tokenize augmented SMILES
            # print(len(ast.literal_eval(data["DA_pair_aug"][0])))
            da_aug_list = []
            for i in range(len(data["DA_pair_aug"])):
                da_aug_list.extend(ast.literal_eval(data["DA_pair_aug"][i]))

            ad_aug_list = []
            for i in range(len(data["AD_pair_aug"])):
                ad_aug_list.extend(ast.literal_eval(data["AD_pair_aug"][i]))

            # tokenize together
            tokenized_input_da = tokenizer(
                list(da_aug_list), padding="longest", return_tensors="pt"
            )

            tokenized_input_ad = tokenizer(
                list(ad_aug_list), padding="longest", return_tensors="pt"
            )

            # put tokenized data back into its corresponding indexed list ([[16 SMILES], [16], ...])
            da_aug_tokenized = []
            aug_cap = 0
            inner_list = []
            for input in tokenized_input_da["input_ids"]:
                inner_list.append(input.tolist())
                aug_cap += 1
                if aug_cap == 16:
                    da_aug_tokenized.append(inner_list)
                    inner_list = []
                    aug_cap = 0

            ad_aug_tokenized = []
            aug_cap = 0
            inner_list = []
            for input in tokenized_input_ad["input_ids"]:
                inner_list.append(input.tolist())
                aug_cap += 1
                if aug_cap == 16:
                    ad_aug_tokenized.append(inner_list)
                    inner_list = []
                    aug_cap = 0

            # original data comes from first augmented d-a / a-d pair from each pair
            da_pair_list = []
            for i in range(len(da_aug_tokenized)):
                da_pair_list.append(
                    da_aug_tokenized[i][0]
                )  # PROBLEM: different lengths, therefore cannot np.array nicely
            da_pair_array = np.array(da_pair_list)
            pce_dataset = OPVDataset(da_pair_array, pce_array)
            (self.pce_train, self.pce_val, self.pce_test,) = pce_dataset.get_splits_aug(
                da_aug_tokenized, ad_aug_tokenized, pce_array, seed_val=self.seed_val
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.pce_train,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.pce_val,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.pce_test,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )


def distribution_plot(data_dir):
    df = pd.read_csv(data_dir)
    pce_array = df["PCE(%)"].to_numpy().astype("float32")
    # minimize range of pce between 0-1
    # find max of pce_array
    max_pce = pce_array.max()
    pce_array = pce_array / max_pce

    fig, ax = plt.subplots()
    ax.hist(pce_array, bins=20, rwidth=0.9, color="#607c8e")
    ax.set_title("Experimental_PCE_(%) Distribution")
    ax.set_xlabel("Experimental_PCE_(%)")
    ax.set_ylabel("Frequency")
    plt.show()

    # distribution_plot(DATA_DIR)
    # distribution_plotly(PREDICTION_DIR)


# for transformer
# chembert_model = CHEMBERT
# chembert_tokenizer = CHEMBERT_TOKENIZER

# unique_datatype = {
#     "smiles": 0,
#     "bigsmiles": 0,
#     "selfies": 1,
#     "aug_smiles": 0,
#     "hw_frag": 0,
#     "aug_hw_frag": 0,
#     "brics": 0,
#     "manual": 0,
#     "aug_manual": 0,
#     "fingerprint": 0,
# }

# shuffled = False

# data_module = OPVDataModule(
#     train_batch_size=128,
#     val_batch_size=32,
#     test_batch_size=32,
#     num_workers=4,
#     smiles=unique_datatype["smiles"],
#     bigsmiles=unique_datatype["bigsmiles"],
#     selfies=unique_datatype["selfies"],
#     aug_smiles=unique_datatype["aug_smiles"],
#     hw_frag=unique_datatype["hw_frag"],
#     aug_hw_frag=unique_datatype["aug_hw_frag"],
#     brics=unique_datatype["brics"],
#     manual=unique_datatype["manual"],
#     aug_manual=unique_datatype["aug_manual"],
#     fingerprint=unique_datatype["fingerprint"],
#     fp_radius=3,
#     fp_nbits=512,
#     cv=0,
#     pt_model=None,
#     pt_tokenizer=None,
#     shuffled=shuffled,
#     seed_val=SEED_VAL,
# )
# data_module.setup()
# data_module.prepare_data()
# print("TRAINING SIZE: ", len(data_module.pce_train.dataset))
# test_idx = list(data_module.pce_test.indices)
# print(test_idx)
# print(data_module.pce_array[test_idx])

# distribution_plot(DATA_DIR)

# print(Chem.Descriptors.ExactMolWt("CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(/C=C4\C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)sc3-c3sc4c(c(CCCCCC)cc5c4cc(CCCCCC)c4c6c(sc45)-c4sc(/C=C5\C(=O)c7cc(F)c(F)cc7C5=C(C#N)C#N)cc4C6(c4ccc(CCCCCC)cc4)c4ccc(CCCCCC)cc4)c32)cc1"))
