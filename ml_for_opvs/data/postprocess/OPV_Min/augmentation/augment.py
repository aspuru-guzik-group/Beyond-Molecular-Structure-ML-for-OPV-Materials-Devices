import ast  # for str -> list conversion
import numpy as np
import pandas as pd
import pkg_resources
import random
from rdkit import Chem

from opv_ml.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer

MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/augmentation/train_aug_master5.csv"
)


class Augment:
    """
    Class that contains functions to augment SMILES donor-acceptor pairs
    """

    def __init__(self, data):
        """
        Instantiate class with appropriate data.

        Args:
            data: path to master donor-acceptor pair data
        """
        self.data = pd.read_csv(data)

    def pad_input(self, tokenized_array, seq_len):
        """
        Function that pads the reactions with 0 (_PAD) to a fixed length
        PRE-PADDING (features[ii, -len(review) :] = np.array(review)[:seq_len])
        POST-PADDING (features[ii, : len(review)] = np.array(review)[:seq_len])

        Args:
            tokenized_array: tokenized SMILES array to be padded
            seq_len: maximum sequence length for the entire dataset
        
        Returns:
            pre-padded tokenized SMILES
        """
        features = np.zeros(seq_len, dtype=int)
        for i in tokenized_array:
            if len(tokenized_array) != 0:
                features[-len(tokenized_array) :] = np.array(tokenized_array)[:seq_len]
        return features.tolist()

    def aug_smi_doRandom(self, augment_smiles_data, num_of_augment):
        """
        Function that creates augmented DA and AD pairs with X number of augmented SMILES
        Uses doRandom=True for augmentation

        Args:
            augment_smiles_data: SMILES data to be augmented
            num_of_augment: number of augmentations to perform per SMILES

        Returns:
            New .csv with DA_pair_aug, AD_pair_aug, DA_pair_tokenized_list, AD_pair_tokenized_list, and PCE
        """
        # keeps randomness the same
        random.seed(1)
        column_names = [
            "Donor",
            "Donor_SMILES",
            "Acceptor",
            "Acceptor_SMILES",
            "DA_pair_aug",
            "AD_pair_aug",
            "PCE(%)",
        ]
        train_aug_df = pd.DataFrame(columns=column_names)
        train_aug_df["Donor"] = self.data["Donor"]
        train_aug_df["Donor_SMILES"] = self.data["Donor_SMILES"]
        train_aug_df["Acceptor"] = self.data["Acceptor"]
        train_aug_df["Acceptor_SMILES"] = self.data["Acceptor_SMILES"]
        train_aug_df["PCE(%)"] = self.data["PCE(%)"]

        for i in range(len(train_aug_df["Donor"])):
            augmented_da_list = []
            augmented_ad_list = []
            donor_smi = train_aug_df.at[i, "Donor_SMILES"]
            acceptor_smi = train_aug_df.at[i, "Acceptor_SMILES"]

            # keep track of unique donors and acceptors
            unique_donor = [donor_smi]
            unique_acceptor = [acceptor_smi]

            # add original donor-acceptor / acceptor-donor pair
            augmented_da_list.append(donor_smi + "." + acceptor_smi)
            augmented_ad_list.append(acceptor_smi + "." + donor_smi)

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
                    augmented_da_list.append(donor_aug_smi + "." + acceptor_aug_smi)
                    augmented_ad_list.append(acceptor_aug_smi + "." + donor_aug_smi)
                    augmented += 1

            train_aug_df.at[i, "DA_pair_aug"] = augmented_da_list
            train_aug_df.at[i, "AD_pair_aug"] = augmented_ad_list

        train_aug_df.to_csv(augment_smiles_data)

    def aug_smi_tokenize(self, train_aug_data):
        """
        Returns new columns with tokenized SMILES data

        Args:
            train_aug_data: path to augmented data to be tokenized

        Returns:
            new columns to train_aug_master.csv: DA_pair_tokenized_aug, AD_pair_tokenized_aug 
        """
        aug_smi_data = pd.read_csv(train_aug_data)
        # initialize new columns
        aug_smi_data["DA_pair_tokenized_aug"] = " "
        aug_smi_data["AD_pair_tokenized_aug"] = " "
        da_aug_list = []
        for i in range(len(aug_smi_data["DA_pair_aug"])):
            da_aug_list.append(ast.literal_eval(aug_smi_data["DA_pair_aug"][i]))

        ad_aug_list = []
        for i in range(len(aug_smi_data["AD_pair_aug"])):
            ad_aug_list.append(ast.literal_eval(aug_smi_data["AD_pair_aug"][i]))

        # build token2idx dictionary
        # flatten lists
        flat_da_aug_list = [item for sublist in da_aug_list for item in sublist]
        token2idx = Tokenizer().build_token2idx(flat_da_aug_list)
        print(token2idx)

        # get max length of any tokenized pair
        max_length = 0

        for i in range(len(da_aug_list)):
            tokenized_list = []
            da_list = da_aug_list[i]
            for da in da_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 0 for token in da
                ]
                tokenized_list.append(tokenized_smi)
                if len(tokenized_smi) > max_length:
                    max_length = len(tokenized_smi)

        # tokenize augmented data and return new column in .csv
        # TODO: add padding in a systematic way (only at beginning)
        for i in range(len(da_aug_list)):
            tokenized_list = []
            da_list = da_aug_list[i]
            for da in da_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 1 for token in da
                ]
                tokenized_smi = self.pad_input(tokenized_smi, max_length)
                tokenized_list.append(tokenized_smi)
            aug_smi_data.at[i, "DA_pair_tokenized_aug"] = tokenized_list

        for i in range(len(ad_aug_list)):
            tokenized_list = []
            ad_list = ad_aug_list[i]
            for ad in ad_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 1 for token in ad
                ]
                tokenized_smi = self.pad_input(tokenized_smi, max_length)
                tokenized_list.append(tokenized_smi)
            aug_smi_data.at[i, "AD_pair_tokenized_aug"] = tokenized_list

        aug_smi_data.to_csv(train_aug_data, index=False)


# augmenter = Augment(MASTER_DATA)
# augmenter.aug_smi_doRandom(AUGMENT_SMILES_DATA, 4)
# augmenter.aug_smi_tokenize(AUGMENT_SMILES_DATA)

# from rdkit.Chem import Descriptors

# print(
#     Descriptors.ExactMolWt(
#         Chem.MolFromSmiles(
#             "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(/C=C4\C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)sc3-c3sc4c(c(CCCCCC)cc5c4cc(CCCCCC)c4c6c(sc45)-c4sc(/C=C5\C(=O)c7cc(F)c(F)cc7C5=C(C#N)C#N)cc4C6(c4ccc(CCCCCC)cc4)c4ccc(CCCCCC)cc4)c32)cc1"
#         )
#     )
# )

thiophene = Chem.MolFromSmiles("c1cccs1")
for i in range(5):
    print(Chem.MolToSmiles(thiophene, doRandom=True))
