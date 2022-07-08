import ast  # for str -> list conversion
import numpy as np
import pandas as pd
import pkg_resources
import random
from rdkit import Chem

from ml_for_opvs.ML_models.sklearn.tokenizer import Tokenizer

MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/aug_SMILES/train_aug_master.csv")

# AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
#     "ml_for_opvs", "data/input_representation/OPV_Min/aug_SMILES/train_aug_master.parquet")


def pad_input(tokenized_array, seq_len):
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

def aug_smi_doRandom(smiles_data, augment_smiles_data, num_of_augment):
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
    train_aug_df = pd.read_csv(smiles_data)
    train_aug_df["DA_pair_aug"] = ""
    total_aug = 0
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
        while augmented < (num_of_augment-1): # augment X - 1 times because the original is kept. Therefore X times the amount of data is created.
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
                total_aug += 1

        train_aug_df.at[i, "DA_pair_aug"] = augmented_da_list
    
    print("TOTAL AUGMENTED: ", total_aug)

    # train_aug_df.to_parquet(augment_smiles_data, index=False, engine="fastparquet")
    train_aug_df.to_csv(augment_smiles_data, index=False)

def aug_smi_tokenize(train_aug_data):
    """
    Returns new columns with tokenized SMILES data

    Args:
        train_aug_data: path to augmented data to be tokenized

    Returns:
        new columns to train_aug_master.csv: DA_pair_tokenized_aug, AD_pair_tokenized_aug 
    """
    aug_smi_data = pd.read_csv(train_aug_data)
    # aug_smi_data = pd.read_parquet(train_aug_data, engine="fastparquet")
    # print(aug_smi_data.head())

    # initialize new columns
    aug_smi_data["DA_pair_tokenized_aug"] = " "
    da_aug_list = []
    for i in range(len(aug_smi_data["DA_pair_aug"])):
        da_aug_list.append(aug_smi_data["DA_pair_aug"][i])


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
    print(max_length)
    # tokenize augmented data and return new column in .csv
    # TODO: add padding in a systematic way (only at beginning)
    for i in range(len(da_aug_list)):
        tokenized_list = []
        da_list = da_aug_list[i]
        for da in da_list:
            tokenized_smi = [
                token2idx[token] if token in token2idx else 1 for token in da
            ]
            tokenized_smi = pad_input(tokenized_smi, max_length)
            tokenized_list.append(tokenized_smi)
        aug_smi_data.at[i, "DA_pair_tokenized_aug"] = tokenized_list

    # aug_smi_data.fillna(np.NaN, inplace=True)
    # aug_smi_data.to_parquet(train_aug_data, index=False, engine="fastparquet")
    aug_smi_data.to_csv(train_aug_data, index=False)


# aug_smi_doRandom(MASTER_DATA, AUGMENT_SMILES_DATA, 5)
# aug_smi_tokenize(AUGMENT_SMILES_DATA)

