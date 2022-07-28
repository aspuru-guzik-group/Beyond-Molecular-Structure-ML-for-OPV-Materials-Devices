import ast  # for str -> list conversion
import numpy as np
import pandas as pd
import pkg_resources
import random
from rdkit import Chem

MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

MASTER_SMI_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/smiles/master_smiles.csv"
) 

def create_smi_csv(master_data: str, filepath: str):
    """Creates master file with SMILES, SELFIES, and BigSMILES

    Args:
        master_data (str): pre-processed master file
        filepath (str): path to save smiles data.
    """
    master_df: pd.DataFrame = pd.read_csv(master_data)
    master_df["DA_SMILES"] = ""
    master_df["DA_SELFIES"] = ""
    master_df["DA_BigSMILES"] = ""
    for i, row in master_df.iterrows():
        donor_smi = master_df.at[i, "Donor_SMILES"]
        donor_selfies = master_df.at[i, "Donor_SELFIES"]
        donor_bigsmi = master_df.at[i, "Donor_Big_SMILES"]
        acceptor_smi = master_df.at[i, "Acceptor_SMILES"]
        acceptor_selfies = master_df.at[i, "Acceptor_SELFIES"]
        acceptor_bigsmi = master_df.at[i, "Acceptor_Big_SMILES"]

        master_df.at[i, "DA_SMILES"] = donor_smi + "." + acceptor_smi
        master_df.at[i, "DA_SELFIES"] = donor_selfies + "." + acceptor_selfies
        master_df.at[i, "DA_BigSMILES"] = donor_bigsmi + "." + acceptor_bigsmi


    master_df.to_csv(filepath, index=False)


if __name__ == "__main__":
    create_csv(MASTER_DATA, MASTER_SMI_DATA)