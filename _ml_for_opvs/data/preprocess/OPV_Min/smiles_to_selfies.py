import selfies as sf
import pkg_resources
import pandas as pd

CLEAN_DONOR_CSV = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/preprocess/OPV_Min/clean_min_donors.csv"
)

CLEAN_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/preprocess/OPV_Min/clean_min_acceptors.csv"
)


def opv_smiles_to_selfies(donor_data, acceptor_data):
    """
    Function that updates clean donors/acceptors .csv with SELFIES using the corresponding SMILES
    Args:
        donor_data: path to the donor.csv
        acceptor_data: path to the acceptor.csv

    Returns:
        Updates donor and acceptor data with a new column of SELFIES representation
    """
    donor_df = pd.read_csv(donor_data)
    acceptor_df = pd.read_csv(acceptor_data)

    for index, row in donor_df.iterrows():
        donor_selfies = sf.encoder(row["SMILES"])
        donor_df.at[index, "SELFIES"] = donor_selfies

    # print(donor_df.head())
    donor_df.to_csv(donor_data, index=False)

    for index, row in acceptor_df.iterrows():
        acceptor_selfies = sf.encoder(row["SMILES"])
        acceptor_df.at[index, "SELFIES"] = acceptor_selfies

    # print(acceptor_df.head())
    acceptor_df.to_csv(acceptor_data, index=False)


# opv_smiles_to_selfies(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)

# smi = "[R7]OC(=O)c1cc(-c2cc(C(=O)O[R7])c(-c3ccc(C)s3)s2)sc1-c1ccc(C)s1"
# selfie = sf.encoder(smi)
# print(selfie)

# catalysis_smiles_to_selfies(CATALYSIS_MASTER)
