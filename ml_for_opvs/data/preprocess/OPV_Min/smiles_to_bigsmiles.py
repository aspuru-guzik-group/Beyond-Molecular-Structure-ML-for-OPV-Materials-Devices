import pkg_resources
from rdkit import Chem
import pandas as pd

CLEAN_DONOR_CSV = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/clean_min_donors.csv"
)

CLEAN_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/clean_min_acceptors.csv"
)


def smile_to_bigsmile(donor_data, acceptor_data):

    """Function that will convert SMILES to BigSMILES in the acceptor/donor dataframe.
    - replaces any methyl group with ([$]) which is the bonding descriptor for polymers

    Args:
        donor_data: contains labels and various Cheminformatic Representations for donor molecules
        acceptor_data: contains labels and various Cheminformatic Representations for acceptor molecules

    Returns:
        updates donor_data and acceptor_data
    """
    donor_df = pd.read_csv(donor_data)
    acceptor_df = pd.read_csv(acceptor_data)

    # NOTE: two types of SMILES strings
    # 1. Starts with one of the methyl groups, so one of them is "Cc", and the other is "(C)"
    # 2. Starts with R group, so both methyl groups are "(C)"

    for index, row in donor_df.iterrows():
        smile = row["SMILES"]
        if smile[-1] == "C":
            smile = smile[:-1]
            smile = (
                smile + "[$]"
            )  # NOTE: check how we want to format this, in OPV sheets it's $ only
            # smile = smile.replace("(C)", "[$]")
        elif smile[0:2] == "Cc":
            smile = smile[1:]
            smile = "[$]" + smile
        # # type 2
        smile = smile.replace(
            "(C)", "([$])"
        )  # NOTE: check how we want to format this, in OPV sheets it's ($)

        smile = "{" + smile + "}"
        donor_df.at[index, "Big_SMILES"] = smile

    print(donor_df.head())
    donor_df.to_csv(donor_data, index=False)

    for index, row in acceptor_df.iterrows():
        smile = row["SMILES"]
        # if smile[0] == "C":
        #     smile = smile[1:]
        #     smile = (
        #         "[$]" + smile
        #     )  # NOTE: check how we want to format this, in OPV sheets it's $ only
        #     smile = smile.replace("(C)", "[$]")
        # # type 2
        # else:
        #     smile = smile.replace(
        #         "(C)", "[$]"
        #     )  # NOTE: check how we want to format this, in OPV sheets it's ($)

        # smile = "{" + smile + "}"
        acceptor_df.at[index, "Big_SMILES"] = smile

    acceptor_df.to_csv(acceptor_data, index=False)


def sanity_check_bigsmiles(donor_data: str):
    """Checks whether there are only 2 [$] for each donor polymer.

    Args:
        donor_data (str): _description_
    """
    donor_df: pd.DataFrame = pd.read_csv(donor_data)
    for index, row in donor_df.iterrows():
        donor: str = row["Donor"]
        smile: str = row["Big_SMILES"]
        counts: int = smile.count("[$]")
        assert counts == 2, "False BigSMILES, {}".format(donor)


# smile_to_bigsmile(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)
sanity_check_bigsmiles(CLEAN_DONOR_CSV)
# 