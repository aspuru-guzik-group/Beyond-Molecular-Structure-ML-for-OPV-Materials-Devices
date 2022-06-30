from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import pkg_resources
import pandas as pd
import numpy as np

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/master_ml_for_opvs_from_min.csv"
)

FP_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/fingerprint/master_fingerprint.csv"
)


np.set_printoptions(threshold=np.inf)

def create_master_fp(master_data, fp_path, radius: int, nbits: int):
    """
    Create and export dataframe with fingerprint bit vector representations to .csv or .pkl file

    Args:
        fp_path: path to master fingerprint data for training
        radius: radius for creating fingerprints
        nbits: number of bits to create the fingerprints

    Returns:
        new dataframe with fingerprint data for training
    """
    fp_df = pd.read_csv(master_data)

    # Only used when first creating dataframe from master data before
    fp_df.drop(
        [
            "Donor_Big_SMILES",
            "Donor_SELFIES",
            "Acceptor_Big_SMILES",
            "Acceptor_SELFIES",
        ],
        axis=1,
    )

    new_column_da_pair = "DA_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
    fp_df[new_column_da_pair] = " "
    for index, row in fp_df.iterrows():
        da_pair = (
            fp_df.at[index, "Donor_SMILES"]
            + "."
            + fp_df.at[index, "Acceptor_SMILES"]
        )
        da_pair_mol = Chem.MolFromSmiles(da_pair)
        bitvector_da = AllChem.GetMorganFingerprintAsBitVect(
            da_pair_mol, radius, nBits=nbits
        )
        fp_da_list = list(bitvector_da.ToBitString())
        fp_da_map = map(int, fp_da_list)
        fp_da = list(fp_da_map)

        fp_df.at[index, new_column_da_pair] = fp_da

    fp_df.to_csv(fp_path, index=False)


# put master_ml_data first, and then when you create more fingerprints, use fp_data
# create_master_fp(MASTER_ML_DATA, FP_DATA, 3, 512)
