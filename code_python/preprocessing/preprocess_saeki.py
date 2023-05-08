import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from code_python import DATASETS


def assign_ids(smiles_series: pd.Series) -> pd.Series:
    """
    Generates a pandas Series of labels, where each unique SMILES string is mapped to its corresponding label.

    Args:
        smiles_series: A pandas series of SMILES strings

    Returns:
        A pandas series of labels, where each unique SMILES string is mapped to its corresponding label
    """
    # Canonicalize SMILES strings in smiles_series
    smiles_series: pd.Series = smiles_series.apply(lambda x: Chem.CanonSmiles(x))
    # Create a list of unique SMILES strings
    unique_smiles: set = set(smiles_series)
    # Create a list of labels
    labels: list[int] = [i for i in range(len(unique_smiles))]
    # Create a dictionary of SMILES strings to labels
    smiles_to_label: dict[str, int] = dict(zip(unique_smiles, labels))
    # Create a list of labels, where each unique SMILES string is mapped to its corresponding label
    labels: list[int] = [smiles_to_label[smiles] for smiles in smiles_series]
    return pd.Series(labels)


# Import csv version
dataset_csv = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected.csv"
saeki = pd.read_csv(dataset_csv)

# # Import pkl version
dataset_pkl = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected_r5_b512.pkl"
# saeki = pd.read_pickle(dataset_pkl)


# Assign molecule IDs to unique n and p SMILES
saeki["n(labels)"] = assign_ids(saeki["n(SMILES)"])
saeki["p(labels)"] = assign_ids(saeki["p(SMILES)"])

# Create Molecule and fingerprint objects for pickle file
saeki["n(mol)"] = saeki["n(SMILES)"].apply(lambda x: Chem.MolFromSmiles(x))
saeki["p(mol)"] = saeki["p(SMILES)"].apply(lambda x: Chem.MolFromSmiles(x))
saeki["n(FP)"] = saeki["n(mol)"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=512)))
saeki["p(FP)"] = saeki["p(mol)"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=512)))
saeki["n,p(FP)"] = [[*n, *p] for n, p in zip(saeki["n(FP)"], saeki["p(FP)"])]

# Save to pkl
saeki.to_pickle(dataset_pkl)
