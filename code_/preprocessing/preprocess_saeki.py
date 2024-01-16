from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def int_to_alpha(number):
    result = ""
    while number > 0:
        number -= 1
        digit = number % 26
        result = chr(digit + ord('A')) + result
        number //= 26
    return result


def assign_ids(smiles_series: pd.Series, prefix: str) -> pd.Series:
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
    unique_labels: list[str] = [f"{prefix} {int_to_alpha(i)}" for i in range(len(unique_smiles))]
    # Create a dictionary of SMILES strings to labels
    smiles_to_label: dict[str, str] = dict(zip(unique_smiles, unique_labels))
    # Create a list of labels, where each unique SMILES string is mapped to its corresponding label
    labels: list[str] = [smiles_to_label[smiles] for smiles in smiles_series]
    return pd.Series(labels)


# Import csv version
dataset_csv = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected_pipeline.csv"
saeki = pd.read_csv(dataset_csv)

# # Import pkl version
dataset_pkl = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected_pipeline.pkl"
# saeki = pd.read_pickle(dataset_pkl)


# # Assign molecule IDs to unique n and p SMILES
# saeki["n(labels)"] = assign_ids(saeki["n(SMILES)"])
# saeki["p(labels)"] = assign_ids(saeki["p(SMILES)"])

# Create Molecule and fingerprint objects for pickle file
saeki["Acceptor SMILES"] = saeki["Acceptor SMILES"].apply(lambda x: Chem.CanonSmiles(x))
saeki["Donor SMILES"] = saeki["Donor SMILES"].apply(lambda x: Chem.CanonSmiles(x))
saeki["Acceptor"] = assign_ids(saeki["Acceptor SMILES"], "Acceptor")
saeki["Donor"] = assign_ids(saeki["Donor SMILES"], "Donor")
saeki["Acceptor Mol"] = saeki["Acceptor SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
saeki["Donor Mol"] = saeki["Donor SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
saeki["Acceptor ECFP10_2048"] = saeki["Acceptor Mol"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)))
saeki["Donor ECFP10_2048"] = saeki["Donor Mol"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)))
# saeki["n,p(FP)"] = [[*n, *p] for n, p in zip(saeki["n(FP)"], saeki["p(FP)"])]

# Save to pkl
saeki.to_pickle(dataset_pkl)

# Get unique donor and acceptor SMILES
donors = saeki[["Donor SMILES", "Donor"]].drop_duplicates(ignore_index=True)
acceptors = saeki[["Acceptor SMILES", "Acceptor"]].drop_duplicates(ignore_index=True)

saeki_donors = DATASETS / "Saeki_2022_n1318" / "donors.csv"
saeki_acceptors = DATASETS / "Saeki_2022_n1318" / "acceptors.csv"
donors["SMILES"] = donors["Donor SMILES"]
acceptors["SMILES"] = acceptors["Acceptor SMILES"]
donors[["Donor", "SMILES"]].to_csv(saeki_donors, index=False)
acceptors[["Acceptor", "SMILES"]].to_csv(saeki_acceptors, index=False)
