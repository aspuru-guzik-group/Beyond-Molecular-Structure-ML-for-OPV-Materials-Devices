from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from code_python.cleaning.utils import find_identical_molecules


def convert_energy_levels(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the HOMO and LUMO columns to the negative values and returns the dataframe.
    Also renames HOMO and LUMO columns.

    Args:
        dataset: Saeki dataset

    Returns:
        Saeki dataset where energy levels are listed in negative scale.
    """
    energy_levels_cols: list[str] = ["'-HOMO_n(eV)", "'-LUMO_n(eV)", "'-HOMO_p(eV)", "'-LUMO_p(eV)"]
    energy_levels_names: dict[str, str] = {column: column[2:] for column in energy_levels_cols}

    for column in energy_levels_cols:
        dataset[column] = dataset[column].apply(lambda x: -abs(x))
    return dataset.rename(columns=energy_levels_names)


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


if __name__ == "__main__":
    # Import csv version
    dataset_csv = Path(__file__).parent.parent.parent / "datasets" / "Saeki_2022_n1318" / "Saeki_corrected.csv"
    saeki = pd.read_csv(dataset_csv)

    # # Import pkl version
    dataset_pkl = Path(__file__).parent.parent.parent / "datasets" / "Saeki_2022_n1318" / "Saeki_corrected_r5_b512.pkl"
    # saeki = pd.read_pickle(dataset_pkl)

    # # Convert energy level columns to negative scale
    # saeki = convert_energy_levels(saeki)

    # # Check SMILES and manually correct those that produce RDKit errors
    # saeki["n wrong"] = saeki["n(SMILES)"].apply(lambda x: get_incorrect_smiles(x))
    # saeki["p wrong"] = saeki["p(SMILES)"].apply(lambda x: get_incorrect_smiles(x))
    # print("n wrong:", saeki["n wrong"].sum())
    # print(saeki[saeki["n wrong"] == True]["n(SMILES)"])
    # print("p wrong:", saeki["p wrong"].sum())
    # print(saeki[saeki["p wrong"] == True]["p(SMILES)"])

    # Find identical molecules by Tanimoto similarity
    # for material in ["n(SMILES)", "p(SMILES)"]:
    # for material in ["n(SMILES)"]:
    #     print("\n\nExamining materal:", material)
        # for bits in [512, 1024, 2048, 4096, 8192]:
        # for bits in [512]:
        #     for radius in [3, 4, 5, 6, 7]:
        #     for radius in [3, 4, 5]:
        #         find_identical_molecules(saeki[material], radius=radius, bits=bits)
    #     # for bits, radius in zip([512, 1024, 2048, 4096, 8192], [3, 4, 5, 6, 7]):
    #     for bits, radius in zip([1024, 2048, 4096, 8192], [4, 5, 6, 7]):
    #         find_identical_molecules(saeki[material], radius=radius, bits=bits)
    #     find_identical_molecules(saeki[material], radius=8, bits=512)

    # Assign molecule IDs to unique n and p SMILES
    saeki["n(labels)"] = assign_ids(saeki["n(SMILES)"])
    saeki["p(labels)"] = assign_ids(saeki["p(SMILES)"])

    # Save to csv
    saeki.to_csv(dataset_csv, index=False)

    # Create Molecule and fingerprint objects for pickle file
    saeki["n(mol)"] = saeki["n(SMILES)"].apply(lambda x: Chem.MolFromSmiles(x))
    saeki["p(mol)"] = saeki["p(SMILES)"].apply(lambda x: Chem.MolFromSmiles(x))
    saeki["n(FP)"] = saeki["n(mol)"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=512)))
    saeki["p(FP)"] = saeki["p(mol)"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=512)))
    saeki["n,p(FP)"] = [[*n, *p] for n, p in zip(saeki["n(FP)"], saeki["p(FP)"])]

    # Save to pkl
    saeki.to_pickle(dataset_pkl)
