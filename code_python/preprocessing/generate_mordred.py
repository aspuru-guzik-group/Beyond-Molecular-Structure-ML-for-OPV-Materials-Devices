from pathlib import Path

import mordred
import numpy as np
from mordred import Calculator
import mordred.descriptors
import pandas as pd
from rdkit import Chem

from code_python import DATASETS


def get_mordred_descriptors(mordred_descriptors: pd.DataFrame, label: str) -> np.ndarray:
    """
    Get Mordred descriptors for a given label.

    Args:
        mordred_descriptors: DataFrame of Mordred descriptors
        label: Donor or Acceptor

    Returns:
        Mordred descriptors as numpy array
    """
    descriptors: np.ndarray = mordred_descriptors.loc[label].to_numpy(dtype="float", copy=True)
    return descriptors


def generate_mordred_descriptors(donor_structures: pd.DataFrame, acceptor_structures: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Mordred descriptors from "Donor Mol" and "Acceptor Mol" columns in the dataset.
    Remove all columns with nan values, and remove all columns with zero variance.

    Returns:
        DataFrame of filtered Mordred descriptors
    """
    material_smiles: dict[str, pd.DataFrame] = {"Donor":    donor_structures.set_index("Donor"),
                                                "Acceptor": acceptor_structures.set_index("Acceptor")
                                                }
    donor_mols: pd.Series = material_smiles["Donor"]["SMILES"].map(lambda smiles: Chem.MolFromSmiles(smiles))
    acceptor_mols: pd.Series = material_smiles["Acceptor"]["SMILES"].map(
        lambda smiles: Chem.MolFromSmiles(smiles))
    all_mols: pd.Series = pd.concat([donor_mols, acceptor_mols])

    # BUG: Get numpy "RuntimeWarning: overflow encountered in reduce"
    # Generate Mordred descriptors
    print("Generating mordred descriptors...")
    calc: Calculator = Calculator(mordred.descriptors, ignore_3D=True)
    descriptors: pd.Series = all_mols.map(lambda mol: calc(mol))
    mordred_descriptors: pd.DataFrame = pd.DataFrame(descriptors.tolist(), index=all_mols.index)
    # Remove any columns with nan values
    mordred_descriptors.dropna(axis=1, how='any', inplace=True)
    # Remove any columns with zero variance
    mordred_descriptors = mordred_descriptors.loc[:, mordred_descriptors.var() != 0]
    print("Done generating Mordred descriptors.")
    return mordred_descriptors


def assign_mordred(labels: pd.Series, mordred_descriptors: pd.DataFrame) -> pd.Series:
    """
    Assigns Mordred descriptors to the dataset.
    """
    mordred_series: pd.Series = labels.map(lambda mol: get_mordred_descriptors(mordred_descriptors, mol))
    print("Done assigning Mordred descriptors.")
    return mordred_series


def main():
    # Load cleaned donor and acceptor structures
    min_dir: Path = DATASETS / "Min_2020_n558"
    donor_structures_file = min_dir / "cleaned donors.csv"
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures_file = min_dir / "cleaned acceptors.csv"
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)

    # Load dataset
    dataset_pkl = min_dir / "cleaned_dataset.pkl"
    dataset: pd.DataFrame = pd.read_pickle(dataset_pkl)

    # Generate mordred descriptors and remove 0 variance and nan columns
    mordred_descriptors: pd.DataFrame = generate_mordred_descriptors(donor_structures, acceptor_structures)
    mordred_descriptors_used: pd.Series = pd.Series(mordred_descriptors.columns.tolist())

    # Save mordred descriptor IDs
    mordred_csv = min_dir / "mordred_descriptors.csv"
    mordred_descriptors_used.to_csv(mordred_csv)

    for material in ["Donor", "Acceptor"]:
        dataset[f"{material} mordred"] = assign_mordred(dataset[f"{material} Mol"], mordred_descriptors)


if __name__ == "__main__":
    main()
