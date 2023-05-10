import mordred
import numpy as np
from mordred import Calculator
import pandas as pd
from rdkit import Chem


def get_mordred_descriptors(label: str) -> np.ndarray:
    """
    Get Mordred descriptors for a given label.

    Args:
        label: Donor or Acceptor

    Returns:
        Mordred descriptors as numpy array
    """
    descriptors: np.ndarray = mordred_descriptors.loc[label].to_numpy(dtype="float", copy=True)
    return descriptors


def generate_mordred_descriptors() -> pd.DataFrame:
    """
    Generate Mordred descriptors from "Donor Mol" and "Acceptor Mol" columns in the dataset.
    Remove all columns with nan values, and remove all columns with zero variance.

    Returns:
        DataFrame of filtered Mordred descriptors
    """
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


def assign_mordred(labels: pd.Series) -> pd.Series:
    """
    Assigns Mordred descriptors to the dataset.
    """
    mordred_series: pd.Series = labels.map(lambda mol: get_mordred_descriptors(mol))
    print("Done assigning Mordred descriptors.")
    return mordred_series


def main(dataset, material_labels: list[str]):
    mordred_descriptors: pd.DataFrame = generate_mordred_descriptors()
    mordred_descriptors_used: pd.Series = pd.Series(mordred_descriptors.columns.tolist())
    for material in ["Donor", "Acceptor"]:
        dataset[f"{material} mordred"] = assign_mordred(dataset[f"{material} Mol"])


if __name__ == "__main__":
    print("hello world")
