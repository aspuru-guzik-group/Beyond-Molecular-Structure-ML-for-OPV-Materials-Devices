import json
import sys
from pathlib import Path
from typing import Union

import mordred
import mordred.descriptors
import numpy as np
import pandas as pd
from mordred import Calculator
from rdkit import Chem
from rdkit.Chem import Mol

if sys.platform == "linux":
    DATASETS = Path("~/projects/_ml_for_opvs/datasets")
else:
   DATASETS: Path = Path(__file__).parent.parent.parent / "datasets"


def get_mordred_dict(mol: Mol) -> dict[str, float]:
    """
    Get Mordred descriptors for a given molecule.

    Args:
        mol: RDKit molecule

    Returns:
        Mordred descriptors as dictionary
    """
    calc: Calculator = Calculator(mordred.descriptors, ignore_3D=True)
    descriptors: dict[str, float] = calc(mol).asdict()
    return descriptors


def get_mordred_descriptors(mordred_descriptors: pd.DataFrame, label: str) -> np.ndarray:
    """
    Get Mordred descriptors for a given label.

    Args:
        mordred_descriptors: DataFrame of Mordred descriptors
        label: Donor or Acceptor

    Returns:
        Mordred descriptors as numpy array
    """
    descriptors: np.ndarray = mordred_descriptors.loc[label].tolist()
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
    descriptors: pd.Series = all_mols.apply(get_mordred_dict)
    mordred_descriptors: pd.DataFrame = pd.DataFrame.from_records(descriptors, index=all_mols.index)
    # Remove any columns with calculation errors
    mordred_descriptors = mordred_descriptors.infer_objects()
    mordred_descriptors = mordred_descriptors.select_dtypes(exclude=["object"])
    # Remove any columns with nan values
    mordred_descriptors.dropna(axis=1, how='any', inplace=True)
    # Remove any columns with zero variance
    # mordred_descriptors.dropna(axis=1, how='any', inplace=True)
    descriptor_variances: pd.Series = mordred_descriptors.var(numeric_only=True)
    variance_mask: pd.Series = descriptor_variances.eq(0)
    zero_variance: pd.Series = variance_mask[variance_mask == True]
    invariant_descriptors: list[str] = zero_variance.index.to_list()
    mordred_descriptors: pd.DataFrame = mordred_descriptors.drop(invariant_descriptors, axis=1)
    print("Done generating Mordred descriptors.")
    return mordred_descriptors


def assign_mordred(labels: pd.Series, mordred_descriptors: pd.DataFrame) -> pd.Series:
    """
    Assigns Mordred descriptors to the dataset.
    """
    labels = labels
    mordred_series: pd.Series = labels.map(lambda lbl: get_mordred_descriptors(mordred_descriptors, lbl))
    print("Done assigning Mordred descriptors.")
    return mordred_series


class MordredCalculator:
    def __init__(self, donor_structures: pd.DataFrame, acceptor_structures: pd.DataFrame,) -> None:
        donor_structures.set_index("Donor", inplace=True)
        acceptor_structures.set_index("Acceptor", inplace=True)
        donor_mols: pd.Series = donor_structures["SMILES"].map(lambda smiles: Chem.MolFromSmiles(smiles))
        acceptor_mols: pd.Series = acceptor_structures["SMILES"].map(lambda smiles: Chem.MolFromSmiles(smiles))
        self.all_mols: pd.Series = pd.concat([donor_mols, acceptor_mols])
        self.mordred_descriptors_unique: pd.DataFrame = self.generate_mordred_descriptors()

    @property
    def descriptors_used(self) -> list[str]:
        return self.mordred_descriptors_unique.columns.to_list()

    def generate_mordred_descriptors(self) -> pd.DataFrame:
        # BUG: Get numpy "RuntimeWarning: overflow encountered in reduce"
        # Generate Mordred descriptors
        print("Generating mordred descriptors...")
        calc: Calculator = Calculator(mordred.descriptors, ignore_3D=True)
        descriptors: pd.Series = self.all_mols.apply(get_mordred_dict)
        print(descriptors)
        mordred_descriptors: pd.DataFrame = pd.DataFrame.from_records(descriptors, index=self.all_mols.index)
        # Remove any columns with calculation errors
        mordred_descriptors = mordred_descriptors.infer_objects()
        mordred_descriptors = mordred_descriptors.select_dtypes(exclude=["object"])
        # Remove any columns with nan values
        mordred_descriptors.dropna(axis=1, how='any', inplace=True)
        # Remove any columns with zero variance
        # mordred_descriptors.dropna(axis=1, how='any', inplace=True)
        descriptor_variances: pd.Series = mordred_descriptors.var(numeric_only=True)
        variance_mask: pd.Series = descriptor_variances.eq(0)
        zero_variance: pd.Series = variance_mask[variance_mask == True]
        invariant_descriptors: list[str] = zero_variance.index.to_list()
        mordred_descriptors: pd.DataFrame = mordred_descriptors.drop(invariant_descriptors, axis=1)
        print("Done generating Mordred descriptors.")
        return mordred_descriptors

    def assign_descriptors(self, labels: pd.Series) -> pd.DataFrame:
        """
        Assigns Mordred descriptors to the dataset.
        """
        # descriptors: list[pd.Series] = [self.mordred_descriptors_unique.loc[l] for l in labels]
        descriptors = []
        for l in labels:
            d = self.mordred_descriptors_unique.loc[l]
            descriptors.append(d)
        descriptor_df: pd.DataFrame = pd.concat(descriptors, axis=1).transpose().reset_index(drop=True)
        # mordred_series: pd.Series = labels.map(lambda lbl: get_mordred_descriptors(mordred_descriptors, lbl))
        print("Done assigning Mordred descriptors.")
        return descriptor_df


# def run(donor_structures: pd.DataFrame,
#         acceptor_structures: pd.DataFrame,
#         dataset: pd.DataFrame,
#         mordred_csv: Union[Path, str],
#         mordred_pkl: Union[Path, str]
#         ) -> None:
#     # Generate mordred descriptors and remove 0 variance and nan columns
#     mordred_descriptors: pd.DataFrame = generate_mordred_descriptors(donor_structures, acceptor_structures)
#     mordred_descriptors_used: pd.Series = pd.Series(mordred_descriptors.columns.tolist())
#
#     # Save mordred descriptor IDs
#     mordred_descriptors_used.to_csv(mordred_csv)
#
#     for material in ["Donor", "Acceptor"]:
#         dataset[f"{material} mordred"] = assign_mordred(dataset[f"{material}"], mordred_descriptors)
#
#     # Save dataset with mordred descriptors
#     dataset[["Donor mordred", "Acceptor mordred"]].to_pickle(mordred_pkl)

def run(donor_structures: pd.DataFrame,
        acceptor_structures: pd.DataFrame,
        dataset: pd.DataFrame,
        mordred_names: Union[Path, str],
        mordred_pkl: Union[Path, str]
        ) -> None:
    # Generate mordred descriptors and remove 0 variance and nan columns
    mordred_calc: MordredCalculator = MordredCalculator(donor_structures, acceptor_structures)
    mordred_descriptors_used: list[str] = mordred_calc.descriptors_used

    # Save mordred descriptor IDs
    with mordred_names.open("w") as f:
        json.dump(mordred_descriptors_used, f)

    donors_mordred: pd.DataFrame = mordred_calc.assign_descriptors(dataset["Donor"])
    acceptors_mordred: pd.DataFrame = mordred_calc.assign_descriptors(dataset["Acceptor"])

    donors_mordred.columns = [f"Donor mordred {col}" for col in donors_mordred.columns]
    acceptors_mordred.columns = [f"Acceptor mordred {col}" for col in acceptors_mordred.columns]

    # dataset_mordred: pd.DataFrame = pd.concat([donors_mordred, acceptors_mordred], axis=1)
    dataset_mordred: pd.DataFrame = donors_mordred.join(acceptors_mordred)

    # Save dataset with mordred descriptors
    dataset_mordred.to_pickle(mordred_pkl)

def test():
    # Load cleaned donor and acceptor structures
    donor_structures: pd.DataFrame = pd.read_csv("test donors.csv")

    # acceptor_structures_file = min_dir / "cleaned acceptors.csv"
    acceptor_structures: pd.DataFrame = pd.read_csv("test acceptors.csv")

    # Load dataset
    dataset: pd.DataFrame = pd.read_pickle("test dataset.pkl")

    mordred_json = Path("test mordred used.json")
    mordred_pkl = Path("test dataset mordred.pkl")

    run(donor_structures, acceptor_structures, dataset, mordred_json, mordred_pkl)


def main():
    # Load cleaned donor and acceptor structures
    dataset_dir: Path = DATASETS / "Min_2020_n558"
    donor_structures_file = dataset_dir / "cleaned donors.csv"
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures_file = dataset_dir / "cleaned acceptors.csv"
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)

    # Load dataset
    dataset_pkl = dataset_dir / "cleaned_dataset.pkl"
    dataset: pd.DataFrame = pd.read_pickle(dataset_pkl)

    # Save mordred descriptor IDs
    mordred_json = dataset_dir / "mordred_descriptors.json"

    # Save dataset
    mordred_pkl = dataset_dir / "cleaned_dataset_mordred.pkl"

    run(donor_structures, acceptor_structures, dataset, mordred_json, mordred_pkl)


def main_saeki():
    # Load cleaned donor and acceptor structures
    dataset_dir: Path = DATASETS / "Saeki_2022_n1318"
    donor_structures_file = dataset_dir / "donors.csv"
    acceptor_structures_file = dataset_dir / "acceptors.csv"
    dataset_pkl = dataset_dir / "Saeki_corrected_pipeline.pkl"
    mordred_json = dataset_dir / "Saeki_mordred_descriptors.json"
    mordred_pkl = dataset_dir / "Saeki_mordred.pkl"

    # Load dataset
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)
    dataset: pd.DataFrame = pd.read_pickle(dataset_pkl)

    run(donor_structures, acceptor_structures, dataset, mordred_json, mordred_pkl)


def main_hutchison():
    # Load cleaned donor and acceptor structures
    dataset_dir: Path = DATASETS / "Hutchison_2023_n1001"
    donor_structures_file = dataset_dir / "donors.csv"
    acceptor_structures_file = dataset_dir / "acceptors.csv"
    dataset_pkl = dataset_dir / "Hutchison_filtered_dataset_pipeline.pkl"
    mordred_json = dataset_dir / "Hutchison_mordred_descriptors.json"
    mordred_pkl = dataset_dir / "Hutchison_mordred.pkl"

    # Load dataset
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)
    dataset: pd.DataFrame = pd.read_pickle(dataset_pkl)

    run(donor_structures, acceptor_structures, dataset, mordred_json, mordred_pkl)


if __name__ == "__main__":
    test()
    # main()
    # main_saeki()
    main_hutchison()
