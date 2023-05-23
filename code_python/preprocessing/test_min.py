# ATTN: When done, check the following:
#  - All rows that have interlayer have interlayer descriptors
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Mol

from code_python import DATASETS
from code_python.cleaning.utils import find_identical_molecules


def test_tanimoto_similarity(dataset: pd.DataFrame) -> None:
    radius: int = 5
    nbits: int = 512
    for material in ["Acceptor", "Donor"]:
        print(f"Checking {material}s by Tanimoto similarity...")
        overlaps: int = find_identical_molecules(dataset[f"{material} SMILES"], radius=radius, bits=nbits)
        assert overlaps == 0, f"Found {overlaps} identical {material}s by Tanimoto similarity"


def test_has_smiles(dataset: pd.DataFrame) -> None:
    for material in ["Acceptor", "Donor"]:
        print(f"Checking {material}s for missing SMILES...")
        no_smiles = dataset[material][dataset[f"{material} SMILES"].isna()].unique()
        assert len(no_smiles) == 0, f"Found {material}s without SMILES: \n{no_smiles}"


def check_non_null_values(df: pd.DataFrame, column_name: str) -> tuple[bool, pd.DataFrame]:
    """
    Check if all rows in the specified column of a DataFrame have non-null values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to check.

    Returns:
        bool: True if all rows have non-null values, False otherwise.
    """
    column: pd.Series = df[column_name]
    null_rows = df[column.isnull()]
    return column.notnull().all(), null_rows


def test_has_solvent_descriptors(dataset: pd.DataFrame) -> None:
    for solvent in ["solvent", "solvent additive"]:
        print(f"Checking {solvent} descriptors...")
        filtered_dataset: pd.DataFrame = dataset.dropna(subset=[solvent])
        has_descriptors, null_rows = check_non_null_values(filtered_dataset, f"{solvent} descriptors")
        assert has_descriptors, f"Found {solvent}s without descriptors:\n{null_rows}"


def draw_molecule_with_label(smiles: str, label: str):
    mol: Mol = Chem.MolFromSmiles(smiles)
    # Generate the molecule's image
    img = Draw.MolToImage(mol)

    # Show the image with the molecule's index as the title
    plt.imshow(img)
    plt.axis('off')
    plt.title(label)
    plt.show()
    plt.close()


def check_backbone(label: str) -> Optional[str]:
    backbone: str = ""
    while backbone != "y":
        backbone: str = input("Is the backbone correct? (y/n)")
        if backbone == "n":
            correct_smiles: str = input("Enter the correct SMILES:\t")
            # print("New structure:")
            # draw_molecule_with_label(correct_smiles, label)
            return correct_smiles


def check_sidechain(r_group_smiles: str, label: str) -> Optional[str]:
    sidechains: str = ""
    while sidechains != "y":
        sidechains: str = input("Are the sidechains correct? (y/n)")
        if sidechains == "n":
            old_r: str = input("Which R group is incorrect?\t")
            correct_r_group: str = input("Enter the correct R group:\t")
            correct_smiles: str = r_group_smiles.replace(old_r, correct_r_group)
            print(f"New structure:\n{correct_smiles}")
            # draw_molecule_with_label(correct_smiles, label)
            satisfied: str = input("Are you satisfied? (y/n)")
            if satisfied == "y":
                return correct_smiles


def check_structure(r_group_smiles: str, label: str) -> str:
    correct_smiles: str = r_group_smiles
    # Wait for user input before showing the next molecule
    correct: str = ""
    while correct != "y":
        correct: str = input("Is the structure correct? (y/n/exit)")
        if correct == "exit":
            return correct
        if correct == "n":
            # BUG: Can't correct both backbone and sidechains simultaneously
            backbone: Optional[str] = check_backbone(label)
            if backbone is not None:
                correct_smiles = backbone
            sidechains = check_sidechain(r_group_smiles, label)
            if sidechains is not None:
                correct_smiles = sidechains
    # plt.close()
    # if correct_smiles != r_group_smiles:
    #     draw_molecule_with_label(correct_smiles, label)
    return correct_smiles


def verify_structures(series: pd.Series, master_smiles: pd.DataFrame, save_file: Path, last_label: str) -> pd.DataFrame:
    reached_restart: bool = False
    for index, smiles in series.items():
        if index == last_label or last_label == "":
            reached_restart = True
        if not reached_restart:
            continue

        draw_molecule_with_label(smiles, index)
        print(index)
        r_smiles_master: str = master_smiles.at[index, "R_grp_SMILES"]
        if not isinstance(r_smiles_master, str):
            raise TypeError(f"Expected string, got {type(r_smiles_master)} for {index}")
        checked: str = check_structure(r_smiles_master, index)
        if checked == "exit":
            master_smiles.to_csv(save_file)
            sys.exit()  # Exit the program
        else:
            master_smiles.at[index, "R_grp_SMILES"] = checked

        # Save intermediate corrections to file
        master_smiles.to_csv(save_file)
    return master_smiles


def test_correct_structures(dataset: pd.DataFrame, last_structure_label: str) -> None:
    # for material in ["Donor", "Acceptor"]:
    for material in ["Acceptor"]:
        master_file: Path = DATASETS / "Min_2020_n558" / "raw" / f"min_{material.lower()}s_smiles_master_EDITED.csv"
        master_smiles: pd.DataFrame = pd.read_csv(master_file, index_col="Name")

        print(f"Checking {material}s for correct structures...")
        dataset_reindexed: pd.DataFrame = dataset.set_index(material)
        unique_labels: pd.DataFrame = dataset_reindexed[~dataset_reindexed.index.duplicated(keep='first')]
        smi = unique_labels[f"{material} SMILES"]
        master_smiles: pd.DataFrame = verify_structures(smi, master_smiles, master_file, last_structure_label)
        master_smiles.to_csv(master_file)


if __name__ == "__main__":
    min_dir: Path = DATASETS / "Min_2020_n558"
    dataset_file: Path = min_dir / "cleaned_dataset.pkl"
    dataset: pd.DataFrame = pd.read_pickle(dataset_file)

    test_tanimoto_similarity(dataset)
    # test_has_smiles(dataset)
    # test_has_solvent_descriptors(dataset)
    # last_label: str = "IFT-ECA"
    # test_correct_structures(dataset, last_label)
