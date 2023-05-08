from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from code_python import DATASETS
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


if __name__ == "__main__":
    # Import csv version
    dataset_csv = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected.csv"
    saeki = pd.read_csv(dataset_csv)

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

    # Save to csv
    saeki.to_csv(dataset_csv, index=False)
