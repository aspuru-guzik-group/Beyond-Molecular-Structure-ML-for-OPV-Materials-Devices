# ATTN: When done, check the following:
#  - All rows have solvent descriptors
#  - All rows that have solvent additives have solvent additive descriptors
#  - All rows that have interlayer have interlayer descriptors
from pathlib import Path

import pandas as pd

from code_python import DATASETS
from code_python.cleaning.utils import find_identical_molecules


def test_tanimoto_similarity(dataset) -> None:
    radius: int = 5
    nbits: int = 512
    for material in ["Acceptor", "Donor"]:
        print(f"Checking {material}s by Tanimoto similarity...")
        overlaps: int = find_identical_molecules(dataset[f"{material} SMILES"], radius=radius, bits=nbits)
        assert overlaps == 0, f"Found {overlaps} identical {material}s by Tanimoto similarity"


def test_has_smiles(dataset) -> None:
    for material in ["Acceptor", "Donor"]:
        print(f"Checking {material}s for missing SMILES...")
        no_smiles = dataset[material][dataset[f"{material} SMILES"].isna()].unique()
        assert len(no_smiles) == 0, f"Found {material}s without SMILES: \n{no_smiles}"


if __name__ == "__main__":
    min_dir: Path = DATASETS / "Min_2020_n558"
    dataset_file: Path = min_dir / "cleaned_dataset.pkl"
    dataset: pd.DataFrame = pd.read_pickle(dataset_file)

    test_tanimoto_similarity(dataset)
    test_has_smiles(dataset)
