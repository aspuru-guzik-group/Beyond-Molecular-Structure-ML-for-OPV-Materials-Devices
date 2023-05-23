import pandas as pd
from pathlib import Path
from rdkit import Chem

from code_python import DATASETS


def ingest_r_groups(r_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Function that assigns a second-row transition metal or lanthanide to each R group.

    Args:
        r_groups: pandas DataFrame of R group labels and SMILES strings

    Returns:
        pandas DataFrame of R group labels, SMILES strings, and metal labels
    """
    n_labels: int = len(r_groups)
    atomic_nums: list[int] = [*range(39, 49), *range(57, 81), *range(89, 109)]
    metals: list[str] = [Chem.MolToSmiles(Chem.MolFromSmarts(f"[#{num}]")) for num in atomic_nums[:n_labels]]
    r_groups["metal"] = metals
    return r_groups


def r_main():
    r_groups_csv: Path = DATASETS / "Min_2020_n558" / "raw" / "r_groups.csv"
    clean_csv: Path = DATASETS / "Min_2020_n558" / "cleaned R groups.csv"

    r_groups: pd.DataFrame = pd.read_csv(r_groups_csv)
    r_groups: pd.DataFrame = ingest_r_groups(r_groups)
    r_groups.to_csv(clean_csv, index=False)


if __name__ == "__main__":
    r_main()
