from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def get_labels_set(mols_df, column: str, labels: set) -> pd.DataFrame:
    return mols_df[mols_df[column].isin(labels)]


def get_mols(mols_df: pd.DataFrame) -> pd.DataFrame:
    mols_df["Molecule"] = [Chem.MolFromSmiles(mol) for mol in mols_df["SMILES"]]
    return mols_df


def find_duplicate_labels(mols_df: pd.DataFrame, column: str):
    duplicate_labels: List[str] = list(mols_df[mols_df.duplicated(subset=column)][column])

    for label in duplicate_labels:
        dup_label_mols: pd.DataFrame = mols_df[mols_df[column] == label]
        # indices: List[int] = [ix for ix in dup_label_mols.index]
        molecules = [mol for mol in dup_label_mols["Molecule"]]
        legends: List[str] = [f"{label}: {ix}" for ix in dup_label_mols.index]
        Draw.MolsToGridImage(molecules, subImgSize=(400, 400), legends=legends).show()

        drop_ix = input("Which to drop?")
        if drop_ix == "":
            continue
        else:
            drop_ix = eval(drop_ix)
            assert isinstance(drop_ix, int) or isinstance(drop_ix, list), "Wrong input type!"
            if isinstance(drop_ix, int):
                mols_df.drop(labels=[drop_ix], inplace=True)
            elif isinstance(drop_ix, list):
                mols_df.drop(labels=drop_ix, inplace=True)

    return mols_df.drop(labels=["Molecule"], axis=1)


if __name__ == "__main__":
    opv_file = Path.home() / "Downloads" / "OPV ML data extraction.xlsx"
    opv_data = pd.read_excel(opv_file)


    # donors: pd.DataFrame = pd.read_csv("clean_min_donors.csv")
    # donor_labels = set(opv_data["Donor Molecule"])
    # donors = get_mols(get_labels_set(donors, "Donor", donor_labels))
    # cleaned_donors = find_duplicate_labels(donors, "Donor")
    # cleaned_donors.to_csv("clean_min_donors_frfr.csv", index=False)

    acceptors = pd.read_csv("clean_min_acceptors.csv")
    acceptor_labels = set(opv_data["Acceptor Molecule"])
    acceptors = get_mols(get_labels_set(acceptors, "Acceptor", acceptor_labels))
    cleaned_acceptors = find_duplicate_labels(acceptors, "Acceptor")
    cleaned_acceptors.to_csv("clean_min_acceptors_frfr.csv", index=False)
