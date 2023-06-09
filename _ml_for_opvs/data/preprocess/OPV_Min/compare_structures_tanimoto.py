import itertools
from typing import Callable, Dict, List

import pandas as pd
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Mol


def get_mols_and_fps(molecules: pd.DataFrame) -> pd.DataFrame:
    molecules["Molecule"] = [Chem.MolFromSmiles(mol) for mol in molecules["SMILES"]]
    molecules["Fingerprint"] = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in molecules["Molecule"]]
    return molecules


def compare_structures(mols_data: pd.DataFrame, name: str, comp_method: Callable):
    indices: pd.Series = mols_data.index
    fingerprints: pd.Series = mols_data["Fingerprint"]
    names: pd.Series = mols_data[name]
    molecules: pd.Series = mols_data["Molecule"]

    comp_method(indices, names, molecules, fingerprints)

    print("Success! Done.")


def compare_itertools(indices, names, molecules, fingerprints):
    for a, b in itertools.combinations(indices, 2):
        similarity: float = DataStructs.FingerprintSimilarity(fingerprints.loc[a], fingerprints.loc[b])
        if similarity == 1:
            print(names.loc[a], "\t", names.loc[b])
            try:
                Draw.MolsToGridImage((molecules.loc[a], molecules.loc[b]), subImgSize=(300, 300), legends=(names.loc[a], names.loc[b])).show()
            except TypeError:
                print("See above ^")


def compare_loops(indices, names, molecules, fingerprints):
    duplicates: Dict = {}
    for a in indices:
        name = names.loc[a]
        mol = molecules.loc[a]
        duplicates[name] = {"mol": mol, "dupes": []}

        for b in indices:
            similarity: float = DataStructs.FingerprintSimilarity(fingerprints.loc[a], fingerprints.loc[b])
            if similarity == 1:
                print(name, "\t", names.loc[b])
                duplicates[name]["dupes"].append({names.loc[b]: molecules.loc[b]})

        if len(duplicates[name]["dupes"]) > 1:
            dupe_names = set([name] + [list(n.keys())[0] for n in duplicates[name]["dupes"]])
            dupe_mols = set([mol] + [list(m.values())[0] for m in duplicates[name]["dupes"]])
            try:
                Draw.MolsToGridImage(dupe_mols,
                                     subImgSize=(400, 400),
                                     legends=dupe_names
                                     ).show()
            except TypeError:
                print(f"See above ^.\t Index {a}")


if __name__ == "__main__":
    acceptors: pd.DataFrame = pd.read_csv("clean_min_acceptors_frfr.csv")
    acceptors = get_mols_and_fps(acceptors)
    compare_structures(acceptors, "Acceptor", compare_loops)

    donors: pd.DataFrame = pd.read_csv("clean_min_donors_frfr.csv")
    donors = get_mols_and_fps(donors)
    compare_structures(donors, "Donor", compare_loops)
