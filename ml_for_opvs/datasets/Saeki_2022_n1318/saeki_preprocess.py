import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def get_incorrect_smiles(smiles):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    for p in ps:
        # if p.GetType() == 'AtomValenceException':
        #     return True
        return True


def fix_charges(smiles):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType() == 'AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum() == 7 and at.GetFormalCharge() == 0 and at.GetExplicitValence() == 4:
                at.SetFormalCharge(1)
            if at.GetAtomicNum() == 5 and at.GetFormalCharge() == 0 and at.GetExplicitValence() == 4:
                at.SetFormalCharge(-1)
    Chem.SanitizeMol(m)
    return m


# def assign_ids(df):
#     """
#     Iterate through the 'p(SMILES)' and 'n(SMILES)' columns. Calculate the Tanimoto similarity between row N and N+1.
#     If the Tanimoto similarity between the two is equal to 1, assign the same ID to the row.
#     If the Tanimoto similarity is less than 1, assign a new ID to the row.
#     """
#     for i in range(1, len(df)-1):



if __name__ == "__main__":
    # saeki = pd.read_csv("Saeki_corrected.csv")
    # saeki["n wrong"] = saeki["n(SMILES)"].apply(lambda x: get_incorrect_smiles(x))
    # saeki["p wrong"] = saeki["p(SMILES)"].apply(lambda x: get_incorrect_smiles(x))
    # print("n wrong:", saeki["n wrong"].sum())
    # print(saeki[saeki["n wrong"] == True]["n(SMILES)"])
    # print("p wrong:", saeki["p wrong"].sum())
    # print(saeki[saeki["p wrong"] == True]["p(SMILES)"])

    # saeki["n(mol)"] = saeki["n(SMILES)"].apply(lambda x: Chem.MolFromSmiles(x))
    # saeki["p(mol)"] = saeki["p(SMILES)"].apply(lambda x: Chem.MolFromSmiles(x))
    # saeki["n(FP)"] = saeki["n(mol)"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=1024)))
    # saeki["p(FP)"] = saeki["p(mol)"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=1024)))
    saeki = pd.read_pickle("saeki_corrected.pkl")
    saeki["n,p(FP)"] = [[*n, *p] for n, p in zip(saeki["n(FP)"], saeki["p(FP)"])]
    saeki.to_pickle("saeki_corrected.pkl")
    pass
