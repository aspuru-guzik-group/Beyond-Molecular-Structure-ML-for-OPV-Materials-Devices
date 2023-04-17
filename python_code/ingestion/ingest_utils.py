from rdkit import Chem, DataStructs
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


def find_identical_molecules(series, radius, bits):
    # Get unique SMILES
    series = series.apply(lambda x: Chem.CanonSmiles(x))
    molecules = list(set(series))

    # create ECFP fingerprints for all molecules in the series
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), radius, nBits=bits) for mol in molecules]

    # compare all pairs of unique SMILES
    clashes = 0
    for i, mol1 in enumerate(molecules):
        for j, mol2 in enumerate(molecules):
            if j > i:
                fp1 = fps[i]
                fp2 = fps[j]
                sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                if sim == 1:
                    clashes += 1
                    print(f"Molecule {i+1}: {mol1}")
                    print(f"Molecule {j+1}: {mol2}\n")
    print("radius:", radius, "\tbits:", bits, "\t\tmolecule clashes:", clashes, "\n\n")
