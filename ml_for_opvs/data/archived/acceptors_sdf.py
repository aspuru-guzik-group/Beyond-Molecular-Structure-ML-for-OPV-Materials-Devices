from rdkit.Chem import PandasTools
import pandas as pd
import pkg_resources
from rdkit import Chem
from rdkit.Chem.rdmolops import ReplaceSubstructs

SDF_PATH = pkg_resources.resource_filename("r_group", "min_acceptors_with_names.sdf")
SDF_CSV_PATH = pkg_resources.resource_filename("r_group", "min_acceptors_sdf.csv")

frame = PandasTools.LoadSDF(
    SDF_PATH, smilesName="SMILES", molColName="Molecule", includeFingerprints=True
)

patts = {
    "[R1]": "CC(CCCCCC)CCCCCCCC",
    "[R2]": "CCCCCCCC",
    "[R3]": "[Si](CCC)(CCC)(CCC)",
    "[R4]": "CC(CC)CCCC",
    "[R5]": "SCCCCCCCCCCCC",
    "[R6]": "CC(CCCCCCCC)CCCCCCCCCC",
    "[R7]": "SCC(CCCCCC)CCCC",
    "[R8]": "[Si](CC)(CC)(CC)",
    "[R9]": "[Si](C(C)C)(C(C)C)C(C)C",
    "[R10]": "[Si](CCCC)(CCCC)(CCCC)",
    "[R11]": "[Si](C)(C)CCCCCCCC",
    "[R12]": "SCCCCC=C",
    "[R13]": "SCC4CCCCC4",
    "[R14]": "CCCCCC",
    "[R15]": "CCCCCCCCCC",
    "[R19]": "CCCCCCCCCCCCCCCC",
    "[R20]": "CCCCCCCCCCC",
    "[R21]": "C(CCCCCCCCC)CCCCCCC",
    "[R24]": "COCCOC",
    "[R27]": "CCCC",
}

r_grp = {
    "[1*]": "[R1]",
    "[2*]": "[R2]",
    "[3*]": "[R3]",
    "[4*]": "[R4]",
    "[5*]": "[R5]",
    "[6*]": "[R6]",
    "[7*]": "[R7]",
    "[8*]": "[R8]",
    "[9*]": "[R9]",
    "[10*]": "[R10]",
    "[11*]": "[R11]",
    "[12*]": "[R12]",
    "[13*]": "[R13]",
    "[14*]": "[R14]",
    "[15*]": "[R15]",
    "[19*]": "[R19]",
    "[20*]": "[R20]",
    "[21*]": "[R21]",
    "[24*]": "[R24]",
    "[27*]": "[R27]",
}


def PreProcess(acceptor_df):
    smiles_column = acceptor_df["SMILES"]
    # change number* groups to R groups
    i = 0
    while i < len(smiles_column) + 4:
        try:
            string = smiles_column[i]
        except:
            i += 1
            continue
        for r in r_grp:
            string = string.replace(r, r_grp[r])
        smiles_column[i] = string
        i += 1

    return acceptor_df


def Process(acceptor_df):
    smiles_column = acceptor_df["SMILES"]

    i = 0
    while i < len(smiles_column) + 4:
        try:
            string = smiles_column[i]
        except:
            i += 1
            continue
        halogen = "empty"
        first_char = string[0]
        if first_char == "F":
            halogen = "F"
            string = string[4:]
        elif first_char == "C":
            if string[1] == "l":
                halogen = "Cl"
                string = string[6:]
            else:
                halogen = "C"
                string = string[4:]
        elif first_char == "B":
            halogen = "Br"
            string = string[6:]

        elif first_char == "I":
            halogen = "I"
            string = string[4:]

        if halogen != "empty":
            mol_obj = Chem.MolFromSmarts(string)
            indanone_object = Chem.MolFromSmarts("C=C8C(c9ccccc9C\8=C(C#N)\C#N)=O")
            substitute_smiles = "C=C8C(c9cc(" + halogen + ")ccc9C\8=C(C#N)\C#N)=O"
            substitute_object = Chem.MolFromSmarts(substitute_smiles)
            new_mol_object = ReplaceSubstructs(
                mol_obj, indanone_object, substitute_object, replaceAll=True,
            )
            new_smile = Chem.MolToSmiles(new_mol_object[0])
            new_smile = Chem.CanonSmiles(new_smile)
            smiles_column[i] = new_smile

        i += 1
    return acceptor_df


# df = PreProcess(frame)
# df = Process(df)

# df["SMILES"].to_csv(SDF_CSV_PATH)
frame["SMILES"].to_csv("raw.csv")
