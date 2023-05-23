import json

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from code_python import DATASETS


def clean_structures(material: str, master_df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Function that creates a .csv with the correct SMILES, substituted R groups

    Args:
        clean_donor: path to processed donors

    Returns:
        .csv with columns: | Donor | SMILES | SMILES (w/ substituted R) | Big_SMILES | SELFIES
    """
    headers: list[str] = [
        material,
        "SMILES",
        "SMILES w/o R_group replacement",
        "BigSMILES",
        "SELFIES",
    ]
    clean_df: pd.DataFrame = pd.DataFrame(columns=headers)
    labels: list[str] = list(reference["name"])
    # Collect statistics about number of errors
    total: int = 0
    missing: int = 0  # donors not in the OPV but in the data
    errors: int = 0  # number of donors with any error
    for index, row in master_df.iterrows():
        # Ignore structures with error comments
        error_list: list[str] = [
            "not in drive, not in literature",
            "error",
            "wrong structure",
            "same name different structure",
            "not in literature",
        ]
        if row["Comments (Stanley)"] not in error_list:
            clean_df = clean_df.append(
                {
                    material:                         row["Name"],
                    "SMILES":                         row["R_grp_SMILES"],
                    "SMILES w/o R_group replacement": row["R_grp_SMILES"],
                    "SMILES w/o R_group":             " ",
                    "BigSMILES":                      " ",
                    "SELFIES":                        " ",
                },
                ignore_index=True,
            )
        else:
            errors += 1
        if row["Name"] not in labels:
            missing += 1
        total += 1
    print(clean_df.head())
    print("---------------")
    print(
        "missing: ",
        missing,
        "error: ",
        errors,
        "total: ",
        total,
    )
    return clean_df


def replace_in_string(smiles: str, labels: pd.Series) -> str:
    """
    Replaces R groups in a SMILES string by string replacement.

    Args:
        smiles: SMILES string
        labels: pandas Series of R group labels

    Returns:
        SMILES string with R groups replaced with arbitrary metals
    """
    for label, metal in labels.items():
        smiles = smiles.replace(label, metal)
    return smiles


def replace_r_with_arbitrary(structures: pd.DataFrame, r_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Replace R groups with arbitrary metals in SMILES strings.

    Args:
        structures: pandas DataFrame of SMILES strings
        r_groups: pandas DataFrame of R group labels, metal and sidechain SMILES strings

    Returns:
        pandas DataFrame of SMILES strings with R groups replaced with arbitrary metals
    """
    r_labels: pd.DataFrame = r_groups.set_index("label")
    structures["SMILES"] = [replace_in_string(smiles, r_labels["metal"]) for smiles in structures["SMILES"]]
    return structures


def replace_in_molecule(mol: Mol, labels: pd.Series) -> str:
    """
    Replaces arbitrary metals in RDKit Mol object with substructure replacement.

    Args:
        mol: RDKit Mol object
        labels: pandas Series of sidechain SMILES strings

    Returns:
        SMILES string with arbitrary metals replaced with sidechains
    """
    for metal, sidechain in labels.items():
        if metal in Chem.MolToSmiles(mol):
            products = AllChem.ReplaceSubstructs(
                mol,
                Chem.MolFromSmarts(metal),
                Chem.MolFromSmarts(sidechain),
                replaceAll=True,
            )
            mol = products[0]
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def replace_arbitrary_with_sidechain(structures: pd.DataFrame, r_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Replace arbitrary metals with sidechain SMILES strings in SMILES strings.

    Args:
        structures: pandas DataFrame of SMILES strings
        r_groups: pandas DataFrame of R group labels, metal and sidechain SMILES strings

    Returns:
        pandas DataFrame of SMILES strings with arbitrary metals replaced with sidechain SMILES strings
    """
    r_metals: pd.DataFrame = r_groups.set_index("metal")

    molecules: list[Mol] = []
    for smiles in structures["SMILES"]:
        mol = Chem.MolFromSmarts(smiles)
        # Sanitize SMARTS
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        mol.GetRingInfo().NumRings()
        molecules.append(mol)

    structures["SMILES"] = [replace_in_molecule(mol, r_metals["SMILES"]) for mol in molecules]
    return structures


def man_main():
    raw_data = DATASETS / "Min_2020_n558" / "raw"
    r_groups_file = raw_data.parent / "cleaned R groups.csv"
    r_groups = pd.read_csv(r_groups_file)

    # for material in ["Donor", "Acceptor"]:
    for material in ["Acceptor"]:
        master_file = raw_data / f"min_{material.lower()}s_smiles_master_EDITED.csv"
        master_df = pd.read_csv(master_file)
        reference_file = raw_data / f"reference {material.lower()}s.csv"
        reference_df = pd.read_csv(reference_file)

        clean_df: pd.DataFrame = clean_structures(material, master_df, reference_df)
        clean_smiles_df: pd.DataFrame = replace_arbitrary_with_sidechain(
            replace_r_with_arbitrary(clean_df, r_groups),
            r_groups
        )

        clean_file = raw_data.parent / f"cleaned {material.lower()}s.csv"
        clean_smiles_df.to_csv(clean_file, index=False)


if __name__ == "__main__":
    man_main()
