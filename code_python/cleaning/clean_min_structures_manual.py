import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


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
        "SMILES w/o R_group",
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
            # NOTE: add BigSMILES, SELFIES here
            clean_df = clean_df.append(
                {
                    "Donor":                          row["Name"],
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


def replace_r_with_arbitrary(r_group_patterns: dict[str, str], structures: pd.DataFrame) -> pd.DataFrame:
    """
    Replace R group in the clean_min_acceptors.csv

    Args:
        clean_donor: path to processed acceptors

    Returns:
        SMILES column contains acceptors with replaced R groups
    """
    # New R group substitution pattern
    new_patts = {}
    atomic_num = 21
    for k, v in r_group_patterns.items():
        # Only provides the transition metals and lanthanides
        if atomic_num == 31:
            atomic_num = 39
        elif atomic_num == 49:
            atomic_num = 57
        elif atomic_num == 81:
            atomic_num = 89
        mol = Chem.MolFromSmarts(f"[#{atomic_num}]")
        smi = Chem.MolToSmiles(mol)
        new_patts[smi] = k
        atomic_num += 1

    for index, row in structures.iterrows():
        smi = structures.at[index, "SMILES"]
        for key in new_patts.keys():
            smi = smi.replace(new_patts[key], key)
        structures.at[index, "SMILES"] = smi
    return structures


def replace_arbitrary_with_sidechain(r_group_patterns: dict[str, str], structures: pd.DataFrame) -> pd.DataFrame:
    # New R group substitution pattern
    # ATTN: What is this doing? Something about fixing patterns?
    new_patts = {}
    atomic_num = 21
    for k, v in r_group_patterns.items():
        # Only provides the transition metals and lanthanides
        if atomic_num == 31:
            atomic_num = 39
        elif atomic_num == 49:
            atomic_num = 57
        elif atomic_num == 81:
            atomic_num = 89
        mol = Chem.MolFromSmarts(f"[#{atomic_num}]")
        smi = Chem.MolToSmiles(mol)
        new_patts[smi] = v
        atomic_num += 1

    # smiles: pd.Series = structures["SMILES"]
    # index = 0
    for index, row in structures.iterrows():
        smi = row["SMILES"]
        mol = Chem.MolFromSmarts(smi)
        # Sanitize SMARTS
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        mol.GetRingInfo().NumRings()

        for r in new_patts:
            if r in smi:
                products = AllChem.ReplaceSubstructs(
                    mol,
                    Chem.MolFromSmarts(r),
                    Chem.MolFromSmarts(new_patts[r]),
                    replaceAll=True,
                )
                mol = products[0]
        smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        structures.at[index, "SMILES"] = smi
        # index += 1

    return structures


if __name__ == "__main__":
    # TODO: Make sure R groups JSON has all the R groups!!!
    clean_structures("Donor", "clean_min_donors.csv", "donors.csv")
    replace_r_with_arbitrary("donors.csv", "clean_min_donors.csv")
    replace_arbitrary_with_sidechain("donors.csv", "clean_min_donors.csv")
