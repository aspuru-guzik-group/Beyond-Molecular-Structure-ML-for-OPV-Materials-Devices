from pathlib import Path

import pandas as pd

structure_errors: set[str] = {
    "not in drive, not in literature",
    "error",
    "wrong structure",
    "same name different structure",
    "not in literature",
}


class IngestDonor:
    string_representations: list[str] = [
        "Donor",
        "SMILES",
        "SMILES w/o R_group replacement",
        "SMILES w/o R_group",
        "Big_SMILES",
        "SELFIES",
    ]

    def __init__(self, data):
        self.data = data

    # NOTE: Only done once
    def clean_donor(self, clean_donor: Path):
        """
        Function that creates a .csv with the correct SMILES, substituted R groups

        Args:
            clean_donor: path to processed donors

        Returns:
            .csv with columns: | Donor | SMILES | SMILES (w/ substituted R) | Big_SMILES | SELFIES
        """
        clean_df: pd.DataFrame = pd.DataFrame(columns=self.string_representations)
        opv_labels: list[str] = list(self.opv_donor["name"])

        # Collect statistics about number of errors
        total_donors: int = 0
        missing_donors: int = 0  # donors not in the OPV but in the data
        error_donors: int = 0  # number of donors with any error
        for index, row in self.master_donor.iterrows():
            # Ignore structures with error comments
            if row["Comments (Stanley)"] not in structure_errors:
                # NOTE: add BigSMILES, SELFIES here
                clean_df = clean_df.append(
                    {
                        "Donor":                          row["Name"],
                        "SMILES":                         row["R_grp_SMILES"],
                        "SMILES w/o R_group replacement": row["R_grp_SMILES"],
                        "SMILES w/o R_group":             " ",
                        "Big_SMILES":                     " ",
                        "SELFIES":                        " ",
                    },
                    ignore_index=True,
                )
            else:
                error_donors += 1
            if row["Name"] not in opv_labels:
                missing_donors += 1
            total_donors += 1
        print(clean_df.head())
        print("---------------")
        print(
            "missing: ",
            missing_donors,
            "error: ",
            error_donors,
            "total: ",
            total_donors,
        )
        clean_df.to_csv(clean_donor, index=False)

    # ATTN: ONly done once
    def replace_r(self, clean_donor):
        """Replace R group in the clean_min_donors.csv

        Args:
            clean_donor: path to processed donors

        Returns:
            SMILES column contains donors with replaced R groups
        """
        patts = {
            "[R1]":  "CC(CCCCCC)CCCCCCCC",
            "[R2]":  "CCCCCCCC",
            "[R3]":  "[Si](CCC)(CCC)(CCC)",
            "[R4]":  "CC(CC)CCCC",
            "[R5]":  "SCCCCCCCCCCCC",
            "[R6]":  "CC(CCCCCCCC)CCCCCCCCCC",
            "[R7]":  "SCC(CCCCCC)CCCC",
            "[R8]":  "[Si](CC)(CC)(CC)",
            "[R9]":  "[Si](C(C)C)(C(C)C)C(C)C",
            "[R10]": "[Si](CCCC)(CCCC)(CCCC)",
            "[R11]": "[Si](C)(C)CCCCCCCC",
            "[R12]": "SCCCCC=C",
            "[R13]": "SCC4CCCCC4",
            "[R14]": "CCCCCC",
            "[R15]": "CCCCCCCCCC",
            "[R18]": "CCCCC",
            "[R19]": "CCCCCCCCCCCCCCCC",
            "[R20]": "CCCCCCCCCCC",
            "[R21]": "C(CCCCCCCCC)CCCCCCC",
            "[R23]": "CC(CCCCCCCCCCCC)CCCCCCCCCC",
            "[R24]": "COCCOC",
            "[R25]": "CC(CCCCCCCCCCC)CCCCCCCCC",
            "[R26]": "CCC",
            "[R27]": "CCCC",
            "[R28]": "CCC(CC)CCCC",
            "[R29]": "CCCC(CC)CCCC",
        }
        # New R group substitution pattern
        new_patts = {}
        atomic_num = 21
        for k, v in patts.items():
            # Only provides the transition metals and lanthanides
            if atomic_num == 31:
                atomic_num = 39
            elif atomic_num == 49:
                atomic_num = 57
            elif atomic_num == 81:
                atomic_num = 89
            mol = Chem.MolFromSmarts("[#{}]".format(atomic_num))
            smi = Chem.MolToSmiles(mol)
            new_patts[smi] = v
            atomic_num += 1

        clean_df = pd.read_csv(clean_donor)
        donor_smi_list = clean_df["SMILES"]

        index = 0
        for smi in donor_smi_list:
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
            clean_df.at[index, "SMILES"] = smi
            index += 1

        clean_df.to_csv(clean_donor, index=False)

    # ATTN: ??
    def remove_methyl(self, clean_donor):
        """
        Function that checks the number of methyl groups and removes them on donor molecules.
        Only the starting and ending methyl group are deleted.

        Args:
            clean_donor: path to processed donors

        Returns:
            SMILES column contains donors with end methyl groups removed.
        """
        clean_df = pd.read_csv(clean_donor)
        donor_smi_list = clean_df["SMILES"]
        index = 0
        for smi in donor_smi_list:
            donor_mol = Chem.MolFromSmiles(smi)
            n_methyl = 0
            donor_edmol = Chem.EditableMol(donor_mol)
            remove_idx = []
            for atom in donor_edmol.GetMol().GetAtoms():
                if atom.GetDegree() == 1 and atom.GetAtomicNum() == 6:
                    for neighbour in atom.GetNeighbors():
                        if neighbour.GetIsAromatic():
                            n_methyl += 1
                            atom_idx = atom.GetIdx()
                            remove_idx.append(atom_idx)

            # performs action all at once so index doesn't change
            donor_edmol.BeginBatchEdit()
            for idx in remove_idx:
                donor_edmol.RemoveAtom(idx)
            donor_edmol.CommitBatchEdit()

            # Draw.ShowMol(donor_mol, size=(600, 600))
            # Draw.ShowMol(donor_edmol.GetMol(), size=(600, 600))
            donor_smi = Chem.MolToSmiles(donor_edmol.GetMol())
            clean_df.at[index, "SMILES"] = donor_smi
            index += 1

        clean_df.to_csv(clean_donor, index=False)
