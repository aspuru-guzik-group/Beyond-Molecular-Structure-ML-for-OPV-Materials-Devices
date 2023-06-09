from os import error
import pkg_resources
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import csv
import pandas as pd

pd.set_option("display.max_columns", 20)

MASTER_DONOR_CSV = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/preprocess/OPV_Min/min_donors_smiles_master_EDITED.csv"
)
MASTER_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/preprocess/OPV_Min/min_acceptors_smiles_master_EDITED.csv"
)

CLEAN_DONOR_CSV = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/preprocess/OPV_Min/clean_min_donors.csv"
)

CLEAN_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/preprocess/OPV_Min/clean_min_acceptors.csv"
)

# From OPV Google Drive
OPV_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs",
    "data/process/OPV_Min/Machine Learning OPV Parameters - device_params.csv",
)
OPV_DONOR_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/process/OPV_Min/Machine Learning OPV Parameters - Donors.csv"
)
OPV_ACCEPTOR_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs",
    "data/process/OPV_Min/Machine Learning OPV Parameters - Acceptors.csv",
)

MASTER_ML_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
)


class DonorClean:
    """
    Class containing functions that preprocesses the donor data.
    """

    def __init__(self, master_donor, opv_donor):
        """
        Instantiates class with appropriate data

        Args:
            master_donor: path to master donor data (from Excel File)
            opv_donor: path to ML data downloaded from Google Drive shared w/ UCSB

        Returns:
            None
        """
        self.master_donor = pd.read_csv(master_donor)
        self.opv_donor = pd.read_csv(opv_donor)

    def clean_donor(self, clean_donor):
        """
        Function that creates a .csv with the correct SMILES, substituted R groups

        Args:
            clean_donor: path to processed donors

        Returns:
            .csv with columns: | Donor | SMILES | SMILES (w/ substituted R) | Big_SMILES | SELFIES
        """
        headers = [
            "Donor",
            "SMILES",
            "SMILES w/o R_group replacement",
            "SMILES w/o R_group",
            "Big_SMILES",
            "SELFIES",
        ]
        clean_df = pd.DataFrame(columns=headers)
        opv_labels = list(self.opv_donor["name"])
        # Collect statistics about number of errors
        total_donors = 0
        missing_donors = 0  # donors not in the OPV but in the data
        error_donors = 0  # number of donors with any error
        for index, row in self.master_donor.iterrows():
            # Ignore structures with error comments
            error_list = [
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
                        "Donor": row["Name"],
                        "SMILES": row["R_grp_SMILES"],
                        "SMILES w/o R_group replacement": row["R_grp_SMILES"],
                        "SMILES w/o R_group": " ",
                        "Big_SMILES": " ",
                        "SELFIES": " ",
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

    def replace_r(self, clean_donor):
        """Replace R group in the clean_min_donors.csv

        Args:
            clean_donor: path to processed donors

        Returns:
            SMILES column contains donors with replaced R groups
        """
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
        clean_df = pd.read_csv(clean_donor)
        donor_smi_list = clean_df["SMILES"]
        index = 0
        for smi in donor_smi_list:
            for r in patts:
                smi = smi.replace(r, patts[r])
            smi = Chem.CanonSmiles(smi)
            clean_df.at[index, "SMILES"] = smi
            index += 1
        clean_df.to_csv(clean_donor, index=False)

    def delete_r(self, clean_donor):
        """
        Function that deletes R group.

        Args:
            clean_donor: path to processed donors

        Returns:
            SMILES w/o R group column contains donors with deleted R groups
        """
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
        clean_df = pd.read_csv(clean_donor)
        donor_smi_r_list = clean_df["SMILES w/o R_group replacement"]
        index = 0
        for smi in donor_smi_r_list:
            for r in patts:
                smi = smi.replace(r, "*")
            # check if SMILES is valid
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            smi = Chem.CanonSmiles(smi)
            clean_df.at[index, "SMILES w/o R_group"] = smi
            index += 1
        clean_df.to_csv(clean_donor, index=False)

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

    def canon_smi(self, clean_donor):
        """
        Function to canonicalize all the smiles in donor_df to rid of %10...
        Args:
            clean_donor: path to processed donors

        Returns:
            SMILES column that are canonicalized properly
        """
        clean_df = pd.read_csv(clean_donor)
        donor_smi_list = clean_df["SMILES"]

        index = 0
        for smi in donor_smi_list:
            donor_smi = Chem.CanonSmiles(smi)
            clean_df.at[index, "SMILES"] = donor_smi
            index += 1

        clean_df.to_csv(clean_donor, index=False)


class AcceptorClean:
    """
    Class containing functions that process master acceptors .csv file and collect statistics from it
    """

    def __init__(self, master_acceptor, opv_acceptor):
        """
        Instantiates class with appropriate data

        Args:
            master_acceptor: path to master acceptor data (from Excel File)
            opv_acceptor: path to ML data downloaded from Google Drive shared w/ UCSB

        Returns:
            None
        """
        self.master_acceptor = pd.read_csv(master_acceptor)
        self.opv_acceptor = pd.read_csv(opv_acceptor)

    def clean_acceptor(self, clean_acceptor):
        """
        Function that creates a .csv with the correct SMILES, substituted R groups

        Args:
            clean_acceptor: path to processed acceptor

        Returns:
            .csv with columns: | Acceptor | SMILES | SMILES (w/ substituted R) | Big_SMILES | SELFIES
        """
        headers = [
            "Acceptor",
            "SMILES",
        ]
        clean_df = pd.DataFrame(columns=headers)
        opv_labels = list(self.opv_acceptor["name"])
        # Collect statistics about number of errors
        total_acceptors = 0
        missing_acceptors = 0  # acceptors not in the OPV but in the data
        error_acceptors = 0  # number of acceptors with any error
        for index, row in self.master_acceptor.iterrows():
            # Ignore structures with error comments
            error_list = [
                "error",
                "wrong structure",
            ]
            if row["Comments (Stanley)"] not in error_list:
                clean_df = clean_df.append(
                    {
                        "Acceptor": row["Name_Stanley"],
                        "SMILES": row["SMILE"],
                    },
                    ignore_index=True,
                )
            else:
                error_acceptors += 1
            if row["Name_Stanley"] not in opv_labels:
                missing_acceptors += 1
            total_acceptors += 1
        print(clean_df.head())
        print("---------------")
        print(
            "missing: ",
            missing_acceptors,
            "error: ",
            error_acceptors,
            "total: ",
            total_acceptors,
        )
        clean_df.to_csv(clean_acceptor, index=False)

    def replace_r(self, clean_acceptor):
        """
        Replace R group in the clean_min_acceptors.csv

        Args:
            clean_donor: path to processed acceptors

        Returns:
            SMILES column contains acceptors with replaced R groups
        """
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
        clean_df = pd.read_csv(clean_acceptor)
        acceptor_smi_list = clean_df["SMILES"]
        index = 0
        for smi in acceptor_smi_list:
            for r in patts:
                smi = smi.replace(r, patts[r])
            smi = Chem.CanonSmiles(smi)
            clean_df.at[index, "SMILES"] = smi
            index += 1
        clean_df.to_csv(clean_acceptor, index=False)

    def delete_r(self, clean_acceptor):
        """
        Function that deletes R group

        Args:
            clean_acceptor: path to processed acceptors

        Returns:
            SMILES w/o R group column contains acceptors with deleted R groups
        """
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
        clean_df = pd.read_csv(clean_acceptor)
        acceptor_smi_r_list = clean_df["SMILES w/o R_group replacement"]
        index = 0
        for smi in acceptor_smi_r_list:
            for r in patts:
                smi = smi.replace(r, "*")
            # check if SMILES is valid
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            smi = Chem.CanonSmiles(smi)
            clean_df.at[index, "SMILES w/o R_group"] = smi
            index += 1
        clean_df.to_csv(clean_acceptor, index=False)

    def canon_smi(self, clean_acceptor):
        """
        Function to canonicalize all the smiles in acceptor_df to rid of %10...

        Args:
            clean_acceptor: path to processed acceptors

        Returns:
            SMILES column that are canonicalized properly
        """
        clean_df = pd.read_csv(clean_acceptor)
        acceptor_smi_list = clean_df["SMILES"]

        index = 0
        for smi in acceptor_smi_list:
            acceptor_smi = Chem.CanonSmiles(smi)
            clean_df.at[index, "SMILES"] = acceptor_smi
            index += 1

        clean_df.to_csv(clean_acceptor, index=False)


class DAPairs:
    """
    Class containing functions that prepare the Donor-Acceptor Pairs with OPV Data for ML
    """

    def __init__(self, opv_data, donors_data, acceptors_data):
        """
        Instantiates class with appropriate data

        Args:
            opv_data: path to ML data downloaded from Google Drive shared w/ UCSB
            donors_data: path to preprocessed donor data
            acceptors_data: path to preprocessed acceptor data

        Returns:
            None
        """
        self.opv_data = pd.read_csv(opv_data)
        self.donors = pd.read_csv(donors_data)
        self.acceptors = pd.read_csv(acceptors_data)

    def create_master_csv(self, master_csv_path):
        """
        Function that creates the .csv file used for ML project
        
        Args:
            master_csv_path: path to the processed master file for future data representation modifications

        Returns:
            .csv file with columns: | Donor | Donor Input Representations | Acceptor | Acceptor Input Representations | PCE(%) | Voc(V) | Jsc(mA cm^-2) | FF(%) |
        """
        headers = [
            "Donor",
            "Donor_SMILES",
            "Donor_Big_SMILES",
            "Donor_SELFIES",
            "Acceptor",
            "Acceptor_SMILES",
            "Acceptor_Big_SMILES",
            "Acceptor_SELFIES",
            "HOMO_D_eV",
            "LUMO_D_eV",
            "HOMO_A_eV",
            "LUMO_A_eV",
            "D_A_ratio_m_m",
            "solvent",
            "total_solids_conc_mg_mL",
            "solvent_additive",
            "solvent_additive_conc_v_v_percent",
            "active_layer_thickness_nm",
            "annealing_temperature",
            "hole_contact_layer",
            "electron_contact_layer",
            "hole_mobility_blend",
            "electron_mobility_blend",
            "PCE_percent",
            "calc_PCE_percent",
            "Voc_V",
            "Jsc_mA_cm_pow_neg2",
            "FF_percent",
        ]
        master_df = pd.DataFrame(columns=headers)
        donor_avail = list(self.donors["Donor"])
        acceptor_avail = list(self.acceptors["Acceptor"])
        # iterate through data_from_min.csv for donor-acceptor pairs
        for index, row in self.opv_data.iterrows():
            # only keep the rows with available donor and acceptor molecules from clean donors and acceptors
            if (row["Donor Molecule"] in donor_avail) and (
                row["Acceptor Molecule"] in acceptor_avail
            ):
                # get SMILES of donor and acceptor
                donor_row = self.donors.loc[
                    self.donors["Donor"] == row["Donor Molecule"]
                ]
                donor_smile = donor_row["SMILES"].values[0]
                acceptor_row = self.acceptors.loc[
                    self.acceptors["Acceptor"] == row["Acceptor Molecule"]
                ]
                acceptor_smile = acceptor_row["SMILES"].values[0]

                # get SMILES w/o Rgrp replacement of donor and acceptor
                # donor_smile_wo_rgrp_replace = donor_row[
                #     "SMILES w/o R_group replacement"
                # ].values[0]
                # acceptor_smile_wo_rgrp_replace = acceptor_row[
                #     "SMILES w/o R_group replacement"
                # ].values[0]

                # get SMILES w/o Rgrp of donor and acceptor
                # donor_smile_wo_rgrp = donor_row["SMILES w/o R_group"].values[0]
                # acceptor_smile_wo_rgrp = acceptor_row["SMILES w/o R_group"].values[0]

                # get SELFIES of donor and acceptor
                donor_selfies = donor_row["SELFIES"].values[0]
                acceptor_selfies = acceptor_row["SELFIES"].values[0]

                # get BigSMILES of donor and acceptor
                donor_bigsmile = donor_row["Big_SMILES"].values[0]
                acceptor_bigsmile = acceptor_row["Big_SMILES"].values[0]

                # strip whitespace of solvent
                solvent = row["solvent"]
                if isinstance(row["solvent"], str):
                    solvent = solvent.strip()

                # strip whitespace of hole_contact_layer
                hole_contact_layer = row["hole_contact_layer"]
                if isinstance(row["hole_contact_layer"], str):
                    hole_contact_layer = hole_contact_layer.strip()

                # append new donor-acceptor pair to masters dataframe
                master_df = master_df.append(
                    {
                        "Donor": row["Donor Molecule"],
                        "Donor_SMILES": donor_smile,
                        "Donor_Big_SMILES": donor_bigsmile,
                        "Donor_SELFIES": donor_selfies,
                        "Acceptor": row["Acceptor Molecule"],
                        "Acceptor_SMILES": acceptor_smile,
                        "Acceptor_Big_SMILES": acceptor_bigsmile,
                        "Acceptor_SELFIES": acceptor_selfies,
                        "HOMO_D_eV": row["HOMO_D_eV"],
                        "LUMO_D_eV": row["LUMO_D_eV"],
                        "HOMO_A_eV": row["HOMO_A_eV"],
                        "LUMO_A_eV": row["LUMO_A_eV"],
                        "D_A_ratio_m_m": row["D_A_ratio_m_m"],
                        "solvent": solvent,
                        "total_solids_conc_mg_mL": row["total_solids_conc_mg_mL"],
                        "solvent_additive": row["solvent_additive"],
                        "solvent_additive_conc_v_v_percent": row[
                            "solvent_additive_conc (% v/v)"
                        ],
                        "active_layer_thickness_nm": row[
                            "active_layer_thickness_nm"
                        ],
                        "annealing_temperature": row[
                            "temperature of thermal annealing (leave gap if not annealed)"
                        ],
                        "hole_contact_layer": hole_contact_layer,
                        "electron_contact_layer": row["electron_contact_layer"],
                        "hole_mobility_blend": row[
                            "hole_mobility_blend"
                        ],
                        "electron_mobility_blend": row[
                            "electron_mobility_blend"
                        ],
                        "PCE_percent": row["PCE_percent"],
                        "calc_PCE_percent": row["calc_PCE"],
                        "Voc_V": row["Voc_V"],
                        "Jsc_mA_cm_pow_neg2": row["Jsc_mA_cm_pow_neg2"],
                        "FF_percent": row["FF_percent"],
                    },
                    ignore_index=True,
                )
        master_df.to_csv(master_csv_path)

    def fill_empty_values(self, master_csv_path):
        """
        Function that fills in NaN values because it is reasonable.
        Ex. solvent_additive does not have to be present. Therefore, "N/A" should replace NaN        
        
        Args:
            master_csv_path: path to the processed master file for future data representation modifications

        Returns:
            .csv file with filled reasonable values
        """
        master_data = pd.read_csv(master_csv_path)
        column_names = master_data.columns

        # columns that can have NaN values
        idx_solvent_additive = 16
        idx_solvent_additive_conc = 17
        idx_annealing_temp = 19
        null_master_data = master_data.isna()

        # placeholders
        # N/A for string values, -1 for int,float values
        for index, row in master_data.iterrows():
            if null_master_data.at[index, column_names[idx_solvent_additive]] == True:
                master_data.at[index, column_names[idx_solvent_additive]] = "N/A"
            if (
                null_master_data.at[index, column_names[idx_solvent_additive_conc]]
                == True
            ):
                master_data.at[index, column_names[idx_solvent_additive_conc]] = -1
            if null_master_data.at[index, column_names[idx_annealing_temp]] == True:
                master_data.at[index, column_names[idx_annealing_temp]] = -1

        master_data.to_csv(master_csv_path, index=False)

    def filter_master_csv(
        self, master_csv_path, filtered_master_csv_path, column_idx_list
    ):
        """
        Function that filters the .csv file for rows that contain ONLY present values in the important columns:
        
        
        Args:
            master_csv_path: path to the processed master file for future data representation modifications
            filtered_master_csv_path: path to the filtered master file for future data representation modifications

        Returns:
            .csv file with values that contain all of the parameters
        """
        master_data = pd.read_csv(master_csv_path)
        column_names = master_data.columns
        columns_dict = {}
        index = 0
        while index < len(column_names):
            columns_dict[column_names[index]] = index
            index += 1

        print(columns_dict)

        important_columns = []
        for idx in column_idx_list:
            important_columns.append(column_names[idx])

        # select important columns
        filter_master_data = master_data[important_columns]

        # drop rows if there are any NaN values
        filter_master_data = filter_master_data.dropna(axis=0, how="any")

        filter_master_data.to_csv(filtered_master_csv_path, index=False)

    def convert_str_to_float(self, master_csv_path):
        """
        Converts D_A_ratio and solvent_additive_conc string representation to float
        """
        master_data = pd.read_csv(master_csv_path)
        for index, row in master_data.iterrows():
            ratio_data = master_data.at[index, "D_A_ratio_m_m"]
            solvent_add_conc_data = master_data.at[
                index, "solvent_additive_conc_v_v_percent"
            ]
            # ratio data
            if isinstance(ratio_data, str):
                ratio_list = ratio_data.split(":")
                donor_ratio = float(ratio_list[0])
                acceptor_ratio = float(ratio_list[1])
                float_ratio_data = donor_ratio / acceptor_ratio
                master_data.at[index, "D_A_ratio_m_m"] = round(float_ratio_data, 3)

            # solvent_additive_conc data
            master_data.at[index, "solvent_additive_conc_v_v_percent"] = float(
                solvent_add_conc_data
            )

        master_data.to_csv(master_csv_path, index=False)


# Step 1
# donors = DonorClean(MASTER_DONOR_CSV, OPV_DONOR_DATA)
# donors.clean_donor(CLEAN_DONOR_CSV)

# # # # Step 1b
# donors.replace_r(CLEAN_DONOR_CSV)

# # # # # # # Step 1c - do not include for fragmentation
# # # # # donors.remove_methyl(CLEAN_DONOR_CSV)

# # # # # # Step 1d - canonSMILES to remove %10-%100
# donors.canon_smi(CLEAN_DONOR_CSV)

# # # # # Step 1
# acceptors = AcceptorClean(MASTER_ACCEPTOR_CSV, OPV_ACCEPTOR_DATA)
# acceptors.clean_acceptor(CLEAN_ACCEPTOR_CSV)

# # Step 1b
# acceptors.replace_r(CLEAN_ACCEPTOR_CSV)

# # # # # Step 1d - canonSMILES to remove %10-%100
# acceptors.canon_smi(CLEAN_ACCEPTOR_CSV)

# Step 1e - Fragmentation
# donors.delete_r(CLEAN_DONOR_CSV)
# acceptors.delete_r(CLEAN_ACCEPTOR_CSV)

# Step 2 - ERROR CORRECTION (fill in missing D/A)

# Step 3 - smiles_to_bigsmiles.py & smiles_to_selfies.py


# Step 4
pairings = DAPairs(OPV_DATA, CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)
pairings.create_master_csv(MASTER_ML_DATA)
pairings.create_master_csv(MASTER_ML_DATA_PLOT)

# # # Step 4b - Convert STR -> FLOAT
pairings.convert_str_to_float(MASTER_ML_DATA)
pairings.convert_str_to_float(MASTER_ML_DATA_PLOT)

# Step 4c - Fill empty values for Thermal Annealing, and solvent_additives
pairings.fill_empty_values(MASTER_ML_DATA)

# Step 4d - Remove anomalies!
# Go to _ml_for_opvs > data > error_correction > remove_anomaly.py

# Step 5
# Go to rdkit_frag.py (if needed)
