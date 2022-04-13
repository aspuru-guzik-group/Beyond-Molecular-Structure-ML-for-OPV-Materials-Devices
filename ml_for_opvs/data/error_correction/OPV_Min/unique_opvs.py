from dataclasses import MISSING
from enum import unique
from typing import List
import pkg_resources
import pandas as pd

from ml_for_opvs.data.preprocess.OPV_Min.clean_donors_acceptors import (
    DonorClean,
    AcceptorClean,
)

OPV_MIN = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/process/OPV_Min/Machine Learning OPV Parameters - device_params.csv",
)

# OPV data after pre-processing
OPV_CLEAN = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

CLEAN_DONOR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/clean_min_donors.csv"
)

CLEAN_ACCEPTOR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/clean_min_acceptors.csv"
)

MISSING_SMI_DONOR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/missing_smi_donors.csv"
)
MISSING_SMI_ACCEPTOR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/missing_smi_acceptors.csv"
)

CHEMDRAW_DONOR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/min_donors_smiles_master_EDITED.csv"
)

CHEMDRAW_ACCEPTOR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/min_acceptors_smiles_master_EDITED.csv"
)

COMPARE_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/compare_sheets_chemdraw.csv"
)


class UniqueOPVs:
    """
    Class that contains functions to: 
    i) find unique donors and acceptors
    ii) filter for difference in lists from min and clean data
    iii) filter for donors and acceptors without SMILES
    iv) produce .csv file with missing donor and acceptors
    
    Manually add the SMILES for missing donors and acceptors
    v) add new SMILES to clean_donor or clean_acceptor .csv files
    """

    def __init__(self, opv_min, opv_clean):
        """
        Instantiates class with appropriate data

        Args:
            opv_min: path to opv data from original Min file
            opv_clean: path to opv data from pre-processing

        Returns:
            None
        """
        self.opv_min = pd.read_csv(opv_min)
        self.opv_clean = pd.read_csv(opv_clean)

    def unique_list(self, mol_type, data_option) -> List:
        """
        Returns list of unique donors or acceptors

        Args:
            mol_type: choose between donor or acceptor molecules
            data_option: choose between opv_min or opv_clean data

        Returns:
            List of unique donor or acceptors from Min. data or clean data
        """
        if data_option == "min":  # take data from min
            data = self.opv_min
            label = "Molecule"
        elif data_option == "clean":  # take data from pre-processed data
            data = self.opv_clean

        unique_mols = []
        if mol_type == "D":  # unique list of donors
            if data_option == "min":
                label = "Donor " + label
            else:
                label = "Donor"
            for mol in data[label]:
                if mol not in unique_mols:
                    unique_mols.append(mol)
        elif mol_type == "A":  # unique list of acceptors
            if data_option == "min":
                label = "Acceptor " + label
            else:
                label = "Acceptor"
            for mol in data[label]:
                if mol not in unique_mols:
                    unique_mols.append(mol)

        return unique_mols

    def filter(self, min_list, clean_list) -> List:
        """
        Returns list of missing donors or acceptors
        NOTE: MUST BE BOTH DONORS or ACCEPTORS (Do not compare min_donors and clean_acceptors)

        Args:
            min_list: list of donors or acceptors from Min. data
            clean_list: list of donors or acceptors from clean data

        Returns:
            missing_list: list of missing donors or acceptors
        """
        missing_list = list(set(min_list) - set(clean_list))
        return missing_list

    def filter_SMILES(self, missing_list, mol_type) -> List:
        """
        Returns list of missing donors or acceptors without SMILES

        Args:
            missing_list: list of missing donors or acceptors
            mol_type: choose between donor or acceptor molecules

        Returns:
            missing_smi_list: list of missing donors or acceptors without SMILES
        """
        if mol_type == "D":  # clean donors
            clean_data = pd.read_csv(CLEAN_DONOR)
        elif mol_type == "A":  # clean acceptors
            clean_data = pd.read_csv(CLEAN_ACCEPTOR)

        clean_list = list(clean_data["Label"])

        missing_smi_list = list(set(missing_list) - set(clean_list))
        return missing_smi_list

    def create_missing_csv(self, missing_smi_list, mol_type):
        """
        Produces .csv file with missing donors and acceptors

        Args:
            missing_smi_list: list of missing donors or acceptors without SMILES
            mol_type: choose between donor or acceptor molecules

        Returns:
            .csv file with missing data
        """
        if mol_type == "D":
            label = "Donor"
            path = MISSING_SMI_DONOR
        elif mol_type == "A":
            label = "Acceptor"
            path = MISSING_SMI_ACCEPTOR

        missing_df = pd.DataFrame(columns=[label, "SMILES"])

        missing_df[label] = missing_smi_list
        missing_df["SMILES"] = ""
        missing_df.to_csv(path, index=False)

    def clean_up_missing(
        self,
        missing_smi_donor=MISSING_SMI_DONOR,
        missing_smi_acceptor=MISSING_SMI_ACCEPTOR,
    ):
        """
        Updates missing donor and acceptor with canonical SMILES and replaced R

        Args:
            missing_smi_donor: path to data for missing SMILES donor
            missing_smi_acceptor: path to data for missing SMILES acceptor

        Returns:
            .csv file with clean missing data
        """
        donor_clean = DonorClean(
            OPV_MIN, MISSING_SMI_DONOR
        )  # placeholders, useless values
        acceptor_clean = AcceptorClean(
            OPV_MIN, MISSING_SMI_ACCEPTOR
        )  # placeholders, useless values

        donor_clean.replace_r(missing_smi_donor)
        acceptor_clean.replace_r(missing_smi_acceptor)

    def concat_missing_and_clean(self, missing_data, clean_data, mol_type):
        """
        Concatenate missing and clean data.

        Args:
            missing_data: path to missing SMILES data (filled-in)
            clean_data: path to clean SMILES data
            mol_type: choose between donor or acceptor molecules
            NOTE: must be both donor or both acceptor

        Returns:
            .csv file with clean data and missing data combined together
        """
        missing_df = pd.read_csv(missing_data)
        clean_df = pd.read_csv(clean_data)

        # assuming missing and clean data have no overlaps, we can just concatenate
        if mol_type == "D":
            label = "Donor"
        elif mol_type == "A":
            label = "Acceptor"

        new_clean_df = pd.concat([missing_df, clean_df])

        new_clean_df.to_csv(clean_data, index=False)

    def compare(self, compare_path, chemdraw_d, chemdraw_a):
        """
        Compares D-A pairs from Google Sheets and ChemDraw file.
        Produces new .csv with overlap D-A pairs

        Args:
            compare_path: path to .csv for storing comparison of Google Sheets and ChemDraw donors/acceptors
            chemdraw_d: path to excel sheet with all donors from ChemDraw file
            chemdraw_a: path to excel sheet with all acceptors from ChemDraw file

        Returns:
            .csv file with missing data from Google Sheets and ChemDraw
            columns = ["Label", "Type", "Missing_From", "ChemDraw_Comments"]
            Label: Label for donor/acceptor molecule
            Type: indicates whether donor or acceptor
            Missing_From: Is missing from the Google Sheets or ChemDraw
            ChemDraw_Comments: If there are conflicts or error msgs in the ChemDraw, it will be shown.
        """
        compare_df = pd.DataFrame(
            columns=["Label", "Type", "Missing_From", "ChemDraw_Comments"]
        )
        compare_df["Label"] = ""
        compare_df["Type"] = ""
        compare_df["Missing_From"] = ""
        compare_df["ChemDraw_Comments"] = ""

        chemdraw_donor_df = pd.read_csv(chemdraw_d)
        chemdraw_acceptor_df = pd.read_csv(chemdraw_a)
        chemdraw_donors = list(chemdraw_donor_df["Name"])
        chemdraw_acceptors_aaron = chemdraw_acceptor_df["Name_Aaron"]

        sheets_donors = self.unique_list("D", "min")
        sheets_acceptors = self.unique_list("A", "min")

        donors_missing_from_sheets = list(set(chemdraw_donors) - set(sheets_donors))
        donors_missing_from_chemdraw = list(set(sheets_donors) - set(chemdraw_donors))

        acceptors_missing_from_sheets = list(
            set(chemdraw_acceptors_aaron) - set(sheets_acceptors)
        )
        acceptors_missing_from_chemdraw = list(
            set(sheets_acceptors) - set(chemdraw_acceptors_aaron)
        )

        df_index = 0
        index = 0
        while index < len(donors_missing_from_sheets):
            compare_df.at[df_index, "Label"] = donors_missing_from_sheets[index]
            compare_df.at[df_index, "Type"] = "donor"
            compare_df.at[df_index, "Missing_From"] = "Google Sheets"
            index += 1
            df_index += 1

        index = 0
        while index < len(donors_missing_from_chemdraw):
            compare_df.at[df_index, "Label"] = donors_missing_from_chemdraw[index]
            compare_df.at[df_index, "Type"] = "donor"
            compare_df.at[df_index, "Missing_From"] = "ChemDraw"
            index += 1
            df_index += 1

        index = 0
        while index < len(acceptors_missing_from_sheets):
            compare_df.at[df_index, "Label"] = acceptors_missing_from_sheets[index]
            compare_df.at[df_index, "Type"] = "acceptor"
            compare_df.at[df_index, "Missing_From"] = "Google Sheets"
            index += 1
            df_index += 1

        index = 0
        while index < len(acceptors_missing_from_chemdraw):
            compare_df.at[df_index, "Label"] = acceptors_missing_from_chemdraw[index]
            compare_df.at[df_index, "Type"] = "acceptor"
            compare_df.at[df_index, "Missing_From"] = "ChemDraw"
            index += 1
            df_index += 1

        print("DONORS_MISSING_FROM_SHEETS: ", len(donors_missing_from_sheets))
        print("DONORS_MISSING_FROM_CHEMDRAW: ", len(donors_missing_from_chemdraw))
        print("ACCEPTORS_MISSING_FROM_SHEETS: ", len(acceptors_missing_from_sheets))
        print("ACCEPTORS_MISSING_FROM_CHEMDRAW: ", len(acceptors_missing_from_chemdraw))

        compare_df.to_csv(compare_path, index=False)


# run functions
unique_opvs = UniqueOPVs(opv_min=OPV_MIN, opv_clean=OPV_MIN)

# min_unique_donors = unique_opvs.unique_list("D", "min")
# min_unique_acceptors = unique_opvs.unique_list("A", "min")

# clean_unique_donors = unique_opvs.unique_list("D", "clean")
# clean_unique_acceptors = unique_opvs.unique_list("A", "clean")

# missing_donors = unique_opvs.filter(min_unique_donors, clean_unique_donors)
# missing_acceptors = unique_opvs.filter(min_unique_acceptors, clean_unique_acceptors)

# missing_smi_donors = unique_opvs.filter_SMILES(missing_donors, "D")
# missing_smi_acceptors = unique_opvs.filter_SMILES(missing_acceptors, "A")

# print("NUM_missing_donors: ", len(missing_donors))
# print("NUM_missing_acceptors: ", len(missing_acceptors))

# print("NUM_missing_smi_donors: ", len(missing_smi_donors))
# print("NUM_missing_smi_acceptors: ", len(missing_smi_acceptors))

# unique_opvs.create_missing_csv(missing_smi_donors, "D")

# unique_opvs.clean_up_missing()

# concatenate for donors
# unique_opvs.concat_missing_and_clean(MISSING_SMI_DONOR, CLEAN_DONOR, "D")

# concatenate for acceptors
# unique_opvs.concat_missing_and_clean(MISSING_SMI_ACCEPTOR, CLEAN_ACCEPTOR, "A")

# compare Google Sheets and ChemDraw file
unique_opvs.compare(COMPARE_PATH, CHEMDRAW_DONOR, CHEMDRAW_ACCEPTOR)

