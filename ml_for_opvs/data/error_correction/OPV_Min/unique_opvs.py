from typing import List
import pkg_resources
import pandas as pd

OPV_MIN = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/process/OPV_Min/Machine Learning OPV Parameters - device_params.csv",
)

# OPV data after pre-processing which loss some of the OPVs from the Min. paper
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
        elif data_option == "clean":  # take data from pre-processed data
            data = self.opv_clean

        unique_mols = []
        if mol_type == "D":  # unique list of donors
            for mol in data["Donor"]:
                if mol not in unique_mols:
                    unique_mols.append(mol)
        elif mol_type == "A":  # unique list of acceptors
            for mol in data["Acceptor"]:
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


# run functions
unique_opvs = UniqueOPVs(opv_min=OPV_MIN, opv_clean=OPV_CLEAN)

min_unique_donors = unique_opvs.unique_list("D", "min")
min_unique_acceptors = unique_opvs.unique_list("A", "min")

clean_unique_donors = unique_opvs.unique_list("D", "clean")
clean_unique_acceptors = unique_opvs.unique_list("A", "clean")

missing_donors = unique_opvs.filter(min_unique_donors, clean_unique_donors)
missing_acceptors = unique_opvs.filter(min_unique_acceptors, clean_unique_acceptors)

missing_smi_donors = unique_opvs.filter_SMILES(missing_donors, "D")
missing_smi_acceptors = unique_opvs.filter_SMILES(missing_acceptors, "A")

# print("NUM_missing_donors: ", len(missing_donors))
# print("NUM_missing_acceptors: ", len(missing_acceptors))

# print("NUM_missing_smi_donors: ", len(missing_smi_donors))
# print("NUM_missing_smi_acceptors: ", len(missing_smi_acceptors))

unique_opvs.create_missing_csv(missing_smi_acceptors, "A")
