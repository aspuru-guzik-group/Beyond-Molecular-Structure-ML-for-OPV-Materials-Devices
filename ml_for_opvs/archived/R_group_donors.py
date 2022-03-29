import pkg_resources
from rdkit import Chem
from rdkit.Chem.rdmolops import ReplaceSubstructs
import csv
import pandas as pd

DON_PATH = pkg_resources.resource_filename("opv_ml", "min_donors_smiles.txt")
CSV_PATH = pkg_resources.resource_filename("opv_ml", "min_donors_smilesRgrp.csv")
XLSX_PATH = pkg_resources.resource_filename(
    "opv_ml", "Labelling/min_donors_smiles.xlsx"
)
BIGSMILES_PATH = pkg_resources.resource_filename(
    "opv_ml", "Labelling/min_donors_big_smiles.csv"
)

# NOTE: Need to access masters file and get corresponding SMILES and Label in a .csv file
# Label | SMILES | SMILES (w/ substituted R) | BigSMILES | SELFIES | PCE


class R_donor:
    """
    Class containing functions required to change R group for donor structures
    """

    def __init__(self, donor_path):
        file = open(donor_path, "r")
        self.data = file.read()
        file.close()

    def PreProcess(self):
        """Function that will separate long list of SMILES

        Parameters
        ----------
        None

        Returns
        ----------
        data_list: list of molecules separated by "."
        """
        self.data_list = self.data.split(".")

    def Process(self):
        """Function that will prepare SMILES: 
        - 
        - insert R group
        - account for variable attachment

        Parameters
        ----------
        smiles_list: list of smiles and other accessory

        Returns
        ----------
        smiles_dict: dictionary for name of molecule, SMILES, and molecular object
        """
        # patts dictionary contains all R group to SMILES
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
            "[R23]": "CCC(CCCCCCCCCCCC)CCCCCCCCCC",
            "[R24]": "COCCOC",
            "[R27]": "CCCC",
        }

        i = 0
        while i < len(self.data_list):
            string = self.data_list[i]
            for r in patts:
                string = string.replace(r, patts[r])
            self.data_list[i] = string
            i += 1

        self.smiles_list = []
        i = 0
        while i < len(self.data_list):
            self.smiles_list.append([self.data_list[i]])
            i += 1

    def smiles_to_csv(self, csv_path):
        """Function that will convert smiles to csv.

        Parameters
        ----------
        csv_path: path to where csv is created to store SMILES data

        Returns
        ----------
        None
        """
        with open(csv_path, "w", newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["SMILE"])
            wr.writerows(self.smiles_list)

    def read_from_xlsx(self, xlsx_path):
        """Function that will get data from xlsx file.

        Parameters
        ----------
        xlsx_path: path to location of xlsx

        Returns
        ----------
        df: dataframe for info obtained from xlsx file
        """
        df = pd.read_excel(xlsx_path)
        return df

    def smile_to_bigsmile(self, donor_df, bigsmiles_path):
        """Function that will convert SMILES to BigSMILES in the donor dataframe.
        - replaces any methyl group with ([$]) which is the bonding descriptor for polymers

        Parameters
        ----------
        donor_df: dataframe that contains SMILES and Name of the donor OPV molecules
        bigsmiles_path: path to where csv is created to store donor data with new bigsmiles column

        Returns
        ----------
        bigsmiles_donor_csv: new csv with a new column that contains the BigSMILES
        """
        smiles_list = donor_df["R_grp_SMILES"]
        big_smiles_list = []

        # NOTE: two types of SMILES strings
        # 1. Starts with one of the methyl groups, so one of them is "Cc", and the other is "(C)"
        # 2. Starts with R group, so both methyl groups are "(C)"

        for smile in smiles_list:
            # type 1
            if smile[0] == "C":
                smile = smile[1:]
                smile = (
                    "([$])" + smile
                )  # NOTE: check how we want to format this, in OPV sheets it's $ only
                smile = smile.replace("(C)", "([$])")
            # type 2
            else:
                smile = smile.replace(
                    "(C)", "([$])"
                )  # NOTE: check how we want to format this, in OPV sheets it's ($)

            smile = "{" + smile + "}"
            big_smiles_list.append(smile)

        donor_df["R_grp_BigSMILES"] = big_smiles_list
        donor_df.to_csv(bigsmiles_path)


acceptor = R_donor(DON_PATH)
acceptor.PreProcess()
acceptor.Process()
acceptor.smiles_to_csv(CSV_PATH)
donor_df = acceptor.read_from_xlsx(XLSX_PATH)
acceptor.smile_to_bigsmile(donor_df, BIGSMILES_PATH)

