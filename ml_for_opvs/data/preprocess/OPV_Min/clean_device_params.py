from os import error
import pkg_resources
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import csv
import pandas as pd

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

PARAMETER_INVENTORY = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/device_parameter_inventory.csv"
)


class ParameterClean:
    """
    Class that contains functions to track Categorical parameters. 
    (i.e. chemical compounds have the accurate SMILES representation)
    """

    def __init__(self, master_csv_path):
        self.data = pd.read_csv(master_csv_path)

    def create_inventory(self, parameter_csv_path):
        """
        Function that creates .csv file for inventory which stores unique categorical variables:
        Headers: Name | IUPAC | Type of Parameter (List) | SMILES

        Args:
            parameter_csv_path: path to device parameter data (.csv file)

        Returns:
            .csv file with headers shown above
        """
        # preview of columns and column indexes
        column_names = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(column_names):
            columns_dict[column_names[index]] = index
            index += 1

        print(columns_dict)

        # create dictionary for inventory
        inventory_dict = {"Name": [], "Parameter_Type": []}

        inventory_idx = 0
        for index, row in self.data.iterrows():
            # get data from each row
            solvent_data = self.data.at[index, column_names[14]]
            solvent_add_data = self.data.at[index, column_names[16]]
            hole_contact_data = self.data.at[index, column_names[20]]
            electron_contact_data = self.data.at[index, column_names[21]]

            # update dictionary
            if solvent_data not in inventory_dict["Name"] and isinstance(
                solvent_data, str
            ):
                inventory_dict["Name"].append(solvent_data)
                inventory_dict["Parameter_Type"].append(["Solvent"])
            elif solvent_data in inventory_dict["Name"]:
                duplicate_idx = inventory_dict["Name"].index(solvent_data)
                if "Solvent" not in inventory_dict["Parameter_Type"][duplicate_idx]:
                    inventory_dict["Parameter_Type"][duplicate_idx].append("Solvent")

            if solvent_add_data not in inventory_dict["Name"] and isinstance(
                solvent_add_data, str
            ):
                inventory_dict["Name"].append(solvent_add_data)
                inventory_dict["Parameter_Type"].append(["Solvent_Additive"])
            elif solvent_add_data in inventory_dict["Name"]:
                duplicate_idx = inventory_dict["Name"].index(solvent_add_data)
                if (
                    "Solvent_Additive"
                    not in inventory_dict["Parameter_Type"][duplicate_idx]
                ):
                    inventory_dict["Parameter_Type"][duplicate_idx].append(
                        "Solvent_Additive"
                    )

            if hole_contact_data not in inventory_dict["Name"] and isinstance(
                hole_contact_data, str
            ):
                inventory_dict["Name"].append(hole_contact_data)
                inventory_dict["Parameter_Type"].append(["Hole_Contact_Layer"])
            elif hole_contact_data in inventory_dict["Name"]:
                duplicate_idx = inventory_dict["Name"].index(hole_contact_data)
                if (
                    "Hole_Contact_Layer"
                    not in inventory_dict["Parameter_Type"][duplicate_idx]
                ):
                    inventory_dict["Parameter_Type"][duplicate_idx].append(
                        "Hole_Contact_Layer"
                    )

            if electron_contact_data not in inventory_dict["Name"] and isinstance(
                electron_contact_data, str
            ):
                inventory_dict["Name"].append(electron_contact_data)
                inventory_dict["Parameter_Type"].append(["Electron_Contact_Layer"])
            elif electron_contact_data in inventory_dict["Name"]:
                duplicate_idx = inventory_dict["Name"].index(electron_contact_data)
                if (
                    "Electron_Contact_Layer"
                    not in inventory_dict["Parameter_Type"][duplicate_idx]
                ):
                    inventory_dict["Parameter_Type"][duplicate_idx].append(
                        "Electron_Contact_Layer"
                    )

        # Create dataframe from inventory dictionary
        inventory_df = pd.DataFrame(inventory_dict)
        inventory_df["IUPAC"] = ""
        inventory_df["SMILES"] = ""

        inventory_df.to_csv(parameter_csv_path, index=False)
    
    # TODO: clean up thermal annealing and solvent additive conc.
    # TODO: add this to auto_generate_data.py


param_clean = ParameterClean(MASTER_ML_DATA)
param_clean.create_inventory(PARAMETER_INVENTORY)
