from typing import List
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import column

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)


class MissingParameters:
    """
    Class that contains functions to evaluate any parameter. Potential tools:
    - Find missing parameters and return the D-A pair
    - Automate processing of parameter
    """

    def __init__(self, master_data):
        """
        Instantiates class with appropriate data

        Args:
            master_data: path to opv data from pre-processing

        Returns:
            None
        """
        self.data = pd.read_csv(master_data)

    def search_missing(self, column_names):
        """
        Searches for D-A pairs with missing values in the specified column.

        Args:
            column_names: columns that we want to search for

        Returns:
            missing_df: Dataframe with Donor | Acceptor | Column Names (w/ values)
        """
        pass

    def search_duplicate(self, mol_type):
        """
        Search for duplicate D or A with different HOMO/LUMO values for the specified columns.

        Args:
            mol_type: search for "D" - donor or "A" - acceptor

        Returns:
            missing_df: Dataframe with Donor | HOMO_D | LUMO_D or Acceptor | HOMO_A | LUMO_A
        """
        if mol_type == "D":
            column_names = ["Donor", "HOMO_D (eV)", "LUMO_D (eV)"]
        elif mol_type == "A":
            column_names = ["Acceptor", "HOMO_A (eV)", "LUMO_A (eV)"]
        missing_df = pd.DataFrame(columns=column_names)
        missing_idx = 0
        duplicate_df = self.data.duplicated(
            subset=column_names[0], keep=False
        ).to_frame()
        for index, row in duplicate_df.iterrows():
            if row[0] == True:
                missing_df.at[missing_idx, column_names[0]] = self.data.at[
                    index, column_names[0]
                ]
                missing_df.at[missing_idx, column_names[1]] = self.data.at[
                    index, column_names[1]
                ]
                missing_df.at[missing_idx, column_names[2]] = self.data.at[
                    index, column_names[2]
                ]
                missing_idx += 1

        return missing_df

    def plot_all(self, plot_df):
        """
        For each unique D or A, the different values for HOMO/LUMO are plotted as a histogram.

        Args:
            plot_df: dataframe with Donor or Acceptor | Column Names (w/ values)

        Returns:
            .png plot of all the HOMO and LUMO distributions
        """
        # curate dictionary with unique donor/acceptors and their corresponding HOMO/LUMO values
        unique_homo_dict = {}
        unique_lumo_dict = {}
        for index, row in plot_df.iterrows():
            if row[0] not in unique_homo_dict:
                unique_homo_dict[row[0]] = [row[1]]
            else:
                unique_homo_dict[row[0]].append(row[1])
            if row[0] not in unique_lumo_dict:
                unique_lumo_dict[row[0]] = [row[2]]
            else:
                unique_lumo_dict[row[0]].append(row[2])

        x_homo_columns = round(np.sqrt(len(unique_homo_dict)))
        y_homo_rows = round(np.sqrt(len(unique_homo_dict))) + 1
        x_lumo_columns = round(np.sqrt(len(unique_lumo_dict)))
        y_lumo_rows = round(np.sqrt(len(unique_lumo_dict))) + 1

        # plot HOMO
        # NOTE: count number of values and number of nan values
        fig, axs = plt.subplots(
            y_homo_rows, x_homo_columns, figsize=(y_homo_rows * 3, x_homo_columns * 4)
        )
        x_idx = 0
        y_idx = 0
        for homo in unique_homo_dict:
            homo_array = unique_homo_dict[homo]
            num_homo = len(homo_array)
            num_of_nan = 0
            for boolean in np.isnan(homo_array):
                if boolean:
                    num_of_nan += 1

        # plot LUMO


missing = MissingParameters(MASTER_ML_DATA)
donor_dup = missing.search_duplicate("D")
missing.plot_all(donor_dup)
# acceptor_dup = missing.search_duplicate("A")
# missing.plot_all(acceptor_dup)
