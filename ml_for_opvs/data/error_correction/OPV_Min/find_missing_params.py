from typing import List
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

HOMO_D_DISTRIBUTION = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/homo_donor_distribution_plot.png"
)

LUMO_D_DISTRIBUTION = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/lumo_donor_distribution_plot.png"
)

HOMO_A_DISTRIBUTION = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/homo_acceptor_distribution_plot.png"
)

LUMO_A_DISTRIBUTION = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/lumo_acceptor_distribution_plot.png"
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

    def plot_all(self, plot_df, homo_path, lumo_path):
        """
        For each unique D or A, the different values for HOMO/LUMO are plotted as a histogram.

        Args:
            plot_df: dataframe with Donor or Acceptor | Column Names (w/ values)
            homo_path: path to distribution plot of HOMO values
            lumo_path: path to distribution plot of LUMO values

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
            if x_idx == x_homo_columns:
                x_idx = 0
                y_idx += 1
            homo_array = unique_homo_dict[homo]
            num_homo = len(homo_array)
            num_of_nan = 0
            for boolean in np.isnan(homo_array):
                if boolean:
                    num_of_nan += 1
            filtered_homo_array = [item for item in homo_array if not np.isnan(item)]
            axs[y_idx, x_idx].set_title(homo)
            n, bins, patches = axs[y_idx, x_idx].hist(filtered_homo_array, bins=30)
            total = "Total: " + str(num_homo) + "\n" + "Total nan: " + str(num_of_nan)
            anchored_text = AnchoredText(total, loc="lower right")
            axs[y_idx, x_idx].add_artist(anchored_text)
            x_idx += 1
        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.3  # the amount of width reserved for blank space between subplots
        hspace = 0.6  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.savefig(homo_path)

        # plot LUMO
        fig, axs = plt.subplots(
            y_lumo_rows, x_lumo_columns, figsize=(y_lumo_rows * 3, x_lumo_columns * 4)
        )
        x_idx = 0
        y_idx = 0
        for lumo in unique_lumo_dict:
            if x_idx == x_lumo_columns:
                x_idx = 0
                y_idx += 1
            lumo_array = unique_lumo_dict[lumo]
            num_lumo = len(lumo_array)
            num_of_nan = 0
            for boolean in np.isnan(lumo_array):
                if boolean:
                    num_of_nan += 1
            filtered_lumo_array = [item for item in lumo_array if not np.isnan(item)]
            axs[y_idx, x_idx].set_title(lumo)
            n, bins, patches = axs[y_idx, x_idx].hist(filtered_lumo_array, bins=30)
            total = "Total: " + str(num_lumo) + "\n" + "Total nan: " + str(num_of_nan)
            anchored_text = AnchoredText(total, loc="lower right")
            axs[y_idx, x_idx].add_artist(anchored_text)
            x_idx += 1
        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.3  # the amount of width reserved for blank space between subplots
        hspace = 0.6  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.savefig(lumo_path)


missing = MissingParameters(MASTER_ML_DATA)
donor_dup = missing.search_duplicate("D")
missing.plot_all(donor_dup, HOMO_D_DISTRIBUTION, LUMO_D_DISTRIBUTION)
acceptor_dup = missing.search_duplicate("A")
missing.plot_all(acceptor_dup, HOMO_A_DISTRIBUTION, LUMO_A_DISTRIBUTION)
