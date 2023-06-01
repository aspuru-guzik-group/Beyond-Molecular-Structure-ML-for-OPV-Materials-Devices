from typing import List
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import math

from torch import unique

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
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

HOMO_D_DISTRIBUTION_APPROX = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/homo_approx_donor_distribution_plot.png"
)

LUMO_D_DISTRIBUTION_APPROX = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/lumo_approx_donor_distribution_plot.png"
)

HOMO_A_DISTRIBUTION_APPROX = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/homo_approx_acceptor_distribution_plot.png"
)

LUMO_A_DISTRIBUTION_APPROX = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/lumo_approx_acceptor_distribution_plot.png"
)

STATS_D_HOMO = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/stats_homo_donor.csv"
)

STATS_D_LUMO = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/stats_lumo_donor.csv"
)

STATS_A_HOMO = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/stats_homo_acceptor.csv"
)

STATS_A_LUMO = pkg_resources.resource_filename(
    "ml_for_opvs", "data/error_correction/OPV_Min/stats_lumo_acceptor.csv"
)


class MissingParameters:
    """
    Class that contains functions to evaluate_model any parameter. Potential tools:
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
            column_names = ["Donor", "HOMO_D_eV", "LUMO_D_eV"]
        elif mol_type == "A":
            column_names = ["Acceptor", "HOMO_A_eV", "LUMO_A_eV"]
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

        # find min and max value of HOMO and LUMO
        column_names = plot_df.columns
        homo_min = math.floor(min(plot_df[column_names[1]]))
        homo_max = math.ceil(max(plot_df[column_names[1]]))
        lumo_min = math.floor(min(plot_df[column_names[2]]))
        lumo_max = math.ceil(max(plot_df[column_names[2]]))
        print(homo_min, homo_max, lumo_min, lumo_max)

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
            w = float((homo_max - homo_min) / 40)
            bins = np.arange(homo_min, homo_max + w, w)
            n, bins, patches = axs[y_idx, x_idx].hist(filtered_homo_array, bins=bins)
            total = "Total: " + str(num_homo) + "\n" + "Total nan: " + str(num_of_nan)
            anchored_text = AnchoredText(total, loc="upper left")
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
            w = float((lumo_max - lumo_min) / 40)
            bins = np.arange(lumo_min, lumo_max + w, w)
            n, bins, patches = axs[y_idx, x_idx].hist(filtered_lumo_array, bins=bins)
            total = "Total: " + str(num_lumo) + "\n" + "Total nan: " + str(num_of_nan)
            anchored_text = AnchoredText(total, loc="upper left")
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

    def avg_stdev_for_homolumo(self, duplicate_df, stats_homo_path, stats_lumo_path):
        """
        Calculates the average and standard deviation for each distinct donor/acceptor with duplicates

        Args:
            duplicate_df: dataframe with Donor or Acceptor | Column Names (w/ values)
            stats_homo_path: path to .csv file with average and standard deviation of HOMO for each distinct donor or acceptor
            stats_lumo_path: path to .csv file with average and standard deviation of LUMO for each distinct donor or acceptor

        Returns:
            .csv file
        """
        column_names = duplicate_df.columns
        stats_homo_df = pd.DataFrame(columns=[column_names[0]])
        stats_homo_df["HOMO_avg"] = ""
        stats_homo_df["HOMO_std"] = ""
        stats_lumo_df = pd.DataFrame(columns=[column_names[0]])
        stats_lumo_df["LUMO_avg"] = ""
        stats_lumo_df["LUMO_std"] = ""
        # curate dictionary with unique donor/acceptors and their corresponding HOMO/LUMO values
        unique_homo_dict = {}
        unique_lumo_dict = {}
        for index, row in duplicate_df.iterrows():
            if row[0] not in unique_homo_dict:
                unique_homo_dict[row[0]] = [row[1]]
            else:
                unique_homo_dict[row[0]].append(row[1])
            if row[0] not in unique_lumo_dict:
                unique_lumo_dict[row[0]] = [row[2]]
            else:
                unique_lumo_dict[row[0]].append(row[2])

        idx = 0
        for homo_mol in unique_homo_dict:
            homo_list = unique_homo_dict[homo_mol]
            filtered_homo_array = [item for item in homo_list if not np.isnan(item)]
            print(filtered_homo_array)
            try:
                homo_avg = np.mean(filtered_homo_array)
                homo_std = np.std(filtered_homo_array)
            except:
                print("NO VALUES")
            else:
                stats_homo_df.at[idx, column_names[0]] = homo_mol
                stats_homo_df.at[idx, "HOMO_avg"] = homo_avg
                stats_homo_df.at[idx, "HOMO_std"] = homo_std
                idx += 1

        idx = 0
        for lumo_mol in unique_lumo_dict:
            lumo_list = unique_lumo_dict[lumo_mol]
            filtered_lumo_array = [item for item in lumo_list if not np.isnan(item)]
            try:
                lumo_avg = np.mean(filtered_lumo_array)
                lumo_std = np.std(filtered_lumo_array)
            except:
                print("NO VALUES")
            else:
                stats_lumo_df.at[idx, column_names[0]] = lumo_mol
                stats_lumo_df.at[idx, "LUMO_avg"] = lumo_avg
                stats_lumo_df.at[idx, "LUMO_std"] = lumo_std
                idx += 1

        stats_homo_df.to_csv(stats_homo_path, index=False)
        stats_lumo_df.to_csv(stats_lumo_path, index=False)


# missing = MissingParameters(MASTER_ML_DATA_PLOT)

# donor_dup = missing.search_duplicate("D")
# missing.plot_all(donor_dup, HOMO_D_DISTRIBUTION, LUMO_D_DISTRIBUTION)
# acceptor_dup = missing.search_duplicate("A")
# missing.plot_all(acceptor_dup, HOMO_A_DISTRIBUTION, LUMO_A_DISTRIBUTION)

# missing.avg_stdev_for_homolumo(donor_dup, STATS_D_HOMO, STATS_D_LUMO)
# missing.avg_stdev_for_homolumo(acceptor_dup, STATS_A_HOMO, STATS_A_LUMO)

missing = MissingParameters(MASTER_ML_DATA)

donor_dup = missing.search_duplicate("D")
missing.plot_all(donor_dup, HOMO_D_DISTRIBUTION_APPROX, LUMO_D_DISTRIBUTION_APPROX)
acceptor_dup = missing.search_duplicate("A")
missing.plot_all(acceptor_dup, HOMO_A_DISTRIBUTION_APPROX, LUMO_A_DISTRIBUTION_APPROX)