from cmath import nan
import math
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import itertools
from sklearn.metrics import mean_squared_error

# IMPORTANT: check if data has replaced with -1 and N/A values. DO NOT USE THAT!

# OPV data after pre-processing
MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
)

CORRELATION_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/correlation_parity_plot.png"
)

CORRELATION_HEATMAP_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/correlation_heatmap_plot.png"
)

RMSE_HEATMAP_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/rmse_heatmap_plot.png"
)


class Correlation:
    """
    Class that contains all functions for creating correlations between each variable.
    """

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def parity_plot(self):
        """
        Function that plots the parity plots between each variable.
        NOTE: you must know the variable names beforehand

        Args:
            None

        Returns:
            Parity plots of each relationship.
            Layout will be:
            Var 1|X     PLOT  PLOT
            Var 2|PLOT  X     PLOT
            Var 3|PLOT  PLOT  X
                 |Var 1|Var 2|Var 3|
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)
        # select which columns you want to plot
        column_idx_first = 9
        column_idx_last = 28 + 1

        x_columns = column_idx_last - column_idx_first
        y_rows = column_idx_last - column_idx_first

        column_range = range(column_idx_first, column_idx_last)
        permutations = list(itertools.permutations(column_range, 2))

        fig, axs = plt.subplots(y_rows, x_columns, figsize=(100, 100))

        for pair in permutations:
            column_idx_0 = pair[0]
            column_idx_1 = pair[1]
            column_name_0 = columns[column_idx_0]
            column_name_1 = columns[column_idx_1]
            column_0 = self.data[column_name_0]
            column_1 = self.data[column_name_1]
            # handle unequal number of data points
            # mask values with True or False if nan
            isna_column_0 = column_0.isna().tolist()
            isna_column_1 = column_1.isna().tolist()
            filtered_column_0 = []
            filtered_column_1 = []
            index = 0
            while index < len(column_0):
                if not isna_column_0[index] and not isna_column_1[index]:
                    filtered_column_0.append(column_0[index])
                    filtered_column_1.append(column_1[index])
                index += 1

            # subplot
            x_axis_idx = column_idx_0 - column_idx_first
            y_axis_idx = column_idx_1 - column_idx_first
            axs[x_axis_idx, y_axis_idx].scatter(
                filtered_column_1, filtered_column_0, s=1
            )

            # set xlabel and ylabel
            axs[x_axis_idx, y_axis_idx].set_xlabel(column_name_1)
            axs[x_axis_idx, y_axis_idx].set_ylabel(column_name_0)

            # handle different data types (str, float)
            if isinstance(filtered_column_0[0], float) and isinstance(
                filtered_column_1[0], float
            ):
                # find slope and y-int of linear line of best fit
                m, b = np.polyfit(filtered_column_1, filtered_column_0, 1,)
                # find correlation coefficient
                corr_coef = np.corrcoef(filtered_column_1, filtered_column_0,)[0, 1]
                # find rmse
                rmse = np.sqrt(
                    mean_squared_error(filtered_column_1, filtered_column_0,)
                )
                axs[x_axis_idx, y_axis_idx].plot(
                    np.array(filtered_column_1),
                    m * np.array(filtered_column_1) + b,
                    color="black",
                )
                textstr = (
                    "R: "
                    + str(round(corr_coef, 3))
                    + "  "
                    + "RMSE: "
                    + str(round(rmse, 3))
                )
                anchored_text = AnchoredText(textstr, loc="lower right")
                axs[x_axis_idx, y_axis_idx].add_artist(anchored_text)

        plt.savefig(CORRELATION_PLOT)

    def heatmap_plot(self, option):
        """
        Function that plots the R and RMSE between each variable.
        NOTE: you must know the variable names beforehand

        Args:
            option: input whether you want to calculate R or RMSE

        Returns:
            Parity plots of each relationship.
            Layout will be:
            Var 1|X     R     R
            Var 2|R     X     R
            Var 3|R     R     X
                 |Var 1|Var 2|Var 3|
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)
        # select which columns you want to plot
        column_idx_first = 9
        column_idx_last = 28 + 1

        x_columns = column_idx_last - column_idx_first
        y_rows = column_idx_last - column_idx_first

        column_range = range(column_idx_first, column_idx_last)
        column_name_list = columns[column_idx_first:column_idx_last]
        permutations = list(itertools.permutations(column_range, 2))

        fig, ax = plt.subplots(figsize=(20, 20))

        if option == "r":
            heatmap_array = np.ones((y_rows, x_columns))
        elif option == "rmse":
            heatmap_array = np.zeros((y_rows, x_columns))

        for pair in permutations:
            column_idx_0 = pair[0]
            column_idx_1 = pair[1]
            column_name_0 = columns[column_idx_0]
            column_name_1 = columns[column_idx_1]
            column_0 = self.data[column_name_0]
            column_1 = self.data[column_name_1]
            # handle unequal number of data points
            # mask values with True or False if nan
            isna_column_0 = column_0.isna().tolist()
            isna_column_1 = column_1.isna().tolist()
            filtered_column_0 = []
            filtered_column_1 = []
            index = 0
            while index < len(column_0):
                if not isna_column_0[index] and not isna_column_1[index]:
                    filtered_column_0.append(column_0[index])
                    filtered_column_1.append(column_1[index])
                index += 1

            # indexes needed to fill in array appropriately
            x_axis_idx = column_idx_0 - column_idx_first
            y_axis_idx = column_idx_1 - column_idx_first

            # handle different data types (str, float)
            if isinstance(filtered_column_0[0], float) and isinstance(
                filtered_column_1[0], float
            ):
                if option == "r":
                    # find correlation coefficient
                    result = np.corrcoef(filtered_column_1, filtered_column_0,)[0, 1]
                elif option == "rmse":
                    # find rmse
                    result = np.sqrt(
                        mean_squared_error(filtered_column_1, filtered_column_0,)
                    )
                heatmap_array[x_axis_idx, y_axis_idx] = round(result, 3)
            else:
                heatmap_array[x_axis_idx, y_axis_idx] = 0

        im = ax.imshow(heatmap_array)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(column_name_list)))
        ax.set_yticks(np.arange(len(column_name_list)))
        ax.set_xticklabels(column_name_list)
        ax.set_yticklabels(column_name_list)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(column_name_list)):
            for j in range(len(column_name_list)):
                text_r = ax.text(
                    j, i, heatmap_array[i, j], ha="center", va="center", color="w"
                )

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel(option, rotation=-90, va="bottom")

        if option == "r":
            ax.set_title("Heatmap of Correlation Coefficient between Device Parameters")
            fig.tight_layout()
            plt.savefig(CORRELATION_HEATMAP_PLOT)

        elif option == "rmse":
            ax.set_title("Heatmap of RMSE between Device Parameters")
            fig.tight_layout()
            plt.savefig(RMSE_HEATMAP_PLOT)


corr_plot = Correlation(MASTER_ML_DATA_PLOT)
corr_plot.parity_plot()
# corr_plot.heatmap_plot("r")

