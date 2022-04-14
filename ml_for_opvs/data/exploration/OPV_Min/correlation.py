from cmath import nan
import math
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import itertools
from sklearn.metrics import mean_squared_error

# OPV data after pre-processing
OPV_CLEAN = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

CORRELATION_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/correlation_plot.png"
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
        column_idx_last = 27 + 1

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


corr_plot = Correlation(OPV_CLEAN)
corr_plot.parity_plot()

