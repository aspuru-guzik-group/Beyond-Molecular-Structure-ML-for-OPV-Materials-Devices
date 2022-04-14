import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from ml_for_opvs.data.preprocess.OPV_Min.clean_donors_acceptors import OPV_DATA

# OPV data after pre-processing
OPV_CLEAN = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)


class Distribution:
    """
    Class that contains functions to determine the distribution of each variable in the dataset.
    Each dataset will have slightly different variable names.
    Must be able to handle numerical and categorical variables.
    """

    def __init__(self, data):
        self.data = pd.read_csv(data)

    def histogram(self):
        """
        Function that plots the histogram of all variables in the dataset
        NOTE: you must know the variable names beforehand

        Args:
            None

        Returns:
            Histogram plots of all the variables.
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)
        # select which columns you want to plot in the histogram
        column_idx_first = 9
        column_idx_last = 27 + 1

        # prepares the correct number of (x,y) subplots
        num_columns = column_idx_last - column_idx_first
        x_columns = round(np.sqrt(num_columns))
        if x_columns == np.floor(np.sqrt(num_columns)):
            y_rows = x_columns + 1
        elif x_columns == np.ceil(np.sqrt(num_columns)):
            y_rows = x_columns
        print(x_columns, y_rows)

        fig, axs = plt.subplots(y_rows, x_columns)
        fig.tight_layout()
        column_range = range(column_idx_first, column_idx_last)

        x_idx = 0
        y_idx = 0
        for i in column_range:
            current_column = columns[i]
            current_val_list = self.data[current_column].tolist()
            current_val_list = [
                item for item in current_val_list if not (pd.isnull(item)) == True
            ]

            axs[y_idx, x_idx].set_title(current_column)
            axs[y_idx, x_idx].hist(current_val_list)
            stepsize = round(len(current_val_list) / 5)
            start, end = axs[y_idx, x_idx].get_ylim()
            axs[y_idx, x_idx].yaxis.set_ticks(np.arange(start, end, stepsize))
            total = "Total: " + str(len(current_val_list))

            anchored_text = AnchoredText(total, loc="lower right")
            axs[y_idx, x_idx].add_artist(anchored_text)
            if isinstance(current_val_list[0], str):
                axs[y_idx, x_idx].tick_params(rotation=45)
                axs[y_idx, x_idx].tick_params(axis="x", labelsize=6)
            y_idx += 1
            if y_idx == y_rows:
                y_idx = 0
                x_idx += 1

        plt.show()


dist = Distribution(OPV_CLEAN)

dist.histogram()
