import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from cmath import nan

# OPV data after pre-processing
MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
)

DISTRIBUTION_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/distribution_plot.png"
)

DONOR_SOLVENT_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/donor_solvent_heatmap.png"
)

ACCEPTOR_SOLVENT_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/acceptor_solvent_heatmap.png"
)

DONOR_ACCEPTOR_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "data/exploration/OPV_Min/donor_acceptor_heatmap.png"
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
        column_idx_last = 37 + 1

        # prepares the correct number of (x,y) subplots
        num_columns = column_idx_last - column_idx_first
        x_columns = round(np.sqrt(num_columns))
        if x_columns == np.floor(np.sqrt(num_columns)):
            y_rows = x_columns + 1
        elif x_columns == np.ceil(np.sqrt(num_columns)):
            y_rows = x_columns
        print(x_columns, y_rows)

        fig, axs = plt.subplots(y_rows, x_columns, figsize=(y_rows * 3, x_columns * 4))
        column_range = range(column_idx_first, column_idx_last)

        x_idx = 0
        y_idx = 0
        for i in column_range:
            current_column = columns[i]
            current_val_list = self.data[current_column].tolist()
            current_val_list = [
                item for item in current_val_list if not (pd.isnull(item)) == True
            ]
            unique_val_list = list(set(current_val_list))
            axs[y_idx, x_idx].set_title(current_column)
            if isinstance(current_val_list[0], str):
                n, bins, patches = axs[y_idx, x_idx].hist(
                    current_val_list, bins=len(unique_val_list) * 2
                )
            elif isinstance(current_val_list[0], float):
                n, bins, patches = axs[y_idx, x_idx].hist(current_val_list, bins=30)
            start = 0
            end = n.max()
            stepsize = end / 5
            y_ticks = list(np.arange(start, end, stepsize))
            y_ticks.append(end)
            axs[y_idx, x_idx].yaxis.set_ticks(y_ticks)
            total = "Total: " + str(len(current_val_list))
            anchored_text = AnchoredText(total, loc="lower right")
            axs[y_idx, x_idx].add_artist(anchored_text)
            if isinstance(current_val_list[0], str):
                axs[y_idx, x_idx].tick_params(axis="x", labelrotation=90)
                axs[y_idx, x_idx].tick_params(axis="x", labelsize=6)
            y_idx += 1
            if y_idx == y_rows:
                y_idx = 0
                x_idx += 1

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.3  # the amount of width reserved for blank space between subplots
        hspace = 0.6  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.savefig(DISTRIBUTION_PLOT)

    def x_y_heatmap(self, x, y, x_y_path):
        """
        Function that plots the heatmap of any 2 variables in the dataset
        NOTE: you must know the variable names beforehand

        Args:
            2 column names and file path we want to save to

        Returns:
            Heatmaps of 2 chosen variables.
        """
        unique_x_y_dict = {}
        for index, row in self.data.iterrows():
            if (
                str(self.data.at[index, x]) != "nan"
                and str(self.data.at[index, y]) != "nan"
            ):
                if (
                    self.data.at[index, x],
                    self.data.at[index, y],
                ) not in unique_x_y_dict:
                    unique_x_y_dict[
                        (self.data.at[index, x], self.data.at[index, y])
                    ] = 1
                elif (
                    self.data.at[index, x],
                    self.data.at[index, y],
                ) in unique_x_y_dict:
                    unique_x_y_dict[
                        (self.data.at[index, x], self.data.at[index, y])
                    ] += 1

        unique_x = set()
        unique_y = set()
        for x_y in unique_x_y_dict:
            if x_y[0] not in unique_x:
                unique_x.add(x_y[0])
            if x_y[1] not in unique_y:
                unique_y.add(x_y[1])
        unique_x = list(unique_x)
        unique_y = list(unique_y)

        heatmap_array = np.zeros((len(unique_y), len(unique_x)))
        for x_y in unique_x_y_dict:
            x_idx = unique_x.index(x_y[0])
            y_idx = unique_y.index(x_y[1])
            freq = unique_x_y_dict[x_y]
            heatmap_array[y_idx, x_idx] = freq

        fig, ax = plt.subplots(figsize=(200, 100))
        heatmap_array_masked = np.ma.masked_where(heatmap_array == 0, heatmap_array)
        im = ax.imshow(heatmap_array_masked)
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(unique_x)))
        ax.set_yticks(np.arange(len(unique_y)))
        ax.set_xticklabels(unique_x)
        ax.set_yticklabels(unique_y)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                text_r = ax.text(
                    i,
                    j,
                    heatmap_array_masked[j, i],
                    ha="center",
                    va="center",
                    color="w",
                )
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.35)
        cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

        ax.set_title("Heatmap of {} and {} Pair Frequency".format(x, y))
        fig.tight_layout()
        plt.savefig(x_y_path)


# dist = Distribution(MASTER_ML_DATA_PLOT)

# dist.histogram()

# dist.x_y_heatmap("Donor", "Acceptor", DONOR_ACCEPTOR_PATH)

# df = pd.read_csv(MASTER_ML_DATA_PLOT)
# print(max(df["hole_mobility_blend"]))