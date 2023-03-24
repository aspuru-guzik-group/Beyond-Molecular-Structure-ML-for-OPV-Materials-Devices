import itertools

import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

# OPV data after pre-processing
MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/preprocess/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv",
)

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv",
)


def plot_feature_distributions(dataset: pd.DataFrame, drop_columns: list = None):
    """
    Function that plots the distribution of all variables in the dataset. The function only drops nan values in one column at a time.
    The function then plots the distributions in one figure, and saves the figure as a png file. The figure should have a length-to-width ratio of 3:2.

    Args:
        dataset: Pre-processed dataset of OPV device parameters
    """
    df = dataset.copy()  # make a copy of the dataset
    df.drop(drop_columns, axis=1, inplace=True)  # drop the columns that won't be used in the analysis

    df_columns = len(df.columns)  # get the number of columns in the dataset
    # find the number of rows and columns of the figure
    num_columns = df_columns
    x_columns = round(np.sqrt(num_columns))
    if x_columns == np.floor(np.sqrt(num_columns)):
        y_rows = x_columns + 1  # if the number of columns is a perfect square, then the number of rows is one more than the number of columns
    elif x_columns == np.ceil(np.sqrt(num_columns)):
        y_rows = x_columns
    fig, axs = plt.subplots(y_rows, x_columns, figsize=(y_rows * 6, x_columns * 2))  # prepare the figure
    column_range: list = list(range(0, df_columns))  # get the range of columns to plot
    x_idx = 0  # index of the current column
    y_idx = 0  # index of the current row
    for i in column_range:
        current_column = df.columns[i]
        current_val_list = df[current_column].tolist()
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
    plt.savefig("feature_distributions.png")


def plot_heatmap_of_pair_frequency(dataset: pd.DataFrame, column1: str, column2: str):
    """
    Function that calculates the frequency of each pair of variables found in two specified columns of the dataset.
    The function then plots the heatmap of the frequency.
    """
    df = dataset.copy()  # make a copy of the dataset
    df = df[[column1, column2]]  # only keep the two columns we are interested in
    df = df.dropna()  # drop all rows with missing values
    df = df.reset_index(drop=True)  # reset the index of the dataframe
    df = df.astype(str)  # convert all values to string
    df = df.groupby([column1, column2]).size().reset_index(
        name='Frequency')  # calculate the frequency of each pair of variables
    df = df.pivot(index=column1, columns=column2,
                  values='Frequency')  # pivot the dataframe to make it suitable for plotting
    # df = df.fillna(0) # fill all missing values with 0
    # df = df.astype(int)  # convert all values to integer
    df = df.sort_index(axis=1)  # sort the columns in ascending order
    df = df.sort_index(axis=0)  # sort the rows in ascending order
    df = df.reindex(sorted(df.columns), axis=1)  # sort the columns in ascending order
    df = df.reindex(sorted(df.index), axis=0)  # sort the rows in ascending order

    # plot the heatmap
    fig, ax = plt.subplots(figsize=(3000, 1000))
    im = ax.imshow(df)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(df.columns)):
        for j in range(len(df.index)):
            text = ax.text(i, j, df.iloc[j, i], ha="center", va="center", color="w")
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.35)
    cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

    ax.set_title(f"Heatmap of {column1} and {column2} pair frequency")
    fig.tight_layout()
    plt.savefig(f"heatmap_{column1}_{column2}_frequency.png")


if __name__ == "__main__":
    opv_dataset = pd.read_csv(MASTER_ML_DATA)

    pairs_list = ["Donor", "Acceptor", "solvent"]
    # find every non-symmetric combination in pairs_list
    pairs_combinations = list(itertools.combinations(pairs_list, 2))
    for pair in pairs_combinations:
        plot_heatmap_of_pair_frequency(opv_dataset, pair[0], pair[1])

    plot_feature_distributions(opv_dataset, ["ref", "Donor", "Acceptor", "Donor_SMILES", "Donor_Big_SMILES", "Donor_SELFIES", "Acceptor_SMILES", "Acceptor_Big_SMILES", "Acceptor_SELFIES"])
