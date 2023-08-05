import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_filepath: Path = (
    Path(__file__).parent.parent.parent.parent.resolve()
    / "datasets"
    / "Min_2020_n558"
    / "cleaned_dataset.csv"
)

d_a_heatmap_filepath: Path = (
    Path(__file__).parent.resolve() / "heatmap_Donor_Acceptor_frequency.png"
)

data: pd.DataFrame = pd.read_csv(data_filepath)


def x_y_heatmap(data, x, y, x_y_path):
    """
    Function that plots the heatmap of any 2 variables in the dataset
    NOTE: you must know the variable names beforehand

    Args:
        2 column names and file path we want to save to

    Returns:
        Heatmaps of 2 chosen variables.
    """
    unique_x_y_dict = {}
    for index, row in data.iterrows():
        if str(data.at[index, x]) != "nan" and str(data.at[index, y]) != "nan":
            if (
                data.at[index, x],
                data.at[index, y],
            ) not in unique_x_y_dict:
                unique_x_y_dict[(data.at[index, x], data.at[index, y])] = 1
            elif (
                data.at[index, x],
                data.at[index, y],
            ) in unique_x_y_dict:
                unique_x_y_dict[(data.at[index, x], data.at[index, y])] += 1

    # get sorted list of unique donor and acceptor by HOMO and LUMO levels
    if x == "Donor":
        data: pd.DataFrame = data.sort_values("HOMO_D (eV)", ascending=False)
    else:
        data: pd.DataFrame = data.sort_values("LUMO_A (eV)", ascending=False)
    unique_x_sorted: list = list(data[x].unique())
    if y == "Acceptor":
        data: pd.DataFrame = data.sort_values("LUMO_A (eV)", ascending=False)
    else:
        data: pd.DataFrame = data.sort_values("HOMO_D (eV)", ascending=False)
    unique_y_sorted: list = list(data[y].unique())

    heatmap_array = np.zeros((len(unique_y_sorted), len(unique_x_sorted)))
    for x_y in unique_x_y_dict:
        x_idx = unique_x_sorted.index(x_y[0])
        y_idx = unique_y_sorted.index(x_y[1])
        freq = unique_x_y_dict[x_y]
        heatmap_array[y_idx, x_idx] = freq

    fig, ax = plt.subplots(figsize=(150, 75))
    heatmap_array_masked = np.ma.masked_where(heatmap_array == 0, heatmap_array)
    im = ax.imshow(heatmap_array_masked)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(unique_x_sorted)))
    ax.set_yticks(np.arange(len(unique_y_sorted)))
    ax.set_xticklabels(unique_x_sorted)
    ax.set_yticklabels(unique_y_sorted)
    ax.set_ylabel(y)
    ax.set_xlabel(x)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(unique_x_sorted)):
        for j in range(len(unique_y_sorted)):
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


x_y_heatmap(data, "Acceptor", "Donor", d_a_heatmap_filepath)
