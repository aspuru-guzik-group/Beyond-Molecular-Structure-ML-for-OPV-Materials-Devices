import argparse
from ast import Str
from copy import deepcopy
from email import generator
from pathlib import Path
from typing import Iterable
from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import textwrap
import json

from ml_for_opvs.visualization.path_utils import (
    gather_results,
    path_to_result,
)


def heatmap(config: dict):
    """
    Args:
        config: outlines the parameters to select for the appropriate configurations for comparison
    """
    with open(config["config_path"]) as f:
        plot_config: dict = json.load(f)
    plot_config: dict = plot_config[config["config_name"]]

    # Combine 2 dictionaries together
    config.update(plot_config)
    print(config)

    summary_paths: list[Path] = path_to_result(config, "summary")

    summary: pd.DataFrame = gather_results(summary_paths)
    # Plot Axis
    fig, ax = plt.subplots(figsize=(14, 9))
    # Title
    ax.set_title("Heatmap of {}".format(config["datasets"][0]))
    # display all the rows and columns in pandas
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.precision",
        3,
    ):
        # print(summary)
        pass

    mean_metric: str = config["metrics"] + "_mean"
    std_metric: str = config["metrics"] + "_std"

    # Order of X/Y-axes
    # x = [
    #     # "MLR", # ignored for rmse and mae
    #     "Lasso",
    #     "KRR",
    #     "KNN",
    #     "SVM",
    #     "RF",
    #     "XGBoost",
    #     "NGBoost",
    #     "GP",
    #     "NN",
    #     "GNN",
    # ]
    # y = [
    #     "DA_gnn",
    #     "DA_graphembed",
    #     "DA_mordred_pca",
    #     "DA_mordred",
    #     "DA_FP_radius_3_nbits_1024",
    #     "DA_tokenized_BRICS",
    #     "DA_SELFIES",
    #     "DA_BigSMILES",
    #     "DA_SMILES",
    #     "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV",
    # ]
    x= ["RF", "RF_ensemble", "RF_multi", "XGBoost", "XGBoost_ensemble", "XGBoost_multi", "SVM", "SVM_ensemble", "SVM_multi"]
    y = ["DA_FP_radius_3_nbits_1024"]

    # Heatmap
    mean_summary: pd.DataFrame = summary.pivot("Feature_Names", "Model", mean_metric)
    mean_summary: pd.DataFrame = mean_summary.reindex(index=y, columns=x)
    summary_annotated: pd.DataFrame = deepcopy(summary)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.precision",
        3,
    ):
        print(summary)
    for index, row in summary.iterrows():
        m: float = round(summary.at[index, mean_metric], 2)
        s: float = round(summary.at[index, std_metric], 2)
        annotate_label: str = str(m) + "\n" + "(±" + str(s) + ")"
        summary_annotated.at[index, "annotate_label"] = annotate_label

    summary_annotated: pd.DataFrame = summary_annotated.pivot(
        "Feature_Names", "Model", "annotate_label"
    )
    summary_annotated: pd.DataFrame = summary_annotated.reindex(index=y, columns=x)
    summary_annotated: np.ndarray = summary_annotated.to_numpy()

    sns.heatmap(
        mean_summary,
        annot=summary_annotated,
        cmap="viridis",
        fmt="",
        cbar_kws={"label": "{} (±{})".format(mean_metric, std_metric)},
    )

    # for plotting/saving
    plt.tight_layout()
    plot_path: Path = Path(config["plot_path"])
    plot_path: Path = plot_path / "{}_{}_heatmap.png".format(
        config["config_name"], config["metrics"]
    )
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_training",
        type=str,
        help="Filepath to directory called 'training' which contains all outputs from train.py",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Filepath to config.json which contains most of the necessary parameters to create informative barplots",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Input key of config you'd like to visualize. It is specified in barplot_config.json",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        help="Directory path to location of plotting.",
    )
    args = parser.parse_args()
    config = vars(args)
    heatmap(config)
