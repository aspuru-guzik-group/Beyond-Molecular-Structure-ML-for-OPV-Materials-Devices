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

from da_for_polymers.visualization.path_utils import (
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
    fig, ax = plt.subplots()
    # Title
    ax.set_title("Heatmap of {}".format(config["datasets"][0]))

    # Heatmap
    summary: pd.DataFrame = summary.pivot("Features", "Model", config["metrics"])
    sns.heatmap(
        summary,
        annot=True,
        cmap="mako",
        fmt=".4f",
        cbar_kws={"label": "{}".format(config["metrics"])},
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
