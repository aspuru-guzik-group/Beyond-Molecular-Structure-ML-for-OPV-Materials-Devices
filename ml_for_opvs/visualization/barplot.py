import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from textwrap import wrap
import textwrap

from python_code.visualization.path_utils import (
    gather_results,
    path_to_result,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

### MAIN FUNCTION
def wrap_labels(ax, width: int = 10):
    print(ax)
    labels: list = []
    for label in ax.get_xticklabels():
        text: str = label.get_text()
        text_split: list = text.split(",")
        wrapped_text_split: list = []
        idx = 0
        for text in text_split:
            if len(text) > width:
                text: str = textwrap.fill(text, width)
            wrapped_text_split.append(text)
            if idx < len(text_split):
                wrapped_text_split.append("\n")
            idx += 1
        final_text: str = "".join(wrapped_text_split)
        labels.append(final_text)
    ax.set_xticklabels(labels, rotation=0)


def barplot(config: dict):
    """Creates a bar plot of the model performance from several configurations.
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    Returns:
        bar_plot: saves a bar plot comparison of all the configurations in the current working directory.
    """
    with open(config["config_path"]) as f:
        plot_config: dict = json.load(f)
    plot_config: dict = plot_config[config["config_name"]]

    # Combine 2 dictionaries together
    config.update(plot_config)
    # print(config)

    progress_paths: list[Path] = path_to_result(config, "progress_report")

    summary: pd.DataFrame = gather_results(progress_paths)
    summary: pd.DataFrame = summary.sort_values(config["hue"])

    # Plot Axis
    size: list[str] = config["size"].split(",")
    size: list[int] = [int(x) for x in size]
    size: tuple = tuple(size)
    fig, ax = plt.subplots(figsize=size)
    # Title
    ax.set_title("Barplot of {}".format(config["config_name"]))  # config["models"][0])
    # Barplot
    sns.set_style("whitegrid")
    print(summary)

    if "data" in config["config_name"]:
        print("True")
        sns.barplot(
            x=summary[config["x"]],
            y=summary[config["metrics"]],
            ax=ax,
            hue=summary[config["hue"]],
        )
        for container in ax.containers:
            ax.bar_label(container)
    else:
        sns.barplot(
            x=summary[config["x"]],
            y=summary[config["metrics"]],
            ci="sd",
            ax=ax,
            hue=summary[config["hue"]],
            capsize=0.08,
            errwidth=0.8,
        )
        # Y-axis Limits
        min_yval: float = min(summary[config["metrics"]])
        # min_idx_yval: int = np.argmin(summary[config["metrics"]])
        # min_yval: float = min_yval - list(summary["r_std"])[min_idx_yval]
        # min_yval: float = min_yval * 0.9
        max_yval: float = max(summary[config["metrics"]])
        ax.set_ylim(min_yval, max_yval + 0.1)

    # extract labels
    labels = ax.get_xticklabels()
    # fix the labels
    for v in labels:
        text = v.get_text()
        text = "\n".join(wrap(text, 25))
        v.set_text(text)

    ax.set_xticklabels(labels)

    # annotations on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=10)
    # for plotting/saving
    plt.tight_layout()
    plot_path: Path = Path(config["plot_path"])
    plot_path: Path = plot_path / "{}_{}_barplot.png".format(
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
    parser.add_argument(
        "--size",
        type=str,
        help="Size of barplot. Please specify only two numbers separated by a comma.",
    )
    args = parser.parse_args()
    config = vars(args)
    barplot(config)

# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
