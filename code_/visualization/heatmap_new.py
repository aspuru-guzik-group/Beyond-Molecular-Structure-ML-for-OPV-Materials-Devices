import json
from itertools import product
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 16})

score_bounds: dict[str, int] = {"r": 1, "r2": 1, "mae": 25, "rmse": 25}

var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}


def get_results_from_file(results_dir: Path, model: str, score: str, var: str) -> tuple[float, float]:
    """
    Args:
        root_dir: Root directory containing all results
        representation: Representation for which to get scores
        model: Model for which to get scores.
        score: Score to plot
        var: Variance to plot

    Returns:
        Average and variance of score
    """
    # assert results_dir.exists(), f"Results directory does not exist: {results_dir}"
    score_file: list[Path] = list(results_dir.glob(f"{model}_scores.json"))
    if len(score_file) == 0:
        avg, std = np.nan, np.nan
    else:
        # for f in root_dir.glob(f"{model}_{representation}*_scores.json"):
        #     with open(f) as json_file:
        #         data: dict = json.load(json_file)
        file = score_file[0]
        with open(file, "r") as f:
            data = json.load(f)
        avg = data[f"{score}_avg"]
        if var == "stdev":
            std = data[f"{score}_stdev"]
        elif var == "stderr":
            std = data[f"{score}_stderr"]
        else:
            raise ValueError(f"Unknown variance type: {var}")
        # se = data[f"{score}_stderr"]

    avg: float = np.nan if abs(avg) > score_bounds[score] else avg
    std: float = np.nan if abs(std) > score_bounds[score] else std
    # se: float = np.nan if abs(se) > score_bounds[score] else se

    if score in ["mae", "rmse"]:
        avg, std = abs(avg), abs(std)
    return avg, std


def generate_annotations(num: float) -> str:
    """
    Args:
        num: Number to annotate

    Returns:
        String to annotate heatmap
    """
    if isinstance(num, float) and not np.isnan(num):
        num_txt: str = f"{round(num, 2)}"
    else:
        num_txt = "NaN"
    return num_txt


model_abbrev_to_full: dict[str, str] = {
    "MLR": "Linear Regression",
    "KRR": "Kernel Ridge",
    "KNN": "K-Nearest Neighbors",
    "SVR": "Support Vector Machine",
    "RF":  "Random Forest",
    "XGB": "Gradient Boosted Trees",
    "HGB": "Histogram Gradient Boosting",
    "NGB": "Natural Gradient Boosting",
    "GP":  "Gaussian Process",
    "NN":  "Neural Network",
    "GNN": "Graph Neural Network",
}


def _create_heatmap(root_dir: Path,
                    score: str, var: str,
                    x_labels: list[str], y_labels: list[str],
                    figsize: tuple[int, int],
                    fig_title: str,
                    x_title: str,
                    y_title: str,
                    fname: str,
                    ) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        figsize: Figure size
        fig_title: Figure title
        x_title: X-axis title
        y_title: Y-axis title
        fname: Filename to save figure
    """
    avg_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    std_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    annotations: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)

    for rep, model in product(x_labels, y_labels):
        p = root_dir / f"features_{rep}"
        avg, std = get_results_from_file(p, model, score, var)
        avg_scores.at[model, rep] = avg
        std_scores.at[model, rep] = std

    for x, y in product(x_labels, y_labels):
        avg: float = avg_scores.loc[y, x]
        std: float = std_scores.loc[y, x]
        avg_txt: str = generate_annotations(avg)
        std_txt: str = generate_annotations(std)
        annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"

    avg_scores = avg_scores.astype(float)
    annotations = annotations.astype(str)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    palette: str = "viridis" if score in ["r", "r2"] else "viridis_r"
    custom_cmap = sns.color_palette(palette, as_cmap=True)
    custom_cmap.set_bad(color="gray")
    hmap = sns.heatmap(avg_scores,
                       annot=annotations,
                       fmt="",
                       cmap=custom_cmap,
                       cbar=True,
                       ax=ax,
                       mask=avg_scores.isnull(),
                       annot_kws={"fontsize": 12})

    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(avg_scores.columns)) + 0.5)
    ax.set_yticks(np.arange(len(avg_scores.index)) + 0.5)
    x_tick_labels: list[str] = [col.title() for col in avg_scores.columns]
    y_tick_labels: list[str] = [model_abbrev_to_full[x] for x in avg_scores.index]
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_tick_labels, rotation=0, ha="right")

    # Set plot and axis titles
    plt.title(fig_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    # Set colorbar title
    score: str = "$R^2$" if score == "r2" else score
    cbar = hmap.collections[0].colorbar
    cbar.set_label(f"Average {score.upper()} ± {var_titles[var]}", rotation=270, labelpad=20)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(root_dir / f"{fname}.png", dpi=600)

    # Show the heatmap
    plt.show()


def create_grid_search_heatmap(root_dir: Path, score: str, var: str) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
    """
    # Collect x-axis labels from directory names
    # y_labels: List[str] = ["fabrication only", "OHE", "material properties", "SMILES", "SELFIES", "ECFP",
    #                        "mordred", "GNN"][::-1]
    # x_labels: List[str] = ["MLR", "KRR", "KNN", "SVR", "RF", "XGB", "HGB", "NGB", "GP", "NN", "GNN"]
    x_labels: List[str] = ["fabrication only", "OHE", "material properties", "SMILES", "SELFIES", "ECFP",
                           "mordred", "GNN"]
    y_labels: List[str] = ["MLR", "KRR", "KNN", "SVR", "RF", "XGB", "HGB", "NGB", "GP", "NN", "GNN"][::-1]

    target: str = ", ".join(root_dir.name.split("_")[1:])
    score_txt: str = "$R^2$" if score == "r2" else score.upper()

    _create_heatmap(root_dir,
                    score, var,
                    x_labels=x_labels, y_labels=y_labels,
                    figsize=(12, 8),
                    fig_title=f"Average {score_txt} Scores for Models Predicting {target}",
                    x_title="Structural Representations",
                    y_title="Machine Learning Regression Models",
                    fname=f"model-representation search heatmap_{score}"
                    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    for target in ["PCE", "Voc", "Jsc", "FF"]:
        results = root / "results" / f"target_{target}"

        # Use pathlib glob to get all directories in results
        directory_paths: List[Path] = [dir for dir in results.glob("*") if dir.is_dir()]

        # Create heatmap
        for score in ["r", "r2", "rmse", "mae"]:
            create_grid_search_heatmap(results, score, var="stderr")
