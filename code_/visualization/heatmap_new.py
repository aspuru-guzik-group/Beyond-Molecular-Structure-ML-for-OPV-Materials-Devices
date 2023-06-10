import json
import pickle
from itertools import product
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

score_bounds: dict[str, int] = {"r": 1, "r2": 1, "mae": 10, "rmse": 10}


def get_results_from_file(root_dir: Path, representation: str, model: str, score: str) -> tuple[float, float]:
    """
    Args:
        struct_rep: Structural representation
        model: Machine learning model

    Returns:
        Average score from JSON file
    """
    score_file: list[Path] = list(root_dir.glob(f"{model}_{representation}*_scores.json"))
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
        std = data[f"{score}_stdev"]

    avg: float = np.nan if abs(avg) > score_bounds[score] else avg
    std: float = np.nan if abs(std) > score_bounds[score] else std
    return avg, std


def generate_annotations(num: float) -> str:
    if isinstance(num, float) and not np.isnan(num):
        num_txt: str = f"{round(num, 2)}"
    else:
        num_txt = "NaN"
    return num_txt


def create_grid_search_heatmap(root_dir: Path, score: str) -> None:
    # Collect x-axis labels from directory names
    # TODO: MOre flfexible
    y_labels: List[str] = ["fabrication only", "OHE", "material properties", "SMILES", "SELFIES", "BRICS", "ECFP5-2048",
                           "mordred", "GNN"][::-1]
    x_labels: List[str] = ["MLR", "Lasso", "KRR", "KNN", "SVR", "RF", "XGB", "HGB", "NGB", "GP", "NN", "GNN"]

    avg_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    std_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)

    for rep, model in product(y_labels, x_labels):
        p = root_dir / rep
        avg, std = get_results_from_file(p, rep, model, score)
        avg_scores.at[rep, model] = avg
        std_scores.at[rep, model] = std

    annotations: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    for x, y in product(x_labels, y_labels):
        avg: float = avg_scores.loc[y, x]
        std: float = std_scores.loc[y, x]
        avg_txt: str = generate_annotations(avg)
        std_txt: str = generate_annotations(std)
        annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"

    avg_scores = avg_scores.astype(float)
    annotations = annotations.astype(str)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    custom_cmap = sns.color_palette("viridis", as_cmap=True)
    custom_cmap.set_bad(color="gray")
    hmap = sns.heatmap(avg_scores,
                       annot=annotations,
                       fmt="",
                       cmap=custom_cmap,
                       cbar=True,
                       ax=ax,
                       mask=avg_scores.isnull())

    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(avg_scores.columns)) + 0.5)
    ax.set_yticks(np.arange(len(avg_scores.index)) + 0.5)
    ax.set_xticklabels(avg_scores.columns, rotation=45, ha='right')
    ax.set_yticklabels(avg_scores.index, rotation=0, ha='right')

    # Set plot and axis titles
    plt.title(f"Average Values and Standard Deviations of {score.upper()} Scores")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Structural Representations")
    # Set colorbar title
    cbar = hmap.collections[0].colorbar
    cbar.set_label(f"Average {score.upper()} (± Standard Deviation)", rotation=270, labelpad=20)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(root_dir / f"heatmap_{score}.png", dpi=600)

    # Show the heatmap
    plt.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    results = root / "results" / "structure_only" / "hyperopt"

    # Use pathlib glob to get all directories in results
    directory_paths: List[Path] = [dir for dir in results.glob("*") if dir.is_dir()]

    # Create heatmap
    for score in ["r", "r2", "rmse", "mae"]:
        create_grid_search_heatmap(results, score)
