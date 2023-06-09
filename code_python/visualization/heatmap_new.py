import json
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Generator, List, Union


def get_results_from_file(root_dir: Path, representation: str, model: str, score: str) -> tuple[float, float]:
    """
    Args:
        struct_rep: Structural representation
        model: Machine learning model

    Returns:
        Average score from JSON file
    """
    for f in root_dir.glob(f"{model}_{representation}*_scores.json"):
        with open(f) as json_file:
            data: dict = json.load(json_file)
        avg = data[f"{score}_avg"]
        std = data[f"{score}_stdev"]
    return avg, std


def generate_annotations(num: float) -> str:
    if isinstance(num, float) and not np.isnan(num):
        num_txt: str = f"{round(num, 2)}"
    else:
        num_txt = "NaN"
    return num_txt


def create_model_struct_heatmap(root_dir: Path, score: str) -> None:
    # Collect x-axis labels from directory names
    # TODO: MOre flfexible
    y_labels: List[str] = ["OHE", "material properties", "SMILES", "SELFIES", "BRICS", "ECFP5-2048", "mordred"]
    x_labels: List[str] = ["MLR", "Lasso", "KRR", "KNN", "SVR", "RF", "XGB", "NGB"]

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

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    avg_scores = avg_scores.astype(float).T
    annotations = annotations.astype(str).T  # TODO: Correct pivoting
    sns.heatmap(avg_scores, annot=annotations, fmt="", cmap="viridis", cbar=True, ax=ax, mask=avg_scores.isnull())

    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticks(np.arange(len(x_labels)) + 0.5)
    ax.set_xticklabels(y_labels, rotation=45, ha='right')
    ax.set_yticklabels(x_labels)

    # Set plot title and adjust layout
    plt.title("Heatmap with Average Values and Standard Deviations")
    plt.tight_layout()

    # Show the heatmap
    plt.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    results = root / "results" / "structure_only"

    # Use pathlib glob to get all directories in results
    directory_paths: List[Path] = [dir for dir in results.glob("*") if dir.is_dir()]

    # Create heatmap
    create_model_struct_heatmap(results, "r")
