import json
from itertools import product
from pathlib import Path
from typing import List

# import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

HERE: Path = Path(__file__).parent
filters = HERE.parent / "training" / "filters.json"
with open(filters, "r") as f:
    FILTERS: dict = json.load(f)

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 16})

score_bounds: dict[str, int] = {"r": 1, "r2": 1, "mae": 25, "rmse": 25}

var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}


def get_results_from_file(results_dir: Path, score: str, var: str, impute: bool = False) -> tuple[float, float]:
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
    pattern: str = "*imputer_scores.json" if impute else "*_scores.json"
    score_files: list[Path] = list(results_dir.rglob(pattern))

    for file in score_files:
        if not file.exists():
            features, model = None, None
            avg, std = np.nan, np.nan
        else:
            if impute:
                imputer_txt: str = file.stem.split("_")[1]
                features = " ".join(imputer_txt.split(" ")[:-1])
                pass
            else:
                features = file.parent.name.split("_")[-1]
            # features =  if impute else file.parent.name.split("_")[-1]
            model = file.name.split("_")[0]

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
        yield features, model, avg, std


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
    "MLR": "Linear",
    "KRR": "Kernel Ridge",
    "KNN": "K-Nearest Neighbors",
    "SVR": "Support Vector",
    "RF":  "Random Forest",
    "XGB": "Gradient Boosted Trees",
    "HGB": "Histogram Gradient Boosting",
    "NGB": "Natural Gradient Boosting",
    "GP":  "Gaussian Process",
    "NN":  "Neural Network",
    "ANN": "Artificial Neural Network",
    "GNN": "Graph Neural Network",
}


def _create_heatmap(root_dir: Path,
                    score: str, var: str,
                    x_labels: list[str], y_labels: list[str],
                    parent_dir_labels: list[str],
                    figsize: tuple[int, int],
                    fig_title: str,
                    x_title: str,
                    y_title: str,
                    fname: str,
                    vmin: float = None,
                    vmax: float = None,
                    **kwargs
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

    # for parent, model in product(parent_dir_labels, y_labels):
    for parent in parent_dir_labels:
        p = root_dir / parent
        for feats, model, avg, std in get_results_from_file(p, score, var, **kwargs):
            if model is None:
                continue
            if model not in annotations.index or feats not in annotations.columns:
                continue
            avg_scores.at[model, feats] = avg
            std_scores.at[model, feats] = std

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
    # # palette: str = "cmc.batlow" if score in ["r", "r2"] else "cmc.batlow_r"
    # palette: str = "cmc.batlow_r" if score in ["r", "r2"] else "cmc.batlow"
    custom_cmap = sns.color_palette(palette, as_cmap=True)
    custom_cmap.set_bad(color="lightgray")
    hmap = sns.heatmap(avg_scores,
                       annot=annotations,
                       fmt="",
                       cmap=custom_cmap,
                       cbar=True,
                       vmin=vmin,
                       vmax=vmax,
                       ax=ax,
                       mask=avg_scores.isnull(),
                       annot_kws={"fontsize": 12})

    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(avg_scores.columns)) + 0.5)
    ax.set_yticks(np.arange(len(avg_scores.index)) + 0.5)
    x_tick_labels: list[str] = [col for col in avg_scores.columns]
    y_tick_labels: list[str] = [model_abbrev_to_full[x] for x in avg_scores.index]
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_tick_labels, rotation=0, ha="right")

    # Set plot and axis titles
    plt.title(fig_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    # Set colorbar title
    score_txt: str = "$R^2$" if score == "r2" else score
    cbar = hmap.collections[0].colorbar
    cbar.set_label(f"Average {score_txt.upper()} ± {var_titles[var]}", rotation=270, labelpad=20)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(root_dir / f"{fname}.png", dpi=600)

    # Show the heatmap
    # plt.show()
    plt.close()


def create_grid_search_heatmap(root_dir: Path, score: str, var: str) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
    """
    x_labels: List[str] = ["fabrication only", "OHE", "material properties", "SMILES", "SELFIES", "ECFP",
                           "mordred", "graph embeddings", "GNN"]
    y_labels: List[str] = ["MLR", "KRR", "KNN", "SVR", "RF", "XGB", "HGB", "NGB", "GP", "NN", "ANN", "GNN"][::-1]
    parent_dir_labels: list[str] = [f"features_{x}" for x in x_labels]

    target: str = ", ".join(root_dir.name.split("_")[1:])
    score_txt: str = "$R^2$" if score == "r2" else score.upper()

    _create_heatmap(root_dir,
                    score, var,
                    x_labels=x_labels, y_labels=y_labels,
                    parent_dir_labels=parent_dir_labels,
                    figsize=(12, 8),
                    fig_title=f"Average {score_txt} Scores for Models Predicting {target}",
                    x_title="Structural Representations",
                    y_title="Regression Models",
                    fname=f"model-representation search heatmap_{score}"
                    )


def create_fabrication_grid_heatmap(root_dir: Path, score: str, var: str) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
    """
    x_labels: list[str] = ["material properties", "fabrication", "device architecture"]
    y_labels: list[str] = ["SVR", "RF", "XGB", "HGB", "NGB", "NN", "ANN"][::-1]

    representations: list[str] = ["ECFP", "mordred"]
    # Create the product of representations and x_labels where they're joined as f"{rep}_{x_label}"
    feature_labels: list[str] = [f"{rep}-{x_label}" for rep, x_label in product(representations, x_labels)]
    parent_dir_labels: list[str] = [f"features_{x}" for x in feature_labels]

    target: str = ", ".join(root_dir.name.split("_")[1:])
    score_txt: str = "$R^2$" if score == "r2" else score.upper()

    _create_heatmap(root_dir,
                    score, var,
                    x_labels=feature_labels, y_labels=y_labels,
                    parent_dir_labels=parent_dir_labels,
                    figsize=(12, 8),
                    fig_title=f"Average {score_txt} Scores for Models Predicting {target}",
                    x_title="Structural Representations",
                    y_title="Regression Models",
                    fname=f"model-processing search heatmap_{score}"
                    )


# def _create_subspace_heatmap(root_dir: Path,
#                     score: str, var: str,
#                     x_labels: list[str], y_labels: list[str],
#                     parent_dir_labels: list[str],
#                     figsize: tuple[int, int],
#                     fig_title: str,
#                     x_title: str,
#                     y_title: str,
#                     fname: str,
#                     ) -> None:
#     avg_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
#     std_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
#     annotations: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
#
#     for parent, model in product(parent_dir_labels, y_labels):
#         p = root_dir / parent
#         feats = parent.split("_")[-1]
#         avg, std = get_results_from_file(p, model, score, var)
#         avg_scores.at[model, feats] = avg
#         std_scores.at[model, feats] = std
#
#     for x, y in product(x_labels, y_labels):
#         avg: float = avg_scores.loc[y, x]
#         std: float = std_scores.loc[y, x]
#         avg_txt: str = generate_annotations(avg)
#         std_txt: str = generate_annotations(std)
#         annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"
#
#     avg_scores = avg_scores.astype(float)
#     annotations = annotations.astype(str)
#
#     # Create heatmap
#     fig, ax = plt.subplots(figsize=figsize)
#     palette: str = "viridis" if score in ["r", "r2"] else "viridis_r"
#     custom_cmap = sns.color_palette(palette, as_cmap=True)
#     custom_cmap.set_bad(color="gray")
#     hmap = sns.heatmap(avg_scores,
#                        annot=annotations,
#                        fmt="",
#                        cmap=custom_cmap,
#                        cbar=True,
#                        ax=ax,
#                        mask=avg_scores.isnull(),
#                        annot_kws={"fontsize": 12})
#
#     # Set axis labels and tick labels
#     ax.set_xticks(np.arange(len(avg_scores.columns)) + 0.5)
#     ax.set_yticks(np.arange(len(avg_scores.index)) + 0.5)
#     x_tick_labels: list[str] = [col for col in avg_scores.columns]
#     y_tick_labels: list[str] = [model_abbrev_to_full[x] for x in avg_scores.index]
#     ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
#     ax.set_yticklabels(y_tick_labels, rotation=0, ha="right")
#
#     # Set plot and axis titles
#     plt.title(fig_title)
#     ax.set_xlabel(x_title)
#     ax.set_ylabel(y_title)
#     # Set colorbar title
#     score_txt: str = "$R^2$" if score == "r2" else score
#     cbar = hmap.collections[0].colorbar
#     cbar.set_label(f"Average {score_txt.upper()} ± {var_titles[var]}", rotation=270, labelpad=20)
#
#     # Adjust layout and save figure
#     plt.tight_layout()
#     plt.savefig(root_dir / f"{fname}.png", dpi=600)
#
#     # Show the heatmap
#     plt.show()

def create_subspace_grid_heatmap(root_dir: Path, score: str, var: str) -> None:
    spaces: list[str] = ["fabrication", "device architecture"]
    y_labels: list[str] = ["SVR", "RF", "XGB", "HGB", "NGB", "NN"][::-1]

    representations: list[str] = ["ECFP", "mordred"]
    # Create the product of representations and x_labels where they're joined as f"{rep}_{x_label}"
    feature_labels: list[str] = [f"{rep}-{space}" for rep, space in product(representations, spaces)]

    target: str = ", ".join(root_dir.name.split("_")[1:])
    score_txt: str = "$R^2$" if score == "r2" else score.upper()

    # for feats in spaces:
    #     space = feats.split("-")[1]
    #     space_dir = root_dir / f"features_{feats}"
    #
    #     feature_labels: list[str] = [subspace for subspace in FILTERS[space]]
    # parent_dir_labels: list[str] = [f"features_{x}" for x in feature_labels]

    for feats in feature_labels:
        space = feats.split("-")[1]
        subspace_labels: list[str] = [feats, *FILTERS[space][:-1]]
        parent_dir_labels: list[str] = [f"features_{feats}"]
        # root_dir = root_dir / f"features_{feats}"

        _create_heatmap(root_dir,
                        score, var,
                        x_labels=subspace_labels, y_labels=y_labels,
                        parent_dir_labels=parent_dir_labels,
                        figsize=(12, 8),
                        fig_title=f"Average {score_txt} Scores for Models Predicting {target} with {feats.replace('-', ', ')} Features",
                        x_title="Structural Representations",
                        y_title="Regression Models",
                        fname=f"{feats} subspace search heatmap_{score}",
                        )


def create_impute_grid_heatmap(root_dir: Path, score: str, var: str) -> None:
    # imputers: list[str] = ["mean", "median", "most-frequent", "uniform KNN", "distance KNN", "iterative"]
    y_labels: list[str] = ["RF", "XGB", "HGB", "NGB"][::-1]

    representations: list[str] = ["ECFP", "mordred"]
    # Create the product of representations and x_labels where they're joined as f"{rep}_{x_label}"
    feature_labels: list[str] = ["mean", "median", "most-frequent", "uniform KNN", "distance KNN", "iterative"]

    target: str = ", ".join(root_dir.name.split("_")[1:])
    score_txt: str = "$R^2$" if score == "r2" else score.upper()

    for feats in feature_labels:
        # space = feats.split("-")[1]
        # subspace_labels: list[str] = [feats, *FILTERS[space][:-1]]
        parent_dir_labels: list[str] = ["features_mordred-device architecture"]
        # root_dir = root_dir / f"features_{feats}"

        _create_heatmap(root_dir,
                        score, var,
                        x_labels=feature_labels, y_labels=y_labels,
                        parent_dir_labels=parent_dir_labels,
                        figsize=(12, 8),
                        fig_title=f"Average {score_txt} Scores for Models Predicting {target} with Imputing",
                        x_title="Structural Representations",
                        y_title="Regression Models",
                        fname=f"impute search heatmap_{score}",
                        impute=True
                        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent

    # Model, representation and processing search heatmaps
    # for target in ["Voc", "Jsc", "FF"]:
    for target in ["PCE"]:
        results = root / "results" / f"target_{target}"

        # Create heatmap
        # for score in ["r", "r2", "rmse", "mae"]:
        #     create_grid_search_heatmap(results, score, var="stderr")

        for score in ["r", "r2", "rmse", "mae"]:
            create_fabrication_grid_heatmap(results, score, var="stderr")

    # # Subspace search heatmaps
    # for target in ["Voc", "Jsc", "FF"]:
    #     results = root / "results" / f"target_{target}"

    #     # Create heatmap
    #     for score in ["r", "r2", "rmse", "mae"]:
    #         create_subspace_grid_heatmap(results, score, var="stderr")

    # Impute search heatmaps
    # for target in ["PCE"]:
    #     results = root / "results" / f"target_{target}"
    #
    #     # Create heatmap
    #     for score in ["r", "r2", "rmse", "mae"]:
    #         create_impute_grid_heatmap(results, score, var="stderr")
