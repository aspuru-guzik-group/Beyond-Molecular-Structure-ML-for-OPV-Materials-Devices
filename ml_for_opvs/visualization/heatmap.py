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
    fig, ax = plt.subplots(figsize=(14, 6))
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
    if config["config_name"] == "grid_search":
        x = [
            "MLR",  # ignored for rmse and mae
            "Lasso",
            "KRR",
            "KNN",
            "SVM",
            "RF",
            "XGBoost",
            "NGBoost",
            "GP",
            "NN",
            "GNN",
        ]
        y = [
            "DA_gnn",
            "DA_graphembed",
            "DA_mordred_pca",
            "DA_mordred",
            "DA_FP_radius_3_nbits_1024",
            "DA_tokenized_BRICS",
            "DA_SELFIES",
            "DA_BigSMILES",
            "DA_SMILES",
            "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV",
            # "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV",
            "DA_ohe",
        ]
        x_name = "Model"
        y_name = "Feature_Names"
    elif config["config_name"] == "grid_search_ensemble":
        x = [
            "RF",
            "RF_ensemble",
            "XGBoost",
            "XGBoost_ensemble",
            "SVM",
            "SVM_ensemble",
        ]
        y = ["DA_FP_radius_3_nbits_1024"]
        x_name = "Model"
        y_name = "Target"
    elif config["config_name"] == "grid_search_target":
        x = [
            "RF",
            "XGBoost",
            "SVM",
            "GP",
            "NGBoost",
            "GP_joint",
        ]
        y = ["FF_percent", "Voc_V", "Jsc_mA_cm_pow_neg2", "calc_PCE_percent,FF_percent,Jsc_mA_cm_pow_neg2,Voc_V"]
        x_name = "Model"
        y_name = "Target"
        summary = summary.replace({"RF_ensemble": "RF", "XGBoost_ensemble": "XGBoost", "SVM_ensemble": "SVM", "GP_ensemble": "GP", "NGBoost_ensemble": "NGBoost"})
    elif config["config_name"] == "feature_comparison":
        x = ["RF_ensemble", "XGBoost_ensemble", "SVM_ensemble"]
        # y = [
        #     "DA_FP_radius_3_nbits_1024",
        #     "DA_FP_radius_3_nbits_1024,HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV",
        #     "DA_FP_radius_3_nbits_1024,HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,solvent_additive_conc_v_v_percent",
        # ]
        y = [
            "result_device_wo_thickness",
            "result_fabrication_wo_solid",
            "result_molecules_only",
        ]
        x_name = "Model"
        y_name = "Features"
    elif config["config_name"] == "fab_wo_solid_comparison":
        x = ["RF", "XGBoost", "SVM"]
        summary = summary.replace({"DA_FP_radius_3_nbits_1024": "Fingerprint", "DA_FP_radius_3_nbits_1024,HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV": "Fingerprint + Material", "DA_FP_radius_3_nbits_1024,HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,solvent_additive_conc_v_v_percent": "Fingerprint + Material + Fabrication", "fabrication_wo_solid,solvent_properties": "Fingerprint + Material + Fabrication + Solvent Properties"})
        y = ["Fingerprint", "Fingerprint + Material", "Fingerprint + Material + Fabrication", "Fingerprint + Material + Fabrication + Solvent Properties"]
        x_name = "Model"
        y_name = "Feature_Names"
    # Heatmap
    # NOTE: You have to change the pivot columns depending on your plot
    # remove NaN rows from y_name
    summary = summary.dropna(subset=y_name)
    print(f"{summary.columns=}")
    summary.to_csv("test.csv")
    mean_summary: pd.DataFrame = summary.pivot(x_name, y_name, mean_metric)
    mean_summary: pd.DataFrame = mean_summary.reindex(index=x, columns=y)

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

    # NOTE: You have to change the pivot columns depending on your plot!
    summary_annotated: pd.DataFrame = summary_annotated.pivot(
        x_name, y_name, "annotate_label"
    )
    # mean_summary: pd.DataFrame = mean_summary.T
    # summary_annotated: pd.DataFrame = summary_annotated.T
    summary_annotated: pd.DataFrame = summary_annotated.reindex(index=x, columns=y)
    mean_summary: pd.DataFrame = mean_summary.T
    summary_annotated: pd.DataFrame = summary_annotated.T

    summary_annotated: np.ndarray = summary_annotated.to_numpy()
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.precision",
        3,
    ):
        print(mean_summary)
        print(summary_annotated)
    if config["metrics"] == "r":
        colorbar_label = "$\;R$"
    elif config["metrics"] == "r2":
        colorbar_label = "$\;R^2$"
    ax = sns.heatmap(
        mean_summary,
        annot=summary_annotated,
        cmap="viridis",
        fmt="",
        cbar_kws={
            "label": f"Average of {colorbar_label} \n ($±\;$Standard Deviation of {colorbar_label})"
        },
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
