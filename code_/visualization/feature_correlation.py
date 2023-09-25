import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score
from typing import Callable, Optional

from code_ import DATASETS, FIGURES
from code_.training import unroll_lists_to_columns

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 12})


def calculate_pearson(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlation matrix for a given dataframe."""
    return df.corr(method="pearson")


def calculate_r2(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate R2 correlation matrix for a given dataframe."""
    corr_matrix: np.ndarray = np.zeros((len(df.columns), len(df.columns)))
    for i, col_i in enumerate(df.columns):
        for j, col_j in enumerate(df.columns):
            corr_matrix[i, j] = r2_score(df[col_i], df[col_j])
    return pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)


def calculate_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RMSE correlation matrix for a given dataframe."""
    corr_matrix: np.ndarray = np.zeros((len(df.columns), len(df.columns)))
    for i, col_i in enumerate(df.columns):
        for j, col_j in enumerate(df.columns):
            corr_matrix[i, j] = np.sqrt(np.mean((df[col_i] - df[col_j]) ** 2))
    return pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)


def calculate_mae(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MAE correlation matrix for a given dataframe."""
    corr_matrix: np.ndarray = np.zeros((len(df.columns), len(df.columns)))
    for i, col_i in enumerate(df.columns):
        for j, col_j in enumerate(df.columns):
            corr_matrix[i, j] = np.mean(np.abs(df[col_i] - df[col_j]))
    return pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)


calculation_factory: dict[str, Callable] = {
    "pearson": calculate_pearson,
    "r2": calculate_r2,
    "rmse": calculate_rmse,
    "mae": calculate_mae,
}

correlation_labels: dict[str, str] = {
    "pearson": "Pearson Correlation (R)",
    "r2": "R2 Correlation",
    "rmse": "RMSE Correlation",
    "mae": "MAE Correlation",
}


class HeatmapGenerator:
    def __init__(
        self, full_df: pd.DataFrame, properties: Optional[list[str]], plot_name: str
    ) -> None:
        if properties:
            self.df: pd.DataFrame = full_df[properties]
        else:
            self.df: pd.DataFrame = full_df

        self.plot_name: str = plot_name

    def calculate_matrix(self, method: str) -> pd.DataFrame:
        """
        Calculate the correlation matrix with the given method.

        Args:
            method: The method to use to calculate the correlation matrix.
                Options are "pearson", "r2", "rmse", and "mae".

        Returns:
            The correlation matrix as a pandas DataFrame.
        """
        return calculation_factory[method](self.df)

    @staticmethod
    def get_colorbar_range(correlation_matrix: pd.DataFrame) -> float:
        """
        Get the largest absoluate value in correlation_matrix and return. If value is less than 1, return 1.

        Args:
            correlation_matrix: The correlation matrix.

        Returns:
            The minimum and maximum values for the colorbar.
        """
        max_val: float = correlation_matrix.abs().max().max()
        if max_val < 1:
            max_val = 1
        return max_val

    def plot(self, method: str, show: bool = True, save: bool = True) -> None:
        """
        Plot the correlation matrix with the given method.

        Args:
            method: The method to use to calculate the correlation matrix.
                Options are "pearson", "r2", "rmse", and "mae".
            show: Whether to show the figure.
            save: Whether to save the figure to a file.

        Returns:
            None
        """
        corr: pd.DataFrame = self.calculate_matrix(method)
        sns.set(style="white")
        cmap = sns.color_palette("icefire", as_cmap=True)
        self.f, self.ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            corr,
            cmap=cmap,
            vmin=-self.get_colorbar_range(corr),
            vmax=self.get_colorbar_range(corr),
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.5, "label": correlation_labels[method]},
            annot_kws={"fontsize": 10},
        )
        self.ax.set_title(f"Heatmap of {self.plot_name.title()} using {method.title()}")
        plt.tight_layout()

        if show:
            plt.show()
        if save:
            self.save_fig(method)

    def save_fig(self, method: str) -> None:
        """
        Save the figure to a file.

        Args:
            method: The method to use to calculate the correlation matrix.
                Options are "pearson", "r2", "rmse", and "mae".
        """
        figure_dir: Path = FIGURES / "Min"
        self.f.savefig(
            figure_dir / f"correlation_{self.plot_name.lower()}_{method.lower()}.png"
        )
        plt.close()


def plot_all_methods(
    df: pd.DataFrame, properties: Optional[list[str]], plot_name: str
) -> None:
    """
    Plot the correlation matrix for all methods.

    Args:
        file: The path to the dataframe to plot.
        properties: The properties to plot.
        plot_name: The name of the plot.

    Returns:
        None
    """
    heatmap: HeatmapGenerator = HeatmapGenerator(df, properties, plot_name)
    for method in calculation_factory.keys():
        try:
            heatmap.plot(method)
        except ValueError:
            print(f"Could not plot {method} for {plot_name}")


def plot_solvent_correlations(df: pd.DataFrame) -> None:
    with open(DATASETS / "Min_2020_n558" / "selected_properties.json", "r") as f:
        solvent_descriptors: list[str] = json.load(f)["solvent"]
    for descriptors in ["solvent descriptors", "solvent additive descriptors"]:
        new_cols: list[str] = [f"solvent {d}" for d in solvent_descriptors]
        solvents: pd.DataFrame = unroll_lists_to_columns(df, [descriptors], new_cols)
        solvent_correlations: pd.DataFrame = pd.concat(
            [
                solvents,
                df[["Voc (V)", "Jsc (mA cm^-2)", "FF (%)", "calculated PCE (%)"]],
            ],
            axis=1,
        )
        plot_all_methods(solvent_correlations, None, descriptors)


def plot_processing_correlations(df: pd.DataFrame) -> None:
    device_feats: list[str] = [
        "D:A ratio (m/m)",
        "Active layer spin coating speed (rpm)",
        "total solids conc. (mg/mL)",
        "solvent additive conc. (% v/v)",
        "active layer thickness (nm)",
        "temperature of thermal annealing",
        "annealing time (min)",
        "HTL energy level (eV)",
        "HTL thickness (nm)",
        "ETL energy level (eV)",
        "ETL thickness (nm)",
        "hole mobility blend (cm^2 V^-1 s^-1)",
        "electron mobility blend (cm^2 V^-1 s^-1)",
        "hole:electron mobility ratio",
        "Voc (V)",
        "Jsc (mA cm^-2)",
        "FF (%)",
        "calculated PCE (%)",
    ]
    plot_all_methods(df, device_feats, "device fabrication")


def plot_material_correlations(df: pd.DataFrame) -> None:
    material_props: list[str] = [
        "Donor PDI",
        "Donor Mn (kDa)",
        "Donor Mw (kDa)",
        "HOMO_D (eV)",
        "LUMO_D (eV)",
        "Ehl_D (eV)",
        "Eg_D (eV)",
        "HOMO_A (eV)",
        "LUMO_A (eV)",
        "Ehl_A (eV)",
        "Eg_A (eV)",
        "Voc (V)",
        "Jsc (mA cm^-2)",
        "FF (%)",
        "calculated PCE (%)",
    ]
    plot_all_methods(df, material_props, "material properties")


if __name__ == "__main__":
    df_path: Path = DATASETS / "Min_2020_n558" / "cleaned_dataset.pkl"
    opv_dataset: pd.DataFrame = pd.read_pickle(df_path)

    # plot_solvent_correlations(opv_dataset)
    # plot_processing_correlations(opv_dataset)
    plot_material_correlations(opv_dataset)
