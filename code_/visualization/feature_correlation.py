import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional

# from code_ import DATASETS, FIGURES
# from code_.training import unroll_lists_to_columns
# import sys
# sys.path.append("../training")
# # from code_.training import pipeline_utils
# from code_.training.pipeline_utils import unroll_lists_to_columns

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
FIGURES = Path(__file__).resolve().parents[2] / "figures"

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 16})


def calculate_pearson(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlation matrix for a given dataframe."""
    return df.corr(method="pearson")


def calculate_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlation matrix for a given dataframe."""
    return df.corr(method="spearman")


calculation_factory: dict[str, Callable] = {
    "pearson": calculate_pearson,
    "spearman": calculate_spearman,
}

correlation_labels: dict[str, str] = {
    "pearson": "Pearson Correlation (R)",
    "spearman": r"Spearman Correlation ($\rho$)",
}


def annotate_heatmap(data, cmap, **kwargs):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i == j:  # Check if it's the diagonal element (self-correlation)
                color = 'lightgrey'
                text = ''
            else:
                color = cmap(data.iloc[i, j])
                text = f"{data.iloc[i, j]:.1f}"
            plt.text(j + 0.5, i + 0.5, text, **kwargs, ha="center", va="center", color=color)


def unroll_lists_to_cols_old(df: pd.DataFrame, unroll_cols: list[str], new_col_names: list[str]) -> pd.DataFrame:
    """
    Unroll a list of lists into columns of a DataFrame.

    Args:
        df: DataFrame to unroll.
        unroll_cols: List of columns containing list to unroll.
        new_col_names: List of new column names.

    Returns:
        DataFrame with unrolled columns.
    """
    rolled_cols: pd.DataFrame = df[unroll_cols]
    unrolled_df: pd.DataFrame = pd.concat([rolled_cols[col].apply(pd.Series) for col in rolled_cols.columns], axis=1)
    unrolled_df.columns = new_col_names
    return unrolled_df


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
        heatmap = sns.heatmap(
            corr,
            cmap=cmap,
            vmin=-self.get_colorbar_range(corr),
            vmax=self.get_colorbar_range(corr),
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".1f",
            cbar_kws={"shrink": 0.5, "label": correlation_labels[method]},
            annot_kws={"fontsize": 12},
        )

        self.ax.set_title(f"Heatmap of {self.plot_name.title()} using {method.title()}")

        # Rotate x-axis labels by 45 degrees
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(), rotation=45, horizontalalignment='right'
        )

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
            figure_dir / f"correlation_{self.plot_name.lower()}_{method.lower()}.png",
            dpi=300
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
        solvents: pd.DataFrame = unroll_lists_to_cols_old(df, [descriptors], new_cols)
        solvent_correlations: pd.DataFrame = pd.concat(
            [
                solvents,
                df[["Voc (V)", "Jsc (mA cm^-2)", "FF (%)", "calculated PCE (%)"]],
            ],
            axis=1,
        )
        plot_all_methods(solvent_correlations, None, descriptors)


def plot_processing_correlations(df: pd.DataFrame) -> None:
    process_feats: list[str] = [
        "D:A ratio (m/m)",
        "Active layer spin coating speed (rpm)",
        "total solids conc. (mg/mL)",
        "solvent additive conc. (% v/v)",
        "active layer thickness (nm)",
        "temperature of thermal annealing",
        "annealing time (min)",
        "Voc (V)",
        "Jsc (mA cm^-2)",
        "FF (%)",
        "calculated PCE (%)",
    ]
    plot_all_methods(df, process_feats, "processing")


def plot_device_correlations(df: pd.DataFrame) -> None:
    device_feats: list[str] = [
        "HTL energy level (eV)",
        "HTL thickness (nm)",
        "ETL energy level (eV)",
        "ETL thickness (nm)",
        "Voc (V)",
        "Jsc (mA cm^-2)",
        "FF (%)",
        "calculated PCE (%)",
    ]
    plot_all_methods(df, device_feats, "device architecture")


def plot_electrical_correlations(df: pd.DataFrame) -> None:
    electrical_feats: list[str] = [
        "hole mobility blend (cm^2 V^-1 s^-1)",
        "electron mobility blend (cm^2 V^-1 s^-1)",
        "hole:electron mobility ratio",
        "log hole mobility blend (cm^2 V^-1 s^-1)",
        "log electron mobility blend (cm^2 V^-1 s^-1)",
        "log hole:electron mobility ratio",
        "Voc (V)",
        "Jsc (mA cm^-2)",
        "FF (%)",
        "calculated PCE (%)",
    ]
    plot_all_methods(df, electrical_feats, "electrical characteristics")


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

    plot_material_correlations(opv_dataset)
    plot_processing_correlations(opv_dataset)
    plot_device_correlations(opv_dataset)
    plot_electrical_correlations(opv_dataset)
    plot_solvent_correlations(opv_dataset)

