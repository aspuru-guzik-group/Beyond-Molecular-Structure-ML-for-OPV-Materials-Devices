import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score
from typing import Callable


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


calculation_factory: dict[str, Callable] = {"pearson": calculate_pearson,
                                            "r2":      calculate_r2,
                                            "rmse":    calculate_rmse,
                                            "mae":     calculate_mae,
                                            }

correlation_labels: dict[str, str] = {"pearson": "Pearson Correlation (R)",
                                      "r2":      "R2 Correlation",
                                      "rmse":    "RMSE Correlation",
                                      "mae":     "MAE Correlation",
                                      }


class HeatmapGenerator:
    def __init__(self, file: Path, properties: list[str], plot_name: str) -> None:
        self.full_df: pd.DataFrame = pd.read_csv(file)
        self.df: pd.DataFrame = self.full_df[properties]
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
        sns.heatmap(corr, cmap=cmap, vmin=-self.get_colorbar_range(corr), vmax=self.get_colorbar_range(corr), center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f",
                    cbar_kws={"shrink": .5, "label": correlation_labels[method]},
                    annot_kws={"fontsize": 10})
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
        self.f.savefig(f"correlation_{self.plot_name.lower()}_{method.lower()}.png")
        plt.close()


def plot_all_methods(file: Path, properties: list[str], plot_name: str) -> None:
    """
    Plot the correlation matrix for all methods.

    Args:
        file: The path to the dataframe to plot.
        properties: The properties to plot.
        plot_name: The name of the plot.

    Returns:
        None
    """
    heatmap: HeatmapGenerator = HeatmapGenerator(file, properties, plot_name)
    for method in calculation_factory.keys():
        heatmap.plot(method)


if __name__ == "__main__":
    df_path: Path = Path.home() / "Downloads" / "HSPiP solvent properties.csv"
    props: list[str] = ["dipole", "dD", "dP", "dH", "dHDon", "dHAcc", "MW", "Density", "BPt", "MPt", "logKow",
                        "RI", "Trouton", "RER", "ParachorGA", "RD", "DCp", "log n", "SurfTen"]
    plot_all_methods(df_path, props, "solvent properties")
