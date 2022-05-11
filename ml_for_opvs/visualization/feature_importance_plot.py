import numpy as np
import pkg_resources
import pandas as pd
import copy
import matplotlib.pyplot as plt

FEATURE_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "ML_models/sklearn/RF/OPV_Min/"
)

PLOT_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "ML_models/sklearn/RF/OPV_Min/",
)

CURRENT_PATH = "PCE_device_solv_only_opv_rf_feature_impt"

FEATURE_DIR = FEATURE_DIR + CURRENT_PATH + ".csv"

PLOT_PATH = PLOT_PATH + CURRENT_PATH + ".png"


class PLOT_FEATURE:
    """
    Class that contains functions to properly calculate the feature importance and visualize them.
    """

    def __init__(self, feature_path):
        """
        Args:
            feature_path: path to feature importance data from 5-fold cv
        """
        self.data = pd.read_csv(feature_path)

    def calc_avg_std(self) -> dict:
        """
        Function that calculates the average and standard deviation for each feature across the k-folds.
        Args:
            None
        Returns:
            avg_feature_impt: average across k-folds for each feature
            std_feature_impt: standard deviation across k-folds for each feature
        """
        avg_feature_impt = {}
        std_feature_impt = {}
        for index, row in self.data.iterrows():
            avg_feature_impt[row.values[0]] = np.mean(row.values[1 : len(row)])
            std_feature_impt[row.values[0]] = np.std(row.values[1 : len(row)])

        return avg_feature_impt, std_feature_impt

    def plot_feature_impt(self, avg_dict, std_dict, plot_path):
        """
        Function that plots the average and standard deviation of the features, and correctly labels each feature
        Args:
            plot_path: path to feature importance data from 5-fold cv
        """
        avg_df = pd.DataFrame.from_dict(avg_dict, orient="index")
        std_df = pd.DataFrame.from_dict(std_dict, orient="index")

        avg_df.plot(yerr=std_df, kind="bar", ylabel="Importance")
        plt.tight_layout()
        plt.savefig(plot_path)


feature_plot = PLOT_FEATURE(FEATURE_DIR)
avg_feature_impt, std_feature_impt = feature_plot.calc_avg_std()
print(avg_feature_impt)
feature_plot.plot_feature_impt(avg_feature_impt, std_feature_impt, PLOT_PATH)
