from cmath import nan
from turtle import color
import numpy as np
import pkg_resources
import pandas as pd
import copy
import matplotlib.pyplot as plt
import itertools

RESULTS_PATH = pkg_resources.resource_filename("ml_for_opvs", "opv_results_v2.csv")


class PLOT_RESULTS:
    """
    Class that contains functions to plot the results in intuitive ways:
    It should have customizable filters for any combination of parameters to plot,
    plot barplots or heatmaps
    """

    def __init__(self, results_path):
        self.results = pd.read_csv(results_path)

    def filter_results(self, model, datatype, parameter, target, stat):
        """
        A customized bar plot that plots the statistics (R or RMSE) on the y-axis, 
        and the corresponding parameters used on the x-axis.
        NOTE: the argument that is specified as "None" will be the one you compare against.
        Args:
            model: choose one model to compare
            datatype: choose one datatype to compare
            parameter: choose one parameter to compare
            target: choose one target value to compare
            stat: choose between R or RMSE
        Returns:
            avg_array, std_array, xlabels, ylabel
        """
        arg_list = [model, datatype, parameter, target]
        arg_list = list(filter((None).__ne__, arg_list))
        print(arg_list)
        avg_array = []
        std_array = []
        xlabel = []
        num_of_data_array = []
        # baseline_comparison
        if stat == "R":
            # Min paper
            avg_array.append(0.71)
            std_array.append(0)
            xlabel.append("BRT_Fragments" + "\n" + "PCE_None")
            num_of_data_array.append(565)
            # Saeki paper
            avg_array.append(0.85)
            std_array.append(0.02)
            xlabel.append("RF_Fingerprints_PCE" + "\n" + "Material_Properties")
            num_of_data_array.append(566)
        ylabel = stat
        stat_avg = stat + "_mean"
        stat_std = stat + "_std"
        for index, row in self.results.iterrows():
            feature_list = [
                self.results.at[index, "Model"],
                self.results.at[index, "Datatype"],
                self.results.at[index, "Parameter"],
                self.results.at[index, "Target"],
            ]
            # 1 feature
            if len(arg_list) == 1:
                for comb in list(itertools.combinations(feature_list, 1)):
                    feature_comb_set = set(comb)
                    if set(arg_list) == feature_comb_set:
                        if str(self.results.at[index, stat_avg]) != "nan":
                            avg_array.append(round(self.results.at[index, stat_avg], 3))
                            std_array.append(self.results.at[index, stat_std])
                            num_of_data_array.append(
                                self.results.at[index, "num_of_data"]
                            )
                            xlabel.append(
                                feature_list[0]
                                + "_"
                                + feature_list[1]
                                + "\n"
                                + feature_list[2]
                                + "_"
                                + feature_list[3]
                            )
            # 2 feature
            elif len(arg_list) == 2:
                for comb in list(itertools.combinations(feature_list, 2)):
                    feature_comb_set = set(comb)
                    if set(arg_list) == feature_comb_set:
                        if str(self.results.at[index, stat_avg]) != "nan":
                            avg_array.append(round(self.results.at[index, stat_avg], 3))
                            std_array.append(self.results.at[index, stat_std])
                            num_of_data_array.append(
                                self.results.at[index, "num_of_data"]
                            )
                            xlabel.append(
                                feature_list[0]
                                + "_"
                                + feature_list[1]
                                + "\n"
                                + feature_list[2]
                                + "_"
                                + feature_list[3]
                            )
            # 3 feature
            elif len(arg_list) == 3:
                for comb in list(itertools.combinations(feature_list, 3)):
                    feature_comb_set = set(comb)
                    if set(arg_list) == feature_comb_set:
                        if str(self.results.at[index, stat_avg]) != "nan":
                            avg_array.append(round(self.results.at[index, stat_avg], 3))
                            std_array.append(self.results.at[index, stat_std])
                            num_of_data_array.append(
                                self.results.at[index, "num_of_data"]
                            )
                            xlabel.append(
                                feature_list[0]
                                + "_"
                                + feature_list[1]
                                + "\n"
                                + feature_list[2]
                                + "_"
                                + feature_list[3]
                            )
            # 4 feature
            elif len(arg_list) == 4:
                for comb in list(itertools.combinations(feature_list, 4)):
                    feature_comb_set = set(comb)
                    if set(arg_list) == feature_comb_set:
                        if str(self.results.at[index, stat_avg]) != "nan":
                            avg_array.append(round(self.results.at[index, stat_avg], 3))
                            std_array.append(self.results.at[index, stat_std])
                            num_of_data_array.append(
                                self.results.at[index, "num_of_data"]
                            )
                            xlabel.append(
                                feature_list[0]
                                + "_"
                                + feature_list[1]
                                + "\n"
                                + feature_list[2]
                                + "_"
                                + feature_list[3]
                            )

        return avg_array, std_array, xlabel, ylabel, num_of_data_array

    def barplot(self, avg_array, std_array, xlabel, ylabel, num_of_data_array):
        """
        Produces twin barplot from results given by self.filter_results
        Args:
            avg_array: array of results' average
            std_array: array of results' std
            xlabel: list of labels corresponding to results
            ylabel: are we measuring R or RMSE?
            num_of_data_array: number of datapoints for each condition
        Returns:
            .png barplot of averages and standard deviation
        """
        fig, ax1 = plt.subplots()
        color_list = [None] * len(avg_array)
        idx = 0
        while idx < len(color_list):
            if idx < 2:
                color_list[idx] = "orange"
            else:
                color_list[idx] = "blue"
            idx += 1

        idx = 0
        while idx < len(avg_array):
            if str(avg_array[idx]) == "nan":
                nan_index = idx
                break
            idx += 1

        try:
            x_axis_positions = np.arange(len(avg_array[0:nan_index]))
            avg_array = avg_array[0:nan_index]
            std_array = std_array[0:nan_index]
            num_of_data_array = num_of_data_array[0:nan_index]
            xlabel = xlabel[0:nan_index]
        except:
            x_axis_positions = np.arange(len(avg_array))

        bar1 = ax1.bar(
            x=x_axis_positions - 0.2,
            height=avg_array,
            width=0.4,
            yerr=std_array,
            color="blue",
        )
        ax1.set_ylabel(ylabel, color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_xticklabels(xlabel)
        ax1.set_xticks(x_axis_positions)
        ax1.bar_label(bar1, avg_array)

        ax2 = ax1.twinx()
        ax2.set_ylabel("number of datapoints", color="green")
        bar2 = ax2.bar(
            x=x_axis_positions + 0.2,
            height=num_of_data_array,
            width=0.4,
            color="green",
        )
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.bar_label(bar2, num_of_data_array)

        plt.show()


plot = PLOT_RESULTS(RESULTS_PATH)
avg_array, std_array, xlabel, ylabel, num_of_data_array = plot.filter_results(
    "RF", "Fingerprints (r=3, bits=512)", "electronic", None, "R"
)
print(avg_array, std_array, xlabel, ylabel, num_of_data_array)
plot.barplot(avg_array, std_array, xlabel, ylabel, num_of_data_array)

