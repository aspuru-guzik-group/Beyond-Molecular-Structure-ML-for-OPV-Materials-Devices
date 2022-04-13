import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt

from ml_for_opvs.data.preprocess.OPV_Min.clean_donors_acceptors import OPV_DATA

# OPV data after pre-processing
OPV_CLEAN = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)


class Distribution:
    """
    Class that contains functions to determine the distribution of each variable in the dataset.
    Each dataset will have slightly different variable names.
    Must be able to handle numerical and categorical variables.
    """

    def __init__(self, data):
        self.data = pd.read_csv(data)

    def histogram(self):
        """
        Function that plots the histogram of all variables in the dataset
        NOTE: you must know the variable names beforehand

        Args:
            None

        Returns:
            Histogram plots of all the variables.
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)
        # select which columns you want to plot in the histogram
        column_range = range(9, 28)
        for i in column_range:
            current_column = columns[i]


dist = Distribution(OPV_CLEAN)

dist.histogram()
