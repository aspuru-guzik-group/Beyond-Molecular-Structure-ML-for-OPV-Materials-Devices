from typing import List
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)


class MissingParameters:
    """
    Class that contains functions to evaluate any parameter. Potential tools:
    - Find missing parameters and return the D-A pair
    - Automate processing of parameter
    """

    def __init__(self, master_data):
        """
        Instantiates class with appropriate data

        Args:
            master_data: path to opv data from pre-processing

        Returns:
            None
        """
        self.data = pd.read_csv(master_data)

    def search_missing(self, column_names):
        """
        Searches for D-A pairs with missing values in the specified column.

        Args:
            column_names: columns that we want to search for

        Returns:
            missing_df: Dataframe with Donor | Acceptor | Column Names (w/ values)
        """
        pass
