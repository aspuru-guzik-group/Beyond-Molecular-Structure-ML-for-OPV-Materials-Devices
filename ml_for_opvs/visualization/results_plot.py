import numpy as np
import pkg_resources
import pandas as pd
import copy
import matplotlib.pyplot as plt


RESULTS_PATH = pkg_resources.resource_filename("ml_for_opvs", "opv_results_v2.csv")

class PLOT_RESULTS:
    """
    Class that contains functions to plot the results in intuitive ways:
    It should have customizable filters for any combination of parameters to plot,
    plot barplots or heatmaps
    """
    def __init__(self, results_path):
        self.results = pd.read_csv(results_path
        )