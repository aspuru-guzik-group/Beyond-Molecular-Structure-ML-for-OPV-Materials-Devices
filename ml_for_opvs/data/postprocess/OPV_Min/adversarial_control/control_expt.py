from doctest import master
from msilib.schema import Control
import pandas as pd
import numpy as np
import pkg_resources

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/hw_frag/train_frag_master.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)


class ControlExperiments:
    """
    Class that contains functions for creating control experiments in the data.
    For example: random shuffling, barcode representations, noise injection
    """

    def __init__(self, data_path):
        """
        Instantiate class with appropriate data.

        Args:
            data_path: path to training data to be shuffled

        Returns:
            None
        """
        self.data_path = data_path

    def shuffle(self):
        """
        Shuffles the PCE(%) by randomly rearranging the PCE(%) for each D-A pair
        
        Args:
            None
        
        Returns:
            The input .csv file with a new column of randomly shuffled PCE
        """
        seed = 0
        np.random.seed(seed)
        main_df = pd.read_csv(self.data_path)
        pre_shuffle_pce = main_df["PCE(%)"].values
        post_shuffle_pce = np.random.permutation(pre_shuffle_pce)

        for i in range(len(pre_shuffle_pce)):
            if pre_shuffle_pce[i] == post_shuffle_pce[i]:
                seed += 1
                np.random.seed(seed)
                post_shuffle_pce = np.random.permutation(pre_shuffle_pce)

        main_df["PCE(%)_shuffled"] = post_shuffle_pce

        main_df.to_csv(self.data_path, index=False)

    def barcode(self):
        "Returns the input .csv file with a new column of barcode representations"
        pass

    def noisy_pce(self):
        "Returns the input .csv file with a new column of noisy PCE"
        # might have to do while training
        pass


ctrl = ControlExperiments(FP_MASTER_DATA)
ctrl.shuffle()
