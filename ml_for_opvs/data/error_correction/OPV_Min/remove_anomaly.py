import pkg_resources
import pandas as pd

# OPV data after pre-processing
MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)
MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
)
# All postprocessing data too!
AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/augmentation/train_aug_master4.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)


class Anomaly:
    """
    Class that contains functions for removing anomalies from the data.
    These anomalies were discussed thoroughly accounting for the impact of removing a few datapoints.
    """

    def __init__(self, master_data):
        self.data = pd.read_csv(master_data)

    def remove_anomaly(self, master_data_path):
        """
        Function that removes anomalies (custom function)
        Args:
            master_data_path: path back to main opv file for ML training/plotting
        Returns:
            .csv file with anomalies removed from main opv file
        """
        delete_index = []
        for index, row in self.data.iterrows():
            if "CuI" in row.values:
                delete_index.append(index)
            elif "BrA" in row.values:
                delete_index.append(index)
            elif "F4-TCNQ" in row.values:
                delete_index.append(index)
            elif "TiO2:TOPD" in row.values:
                delete_index.append(index)
            elif self.data.at[index, "hole mobility blend (cm^2 V^-1 s^-1)"] > 0.02:
                delete_index.append(index)
            elif self.data.at[index, "electron mobility blend (cm^2 V^-1 s^-1)"] > 0.1:
                delete_index.append(index)

        self.data = self.data.drop(self.data.index[delete_index])

        self.data.to_csv(master_data_path, index=False)

    def correct_anomaly(self, master_data_path):
        """
        Function that standardizes the labels!
        Args:
            master_data_path: path back to main opv file for ML training/plotting
        Returns:
            .csv file with labels corrected from main opv file
        """
        for index, row in self.data.iterrows():
            if self.data.at[index, "solvent"] == "DCB":
                self.data.at[index, "solvent"] = "o-DCB"
            if self.data.at[index, "solvent"] == "CF:DCB (80:20)":
                self.data.at[index, "solvent"] = "o-DCB:CF (4:1)"
            if self.data.at[index, "hole contact layer"] == "MoO3":
                self.data.at[index, "hole contact layer"] = "MoOx"

        self.data.to_csv(master_data_path, index=False)


# anomaly = Anomaly(MASTER_ML_DATA)
# anomaly.remove_anomaly(MASTER_ML_DATA)

# anomaly = Anomaly(MASTER_ML_DATA_PLOT)
# anomaly.remove_anomaly(MASTER_ML_DATA_PLOT)

# anomaly = Anomaly(AUG_SMI_MASTER_DATA)
# anomaly.remove_anomaly(AUG_SMI_MASTER_DATA)

# anomaly = Anomaly(BRICS_MASTER_DATA)
# anomaly.remove_anomaly(BRICS_MASTER_DATA)

# anomaly = Anomaly(MANUAL_MASTER_DATA)
# anomaly.remove_anomaly(MANUAL_MASTER_DATA)

# anomaly = Anomaly(FP_MASTER_DATA)
# anomaly.remove_anomaly(FP_MASTER_DATA)
