import pkg_resources
import pandas as pd

# OPV data after pre-processing
MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)
MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
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


anomaly = Anomaly(MASTER_ML_DATA)
anomaly.remove_anomaly(MASTER_ML_DATA)

anomaly = Anomaly(MASTER_ML_DATA_PLOT)
anomaly.remove_anomaly(MASTER_ML_DATA_PLOT)
