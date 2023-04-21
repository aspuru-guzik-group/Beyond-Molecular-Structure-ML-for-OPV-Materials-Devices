import pandas as pd
import pkg_resources

from code_python.visualization.feature_distribution import plot_feature_distributions
# from ml_for_opvs.data.exploration.OPV_Min.correlation import

"""
Load the Saeki dataset and plot the feature distributions
"""

# Load the Saeki dataset
SAEKI_DATA = MASTER_ML_DATA_PLOT = pkg_resources.resource_filename(
    "ml_for_opvs",
    "datasets/Saeki_2022_n1318/Saeki_corrected.csv"
)
saeki_df = pd.read_csv(SAEKI_DATA)

# Plot the feature distributions
plot_feature_distributions(saeki_df, drop_columns=["ID", "Ref", "n(SMILES)", "p(SMILES)", "DonorAcceptor"])
