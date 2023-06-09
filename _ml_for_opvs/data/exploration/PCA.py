import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pkg_resources
import matplotlib.pyplot as plt

# OPV data after pre-processing
MASTER_ML_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)


def analyze_principal_components(dataset: pd.DataFrame):
    """
    Function that carries out principle component analysis
    NOTE: we only work with numerical variables in this analysis

    Args:
        file path we want to save the plots to
    Returns:
        plots of each feature's contribution to PC
    """
    # Create a proper feature matrix for PCA by dropping all outputs and categorical variables
    # TODO: Drop columns based on types rather than names
    df_cont_var = dataset.drop(
        columns=['Unnamed: 0', 'Donor', 'Donor_SMILES', 'Donor_Big_SMILES', 'Donor_SELFIES', 'Acceptor',
                 'Acceptor_SMILES', 'Acceptor_Big_SMILES', 'Acceptor_SELFIES', 'solvent', 'solvent_additive',
                 'hole_contact_layer', 'electron_contact_layer', 'calc_PCE_percent', 'Jsc_mA_cm_pow_neg2', 'Voc_V',
                 'FF_percent', 'PCE_percent'])
    # Drop rows with any empty entries
    df = df_cont_var.dropna(how='any')

    # Data should be centered and normalized before carrying out PCA
    # Normalized data
    df_ctr = (df - df.mean()) / df.std()
    # Compute principal components, set the number of components to be equal to number of columns in the dataset
    pca = PCA(n_components=df_ctr.shape[1])
    pca.fit(df_ctr)

    # Store proportion of variance explained by PCs as a dataframe
    pca_var_explained = pd.DataFrame({'Proportion of variance explained': pca.explained_variance_ratio_})

    # Add component number as a new column
    pca_var_explained['Component'] = np.arange(1, df_ctr.shape[1] + 1)

    # Add cumulative variance explained as a new column
    pca_var_explained['Cumulative variance explained'] = pca_var_explained[
        'Proportion of variance explained'].cumsum()
    # print(pca_var_explained)
    # Plot the % explained variance as a function of PC, and the cumulative variance explained by PCs
    pca_var_explained_plot1 = pca_var_explained.plot.line(x='Component', y='Proportion of variance explained')
    plt.savefig('pca_var_explained.png')
    pca_var_explained_plot2 = pca_var_explained.plot.line(x='Component', y='Cumulative variance explained')
    plt.savefig('pca_var_explained_cum.png')

    # Store the loadings of each feature (how much each feature contributes to each PC) as a dataframe with appropriate names
    loading_df = pd.DataFrame(pca.components_).transpose().rename(
        columns={0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'}  # add entries for each selected component
    ).loc[:, ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]  # slice just components of interest, here we only pick 5 PCs

    # Add a column with the feature names
    loading_df['Variable'] = df_ctr.columns.values
    # print
    print(loading_df)

    # Plot the loading of features in the chosen 5 PCs
    loading_df_plot = loading_df.plot.barh(y=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], x='Variable', figsize=(10, 8))
    plt.tight_layout()
    plt.savefig('feature_loading_pca.png')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv(MASTER_ML_DATA)
    analyze_principal_components(data)
