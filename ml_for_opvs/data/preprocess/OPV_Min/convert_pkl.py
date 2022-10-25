import pkg_resources
import pandas as pd

GRAPH_PKL = pkg_resources.resource_filename("ml_for_opvs", "../gary/data/min_graph.pkl")

MORDRED_PKL = pkg_resources.resource_filename(
    "ml_for_opvs", "../gary/data/min_mordred.pkl"
)

PCA_MORDRED_PKL = pkg_resources.resource_filename(
    "ml_for_opvs", "../gary/data/min_pca_mordred.pkl"
)

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

MORDRED_PREPROCESSED = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/mordred/master_mordred.csv"
)

PCA_MORDRED_PREPROCESSED = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/input_representation/OPV_Min/mordred_pca/master_mordred_pca.csv",
)

GRAPH_PREPROCESSED = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/graphembed/master_graphembed.csv"
)


def convert_pkl_to_data(path: str, master_path: str, preprocessed_path: str, name: str):
    """Read ORDERED pickle into pandas DataFrame, and add all the parameters from the master file.

    Args:
        path (str): _description_
        master_path (str): path to master file in preprocessed.
        preprocessed_path (str): path to preprocessed file for training.
    """
    try:
        data: pd.DataFrame = pd.DataFrame.from_dict(pd.read_pickle(path))
        for index, row in data.iterrows():
            print(data.at[index, "donor"])
    except:
        data: dict = pd.read_pickle(path)
        data: pd.DataFrame = pd.DataFrame({k: list(v) for k, v in data.items()})
        data["DA_{}".format(name)] = ""
        for index, row in data.iterrows():
            data.at[index, "donor"] = list(data.at[index, "donor"])
            data.at[index, "acceptor"] = list(data.at[index, "acceptor"])
            concatenated = data.at[index, "donor"]
            concatenated.extend(data.at[index, "acceptor"])
            data.at[index, "DA_{}".format(name)] = concatenated

    master_data: pd.DataFrame = pd.read_csv(master_path)
    # select only parameters
    master_data: pd.DataFrame = master_data.iloc[:, 9:]
    concat_data: pd.DataFrame = pd.concat([data, master_data], axis=1)
    # print(concat_data)
    concat_data.to_csv(preprocessed_path, index=False)


# convert_pkl_to_data(GRAPH_PKL, MASTER_ML_DATA, GRAPH_PREPROCESSED)
convert_pkl_to_data(MORDRED_PKL, MASTER_ML_DATA, MORDRED_PREPROCESSED, "mordred")
convert_pkl_to_data(
    PCA_MORDRED_PKL, MASTER_ML_DATA, PCA_MORDRED_PREPROCESSED, "mordred_pca"
)
