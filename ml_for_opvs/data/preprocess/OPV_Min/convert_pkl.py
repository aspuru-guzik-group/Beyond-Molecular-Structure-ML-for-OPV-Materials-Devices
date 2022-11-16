from pathlib import Path
import pkg_resources
import pandas as pd
import numpy as np

GRAPH_PKL = pkg_resources.resource_filename("ml_for_opvs", f"../gary/trained_results")

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
    "ml_for_opvs",
    "data/input_representation/OPV_Min/graphembed/processed_graphembed_molecules_only/KFold",
)

# Load pkl
# Check out what it looks like, dict -> dataframe
# Add the device parameters, etc. (automated)


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


def convert_pkl_splits_to_data(
    path: str, master_path: str, preprocessed_path: str, name: str
):
    """Read ORDERED pickle into pandas DataFrame, and add all the parameters from the master file.

    Args:
        path (str): _description_
        master_path (str): path to master file in preprocessed.
        preprocessed_path (str): path to preprocessed file for training.
    """
    for fold in range(5):
        fold_path: Path = Path(path) / f"graphembed_split{fold}.pkl"
        data: pd.DataFrame = pd.DataFrame.from_dict(pd.read_pickle(fold_path))
        preprocessed_path: Path = Path(preprocessed_path)

        # combine train and valid
        data["train"]["acceptor"].extend(data["valid"]["acceptor"])
        data["train"]["donor"].extend(data["valid"]["donor"])
        data["train"]["target"].extend(data["valid"]["target"])

        index: int = 0
        train_da_graphembed: list = []
        train_target: list = []
        for index in range(len(data["train"]["acceptor"])):
            data["train"]["donor"][index].squeeze()
            data["train"]["acceptor"][index].squeeze()
            train_da_graphembed.append(
                list(
                    np.concatenate(
                        [
                            data["train"]["donor"][index],
                            data["train"]["acceptor"][index],
                        ],
                        axis=0,
                    )
                )
            )
            train_target.append(data["train"]["target"][index][0])

        train_df: pd.DataFrame = pd.DataFrame()
        train_df[f"DA_{name}"] = train_da_graphembed
        train_df["calc_PCE_percent"] = train_target
        train_df.to_csv(preprocessed_path / f"input_train_{fold}.csv")

        index: int = 0
        test_da_graphembed: list = []
        test_target: list = []
        for index in range(len(data["test"]["acceptor"])):
            test_da_graphembed.append(
                list(
                    np.concatenate(
                        [data["test"]["donor"][index], data["test"]["acceptor"][index]],
                        axis=0,
                    )
                )
            )
            test_target.append(data["test"]["target"][index][0])

        test_df: pd.DataFrame = pd.DataFrame()
        test_df[f"DA_{name}"] = test_da_graphembed
        test_df["calc_PCE_percent"] = test_target
        test_df.to_csv(preprocessed_path / f"input_test_{fold}.csv")

    # TODO: add device parameters!
    # master_data: pd.DataFrame = pd.read_csv(master_path)
    # # select only parameters
    # master_data: pd.DataFrame = master_data.iloc[:, 9:]
    # concat_data: pd.DataFrame = pd.concat([data, master_data], axis=1)
    # # print(concat_data)
    # concat_data.to_csv(preprocessed_path, index=False)


# convert_pkl_to_data(GRAPH_PKL, MASTER_ML_DATA, GRAPH_PREPROCESSED)
# convert_pkl_to_data(MORDRED_PKL, MASTER_ML_DATA, MORDRED_PREPROCESSED, "mordred")
# convert_pkl_to_data(
#     PCA_MORDRED_PKL, MASTER_ML_DATA, PCA_MORDRED_PREPROCESSED, "mordred_pca"
# )
convert_pkl_splits_to_data(GRAPH_PKL, MASTER_ML_DATA, GRAPH_PREPROCESSED, "graphembed")
