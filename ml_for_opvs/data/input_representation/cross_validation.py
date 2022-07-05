"""
Python file that contains functions and classes which inputs a total dataset, and returns
the corresponding train, and validation set. 
"""
import os
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, StratifiedKFold


def main(config):
    """
    Produces cross-validation folds for any training data.

    Args:
        config (dict): Configuration paths and parameters.

    Returns:
        input_train_fold_*: Examples from dataset used for training, created in current directory.
        input_val_fold_*: Examples from dataset used for validation, created in current directory.
    """
    data_path = Path(config["dataset_path"])
    data_filename = data_path.name.replace(".csv", "")
    data_dir = data_path.parents[0]
    data_df = pd.read_csv(config["dataset_path"])
    num_of_folds = config["num_of_folds"]
    seed = config["random_seed"]
    fold_path = data_dir / Path(data_filename) / Path(config["type_of_crossval"])
    fold_path.mkdir(parents=True, exist_ok=True)
    if config["type_of_crossval"] == "KFold":
        kf = KFold(n_splits=num_of_folds, shuffle=True, random_state=seed)
        i = 0
        for train_index, test_index in kf.split(data_df):
            train = data_df.iloc[train_index]
            valid = data_df.iloc[test_index]
            train_dir = fold_path / f"input_train_{i}.csv"
            valid_dir = fold_path / f"input_valid_{i}.csv"
            train.to_csv(train_dir, index=False)
            valid.to_csv(valid_dir, index=False)
            i += 1

    elif config["type_of_crossval"] == "StratifiedKFold":
        kf = StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=seed)
        i = 0
        for train_index, test_index in kf.split(
            data_df, data_df[config["stratified_label"]]
        ):
            train = data_df.iloc[train_index]
            valid = data_df.iloc[test_index]
            train_dir = fold_path / f"input_train_{i}.csv"
            valid_dir = fold_path / f"input_valid_{i}.csv"
            train.to_csv(train_dir, index=False)
            valid.to_csv(valid_dir, index=False)
            i += 1
    else:
        raise ValueError(
            "Wrong KFold operation. Choose between KFold or StratifiedKFold."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Absolute filepath to complete dataset for cross-validation",
    )
    parser.add_argument(
        "--num_of_folds",
        type=int,
        help="Number of folds to create for cross-validation",
    )
    parser.add_argument(
        "--type_of_crossval",
        type=str,
        default="KFold",
        help="Select between KFold and StratifiedKFold",
    )
    parser.add_argument(
        "--stratified_label",
        type=str,
        help="If StratifiedKFold is selected, it is necessary.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=22,
        help="Random seed initialization to ensure folds are consistent.",
    )
    args = parser.parse_args()
    config = vars(args)
    main(config)

### EXAMPLE USE
"""
python ../../cross_validation.py --dataset_path ~/Research/Repos/ml_for_opvs/ml_for_opvs/data/input_representation/OPV_Min/aug_SMILES/train_aug_master5.csv --num_of_folds 5 --type_of_crossval KFold
"""
