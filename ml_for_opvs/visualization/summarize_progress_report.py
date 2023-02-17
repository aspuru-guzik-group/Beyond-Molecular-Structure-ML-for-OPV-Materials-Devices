import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def summarize(config: dict):
    """Summarize any progress report into a summary file with the correct details.

    Args:
        config (dict): _description_
    """
    progress_path: Path = Path(config["progress_path"])
    missing_columns: list = str(config["progress_path"]).split("/")
    progress: pd.DataFrame = pd.read_csv(config["progress_path"])
    r_mean: float = np.mean(progress["r"])
    r_std: float = np.std(progress["r"])
    r2_mean: float = np.mean(progress["r2"])
    r2_std: float = np.std(progress["r2"])
    rmse_mean: float = np.mean(progress["rmse"])
    rmse_std: float = np.std(progress["rmse"])
    mse_mean: float = np.mean(progress["mse"])
    mse_std: float = np.std(progress["mse"])

    summary_dict: dict = {
        "Dataset": ["OPV_Min"],
        "num_of_folds": [len(progress)],
        "Features": [missing_columns[-3]],
        "Targets": [missing_columns[-2]],
        "Model": [missing_columns[-4]],
        "r_mean": [r_mean],
        "r_std": [r_std],
        "r2_mean": [r2_mean],
        "r2_std": [r2_std],
        "rmse_mean": [rmse_mean],
        "rmse_std": [rmse_std],
        "mse_mean": [mse_mean],
        "mse_std": [mse_std],
        "num_of_data": [559],
    }

    summary: pd.DataFrame = pd.DataFrame.from_dict(summary_dict)

    summary_path: Path = progress_path.parent / "summary.csv"
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--progress_path", type=str)

    args = argparser.parse_args()
    config = vars(args)
    summarize(config)
