from email import generator
from pathlib import Path
import pandas as pd
import os

### HELPER FUNCTIONS


def handle_paths(parent_path: Path, config: dict, directory_names: str) -> generator:
    """Traverses through the directories such that the paths with correct configurations are extracted.

    Args:
        parent_path (Path): _description_
        config (dict): outlines the parameters to select for the appropriate configurations for comparison
        new_paths (str): _description_

    Returns:
        next_path: _description_
    """
    new_paths = config[directory_names]
    if len(new_paths) == 0:
        if parent_path.name == "log":
            pass
        else:
            print("All {} will be plotted.".format(directory_names))
            new_paths: list[Path] = parent_path.iterdir()
            for new_path in new_paths:
                yield new_path
    else:
        for new_path in new_paths:
            next_path: Path = parent_path / new_path
            yield next_path


def path_to_result(config: dict, result_file: str) -> list[Path]:
    """
    For configurations with an empty value, all of the configurations will be plotted.
    Args:
        config: outlines the parameters to select for the appropriate configurations for comparison
            datasets: dataset to plot
            models: ML models to plot
            input_representations: molecular representation (ex. SMILES, fingerprint)
            features: additional descriptors of the molecule or device
            input_names: exact name of the inputs used for training
            target_names: exact name of target used for prediction
            metrics: exact name of metric to plot (ex. r, r2, rmse, mae)
        result_file: which file do you want? progress_report.csv or summary.csv
    Returns:
        progress_report_paths: all the paths to the summary files for plotting
    """
    result_paths: list[Path] = []
    results_path: Path = Path(config["path_to_training"])
    dataset_paths: generator = handle_paths(results_path, config, "datasets")
    # print("dataset_path")
    # print(list(dataset_paths))
    for dataset_path in dataset_paths:
        input_rep_paths: generator = handle_paths(
            dataset_path, config, "input_representations"
        )
        # print("input_rep_paths")
        # print(list(input_rep_paths))
        for input_rep_path in input_rep_paths:
            try:
                features: list[str] = config["features"]
                if len(features) == 0:
                    raise ValueError
                else:
                    feature_paths: list[Path] = handle_paths(input_rep_path, config, "features")
            except:
                print("All features will be plotted.")
                feature_paths: list[Path] = input_rep_path.iterdir()
                # print("feature_paths")
                # print(list(feature_paths))
            finally:
                # print(f"{list(feature_paths)=}")
                for feature_path in feature_paths:
                    # print(feature_path)
                    model_paths: generator = handle_paths(feature_path, config, "models")
                    # print("model_paths")
                    # print(list(model_paths))
                    for model_path in model_paths:
                        # print(model_path)
                        try:
                            inputs: list[str] = config["input_names"]
                            if len(inputs) == 0:
                                raise ValueError
                            else:
                                input_paths: list[Path] = handle_paths(model_path, config, "input_names")
                        except:
                            print("All inputs will be plotted.")
                            input_paths: list[Path] = model_path.iterdir()
                            # print("input_paths")
                            # print(list(input_paths))
                            for input_path in input_paths:
                                # print(feature_path)
                                target_paths: generator = handle_paths(
                                    input_path, config, "target_names"
                                )
                                # print("target_path", list(target_paths))
                                for target_path in target_paths:
                                    result_path: Path = target_path / "{}.csv".format(
                                        result_file
                                    )
                                    result_paths.append(result_path)
                        else:
                            # print("feature_paths")
                            # print(feature_paths)
                            for input_path in input_paths:
                                # print(feature_path)
                                input_path_split: list[str] = str(feature_path).split(",")
                                if inputs == input_path_split[1:]:
                                    target_paths: generator = handle_paths(
                                        input_path, config, "target_names"
                                    )
                                for target_path in target_paths:
                                    # TODO: change to progress report
                                    # TODO: add Dataset, Features, Model, Target, Feature Length, and Num of data
                                    result_path: Path = target_path / "{}.csv".format(
                                        result_file
                                    )
                                    result_paths.append(result_path)
    print(f"{result_paths=}")
    return result_paths

def gather_results(progress_report_paths: list[Path]) -> pd.DataFrame:
    """
    Args:
        progress_report_paths (list[Path]): _description_
        summary_paths (list[Path]): _description_
    """
    progress_full: pd.DataFrame = pd.DataFrame(
        columns=[
            "Dataset",
            "Input_Representation",
            "Features",
            "Feature_Names",
            "Model",
            "Target",
            "fold",
            "r",
            "r2",
            "rmse",
            "mae",
            "feature_length",
            "num_of_data",
        ]
    )
    for progress_path in progress_report_paths:
        # find missing columns from path
        missing_columns: list = str(progress_path).split("/")

        # Get data from summary.csv
        if progress_path.name == "progress_report.csv":
            parent_path: Path = progress_path.parent
            summary_path: Path = parent_path / "summary.csv"
            summary: pd.DataFrame = pd.read_csv(summary_path)

        progress: pd.DataFrame = pd.read_csv(progress_path)
        progress["Dataset"] = missing_columns[-7]
        progress["Input_Representation"] = missing_columns[-6]
        progress["Features"] = missing_columns[-5]
        progress["Feature_Names"] = missing_columns[-3]
        progress["Model"] = missing_columns[-4]
        progress["Target"] = missing_columns[-2]
        if progress_path.name == "progress_report.csv":
            # progress["feature_length"] = summary["feature_length"].values[0]
            progress["num_of_data"] = summary["num_of_data"].values[0]
        progress_full: pd.DataFrame = pd.concat(
            [progress, progress_full], ignore_index=True
        )
    return progress_full

def path_to_results_recursive(root_dir: str):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        offset = len(dirpath.split(os.sep))
        print("    " * (offset - 1), dirpath, sep="")
        for a_file in filenames:
            print("    " * offset, a_file, sep="")


# cwd = Path.cwd()
# root_dir = cwd.parent / "training"
# path_to_results_recursive(root_dir)