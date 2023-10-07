from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "results" / "target_PCE" / "features_ECFP"
DATASETS = ROOT / "datasets"
dataset_ground_truth_csv = DATASETS / "Min_2020_n558" / "cleaned_dataset.csv"

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 18})


def get_score_from_json(json_file: Path) -> tuple[float, float]:
    with open(json_file, "r") as f:
        scores: dict = json.load(f)
    r2_avg = scores["r2_avg"]
    r2_stderr = scores["r2_stderr"]
    return r2_avg, r2_stderr


def outlier_analysis(folder_path: Path) -> list[int]:
    outlier_tracker: dict = {}
    for file in folder_path.iterdir():
        if file.suffix == ".json":
            r2_avg, r2_stderr = get_score_from_json(file)
            if r2_avg > 0.55:
                model_name: str = file.stem.split("_")[0]
                # average predictions
                predictions_file: Path = file.parent / f"{model_name}_predictions.csv"
                predictions: pd.DataFrame = pd.read_csv(predictions_file)
                predictions["avg"] = predictions.mean(axis=1)
                # add ground truth
                ground_truth: pd.DataFrame = pd.read_csv(dataset_ground_truth_csv)
                predictions["calculated PCE (%)"] = ground_truth["calculated PCE (%)"]
                # calculate difference
                predictions["absolute_difference"] = abs(
                    predictions["calculated PCE (%)"] - predictions["avg"]
                )
                for i, row in predictions.iterrows():
                    if row["absolute_difference"] > 3:
                        if i not in outlier_tracker:
                            outlier_tracker[i] = 1
                        else:
                            outlier_tracker[i] += 1
    # remove outliers that are only outliers for 2 or less models
    for key in list(outlier_tracker.keys()):
        if outlier_tracker[key] < 3:
            del outlier_tracker[key]
    ground_truth.loc[outlier_tracker.keys()].to_csv(
        RESULTS / f"outliers.csv", index=True
    )


def plot_outlier_distribution(outlier_csv: Path, column_name: str):
    outliers: pd.DataFrame = pd.read_csv(outlier_csv)
    fig, ax = plt.subplots()
    ax.set_title("outlier_" + column_name)
    ax.hist(outliers[column_name], bins=20, align="mid")
    total = "Total: " + str(len(outliers[column_name]))
    anchored_text = AnchoredText(total, loc="upper right")
    ax.add_artist(anchored_text)
    plt.tight_layout()
    plt.savefig(outlier_csv.parent / f"outlier_{column_name}_distribution.png", dpi=400)


if __name__ == "__main__":
    outlier_analysis(RESULTS)
    plot_outlier_distribution(RESULTS / "outliers.csv", "calculated PCE (%)")
