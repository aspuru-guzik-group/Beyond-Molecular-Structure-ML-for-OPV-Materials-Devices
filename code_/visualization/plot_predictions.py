import json
from math import ceil
from pathlib import Path

# import cmcrameri.cm as cmc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASETS = ROOT / "datasets"

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"],
              "size": 12
              })


def get_predictions(directory: Path, ground_truth_file: Path) -> None:
    true_values: pd.Series = pd.read_csv(ground_truth_file)["calculated PCE (%)"]
    for pred_file in directory.rglob("*_predictions.csv"):
        # print(pred_file)
        model_name: str = pred_file.stem.split("_")[0]
        r2_avg, r2_stderr = get_scores(pred_file.parent, model_name)

        make_predictions_plot(true_values, pred_file, r2_avg, r2_stderr)


def get_scores(dir: Path, model_type: str) -> tuple[float, float]:
    scores_file: Path = dir / f"{model_type}_scores.json"
    with open(scores_file, "r") as f:
        scores: dict = json.load(f)
    r2_avg = round(scores["r2_avg"], 2)
    r2_stderr = round(scores["r2_stderr"], 2)
    return r2_avg, r2_stderr


def make_predictions_plot(true_values: pd.Series, predictions: Path, r2_avg: float, r2_stderr: float) -> None:
    # Load the data from CSV files
    predicted_values = pd.read_csv(predictions)
    seeds = predicted_values.columns

    # There are 7 columns in predicted_values, each corresponding to a different seed
    # Create a Series consisting of the ground truth values repeated 7 times
    true_values_ext = pd.concat([true_values] * len(seeds), ignore_index=True)
    # Create a Series consisting of the predicted values, with the column names as the index
    predicted_values_ext = pd.concat([predicted_values[col] for col in seeds], axis=0, ignore_index=True)

    ext_comb_df = pd.concat([true_values_ext, predicted_values_ext], axis=1)

    # Create the hex-binned plot with value distributions for all y-axis columns
    g = sns.jointplot(data=ext_comb_df, x="calculated PCE (%)", y=0,
                      kind="hex",
                      # cmap=cmc.batlow,
                      # joint_kws={"gridsize": 50, "cmap": "Blues"},
                      # joint_kws={"gridsize": (44, 25)},
                      marginal_kws={"bins": 25},
                      )
    ax_max = ceil(max(ext_comb_df.max()))
    g.ax_joint.plot([0, ax_max], [0, ax_max], ls="--", c=".3")
    # g.ax_joint.annotate(f"$R^2$ = {r2_avg} ± {r2_stderr}",
    #                     # xy=(0.1, 0.9), xycoords='axes fraction',
    #                     # ha='left', va='center',
    #                     # bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}
    #                     )
    # TODO:
    #  kwargs: linewidth=0.2, edgecolor='white',  mincnt=1
    plt.text(0.95, 0.05, f"$R^2$ = {r2_avg} ± {r2_stderr}",
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=g.ax_joint.transAxes,
             )
    # g.plot_marginals(sns.kdeplot, color="blue")
    # Set plot limits to (0, 15) for both axes
    g.set_axis_labels("True PCE (%)", "Predicted PCE (%)")
    g.ax_joint.set_xlim(0, ax_max)
    g.ax_joint.set_ylim(0, ax_max)
    # plt.tight_layout()
    output: Path = predictions.parent / f"{predictions.stem}_plot.png"
    # plt.savefig(output, dpi=600)
    g.savefig(output, dpi=600)
    print(f"Saved {output}")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    # predictions = HERE.parent.parent / "results" / "target_PCE" / "features_ECFP" / "RF_predictions.csv"
    # make_predictions_plot(predictions, 0.87, 0.02)
    dataset_ground_truth_csv = DATASETS / "Min_2020_n558" / "cleaned_dataset.csv"
    # ground_truth_Hutchison_csv = DATASETS / "Hutchison_2023_n1001" / "Hutchison_filtered_dataset_pipeline.csv"
    # ground_truth_Saeki_csv = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected_pipeline.csv"

    get_predictions(ROOT / "results" / "target_PCE", dataset_ground_truth_csv)

    # for result_dir, ground_truth_csv in zip(["results_Hutchison", "results_Saeki"], [ground_truth_Hutchison_csv, ground_truth_Saeki_csv]):
    #     pce_results = ROOT / result_dir
    #     get_predictions(pce_results, ground_truth_csv)
