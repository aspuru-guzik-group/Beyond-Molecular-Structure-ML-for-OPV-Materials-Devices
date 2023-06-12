from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def generate_ci_zscore(ground_truth: pd.Series, z_score: float = 1.96):
    """Generate confidence interval for the predictions.
    https://towardsdatascience.com/generating-confidence-intervals-for-regression-models-2dd60026fbce

    Args:
        ground_truth (pd.Series): _description_
        predictions (pd.Series): _description_

    Returns:
        ci: _description_
    """
    ci = z_score * np.std(ground_truth) / np.sqrt(len(ground_truth))
    return ci

def generate_results_dataset(ground_truth, preds, ci):
    df = pd.DataFrame()
    df["ground_truth"] = ground_truth
    df["prediction"] = preds
    if ci >= 0:
        df["upper"] = ground_truth + ci
        df["lower"] = ground_truth - ci
    else:
        df["upper"] = ground_truth - ci
        df["lower"] = ground_truth + ci

    return df


def generate_ci_quantiles(
    ground_truth: pd.Series, predictions: pd.Series, alpha: float
):
    """Generate confidence interval for the predictions.
    https://towardsdatascience.com/generating-confidence-intervals-for-regression-models-2dd60026fbce

    Args:
        ground_truth (pd.Series): _description_
        predictions (pd.Series): _description_

    Returns:
        ci: _description_
    """
    residuals: list = list(ground_truth - predictions)
    ci = np.quantile(residuals, 1 - alpha)
    return ci


# create a function for creating a scatterplot with confidence intervals
def scatterplot_ci(data_ci: pd.DataFrame, training_path: Path):
    # reorder the dataframe by the ground truth
    data_ci = data_ci.sort_values(by="ground_truth")
    print(data_ci)
    # creates the ground truth line
    ground_truth: pd.Series = data_ci["ground_truth"]
    preds: pd.Series = data_ci["prediction"]
    max_output = max(max(ground_truth), max(preds))
    plt.figure()
    plt.plot([0, max_output], [0, max_output], color="black")
    # Compute Metrics
    r: float = np.corrcoef(ground_truth, preds)[0, 1]
    r2: float = r**2
    mae: float = (ground_truth - preds).abs().mean()
    rmse: float = np.sqrt(np.power(ground_truth - preds, 2.0).mean())
    ax = sns.scatterplot(x=ground_truth, y=preds)
    plt.xlabel("PCE" + " [Ground Truth]")
    plt.ylabel("PCE" + " [Prediction]")
    model: str = training_path.parent.parent.parent.stem
    representation: str = training_path.parent.parent.stem
    category: str = training_path.parent.parent.parent.parent.stem
    target: str = training_path.parent.stem
    plt.title(f"{model} + {representation} \n + {category} + {target}")
    ax.text(
        0.01,
        0.85,
        "MAE:  {:.4E}\nRMSE: {:.4E}\nR:  {:.4F}".format(mae, rmse, r),
        transform=ax.transAxes,
    )
    # plot confidence interval
    ax.fill_between(ground_truth, data_ci["lower"], data_ci["upper"], alpha=0.2)
    # Save figure
    x: int = int(training_path.stem.split("_")[-1])
    plot_path: Path = Path(training_path).parent / f"scatterplot_{x}.png"
    plt.savefig(plot_path)


def collect_outliers(outlier_path: Path, dfs_ci: list[pd.DataFrame]):
    """Collect the outliers from the dataframe.

    Args:
        df_ci (pd.DataFrame): _description_

    Returns:
        outliers: _description_
    """
    outliers_aggregate: pd.DataFrame = pd.DataFrame()
    for df_ci in dfs_ci:
        outliers: pd.DataFrame = df_ci.loc[
            (df_ci["prediction"] > df_ci["upper"])
            | (df_ci["prediction"] < df_ci["lower"])
        ]
        outliers_aggregate: pd.DataFrame = pd.concat([outliers_aggregate, outliers])
    outliers_aggregate.to_csv(outlier_path, index=False)


def histogram_of_outliers(outlier_path: Path):
    """Plot a histogram of the outliers.

    Args:
        outlier_path (Path): _description_
    """
    outliers: pd.DataFrame = pd.read_csv(outlier_path)
    model: str = outlier_path.parent.parent.parent.stem
    representation: str = outlier_path.parent.parent.stem
    category: str = outlier_path.parent.parent.parent.parent.stem
    target: str = outlier_path.parent.stem
    f, ax = plt.subplots()
    sns.histplot(outliers["ground_truth"], bins=14)
    plt.xlabel("Ground Truth")
    plt.ylabel("Count")
    plt.text(0.82,0.94, f"Number of Outliers: {len(outliers)}", horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.title(f"{model} + {representation} \n + {category} + {target}")
    plt.savefig(Path(outlier_path).parent / "histogram_outliers.png")


if __name__ == "__main__":
    targets: list = ["Jsc_mA_cm_pow_neg2", "calc_PCE_percent", "FF_percent", "Voc_V"]
    models = ["RF", "XGBoost"]
    for model in models:
        for target in targets:
            alpha: float = 0.05
            z_score: float = 1.96
            outlier_path: Path = Path(
                f"ml_for_opvs/training/OPV_Min/fingerprint/result_molecules_only/{model}/DA_FP_radius_3_nbits_1024/{target}/outliers.csv"
            )
            dfs_ci: list[pd.DataFrame] = []
            for i in range(0,5):
                training_path: Path = Path(
                    f"ml_for_opvs/training/OPV_Min/fingerprint/result_molecules_only/{model}/DA_FP_radius_3_nbits_1024/{target}/prediction_{i}.csv"
                )
                training: pd.DataFrame = pd.read_csv(training_path)
                # ci = generate_ci_quantiles(
                #     training[f"{target}"], training[f"predicted_{target}"], alpha
                # )
                ci = generate_ci_zscore(training[f"{target}"], z_score)
                df_ci = generate_results_dataset(
                    training[f"{target}"], training[f"predicted_{target}"], ci
                )
                scatterplot_ci(df_ci, training_path)
                dfs_ci.append(df_ci)
                

            collect_outliers(outlier_path, dfs_ci)
            histogram_of_outliers(outlier_path)
