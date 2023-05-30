from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def standard_error_of_estimate(
    ground_truth: pd.Series, predictions: pd.Series
) -> float:
    residuals_squared: list = np.power(ground_truth - predictions, 2)
    sum_of_residuals_squared: float = np.sum(residuals_squared)
    standard_error_of_estimate = np.sqrt(sum_of_residuals_squared / (len(ground_truth)))
    return standard_error_of_estimate


def generate_results_dataset(ground_truth, preds, ci):
    df = pd.DataFrame()
    df["ground_truth"] = ground_truth
    df["prediction"] = preds
    if ci >= 0:
        df["upper"] = preds + ci
        df["lower"] = preds - ci
    else:
        df["upper"] = preds - ci
        df["lower"] = preds + ci

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
    plt.plot([0, max_output], [0, max_output], color="black")
    # Compute Metrics
    r: float = np.corrcoef(ground_truth, preds)[0, 1]
    r2: float = r**2
    mae: float = (ground_truth - preds).abs().mean()
    rmse: float = np.sqrt(np.power(ground_truth - preds, 2.0).mean())
    ax = sns.scatterplot(x=ground_truth, y=preds)
    plt.xlabel("PCE" + " [Ground Truth]")
    plt.ylabel("PCE" + " [Prediction]")
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


def collect_outliers(outlier_path: Path, df_ci: pd.DataFrame):
    """Collect the outliers from the dataframe.

    Args:
        df_ci (pd.DataFrame): _description_

    Returns:
        outliers: _description_
    """
    try:
        outliers_aggregate: pd.DataFrame = pd.read_csv(outlier_path)
    except:
        outliers_aggregate: pd.DataFrame = pd.DataFrame()
    outliers: pd.DataFrame = df_ci.loc[
        (df_ci["ground_truth"] > df_ci["upper"])
        | (df_ci["ground_truth"] < df_ci["lower"])
    ]
    outliers_aggregate: pd.DataFrame = pd.concat([outliers_aggregate, outliers])
    outliers_aggregate.to_csv(outlier_path, index=False)


def histogram_of_outliers(outlier_path: Path):
    """Plot a histogram of the outliers.

    Args:
        outlier_path (Path): _description_
    """
    outliers: pd.DataFrame = pd.read_csv(outlier_path)
    sns.histplot(outliers["ground_truth"], bins=14)
    plt.xlabel("Ground Truth")
    plt.ylabel("Count")
    plt.savefig(Path(outlier_path).parent / "histogram_outliers.png")


if __name__ == "__main__":
    alpha: float = 0.05
    training_path: Path = Path(
        "ml_for_opvs/training/OPV_Min/fingerprint/result_molecules_only/RF/DA_FP_radius_3_nbits_1024/calc_PCE_percent/prediction_4.csv"
    )
    outlier_path: Path = Path(
        "ml_for_opvs/training/OPV_Min/fingerprint/result_molecules_only/RF/DA_FP_radius_3_nbits_1024/calc_PCE_percent/outliers.csv"
    )
    training: pd.DataFrame = pd.read_csv(training_path)
    ci = generate_ci_quantiles(
        training["calc_PCE_percent"], training["predicted_calc_PCE_percent"], alpha
    )
    df_ci = generate_results_dataset(
        training["calc_PCE_percent"], training["predicted_calc_PCE_percent"], ci
    )
    # scatterplot_ci(df_ci, training_path)
    # collect_outliers(outlier_path, df_ci)
    histogram_of_outliers(outlier_path)
