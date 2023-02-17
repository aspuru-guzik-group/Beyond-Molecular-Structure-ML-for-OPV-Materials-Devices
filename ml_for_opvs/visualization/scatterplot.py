import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import re

def scatterplot(config: dict):
    """
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    """
    # TODO: create a scatterplot from one of the prediction files in comparison to the ground truth. In the title, specify which dataset, model, features, etc.
    pred_path: Path = Path(config["predictions"])
    pred_df: pd.DataFrame = pd.read_csv(pred_path)
    gt, pred = pred_df.iloc[:,0], pred_df.iloc[:,1]
    max_output = max(max(gt), max(pred))
    plt.plot([0,max_output], [0,max_output], color="black")
    # Compute Metrics
    r = np.corrcoef(gt, pred)[0,1]
    r2 = r**2
    mae = (gt - pred).abs().mean()
    rmse = np.sqrt(np.power(gt - pred, 2.0).mean())
    ax = sns.scatterplot(x=gt, y=pred)
    plt.xlabel(pred_df.columns[1] + ' [Ground Truth]')
    plt.ylabel(pred_df.columns[1] + ' [Prediction]')
    ax.text(
            0.01,
            0.85,
            'MAE:  {:.4E}\nRMSE: {:.4E}\nR:  {:.4F}'.format(mae, rmse, r),
            transform=ax.transAxes
        )

    # Save figure
    cwd: Path = Path(os.getcwd())
    predictions_int: list = re.findall(r'\d+', pred_path.stem)
    plot_path: Path = cwd / "scatterplot_{}.png".format(predictions_int[0])
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        help="Filepath to predictions. Make sure same parameters, and fold.",
    )
    args = parser.parse_args()
    config = vars(args)
    scatterplot(config)
