import argparse
from pathlib import Path
import matplotlib
import pandas as pd


def scatterplot(config: dict):
    """
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    """
    # TODO: create a scatterplot from one of the prediction files in comparison to the ground truth. In the title, specify which dataset, model, features, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth",
        type=str,
        help="Filepath to ground truth examples. Make sure same parameters, and fold.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Filepath to predictions. Make sure same parameters, and fold.",
    )
    args = parser.parse_args()
    config = vars(args)
    scatterplot(config)
