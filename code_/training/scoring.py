from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

from training_utils import N_FOLDS, SEEDS


def rmse_score(y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the root mean squared error.

    Args:
        y_test: Test targets.
        y_pred: Predicted targets.

    Returns:
        Root mean squared error.
    """
    return mean_squared_error(y_test, y_pred, squared=False)


def np_r(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient.

    Args:
        y_true: Test targets.
        y_pred: Predicted targets.

    Returns:
        Pearson correlation coefficient.
    """
    y_true = y_true.to_numpy().flatten()
    y_pred = y_pred.tolist()
    r = np.corrcoef(y_true, y_pred, rowvar=False)[0, 1]
    return r


def pearson(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = y_true.to_numpy().flatten()
    y_pred = y_pred.flatten()
    r = pearsonr(y_true, y_pred).statistic
    return r


# r_scorer = make_scorer(r_regression, greater_is_better=True)
# r_scorer = make_scorer(np_r, greater_is_better=True)
r_scorer = make_scorer(pearson, greater_is_better=True)
rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def process_scores(scores: dict[int, dict[str, float]]) -> dict[Union[int, str], dict[str, float]]:
    sample_size: int = N_FOLDS * len(SEEDS)  # TODO: Check if this is correct!

    avg_r = round(np.mean([seed["test_r"] for seed in scores.values()]), 2)
    stdev_r = round(np.std([seed["test_r"] for seed in scores.values()]), 2)
    stderr_r = round(stdev_r / np.sqrt(sample_size), 2)
    avg_r2 = round(np.mean([seed["test_r2"] for seed in scores.values()]), 2)
    stdev_r2 = round(np.std([seed["test_r2"] for seed in scores.values()]), 2)
    stderr_r2 = round(stdev_r2 / np.sqrt(sample_size), 2)
    print("Average scores:\t", f"r: {avg_r}±{stderr_r}\t", f"r2: {avg_r2}±{stderr_r2}")

    score_types: list[str] = ["r", "r2", "rmse", "mae"]
    avgs: list[float] = [np.mean([seed[f"test_{score}"] for seed in scores.values()]) for score in score_types]
    stdevs: list[float] = [np.std([seed[f"test_{score}"] for seed in scores.values()]) for score in score_types]
    std_errs: list[float] = [stdev / np.sqrt(sample_size) for stdev in stdevs]
    for score, avg, stdev, stderr in zip(score_types, avgs, stdevs, std_errs):
        scores[f"{score}_avg"] = avg
        scores[f"{score}_stdev"] = stdev
        scores[f"{score}_stderr"] = stderr

    return scores
