from itertools import product
from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics._scorer import r2_scorer
from sklearn.model_selection import cross_val_predict, cross_validate


# from training_utils import N_FOLDS, SEEDS


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
    # TODO: Handling multi-output
    # sample_size: int = 7 * 5  # N seeds * N folds
    sample_size: int = 7  # N seeds since we're essentially doing a paired t-test

    avg_r = round(np.mean([seed["test_r"] for seed in scores.values()]), 2)
    stdev_r = round(np.std([seed["test_r"] for seed in scores.values()]), 2)
    # stderr_r = round(stdev_r / np.sqrt(sample_size), 2)
    avg_r2 = round(np.mean([seed["test_r2"] for seed in scores.values()]), 2)
    stdev_r2 = round(np.std([seed["test_r2"] for seed in scores.values()]), 2)
    # stderr_r2 = round(stdev_r2 / np.sqrt(sample_size), 2)
    print("Average scores:\t", f"r: {avg_r}±{stdev_r}\t", f"r2: {avg_r2}±{stdev_r2}")

    first_key = list(scores.keys())[0]
    score_types: list[str] = [key for key in scores[first_key].keys() if key.startswith("test_")]
    # score_types: list[str] = [score.replace("test_", "") for score in score_types]
    avgs: list[float] = [np.mean([seed[score] for seed in scores.values()]) for score in score_types]
    stdevs: list[float] = [np.std([seed[score] for seed in scores.values()]) for score in score_types]
    std_errs: list[float] = [stdev / np.sqrt(sample_size) for stdev in stdevs]

    # score_types: list[str] = ["r", "r2", "rmse", "mae"]
    # avgs: list[float] = [np.mean([seed[f"test_{score}"] for seed in scores.values()]) for score in score_types]
    # stdevs: list[float] = [np.std([seed[f"test_{score}"] for seed in scores.values()]) for score in score_types]
    # std_errs: list[float] = [stdev / np.sqrt(sample_size) for stdev in stdevs]
    score_types: list[str] = [score.replace("test_", "") for score in score_types]
    for score, avg, stdev, stderr in zip(score_types, avgs, stdevs, std_errs):
        scores[f"{score}_avg"] = abs(avg) if score in ["rmse", "mae"] else avg
        scores[f"{score}_stdev"] = stdev
        scores[f"{score}_stderr"] = stderr

    return scores


def cross_validate_regressor(regressor, X, y, cv) -> tuple[dict[str, float], np.ndarray]:
    # Training and scoring on each fold
    if y.shape[1] > 1:
        regressor.set_output(transform="pandas")
        scores, predictions = cross_validate_multioutput_regressor(regressor, X, y, cv)

    else:
        scores: dict[str, float] = cross_validate(regressor, X, y,
                                                  cv=cv,
                                                  scoring={"r":    r_scorer,
                                                           "r2":   r2_scorer,
                                                           "rmse": rmse_scorer,
                                                           "mae":  mae_scorer},
                                                  # return_estimator=True,
                                                  n_jobs=-1, )

        predictions: np.ndarray = cross_val_predict(regressor, X, y,
                                                    cv=cv,
                                                    n_jobs=-1, )
    return scores, predictions


pce_column = "calculated PCE (%)"
voc_column = "Voc (V)"
jsc_column = "Jsc (mA cm^-2)"
ff_column = "FF (%)"


def r_pce(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    pce_pred = y_pred[:, 0]
    return pearson(pce_true, pce_pred)


def r_voc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    voc_true = y_true[voc_column]
    voc_pred = y_pred[:, 1]
    return pearson(voc_true, voc_pred)


def r_jsc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    jsc_true = y_true[jsc_column]
    jsc_pred = y_pred[:, 2]
    return pearson(jsc_true, jsc_pred)


def r_ff(y_true: pd.Series, y_pred: np.ndarray) -> float:
    ff_true = y_true[ff_column]
    ff_pred = y_pred[:, 3]
    return pearson(ff_true, ff_pred)


def r_pce_eqn(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    voc_pred, jsc_pred, ff_pred = y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    pce_pred = voc_pred * jsc_pred * ff_pred
    return pearson(pce_true, pce_pred)


def r2_pce(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    pce_pred = y_pred[:, 0]
    return r2_score(pce_true, pce_pred)


def r2_voc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    voc_true = y_true[voc_column]
    voc_pred = y_pred[:, 1]
    return r2_score(voc_true, voc_pred)


def r2_jsc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    jsc_true = y_true[jsc_column]
    jsc_pred = y_pred[:, 2]
    return r2_score(jsc_true, jsc_pred)


def r2_ff(y_true: pd.Series, y_pred: np.ndarray) -> float:
    ff_true = y_true[ff_column]
    ff_pred = y_pred[:, 3]
    return r2_score(ff_true, ff_pred)


def r2_pce_eqn(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    voc_pred, jsc_pred, ff_pred = y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    pce_pred = voc_pred * jsc_pred * ff_pred
    return r2_score(pce_true, pce_pred)


def rmse_pce(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    pce_pred = y_pred[:, 0]
    return rmse_score(pce_true, pce_pred)


def rmse_voc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    voc_true = y_true[voc_column]
    voc_pred = y_pred[:, 1]
    return rmse_score(voc_true, voc_pred)


def rmse_jsc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    jsc_true = y_true[jsc_column]
    jsc_pred = y_pred[:, 2]
    return rmse_score(jsc_true, jsc_pred)


def rmse_ff(y_true: pd.Series, y_pred: np.ndarray) -> float:
    ff_true = y_true[ff_column]
    ff_pred = y_pred[:, 3]
    return rmse_score(ff_true, ff_pred)


def rmse_pce_eqn(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    voc_pred, jsc_pred, ff_pred = y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    pce_pred = voc_pred * jsc_pred * ff_pred
    return rmse_score(pce_true, pce_pred)


def mae_pce(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    pce_pred = y_pred[:, 0]
    return mean_absolute_error(pce_true, pce_pred)


def mae_voc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    voc_true = y_true[voc_column]
    voc_pred = y_pred[:, 1]
    return mean_absolute_error(voc_true, voc_pred)


def mae_jsc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    jsc_true = y_true[jsc_column]
    jsc_pred = y_pred[:, 2]
    return mean_absolute_error(jsc_true, jsc_pred)


def mae_ff(y_true: pd.Series, y_pred: np.ndarray) -> float:
    ff_true = y_true[ff_column]
    ff_pred = y_pred[:, 3]
    return mean_absolute_error(ff_true, ff_pred)


def mae_pce_eqn(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pce_true = y_true[pce_column]
    voc_pred, jsc_pred, ff_pred = y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    pce_pred = voc_pred * jsc_pred * ff_pred
    return mean_absolute_error(pce_true, pce_pred)


score_lookup: dict[str, dict[str, Callable]] = {
    "r":    {
        "PCE":     r_pce,
        "Voc":     r_voc,
        "Jsc":     r_jsc,
        "FF":      r_ff,
        "PCE_eqn": r_pce_eqn,
    },
    "r2":   {
        "PCE":     r2_pce,
        "Voc":     r2_voc,
        "Jsc":     r2_jsc,
        "FF":      r2_ff,
        "PCE_eqn": r2_pce_eqn,
    },
    "rmse": {
        "PCE":     rmse_pce,
        "Voc":     rmse_voc,
        "Jsc":     rmse_jsc,
        "FF":      rmse_ff,
        "PCE_eqn": rmse_pce_eqn,
    },
    "mae":  {
        "PCE":     mae_pce,
        "Voc":     mae_voc,
        "Jsc":     mae_jsc,
        "FF":      mae_ff,
        "PCE_eqn": mae_pce_eqn,
    },
}


def get_score_func(score: str, output: str) -> Callable:
    """
    Returns the appropriate scoring function for the given output.
    """
    score_func: Callable = score_lookup[score][output]
    return score_func


greater_lookup: dict[str, bool] = {"r": True, "r2": True, "rmse": False, "mae": False}


def multi_scorer(score: str, output: str) -> Callable:
    """
    Returns the appropriate scorer for the given output.
    """
    score_func: Callable = get_score_func(score, output)
    scorer = make_scorer(score_func=score_func, greater_is_better=greater_lookup[score])
    return scorer


def cross_validate_multioutput_regressor(regressor, X, y, cv) -> tuple[dict[str, float], np.ndarray]:
    # NOTE: This assumes the order of columns in y is [PCE, VOC, JSC, FF].
    """
    Cross-validate a multi-output regressor. Returns R, R2, RMSE, and MAE scores for each of the following:
    - Each output individually (PCE, VOC, JSC, FF)
    - PCE calculated from VOC, JSC, FF
    - All outputs together (PCE, VOC, JSC, FF)  # ATTN: What would this mean?
    """
    # Create scoring dictionary
    scoring: dict[str, Callable] = {f"{score}_{output}": multi_scorer(score, output) for score, output in
                                    product(["r", "r2", "rmse", "mae"], ["PCE", "VOC", "JSC", "FF", "PCE_eqn"])}
    scores = cross_validate(regressor, X, y,
                            cv=cv,
                            scoring={
                                **scoring,
                                "r":    r_scorer,
                                "r2":   r2_scorer,
                                "rmse": rmse_scorer,
                                "mae":  mae_scorer,
                            },
                            n_jobs=-1)

    predictions = cross_val_predict(regressor, X, y,
                                    cv=cv,
                                    n_jobs=-1)
    print(scores)
    return scores, predictions
