import json

import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from code_python import DATASETS

scaler_factory: dict[str, type] = {"MinMax": MinMaxScaler, "Standard": StandardScaler}

regressor_factory: dict[str, type] = {
    "KNN": KNeighborsRegressor,
    "KRR": KernelRidge,
    "Lasso": Lasso,
    "MLR": LinearRegression,
    "RF": RandomForestRegressor,
    "SVR": SVR,
    "XGB": XGBRegressor,
}


def unroll_lists_to_columns(df: pd.DataFrame, unroll_cols: list[str], new_col_names: list[str]) -> pd.DataFrame:
    """
    Unroll a list of lists into columns of a DataFrame.

    Args:
        df: DataFrame to unroll.
        unroll_cols: List of columns containing list to unroll.
        new_col_names: List of new column names.

    Returns:
        DataFrame with unrolled columns.
    """
    rolled_cols: pd.DataFrame = df[unroll_cols]
    unrolled_df: pd.DataFrame = pd.concat([rolled_cols[col].apply(pd.Series) for col in rolled_cols.columns], axis=1)
    unrolled_df.columns = new_col_names
    return unrolled_df


def unroll_solvent_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    with open(DATASETS / "Min_2020_n558" / "selected_properties.json", "r") as f:
        solvent_descriptors: list[str] = json.load(f)["solvent"]

    solvent_cols: list[str] = ["solvent descriptors", "solvent additive descriptors"]
    solent_descriptor_cols: list[str] = [*[f"solvent {d}" for d in solvent_descriptors],
                                         *[f"additive {d}" for d in solvent_descriptors]]
    return unroll_lists_to_columns(df, solvent_cols, solent_descriptor_cols)


def evaluate_model(y_test, y_pred) -> dict[str, float]:
    """
    Score the model.

    Args:
        y_test: Test targets.
        y_pred: Predicted targets.

    Returns:
        Dictionary of scores.
    """
    scores: dict[str, float] = {
        "r": pearsonr(y_test, y_pred)[0],
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred)
    }
    return scores
