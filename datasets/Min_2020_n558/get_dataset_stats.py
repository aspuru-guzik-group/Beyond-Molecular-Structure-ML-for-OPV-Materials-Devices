from scipy import stats as st
import pandas as pd
from pathlib import Path
import numpy as np

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE

dataset = pd.read_csv(DATASETS / "cleaned_dataset.csv")
dataset_stats: str = "cleaned_dataset_stats.csv"

stats: dict = {}
for col in dataset.columns:
    # get values of column
    col_values = dataset[col].values
    if col_values.dtype == np.float64:
        stats[col] = {}
        col_values_no_nan = col_values[~np.isnan(col_values)]
        stats[col]["Mean"] = round(np.mean(col_values_no_nan), 2)
        stats[col]["Std. Dev."] = round(np.std(col_values_no_nan), 2)
        stats[col]["Median"] = round(np.median(col_values_no_nan), 2)
        stats[col]["Mode"] = round(st.mode(col_values_no_nan)[0][0], 2)
        stats[col]["Skew"] = round(st.skew(col_values_no_nan), 2)
        stats[col]["Min"] = round(np.min(col_values_no_nan), 2)
        stats[col]["Max"] = round(np.max(col_values_no_nan), 2)
        stats[col]["Range"] = round(
            np.max(col_values_no_nan) - np.min(col_values_no_nan), 2
        )
        stats[col]["Missing (%)"] = round(
            (np.count_nonzero(np.isnan(col_values)) / len(col_values)) * 100, 2
        )

stats_df = pd.DataFrame.from_dict(
    stats,
    orient="index",
)
print(stats_df.columns)
stats_df.to_csv(DATASETS / dataset_stats)
