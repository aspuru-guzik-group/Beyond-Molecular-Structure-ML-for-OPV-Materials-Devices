import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_filepath: Path = (
    Path(__file__).parent.parent.parent.parent.resolve()
    / "datasets"
    / "Min_2020_n558"
    / "cleaned_dataset.csv"
)

data: pd.DataFrame = pd.read_csv(data_filepath)
print(data.columns)

incorrect_PCE: int = 0
total: int = 0
for index, row in data.iterrows():
    if abs(row["PCE (%)"] - row["calculated PCE (%)"]) > 1:
        incorrect_PCE += 1
    total += 1


print(incorrect_PCE, total)

print(incorrect_PCE * 100 / total)
