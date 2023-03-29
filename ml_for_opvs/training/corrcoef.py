import pandas as pd
import pathlib
import numpy as np

current_dir = pathlib.Path(__file__).parent.absolute()
corr_coefs = []
for x in range(0, 5):
    pred_dir = current_dir / f"prediction_{x}.csv"
    predictions = pd.read_csv(pred_dir)
    corr_coefs.append(
        np.corrcoef(
            predictions["calc_PCE_percent"], predictions["predicted_calc_PCE_percent"]
        )[0, 1]
    )

print(np.mean(corr_coefs))
print(np.std(corr_coefs))
