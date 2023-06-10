import json
from pathlib import Path

import numpy as np
import pandas as pd

target_abbrev: dict[str, str] = {
    "calculated PCE (%)": "PCE",
    "Voc (V)":           "Voc",
    "Jsc (mA cm^-2)":   "Jsc",
    "FF (%)":            "FF",
}


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def save_results(scores: dict[int, dict[str, float]],
                 predictions: pd.DataFrame,
                 results_dir: Path,
                 regressor_type: str,
                 hyperparameter_optimization: bool,
                 ) -> None:
    # sub_dir_name: str = "_".join([str(id) for id in subdir_ids])
    # results_dir = results_dir / "hyperopt" if "hyperopt" in subdir_ids else results_dir
    # sub_dir: Path = results_dir / sub_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    fname_root: str = f"{regressor_type}_hyperopt" if hyperparameter_optimization else regressor_type

    scores_file: Path = results_dir / f"{fname_root}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)

    predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    print("Saved results to:")
    print(scores_file)
    print(predictions_file)