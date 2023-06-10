import json
from pathlib import Path

import numpy as np
import pandas as pd


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
                 subdir_ids: list[str],
                 regressor_type: str,
                 ) -> None:
    sub_dir_name: str = "_".join([str(id) for id in subdir_ids])
    results_dir = results_dir / "hyperopt" if "hyperopt" in subdir_ids else results_dir
    sub_dir: Path = results_dir / sub_dir_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    fname_root: str = f"{regressor_type}_{sub_dir_name}"

    scores_file: Path = sub_dir / f"{fname_root}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)

    # scores_file: Path = sub_dir / f"{fname_root}_scores.pkl"
    # with open(scores_file, "wb") as f:
    #     pickle.dump(scores, f)

    predictions_file: Path = sub_dir / f"{fname_root}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    print("Saved results to:")
    print(scores_file)
    print(predictions_file)