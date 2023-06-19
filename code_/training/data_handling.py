import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE: Path = Path(__file__).resolve().parent

target_abbrev: Dict[str, str] = {
    "calculated PCE (%)": "PCE",
    "Voc (V)":            "Voc",
    "Jsc (mA cm^-2)":     "Jsc",
    "FF (%)":             "FF",
}


def remove_unserializable_keys(d: Dict) -> Dict:
    """Remove unserializable keys from a dictionary."""
    # for k, v in d.items():
    #     if not isinstance(v, (str, int, float, bool, NoneType, tuple, list, np.ndarray, np.floating, np.integer)):
    #         d.pop(k)
    #     elif isinstance(v, dict):
    #         d[k] = remove_unserializable_keys(v)
    # return d
    new_d: dict = {k: v for k, v in d.items() if
                   isinstance(v, (str, int, float, bool, NoneType, np.floating, np.integer))}
    return new_d


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


def _save(scores: Dict[int, Dict[str, float]],
          predictions: pd.DataFrame,
          results_dir: Path,
          regressor_type: str,
          imputer: Optional[str],
          hyperparameter_optimization: bool,
          ) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    fname_root = f"{regressor_type}_{imputer} imputer" if imputer else regressor_type
    fname_root = f"{fname_root}_hyperopt" if hyperparameter_optimization else fname_root

    # fname_root: str = f"{regressor_type}_hyperopt" if hyperparameter_optimization else regressor_type
    print("Filename:", fname_root)

    scores_file: Path = results_dir / f"{fname_root}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)

    predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    print("Saved results to:")
    print(scores_file)
    print(predictions_file)


def save_results(scores: dict,
                 predictions: pd.DataFrame,
                 representation: str,
                 scalar_filter: Optional[str],
                 subspace_filter: Optional[str],
                 target_features: list,
                 regressor_type: str,
                 imputer: Optional[str],
                 hyperparameter_optimization: bool,
                 ) -> None:
    targets_dir: str = "-".join([target_abbrev[target] for target in target_features])

    if representation != scalar_filter:
        feature_ids: list = [feature for feature in [representation, scalar_filter] if feature is not None]
    else:
        feature_ids: list = [representation]
    features_dir: str = "-".join(feature_ids)
    results_dir: Path = HERE.parent.parent / "results" / f"target_{targets_dir}" / f"features_{features_dir}"
    if subspace_filter:
        results_dir = results_dir / f"subspace_{subspace_filter}"

    _save(scores, predictions,
          results_dir=results_dir,
          regressor_type=regressor_type,
          imputer=imputer,
          hyperparameter_optimization=hyperparameter_optimization,
          )
