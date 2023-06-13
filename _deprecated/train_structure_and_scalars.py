from pathlib import Path
from typing import Optional

import pandas as pd

from data_handling import save_results, target_abbrev
from scoring import process_scores
from training_utils import run_structure_and_scalar

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def _structure_and_scalars(representation: str,
                           structural_features: list[str],
                           unroll: dict[str, str],
                           scalar_filter: str,
                           regressor_type: str,
                           target_features: list[str],
                           hyperparameter_optimization: bool,
                           subspace_filter: Optional[str] = None,
                           # subdir_ids: list[Union[str, int]],
                           **kwargs
                           ) -> None:
    dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
    opv_dataset: pd.DataFrame = pd.read_pickle(dataset)

    scores, predictions = run_structure_and_scalar(opv_dataset,
                                                   representation=representation,
                                                   structural_features=structural_features,
                                                   scalar_filter=scalar_filter,
                                                   subspace_filter=subspace_filter,
                                                   target_features=target_features,
                                                   regressor_type=regressor_type,
                                                   hyperparameter_optimization=hyperparameter_optimization,
                                                   unroll=unroll,
                                                   )

    scores = process_scores(scores)

    targets_dir: str = "-".join([target_abbrev[target] for target in target_features])
    all_features: list[str] = [representation, scalar_filter]
    features_dir: str = "-".join(all_features)
    if subspace_filter is not None:
        results_dir: Path = HERE.parent.parent / "results" / f"target_{targets_dir}" / f"features_{features_dir}" / f"subspace_{subspace_filter}"
    else:
        results_dir: Path = HERE.parent.parent / "results" / f"target_{targets_dir}" / f"features_{features_dir}"
    save_results(scores, predictions,
                 results_dir=results_dir,
                 regressor_type=regressor_type,
                 hyperparameter_optimization=hyperparameter_optimization,
                 )


if __name__ == "__main__":
    pass
