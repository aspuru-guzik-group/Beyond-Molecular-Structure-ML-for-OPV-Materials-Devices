from pathlib import Path

import pandas as pd

from training_utils import process_scores, run_structure_and_scalar, save_results


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def main(representation: str, scalar_filter: str, regressor_type: str, target_features: list[str],
         hyperparameter_optimization: bool) -> None:
    dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
    opv_dataset: pd.DataFrame = pd.read_pickle(dataset)

    radius = 6
    n_bits = 4096
    structural_features: list[str] = [f"Donor ECFP{2 * radius}_{n_bits}", f"Acceptor ECFP{2 * radius}_{n_bits}"]
    unroll = {"representation": representation,
              "radius":         radius,
              "n_bits":         n_bits,
              "col_names":      structural_features}

    scores, predictions = run_structure_and_scalar(opv_dataset,
                                                   structural_features=structural_features,
                                                   scalar_filter=scalar_filter,
                                                   target_features=target_features,
                                                   scaler_type="Standard",
                                                   regressor_type=regressor_type,
                                                   unroll=unroll,
                                                   hyperparameter_optimization=hyperparameter_optimization,
                                                   )

    scores = process_scores(scores)

    results_dir: Path = HERE.parent.parent / "results"
    fname_root: str = f"{representation}_{regressor_type}_{scalar_filter}"
    save_results(scores, predictions, results_dir, fname_root)


if __name__ == "__main__":
    main("ECFP", "material properties", "RF",
         target_features=["calculated PCE (%)"],
         hyperparameter_optimization=False
         )
