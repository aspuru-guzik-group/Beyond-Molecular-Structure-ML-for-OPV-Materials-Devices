from pathlib import Path
from typing import Optional

import pandas as pd

from data_handling import save_results
from filter_data import get_appropriate_dataset
from pipeline_utils import radius_to_bits
from scoring import process_scores
from training_utils import run_graphs_only, train_regressor

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
SAEKI = DATASETS / "Saeki_2022_n1318"


def main_ecfp_and_numeric(dataset: pd.DataFrame,
                          regressor_type: str,
                          scalar_filter: str,
                          subspace_filter: Optional[str],
                          target_features: list[str],
                          transform_type: str,
                          hyperparameter_optimization: bool,
                          radius: int = 5) -> None:
    representation: str = "ECFP"
    n_bits = radius_to_bits[radius]
    structural_features: list[str] = [f"Donor ECFP{2 * radius}_{n_bits}",
                                      f"Acceptor ECFP{2 * radius}_{n_bits}"]
    unroll_single_feat = {"representation": representation,
                          "radius":         radius,
                          "n_bits":         n_bits,
                          "col_names":      structural_features}

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=structural_features,
                    unroll=unroll_single_feat,
                    scalar_filter=scalar_filter,
                    subspace_filter=subspace_filter,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    transform_type=transform_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    output_dir_name="results_Saeki",
                    )


def main_mordred_and_numeric(dataset: pd.DataFrame,
                             regressor_type: str,
                             scalar_filter: str,
                             subspace_filter: str,
                             target_features: list[str],
                             transform_type: str,
                             hyperparameter_optimization: bool) -> None:
    representation: str = "mordred"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll_single_feat = {"representation": representation,
                          "data file": SAEKI / "Saeki_mordred.pkl",}

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=structural_features,
                    unroll=unroll_single_feat,
                    scalar_filter=scalar_filter,
                    subspace_filter=subspace_filter,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    transform_type=transform_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    output_dir_name="results_Saeki",
                    )


def main_compare_Saeki(target_feats: list[str], hyperopt: bool = False) -> None:
    dataset_file: Path = SAEKI / "Saeki_corrected_pipeline.pkl"
    opv_dataset: pd.DataFrame = pd.read_pickle(dataset_file)
    transform_type = "Standard"
    filter = "material properties Saeki"
    subspace = None

    for model in ["RF", "HGB", ]:
        # # ECFP
        # main_ecfp_and_numeric(dataset=opv_dataset,
        #                       regressor_type=model,
        #                       scalar_filter=filter,
        #                       subspace_filter=subspace,
        #                       target_features=target_feats,
        #                       transform_type=transform_type,
        #                       hyperparameter_optimization=hyperopt)
        # mordred
        main_mordred_and_numeric(dataset=opv_dataset,
                                 regressor_type=model,
                                 scalar_filter=filter,
                                 subspace_filter=subspace,
                                 target_features=target_feats,
                                 transform_type=transform_type,
                                 hyperparameter_optimization=hyperopt)


if __name__ == "__main__":
    main_compare_Saeki(target_feats=["calculated PCE (%)"], hyperopt=False)
