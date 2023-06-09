import sys
from pathlib import Path
from typing import Union

import pandas as pd

from data_handling import save_results
from models import regressor_factory
from scoring import process_scores
from training_utils import run_structure_and_scalar, run_structure_only

sys.path.append("../pipeline")
from pipeline_utils import radius_to_bits

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def _structure_only(representation: str,
                    structural_features: list[str],
                    unroll: dict[str, str],
                    regressor_type: str,
                    target_features: list[str],
                    hyperparameter_optimization: bool,
                    subdir_ids: list[Union[str, int]]
                    ) -> None:
    dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"  # TODO: Change?
    opv_dataset: pd.DataFrame = pd.read_pickle(dataset)

    scores, predictions = run_structure_only(opv_dataset,
                                             representation=representation,
                                             structural_features=structural_features,
                                             target_features=target_features,
                                             regressor_type=regressor_type,
                                             unroll=unroll,
                                             hyperparameter_optimization=hyperparameter_optimization,
                                             )

    scores = process_scores(scores)

    struct_only_dir: Path = HERE.parent.parent / "results" / "structure_only"
    subdir_ids = subdir_ids + ["hyperopt"] if hyperparameter_optimization else subdir_ids
    save_results(scores, predictions,
                 results_dir=struct_only_dir,
                 subdir_ids=subdir_ids,
                 regressor_type=regressor_type)


def main_ecfp_only(regressor_type: str,
                   target_features: list[str],
                   hyperparameter_optimization: bool,
                   radius: int = 5) -> None:
    representation: str = "ECFP"
    n_bits = radius_to_bits[radius]
    structural_features: list[str] = [f"Donor ECFP{2 * radius}_{n_bits}",
                                      f"Acceptor ECFP{2 * radius}_{n_bits}"]
    unroll = {"representation": representation,
              "radius":         radius,
              "n_bits":         n_bits,
              "col_names":      structural_features}

    _structure_only(representation=representation,
                    structural_features=structural_features,
                    unroll=unroll,
                    regressor_type=regressor_type,
                    target_features=target_features,
                    hyperparameter_optimization=hyperparameter_optimization,
                    subdir_ids=[f"{representation}{radius}-{n_bits}"])


def main_tokenized_only(representation: str, regressor_type: str, target_features: list[str],
                        hyperparameter_optimization: bool) -> None:
    structural_features: list[str] = [f"Donor {representation} token",
                                      f"Acceptor {representation} token"]
    unroll = {"representation": representation}

    _structure_only(representation=representation,
                    structural_features=structural_features,
                    unroll=unroll,
                    regressor_type=regressor_type,
                    target_features=target_features,
                    hyperparameter_optimization=hyperparameter_optimization,
                    subdir_ids=[representation])


def main_ohe_only(regressor_type: str, target_features: list[str],
                  hyperparameter_optimization: bool) -> None:
    representation: str = "OHE"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll = {"representation": representation}

    _structure_only(representation=representation,
                    structural_features=structural_features,
                    unroll=unroll,
                    regressor_type=regressor_type,
                    target_features=target_features,
                    hyperparameter_optimization=hyperparameter_optimization,
                    subdir_ids=[representation])


def main_mordred_only(regressor_type: str, target_features: list[str],
                      hyperparameter_optimization: bool) -> None:
    representation: str = "mordred"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll = {"representation": representation}

    _structure_only(representation=representation,
                    structural_features=structural_features,
                    unroll=unroll,
                    regressor_type=regressor_type,
                    target_features=target_features,
                    hyperparameter_optimization=hyperparameter_optimization,
                    subdir_ids=[representation])


def main_properties_only(regressor_type: str, target_features: list[str],
                         hyperparameter_optimization: bool) -> None:
    representation: str = "material properties"
    mater_props: list[str] = ["HOMO", "LUMO", "Ehl", "Eg"]
    structural_features: list[str] = [*[f"{p}_D (eV)" for p in mater_props],
                                      *[f"{p}_A (eV)" for p in mater_props]]
    unroll = {"representation": representation}

    _structure_only(representation=representation,
                    structural_features=structural_features,
                    unroll=unroll,
                    regressor_type=regressor_type,
                    target_features=target_features,
                    hyperparameter_optimization=hyperparameter_optimization,
                    subdir_ids=[representation])


def main_processing_only(regressor_type: str, target_features: list[str],
                         hyperparameter_optimization: bool) -> None:
    representation: str = "fabrication only"
    structural_features: list[str] = []
    unroll = {}
    scalar_filter: str = "fabrication only"

    dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"  # TODO: Change?
    opv_dataset: pd.DataFrame = pd.read_pickle(dataset)

    scores, predictions = run_structure_and_scalar(opv_dataset,
                                                   representation=representation,
                                                   structural_features=structural_features,
                                                   scalar_filter=scalar_filter,
                                                   scaler_type="Standard",
                                                   target_features=target_features,
                                                   regressor_type=regressor_type,
                                                   unroll=unroll,
                                                   hyperparameter_optimization=hyperparameter_optimization,

                                                   )

    scores = process_scores(scores)

    struct_only_dir: Path = HERE.parent.parent / "results" / "structure_only"
    subdir_ids: list[str] = [representation]
    subdir_ids = subdir_ids + ["hyperopt"] if hyperparameter_optimization else subdir_ids
    save_results(scores, predictions,
                 results_dir=struct_only_dir,
                 subdir_ids=subdir_ids,
                 regressor_type=regressor_type)


def main_grid(hyperopt: bool = False) -> None:
    for model in regressor_factory:
        target_feats: list[str] = ["calculated PCE (%)"]

        # ECFP
        main_ecfp_only(model,
                       target_features=target_feats,
                       hyperparameter_optimization=hyperopt)
        # mordred
        main_mordred_only(model,
                          target_features=target_feats,
                          hyperparameter_optimization=hyperopt)

        # OHE
        main_ohe_only(model,
                      target_features=target_feats,
                      hyperparameter_optimization=hyperopt)

        # tokenized
        for struct_repr in ["BRICS", "SELFIES", "SMILES"]:
            main_tokenized_only(struct_repr,
                                model,
                                target_features=target_feats,
                                hyperparameter_optimization=hyperopt)

        # material properties
        main_properties_only(model,
                             target_features=target_feats,
                             hyperparameter_optimization=hyperopt)

        # processing only
        main_processing_only(model,
                             target_features=target_feats,
                             hyperparameter_optimization=hyperopt)


if __name__ == "__main__":
    # for h_opt in [True, False]:
    #     main_grid(hyperopt=h_opt)

    # main_grid(hyperopt=False)

    main_ecfp_only("KRR",
                   target_features=["calculated PCE (%)"],
                   hyperparameter_optimization=True)
