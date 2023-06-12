from pathlib import Path

import pandas as pd

from training_utils import train_regressor
from pipeline_utils import radius_to_bits

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def main_ecfp_only(dataset: pd.DataFrame,
                   regressor_type: str,
                   target_features: list[str],
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
                    scalar_filter=None,
                    subspace_filter=None,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_tokenized_only(dataset: pd.DataFrame,
                        representation: str,
                        regressor_type: str,
                        target_features: list[str],
                        hyperparameter_optimization: bool) -> None:
    structural_features: list[str] = [f"Donor {representation} token",
                                      f"Acceptor {representation} token"]
    unroll_single_feat = {"representation": representation}

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=structural_features,
                    unroll=unroll_single_feat,
                    scalar_filter=None,
                    subspace_filter=None,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_ohe_only(dataset: pd.DataFrame,
                  regressor_type: str,
                  target_features: list[str],
                  hyperparameter_optimization: bool
                  ) -> None:
    representation: str = "OHE"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll_single_feat = {"representation": representation}

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=structural_features,
                    unroll=unroll_single_feat,
                    scalar_filter=None,
                    subspace_filter=None,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_mordred_only(dataset: pd.DataFrame,
                      regressor_type: str, target_features: list[str],
                      hyperparameter_optimization: bool) -> None:
    representation: str = "mordred"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll_single_feat = {"representation": representation}

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=structural_features,
                    unroll=unroll_single_feat,
                    scalar_filter=None,
                    subspace_filter=None,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_properties_only(dataset: pd.DataFrame,
                         regressor_type: str, target_features: list[str],
                         hyperparameter_optimization: bool) -> None:
    representation: str = "material properties"
    # mater_props: list[str] = ["HOMO", "LUMO", "Ehl", "Eg"]
    # scalar_features: list[str] = [*[f"{p}_D (eV)" for p in mater_props],
    #                               *[f"{p}_A (eV)" for p in mater_props]]
    # unroll_single_feat = {"representation": representation}

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=None,
                    unroll=None,
                    scalar_filter=representation,
                    subspace_filter=None,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_processing_only(dataset: pd.DataFrame,
                         regressor_type: str, target_features: list[str],
                         hyperparameter_optimization: bool) -> None:
    representation: str = "fabrication only"
    structural_features: list[str] = ["solvent descriptors", "solvent additive descriptors"]
    unroll_single_feat: dict[str, str] = {"representation": "solvent",  # TODO: How to unroll multiple columns??
                                          #                           "solv_type":      solv_type
                                          }
    # unroll_multi_feat: list[dict[str, str]] = [{"representation": "solvent",
    #                                             "solv_type":      "solvent",
    #                                             "columns":        ["solvent descriptors"]},
    #                                            {"representation": "solvent",
    #                                             "solv_type":      "solvent additive",
    #                                             "columns":        ["solvent additive descriptors"]}]
    scalar_filter: str = "fabrication only"

    train_regressor(dataset=dataset,
                    representation=representation,
                    structural_features=structural_features,
                    unroll=unroll_single_feat,
                    scalar_filter=scalar_filter,
                    subspace_filter=None,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_grid(target_feats: list[str], hyperopt: bool = False) -> None:
    # for model in regressor_factory:
    for model in ["MLR"]:
        opv_dataset: pd.DataFrame = get_appropriate_dataset(model)

        # ECFP
        main_ecfp_only(dataset=opv_dataset,
                       regressor_type=model,
                       target_features=target_feats,
                       hyperparameter_optimization=hyperopt)
        # mordred
        main_mordred_only(dataset=opv_dataset,
                          regressor_type=model,
                          target_features=target_feats,
                          hyperparameter_optimization=hyperopt)

        # OHE
        main_ohe_only(dataset=opv_dataset,
                      regressor_type=model,
                      target_features=target_feats,
                      hyperparameter_optimization=hyperopt)

        # tokenized
        for struct_repr in ["BRICS", "SELFIES", "SMILES"]:
            main_tokenized_only(dataset=opv_dataset,
                                representation=struct_repr,
                                regressor_type=model,
                                target_features=target_feats,
                                hyperparameter_optimization=hyperopt)

        # material properties
        main_properties_only(dataset=opv_dataset,
                             regressor_type=model,
                             target_features=target_feats,
                             hyperparameter_optimization=hyperopt)

        # processing only
        main_processing_only(dataset=opv_dataset,
                             regressor_type=model,
                             target_features=target_feats,
                             hyperparameter_optimization=hyperopt)


def get_appropriate_dataset(model: str) -> pd.DataFrame:
    if model == "HGB":
        dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
    else:
        dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset.pkl"

    opv_dataset: pd.DataFrame = pd.read_pickle(dataset).reset_index(drop=True)
    return opv_dataset


if __name__ == "__main__":
    # for h_opt in [False, True]:
    #     main_grid(, hyperopt=h_opt)

    # for target in ["calculated PCE (%)", "Voc (V)", "Jsc (mA cm^-2)", "FF (%)"]:
    #     main_grid(target_feats=[target], hyperopt=False)

    # for target in ["calculated PCE (%)", "Voc (V)", "Jsc (mA cm^-2)", "FF (%)"]:
    #     for model in regressor_factory:
    #         main_processing_only(model, target_features=[target], hyperparameter_optimization=False)

    model = "MLR"
    main_processing_only(get_appropriate_dataset(model), model,
                         target_features=["calculated PCE (%)"],
                         hyperparameter_optimization=False)

    # main_ecfp_only("KRR",
    #                target_features=["calculated PCE (%)"],
    #                hyperparameter_optimization=True)
