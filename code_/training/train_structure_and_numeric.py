from pathlib import Path

import pandas as pd

from data_handling import save_results, target_abbrev
from models import regressor_factory
from pipeline_utils import radius_to_bits
from scoring import process_scores
from training_utils import run_graphs_only, train_regressor

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def main_graphs_and_numeric(dataset: pd.DataFrame,
                            regressor_type: str,
                            target_features: list[str],
                            hyperparameter_optimization: bool) -> None:
    """
    Only acceptable for GNNPredictor
    """
    representation: str = "SMILES"
    structural_features: list[str] = [f"Donor SMILES", "Acceptor SMILES"]
    unroll = None

    scores, predictions = run_graphs_only(dataset=dataset,
                                          structural_features=structural_features,
                                          target_features=target_features,
                                          regressor_type=regressor_type,
                                          unroll=unroll,
                                          hyperparameter_optimization=hyperparameter_optimization,
                                          )

    scores = process_scores(scores)

    targets_dir: str = "-".join([target_abbrev[target] for target in target_features])
    features_dir: str = "-".join([representation])
    results_dir: Path = HERE.parent.parent / "results" / f"target_{targets_dir}" / f"features_{features_dir}"
    save_results(scores, predictions,
                 results_dir=results_dir,
                 regressor_type=regressor_type,
                 hyperparameter_optimization=hyperparameter_optimization,
                 )


def main_ecfp_and_numeric(dataset: pd.DataFrame,
                          regressor_type: str,
                          scalar_filter: str,
                          subspace_filter: str,
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
                    scalar_filter=scalar_filter,
                    subspace_filter=subspace_filter,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    hyperparameter_optimization=hyperparameter_optimization,
                    )


def main_mordred_and_numeric(dataset: pd.DataFrame,
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


def main_grid(target_feats: list[str], hyperopt: bool = False) -> None:
    for model in regressor_factory:
        opv_dataset: pd.DataFrame = get_appropriate_dataset(model)

        # TODO: How to iterate over all combinations of filters?
        # TODO: How to save results? Additional datahierarchy?
        filters = ["material properties", "fabrication", "device architecture"]
        for i, filter in enumerate(filters):
            for subspace in

        if model == "GNN":
            # import pdb; pdb.set_trace()
            main_graphs_and_numeric(dataset=opv_dataset,
                                    regressor_type=model,
                                    target_features=target_feats,
                                    hyperparameter_optimization=hyperopt)

        else:
            # ECFP
            main_ecfp_and_numeric(dataset=opv_dataset,
                                  regressor_type=model,
                                  target_features=target_feats,
                                  hyperparameter_optimization=hyperopt)
            # mordred
            main_mordred_and_numeric(dataset=opv_dataset,
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

    main_ecfp_and_numeric(dataset=get_appropriate_dataset("RF"),
                          regressor_type="RF",
                          scalar_filter="device architecture",
                          subspace_filter="material properties",
                          target_features=["calculated PCE (%)"],
                          hyperparameter_optimization=False,
                          radius=5)
