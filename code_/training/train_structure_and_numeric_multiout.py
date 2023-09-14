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


def main_graphs_and_numeric(
    dataset: pd.DataFrame,
    scalar_filter: str,
    subspace_filter: Optional[str],
    regressor_type: str,
    target_features: list[str],
    hyperparameter_optimization: bool,
) -> None:
    """
    Only acceptable for GNNPredictor
    """
    representation: str = "SMILES"
    structural_features: list[str] = ["Donor SMILES", "Acceptor SMILES"]
    unroll = None

    scores, predictions = run_graphs_only(
        dataset=dataset,
        structural_features=structural_features,
        target_features=target_features,
        regressor_type=regressor_type,
        unroll=unroll,
        hyperparameter_optimization=hyperparameter_optimization,
    )

    scores = process_scores(scores)

    save_results(
        scores,
        predictions,
        representation=representation,
        scalar_filter=scalar_filter,
        subspace_filter=subspace_filter,
        target_features=target_features,
        regressor_type=regressor_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_ecfp_and_numeric(
    dataset: pd.DataFrame,
    regressor_type: str,
    scalar_filter: str,
    subspace_filter: Optional[str],
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    radius: int = 5,
) -> None:
    representation: str = "ECFP"
    n_bits = radius_to_bits[radius]
    structural_features: list[str] = [
        f"Donor ECFP{2 * radius}_{n_bits}",
        f"Acceptor ECFP{2 * radius}_{n_bits}",
    ]
    unroll_single_feat = {
        "representation": representation,
        "radius": radius,
        "n_bits": n_bits,
        "col_names": structural_features,
    }

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll_single_feat,
        scalar_filter=scalar_filter,
        subspace_filter=subspace_filter,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_mordred_and_numeric(
    dataset: pd.DataFrame,
    regressor_type: str,
    scalar_filter: str,
    subspace_filter: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    representation: str = "mordred"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll_single_feat = {"representation": representation}

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll_single_feat,
        scalar_filter=scalar_filter,
        subspace_filter=subspace_filter,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_multioutput_grid(target_feats: list[str], hyperopt: bool = False) -> None:
    transform_type = "Standard"

    # for model in ["RF", "XGB", "HGB", "NGB", "NN"]:
    for model in ["ANN"]:
        opv_dataset: pd.DataFrame = get_appropriate_dataset(model)

        main_ecfp_and_numeric(
            dataset=opv_dataset,
            regressor_type=model,
            scalar_filter="device architecture",
            subspace_filter=None,
            target_features=target_feats,
            transform_type=transform_type,
            hyperparameter_optimization=hyperopt,
        )

        # main_mordred_and_numeric(dataset=opv_dataset,
        #                          regressor_type=model,
        #                          scalar_filter="device architecture",
        #                          subspace_filter=None,
        #                          target_features=target_feats,
        #                          transform_type=transform_type,
        #                          hyperparameter_optimization=hyperopt)


if __name__ == "__main__":
    # main_multioutput_grid(
    #         target_feats=["calculated PCE (%)", "Voc (V)", "Jsc (mA cm^-2)", "FF (%)"], hyperopt=False)
    targets_to_try = [
        ["Voc (V)"],
        ["Jsc (mA cm^-2)"],
        ["Voc (V)", "Jsc (mA cm^-2)", "FF (%)"],
    ]
    for target in targets_to_try:
        model = "ANN"
        main_ecfp_and_numeric(
            dataset=get_appropriate_dataset(model),
            regressor_type=model,
            scalar_filter="device architecture",
            subspace_filter=None,
            target_features=target,
            transform_type="Standard",
            hyperparameter_optimization=False,
        )

    # for target in ["calculated PCE (%)", "Voc (V)", "Jsc (mA cm^-2)", "FF (%)"]:
    #     main_representation_and_fabrication_grid(target_feats=[target], hyperopt=False)

    # main_ecfp_and_numeric(dataset=get_appropriate_dataset("RF"),
    #                       regressor_type="RF",
    #                       scalar_filter="device architecture",
    #                       # subspace_filter="material properties",
    #                       subspace_filter=None,
    #                       target_features=["calculated PCE (%)"],
    #                       hyperparameter_optimization=False,
    #                       radius=5)
