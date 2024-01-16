import pandas as pd

from models import regressor_factory
from data_handling import save_results
from filter_data import get_appropriate_dataset
from pipeline_utils import radius_to_bits
from scoring import process_scores
from training_utils import run_graphs_only, train_regressor


# HERE: Path = Path(__file__).resolve().parent
# DATASETS: Path = HERE.parent.parent / "datasets"


# def _structure_only(representation: str,
#                     structural_features: list[str],
#                     unroll: dict[str, str],
#                     regressor_type: str,
#                     target_features: list[str],
#                     hyperparameter_optimization: bool,
#                     # subdir_ids: list[Union[str, int]]
#                     ) -> None:
#     dataset = DATASETS / "Min_2020_n558" / "cleaned_dataset_nans.pkl"
#     opv_dataset: pd.DataFrame = pd.read_pickle(dataset)
#
#     if regressor_type == "GNN":
#         scores, predictions = run_graphs_only(opv_dataset,
#                                              structural_features=structural_features,
#                                              target_features=target_features,
#                                              regressor_type=regressor_type,
#                                              unroll=unroll,
#                                              hyperparameter_optimization=hyperparameter_optimization,
#                                              )
# elif regressor_type == "GP":
#     scores, predictions = run_structure_only(opv_dataset,
#                                             representation=representation,
#                                             structural_features=structural_features,
#                                             target_features=target_features,
#                                             regressor_type=regressor_type,
#                                             unroll=unroll,
#                                             hyperparameter_optimization=hyperparameter_optimization,
#                                             kernel="tanimoto" if "ECFP" in representation else "rbf"
#                                             )
# else:
#     scores, predictions = run_structure_only(opv_dataset,
#                                             representation=representation,
#                                             structural_features=structural_features,
#                                             target_features=target_features,
#                                             regressor_type=regressor_type,
#                                             unroll=unroll,
#                                             hyperparameter_optimization=hyperparameter_optimization,
#                                             )


def main_graphs_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    hyperparameter_optimization: bool,
) -> None:
    """
    Only acceptable for GNNPredictor
    """
    representation: str = "SMILES"
    structural_features: list[str] = [f"Donor SMILES", "Acceptor SMILES"]
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
        scores=scores,
        predictions=predictions,
        representation=representation,
        scalar_filter=None,
        subspace_filter=None,
        regressor_type=regressor_type,
        target_features=target_features,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_ecfp_only(
    dataset: pd.DataFrame,
    regressor_type: str,
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
        scalar_filter=None,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_tokenized_only(
    dataset: pd.DataFrame,
    representation: str,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    structural_features: list[str] = [
        f"Donor {representation} token",
        f"Acceptor {representation} token",
    ]
    unroll_single_feat = {"representation": representation}

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll_single_feat,
        scalar_filter=None,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_ohe_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    representation: str = "OHE"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll_single_feat = {"representation": representation}

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll_single_feat,
        scalar_filter=None,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_mordred_only(
    dataset: pd.DataFrame,
    regressor_type: str,
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
        scalar_filter=None,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_graph_embeddings_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    representation: str = "graph embeddings"
    structural_features: list[str] = ["Donor", "Acceptor"]
    unroll_single_feat = {"representation": representation}

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll_single_feat,
        scalar_filter=None,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_properties_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    representation: str = "material properties"
    # mater_props: list[str] = ["HOMO", "LUMO", "Ehl", "Eg"]
    # scalar_features: list[str] = [*[f"{p}_D (eV)" for p in mater_props],
    #                               *[f"{p}_A (eV)" for p in mater_props]]
    # unroll_single_feat = {"representation": representation}

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=None,
        unroll=None,
        scalar_filter=representation,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_processing_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    representation: str = "fabrication only"
    structural_features: list[str] = [
        "solvent descriptors",
        "solvent additive descriptors",
    ]
    unroll_single_feat: dict[str, str] = {
        "representation": "solvent",
    }
    scalar_filter: str = "fabrication only"

    train_regressor(
        dataset=dataset,
        representation=representation,
        structural_features=structural_features,
        unroll=unroll_single_feat,
        scalar_filter=scalar_filter,
        subspace_filter=None,
        target_features=target_features,
        regressor_type=regressor_type,
        transform_type=transform_type,
        hyperparameter_optimization=hyperparameter_optimization,
    )


def main_representation_model_grid(
    target_feats: list[str], hyperopt: bool = False
) -> None:
    transform_type: str = "Standard"

    for model in ["MLR", "KNN", "SVR", "KRR", "GP", "RF", "XGB", "HGB", "NGB", "NN"]:
        opv_dataset: pd.DataFrame = get_appropriate_dataset(model)

        if model == "GNN":
            # import pdb; pdb.set_trace()
            main_graphs_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                hyperparameter_optimization=hyperopt,
            )

        else:
            # ECFP
            main_ecfp_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                transform_type=transform_type,
                hyperparameter_optimization=hyperopt,
            )
            # mordred
            main_mordred_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                transform_type=transform_type,
                hyperparameter_optimization=hyperopt,
            )

            # graph embeddings
            main_graph_embeddings_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                transform_type=transform_type,
                hyperparameter_optimization=hyperopt,
            )

            # OHE
            main_ohe_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                transform_type=transform_type,
                hyperparameter_optimization=hyperopt,
            )

            # tokenized
            for struct_repr in ["SELFIES", "SMILES"]:
                main_tokenized_only(
                    dataset=opv_dataset,
                    representation=struct_repr,
                    regressor_type=model,
                    target_features=target_feats,
                    transform_type=transform_type,
                    hyperparameter_optimization=hyperopt,
                )

            # material properties
            main_properties_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                transform_type=transform_type,
                hyperparameter_optimization=hyperopt,
            )

            # processing only
            main_processing_only(
                dataset=opv_dataset,
                regressor_type=model,
                target_features=target_feats,
                transform_type=transform_type,
                hyperparameter_optimization=hyperopt,
            )


if __name__ == "__main__":
    # for target in ["calculated PCE (%)", "Voc (V)", "Jsc (mA cm^-2)", "FF (%)"]:
    #     for h_opt in [False, True]:
    #         main_representation_model_grid(target_feats=[target], hyperopt=h_opt)

    # for target in ["Voc (V)", "Jsc (mA cm^-2)", "FF (%)"]:
    #     main_representation_model_grid(target_feats=[target], hyperopt=False)

    # main_representation_model_grid(target_feats=["calculated PCE (%)"], hyperopt=False)
    main_representation_model_grid(target_feats=["calculated PCE (%)"], hyperopt=False)

    # # Run one model
    # transform_type: str = "Standard"
    # model = "ANN"
    # opv_dataset: pd.DataFrame = get_appropriate_dataset(model)
    # main_ecfp_only(dataset=opv_dataset, regressor_type=model, target_features=["calculated PCE (%)"], transform_type=transform_type, hyperparameter_optimization=False)
