import json
from pathlib import Path
from typing import Callable, Union

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, StandardScaler

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"

scaler_factory: dict[str, type] = {"MinMax": MinMaxScaler, "Standard": StandardScaler}


def unroll_lists_to_columns(df: pd.DataFrame, unroll_cols: list[str], new_col_names: list[str]) -> pd.DataFrame:
    """
    Unroll a list of lists into columns of a DataFrame.

    Args:
        df: DataFrame to unroll.
        unroll_cols: List of columns containing list to unroll.
        new_col_names: List of new column names.

    Returns:
        DataFrame with unrolled columns.
    """
    # rolled_cols: pd.DataFrame = df[unroll_cols]
    rolled_cols: pd.DataFrame = df
    unrolled_df: pd.DataFrame = pd.concat([rolled_cols[col].apply(pd.Series) for col in rolled_cols.columns], axis=1)
    unrolled_df.columns = new_col_names
    return unrolled_df


def unroll_solvent_descriptors(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    with open(DATASETS / "Min_2020_n558" / "selected_properties.json", "r") as f:
        solvent_descriptors: list[str] = json.load(f)["solvent"]

    solvent_cols: list[str] = ["solvent descriptors", "solvent additive descriptors"]
    solent_descriptor_cols: list[str] = [*[f"solvent {d}" for d in solvent_descriptors],
                                         *[f"solvent additive {d}" for d in solvent_descriptors]
                                         ]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, solvent_cols, solent_descriptor_cols)
    return new_df


def unroll_fingerprints(df: pd.DataFrame, col_names: list[str] = [], radius: int = 0, n_bits: int = 0,
                        **kwargs) -> pd.DataFrame:
    new_ecfp_col_names: list[str] = [*[f"D EFCP{2 * radius}_bit{i}" for i in range(n_bits)],
                                     *[f"A ECFP{2 * radius}_bit{i}" for i in range(n_bits)]]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_ecfp_col_names)
    return new_df


def get_token_len(df: pd.DataFrame) -> list[int]:
    token_lens: list[int] = []
    for col in df.columns:
        tokenized = df.loc[0, col]
        pass
        if isinstance(tokenized, list):
            n_tokens: int = len(tokenized)
        else:
            n_tokens: int = 1
        token_lens.append(n_tokens)
    return token_lens


def unroll_tokens(df: pd.DataFrame, col_names: list[str] = [], **kwargs) -> pd.DataFrame:
    num_tokens: list[int] = get_token_len(df)
    token_type: str = df.columns[0].split(" ")[1]
    new_token_col_names: list[str] = [*[f"D {token_type} {i}" for i in range(num_tokens[0])],
                                      *[f"A {token_type} {i}" for i in range(num_tokens[1])]]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_token_col_names)
    return new_df


def get_ohe_structures(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # df = df[["Donor", "Acceptor"]]
    ohe: OneHotEncoder = OneHotEncoder(sparse=False).set_output(transform="pandas")
    new_df: pd.DataFrame = ohe.fit_transform(df)
    return new_df


def get_mordred_descriptors(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    with open(DATASETS / "Min_2020_n558" / "cleaned_dataset_mordred.pkl", "rb") as f:
        mordred: pd.DataFrame = pd.read_pickle(f)
    return mordred


def get_material_properties(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df


unrolling_factory: dict[str, Callable] = {"solvent":             unroll_solvent_descriptors,
                                          "solvent additive":    unroll_solvent_descriptors,
                                          "ECFP":                unroll_fingerprints,
                                          "mordred":             get_mordred_descriptors,
                                          "BRICS":               unroll_tokens,
                                          "SELFIES":             unroll_tokens,
                                          "SMILES":              unroll_tokens,
                                          "OHE":                 get_ohe_structures,
                                          "material properties": get_material_properties,
                                          # "GNN":     get_gnn_embeddings,  # TODO: Implement GNN embeddings
                                          }

# class GaussianQuantileTransformer(QuantileTransformer):
#     def __init__(
#             self,
#             *,
#             n_quantiles=1000,
#             output_distribution="normal",
#             ignore_implicit_zeros=False,
#             subsample=10_000,
#             random_state=None,
#             copy=True,
#     ):
#         super().__init__(
#             n_quantiles=n_quantiles,
#             output_distribution=output_distribution,
#             ignore_implicit_zeros=ignore_implicit_zeros,
#             subsample=subsample,
#             random_state=random_state,
#             copy=copy,
#         )


representation_scaling_factory: dict[str, dict[str, Union[Callable, str]]] = {
    "solvent":             {"callable": PowerTransformer,
                            "type":     "Power"},
    "ECFP":                {"callable": MinMaxScaler, "type": "MinMax"},
    "mordred":             {"callable": PowerTransformer,
                            "type":     "Power"},
    "BRICS":               {"callable": MinMaxScaler, "type": "MinMax"},
    "SELFIES":             {"callable": MinMaxScaler, "type": "MinMax"},
    "SMILES":              {"callable": MinMaxScaler, "type": "MinMax"},
    "OHE":                 {"callable": MinMaxScaler, "type": "MinMax"},
    "material properties": {"callable": PowerTransformer, "type":     "Power"},
    # "fabrication only":    {"callable": PowerTransformer,
    #                         "type":     "Power"},
    # "GNN":     {"callable": MinMaxScaler, "type": "MinMax"},
}

radius_to_bits: dict[int, int] = {3: 512, 4: 1024, 5: 2048, 6: 4096}

# def get_feature_scaling(feature: str) -> TransformerMixin:
#     if feature in [
#         "Donor PDI", "Donor Mn (kDa)", "Donor Mw (kDa)",
#         "HOMO_D (eV)", "LUMO_D (eV)", "Eg_D (eV)", "Ehl_D (eV)",
#         "HOMO_A (eV)", "LUMO_A (eV)", "Eg_A (eV)", "Ehl_A (eV)",
#         "active layer thickness (nm)",
#     ]:
#         return QuantileTransformer(output_distribution="normal")
#     elif feature in [
#         "D:A ratio (m/m)", "total solids conc. (mg/mL)", "solvent additive conc. (% v/v)",
#         "temperature of thermal annealing", "annealing time (min)",
#         "HTL energy level (eV)", "ETL energy level (eV)", "HTL thickness (nm)", "ETL thickness (nm)"
#     ]:
#         return MinMaxScaler()
#     else:
#         raise ValueError(f"Feature {feature} not recognized.")


power_features: list[str] = [
    "Donor PDI", "Donor Mn (kDa)", "Donor Mw (kDa)",
    "HOMO_D (eV)", "LUMO_D (eV)", "Eg_D (eV)", "Ehl_D (eV)",
    "HOMO_A (eV)", "LUMO_A (eV)", "Eg_A (eV)", "Ehl_A (eV)",
    "active layer thickness (nm)",
    "log hole mobility blend (cm^2 V^-1 s^-1)", "log electron mobility blend (cm^2 V^-1 s^-1)",
    "log hole:electron mobility ratio",
]
minmax_features: list[str] = [
    "D:A ratio (m/m)", "total solids conc. (mg/mL)", "solvent additive conc. (% v/v)",
    "temperature of thermal annealing", "annealing time (min)",
    "HTL energy level (eV)", "ETL energy level (eV)", "HTL thickness (nm)", "ETL thickness (nm)",
]


def generate_feature_pipeline(transform_name: str, transform: TransformerMixin) -> Pipeline:
    """
    Inserts the transformation into a Pipeline
    """
    # new_pipe: Pipeline = Pipeline(steps=[("minmax", MinMaxScaler()), (transform_name, transform)])
    new_pipe: Pipeline = Pipeline(steps=[(transform_name, transform), ("minmax", MinMaxScaler()), ])
    return new_pipe


def get_feature_pipelines(unrolled_features: list[str],
                          representation: str,
                          numeric_features: list[str]
                          ) -> list[tuple[str, Pipeline, list[str]]]:
    # Establish preprocessing and training pipeline
    solvent_feats: list[str] = [feat for feat in unrolled_features if feat.startswith("solvent")]
    new_struct_feats: list[str] = [feat for feat in unrolled_features if not feat.startswith("solvent")]
    # power_numeric_feats: list[str] = [feat for feat in numeric_features if feat in power_features]
    # minmax_numeric_feats: list[str] = [feat for feat in numeric_features if feat in minmax_features]

    transformers: list[tuple] = []
    if new_struct_feats:
        transformers.append(  # TODO: Update!!
            (representation_scaling_factory[representation]["type"],
             representation_scaling_factory[representation]["callable"](), new_struct_feats)
        )
    if solvent_feats:
        transformers.append(
            ("solvent power transform", generate_feature_pipeline("power", PowerTransformer()), solvent_feats))
    if numeric_features:
        transformers.append(
            ("processing power transform", PowerTransformer(), numeric_features))
    # if minmax_numeric_feats:
    #     transformers.append(("minmax", MinMaxScaler(), minmax_numeric_feats))
    return transformers
