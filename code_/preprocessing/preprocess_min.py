import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import selfies
from rdkit import Chem
from scipy.stats import norm

from preprocess_utils import canonicalize_column, generate_brics, generate_fingerprint, tokenizer_factory

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


class FeatureProcessor:
    def __init__(self,
                 dataset: pd.DataFrame,
                 duplicated_labels: Optional[pd.DataFrame],
                 solvent_properties: pd.DataFrame,
                 interlayer_properties: pd.DataFrame,
                 solvent_descriptors: list[str],
                 ) -> None:
        self.dataset: pd.DataFrame = dataset.drop(columns=["ref number from paper", "reference link"]
                                                  )
        self.duplicated_labels: dict[str, list[str]] = self.get_duplicate_labels(duplicated_labels)
        self.solvent_properties: pd.DataFrame = self.select_descriptors(solvent_properties,
                                                                        selected_descriptors=solvent_descriptors)
        self.solvent_tokens: pd.DataFrame = self.select_descriptors(solvent_properties, selected_descriptors=["Token"])
        self.interlayer_properties: pd.DataFrame = self.select_descriptors(interlayer_properties,
                                                                           selected_descriptors=["Energy Level"])

    def main(self) -> pd.DataFrame:
        # Replace duplicate labels in Donors and Acceptors
        for material in ["Donor", "Acceptor"]:
            self.dataset[material] = self.replace_duplicate_labels(self.dataset[material])

        # Clean up processing features

        # # Calculate hole:electron mobility ratio
        self.dataset["hole:electron mobility ratio"] = self.calculate_hole_electron_mobility_ratio()

        # # Calculate log values related to mobilities
        for mob_column in ["hole mobility blend (cm^2 V^-1 s^-1)", "electron mobility blend (cm^2 V^-1 s^-1)",
                           "hole:electron mobility ratio"]:
            self.dataset[f"log {mob_column}"] = self.get_log(self.dataset[mob_column])

        # # Add calculated PCE
        self.dataset["calculated PCE (%)"]: pd.Series = self.calculate_correct_pce()
        # # Convert FF to fraction rather than percentage
        self.dataset["FF (%)"] = self.dataset["FF (%)"] / 100

        # # Convert strings to floats for DA ratio
        self.dataset["D:A ratio (m/m)"] = self.convert_da_ratio(self.dataset["D:A ratio (m/m)"])

        for material, abbr in zip(["Acceptor", "Donor"], ["A", "D"]):
            for energy in [f"HOMO_{abbr} (eV)", f"LUMO_{abbr} (eV)", f"Eg_{abbr} (eV)"]:
                # # Replace HOMO, LUMO and Eg values with fitted values
                self.dataset[energy] = self.fit_energy_values(material, energy)

            # # Calculate HOMO-LUMO gap
            self.dataset[f"Ehl_{abbr} (eV)"] = self.calculate_homo_lumo_gap(abbr)

        # # Add solvent and additive properties
        for material in ["solvent", "solvent additive"]:
            self.dataset[f"{material} descriptors"] = self.get_solvent_descriptors(material)
            self.dataset[f"{material} token"] = self.get_solvent_tokens(material)

        # Add interlayer energy levels
        for interlayer in ["HTL", "ETL"]:
            self.dataset[f"{interlayer} energy level (eV)"] = self.assign_energy_level(interlayer)

        return self.dataset

    def assign_energy_level(self, material: str) -> pd.Series:
        """
        Creates a Series corresponding to the properties from the DataFrame.

        Args:
            material: Material for which to create Series

        Returns:
            pandas Series of the property
        """
        # Select the desired property column from the properties DataFrame
        property_col: pd.Series = self.interlayer_properties["Energy Level"]

        # Create a new Series using the index from the materials DataFrame
        new_property = pd.Series(index=self.dataset.index, dtype="float")

        # Iterate over the rows of the dataset and assign the property value to the Series
        for index, row in self.dataset.iterrows():
            material_label: str = row[material]
            if pd.isna(material_label):
                new_property[index] = np.nan
            else:
                property_value: float = property_col.loc[material_label]
                new_property[index] = property_value
        return new_property

    def get_solvent_descriptors(self, liquid: str) -> pd.Series:
        # TODO: How to handle values for no solvent additive?
        """
        Creates a Series of lists corresponding to the properties of the label from the DataFrame.

        Args:
            liquid: Material for which to create Series
        """
        descriptors: pd.Series = self.dataset[liquid].apply(lambda x: self.solvent_properties.loc[x].to_numpy())
        return descriptors

    def get_solvent_tokens(self, liquid: str) -> pd.Series:
        """
        Creates a Series of lists corresponding to the properties of the label from the DataFrame.

        Args:
            liquid: Material for which to create Series
        """
        tokens: pd.Series = self.dataset[liquid].map(self.solvent_tokens["Token"])
        return tokens

    def calculate_correct_pce(self) -> pd.Series:
        """
        Calculates the correct PCE from Voc, Jsc and FF values.

        Returns:
            pandas series of calculated PCE values
        """
        calc_pce: pd.Series = self.dataset["Voc (V)"] * self.dataset["Jsc (mA cm^-2)"] * (self.dataset["FF (%)"] / 100)
        return calc_pce

    def calculate_hole_electron_mobility_ratio(self) -> pd.Series:
        """
        Calculates the hole:electron mobility ratio.

        Returns:
            pandas series of hole:electron mobility ratios
        """
        mob_ratio: pd.Series = self.dataset["hole mobility blend (cm^2 V^-1 s^-1)"] / self.dataset[
            "electron mobility blend (cm^2 V^-1 s^-1)"]
        return mob_ratio

    def calculate_homo_lumo_gap(self, material: str) -> pd.Series:
        """
        Calculates the HOMO-LUMO gap.

        Args:
            material: Either donor (D) or acceptor (A)

        Returns:
            pandas series of HOMO-LUMO gaps
        """
        e_hl: pd.Series = self.dataset[f"LUMO_{material} (eV)"] - self.dataset[f"HOMO_{material} (eV)"]
        return e_hl

    @staticmethod
    def convert_da_ratio(ratio_text: pd.Series) -> pd.Series:
        """
        Converts a pandas Series of D:A ratios formatted as X:Y to floats.

        Args:
            ratio_text: pandas Series of D:A ratios

        Returns:
            pandas Series of D:A ratios as floats
        """
        # Split the strings into separate parts
        parts = ratio_text.str.split(':', expand=True)
        d: pd.Series = parts[0].astype(float)
        a: pd.Series = parts[1].astype(float)

        # Convert the parts to floats and calculate the ratio
        ratio: pd.Series = d / a

        # Return the new series
        ratio_float: pd.Series = pd.Series(ratio, index=ratio_text.index)
        return ratio_float

    def fit_energy_values(self, molecule_col: str, energy_col: str) -> pd.Series:
        fitted_values: dict[str, float] = {}
        for molecule in self.dataset[molecule_col].unique():
            avg, _ = self.fit_gaussian(molecule_col, energy_col, molecule)
            fitted_values[molecule] = avg

        return self.replace_fitted_energies(molecule_col, energy_col, fitted_values)

    def fit_gaussian(self, molecule_col: str, energy_col: str, molecule: str, n: int = 10) -> tuple[float, float]:
        """
        Calculates the average energy in the dataset for the molecule.
        If a specific value appears more than N times, replace it with only one occurrence.

        Args:
            molecule_col: Name of molecule column (Donor or Acceptor)
            energy_col: Name of energy value column (HOMO, LUMO, Eg)
            molecule: Label of the molecule or polymer
            n: Arbitrary cut-off for replacing with a single value.

        Returns:
            Mean and variance of the Gaussian fit
        """
        # Filter dataframe to only include rows with the desired molecule
        filtered_df = self.dataset[self.dataset[molecule_col] == molecule]

        # Remove duplicate energy values if they occur more than N times
        value_counts = filtered_df[energy_col].value_counts()
        duplicate_values = value_counts[value_counts > n].index
        filtered_df = filtered_df[~filtered_df[energy_col].isin(duplicate_values)]

        # Fit a Gaussian distribution to the remaining energy values
        mu, std = norm.fit(filtered_df[energy_col].dropna())

        # Return mean and variance
        return mu, std

    @staticmethod
    def get_duplicate_labels(duplicates_df: pd.DataFrame) -> dict[str, list[str]]:
        """
        Converts the duplicate labels DataFrame to a dictionary where
        the key is the label and the value is a list of duplicate labels.

        Args:
            duplicates_df: pandas DataFrame of duplicate labels

        Returns:
            Dictionary of duplicate labels
        """
        duplicates_dict: dict[str, list[str]] = duplicates_df.T.to_dict("list")
        for key, values in duplicates_dict.items():
            duplicates_dict[key] = [value for value in values if not pd.isnull(value)]
        return duplicates_dict

    def replace_duplicate_labels(self, molecules: pd.Series) -> pd.Series:
        """
        Replace the labels in molecules with the key in duplicates if applicable.

        Args:
            molecules: pandas Series of molecule labels

        Returns:
            pandas Series of molecule labels with duplicates replaced
        """
        # Replace duplicate labels with the key
        for key, values in self.duplicated_labels.items():
            molecules = molecules.replace(values, key)
        return molecules

    def replace_fitted_energies(self, molecule_col: str, energy_col: str, fitted_values: dict[str, float]) -> pd.Series:
        """
        Replaces energy values with those obtained from Gaussian fitting.

        Args:
            molecule_col: Name of molecule column (Donor or Acceptor)
            energy_col: Name of energy value column (e.g. HOMO, LUMO, Eg)
            fitted_values: Dictionary of mean values obtained from fitting

        Returns:
            pandas Series of energy values replaced by fitted values
        """
        self.dataset[energy_col] = self.dataset[energy_col].fillna(self.dataset[molecule_col].map(fitted_values))
        return self.dataset[molecule_col].replace(fitted_values)

    @staticmethod
    def select_descriptors(full_properties: pd.DataFrame,
                           selected_descriptors: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Down-select interlayer properties is selected_properties kwarg is provided.

        Args:
            full_properties: solvent properties DataFrame
            selected_descriptors: properties to include

        Returns:
            DataFrame of down-selected interlayer properties
        """
        filtered_properties: pd.DataFrame = full_properties[
            selected_descriptors] if selected_descriptors else full_properties
        return filtered_properties

    def get_log(self, col: pd.Series) -> pd.Series:
        """
        Calculate the log of a column.

        Args:
            col: Name of mobility column (e.g. "Mobility (cm2/Vs)")

        Returns:
            pandas Series of log(mobility)
        """
        log_col: pd.Series = np.log10(col)
        return log_col


class StructureProcessor:

    def __init__(self, dataset: pd.DataFrame, donor_structures: pd.DataFrame,
                 acceptor_structures: pd.DataFrame) -> None:
        self.dataset: pd.DataFrame = dataset
        donor_structures["SMILES"] = canonicalize_column(donor_structures["SMILES"])
        acceptor_structures["SMILES"] = canonicalize_column(acceptor_structures["SMILES"])
        self.material_smiles: dict[str, pd.DataFrame] = {"Donor":    donor_structures.set_index("Donor"),
                                                         "Acceptor": acceptor_structures.set_index("Acceptor")
                                                         }
        self.tokens: dict[str, dict[str, int]] = {}

    def main(self, fp_radii: list[int], fp_bits: list[int]) -> pd.DataFrame:
        for material in ["Donor", "Acceptor"]:
            self.dataset[f"{material} SMILES"] = self.assign_smiles(material, self.material_smiles[material])
            self.dataset[f"{material} SELFIES"] = self.assign_selfies(self.dataset[f"{material} SMILES"])
            # self.dataset[f"{material} BigSMILES"] = self.assign_bigsmiles(material, self.dataset[f"{material} SMILES"])
            self.dataset[f"{material} Mol"] = self.assign_mol(self.dataset[f"{material} SMILES"])
            for r, b in zip(fp_radii, fp_bits):
                self.assign_fingerprints(material, radius=r, nbits=b)
            self.dataset[f"{material} BRICS"] = self.assign_brics(self.dataset[f"{material} Mol"])

        for representation in ["SMILES", "SELFIES", "BRICS"]:
            self.tokenize(representation)

        return self.dataset

    # @staticmethod
    # def assign_bigsmiles(material: str, smiles_series: pd.Series) -> pd.Series:
    #     """
    #     If material is donor, assign BigSMILES. If material is acceptor, assign SMILES.
    #
    #     Args:
    #         material: Donor or Acceptor
    #         smiles_series: Series of SMILES strings
    #
    #     Returns:
    #         Series of BigSMILES strings
    #     """
    #     if material == "Donor":
    #         bigsmiles_series: pd.Series = smiles_series.apply(lambda smiles: convert_to_bigsmiles(smiles))
    #     elif material == "Acceptor":
    #         bigsmiles_series: pd.Series = smiles_series
    #     print(f"Done assigning {material} BigSMILES.")
    #     return bigsmiles_series

    @staticmethod
    def assign_brics(mol_series: pd.Series) -> pd.Series:
        """
        Assigns BRICS fragments to the dataset.

        Args:
            mol_series: Series of RDKit Mol objects

        Returns:
            Series of BRICS fragments
        """
        brics_series: pd.Series = mol_series.map(lambda mol: generate_brics(mol))
        print("Done assigning BRICS fragments.")
        return brics_series

    def assign_fingerprints(self, material: str, radius: int = 3, nbits: int = 1024) -> None:
        """
        Assigns ECFP fingerprints to the dataset.
        """
        self.dataset[f"{material} ECFP{2 * radius}_{nbits}"] = self.dataset[f"{material} Mol"].map(
            lambda mol: generate_fingerprint(mol, radius, nbits))
        print(f"Done assigning {material} ECFP{2 * radius} fingerprints with {nbits} bits.")

    @staticmethod
    def assign_mol(smiles_series: pd.Series) -> pd.Series:
        """
        Converts SMILES to RDKit Mol.

        Args:
            smiles_series: Series of SMILES strings

        Returns:
            Series of RDKit Mol objects
        """
        mol_series: pd.Series = smiles_series.map(lambda smiles: Chem.MolFromSmiles(smiles))
        print("Done assigning RDKit Mol objects.")
        return mol_series

    @staticmethod
    def assign_selfies(smiles_series: pd.Series) -> pd.Series:
        """
        Converts SMILES to SELFIES.

        Args:
            smiles_series: Series of SMILES strings

        Returns:
            Series of SELFIES strings
        """
        selfies_series: pd.Series = smiles_series.map(lambda smiles: selfies.encoder(smiles))
        print("Done assigning SELFIES.")
        return selfies_series

    def assign_smiles(self, material: str, mapping: pd.DataFrame) -> pd.Series:
        """
        Assigns SMILES to the dataset.
        """
        mapping: dict[str, str] = mapping["SMILES"].to_dict()
        mapped_smiles: pd.Series = self.dataset[material].map(mapping)
        print(f"Done assigning {material} SMILES.")
        return mapped_smiles

    def tokenize(self, representation: str) -> None:
        """
        Tokenizes the dataset.

        Args:
            representation: SMILES, SELFIES, or BRICS
        """
        all_representation: pd.Series = pd.concat(
            [self.dataset[f"{material} {representation}"] for material in ["Donor", "Acceptor"]],
            ignore_index=True)

        for material in ["Donor", "Acceptor"]:
            self.dataset[f"{material} {representation} token"], self.tokens[representation] = tokenizer_factory[
                representation](self.dataset[f"{material} {representation}"], all_representation)

        print(f"Done tokenizing {representation}.")


def assign_datatypes(dataset: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    """
    Assigns dtypes to columns in the dataset.
    """
    dataset: pd.DataFrame = dataset.infer_objects()
    for feature, dtype in feature_types.items():
        dataset[feature] = dataset[feature].astype(dtype)
    return dataset


def get_readable_only(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Excludes non-human readable columns from the dataset.

    Args:
        dataset: DataFrame of dataset

    Returns:
        DataFrame of readable columns
    """
    remove: list[str] = ["descriptors", "Mol", "ECFP", "BRICS", "token"]
    df = dataset.loc[:, ~dataset.columns.str.contains('|'.join(remove))]
    print(df.dtypes)
    return df


def pre_main(fp_radii: list[int], fp_bits: list[int], solv_props_as_nan: bool):
    min_dir: Path = DATASETS / "Min_2020_n558"
    min_raw_dir: Path = min_dir / "raw"

    # Whether to treated missing solvent properties as NaN
    if solv_props_as_nan:
        solv_prop_fname = "solvent properties_nan.csv"
        dataset_fname = "cleaned_dataset_nans.pkl"
    else:
        solv_prop_fname = "solvent properties.csv"
        dataset_fname = "cleaned_dataset.pkl"

    # Import raw dataset downloaded from Google Drive
    raw_dataset_file = min_raw_dir / "raw dataset.csv"
    raw_dataset: pd.DataFrame = pd.read_csv(raw_dataset_file, index_col="ref")

    # Get list of duplicated Donor and Acceptor labels
    duplicated_labels_file = min_raw_dir / "duplicate labels.csv"
    duplicated_labels: pd.DataFrame = pd.read_csv(duplicated_labels_file, index_col="Name0")

    # Get selected properties
    selected_properties_file = min_dir / "selected_properties.json"
    with selected_properties_file.open("r") as f:
        selected_properties = json.load(f)

    # Get solvent and solvent additive properties
    solvent_properties_file = min_raw_dir / solv_prop_fname
    solvent_properties: pd.DataFrame = pd.read_csv(solvent_properties_file, index_col="Name")
    selected_solvent_properties: list[str] = selected_properties["solvent"]

    # Get interlayer properties
    interlayer_properties_file = min_raw_dir / "interlayer properties.csv"
    interlayer_properties: pd.DataFrame = pd.read_csv(interlayer_properties_file, index_col="Name")
    selected_interlayer_properties: list[str] = selected_properties["interlayer"]

    # Clean features in the dataset
    dataset: pd.DataFrame = FeatureProcessor(raw_dataset,
                                             duplicated_labels,
                                             solvent_properties,
                                             interlayer_properties,
                                             solvent_descriptors=selected_solvent_properties
                                             ).main()

    # Load cleaned donor and acceptor structures
    donor_structures_file = min_dir / "cleaned donors.csv"
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures_file = min_dir / "cleaned acceptors.csv"
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)

    # Add structural representations to the dataset
    dataset: pd.DataFrame = StructureProcessor(dataset, donor_structures=donor_structures,
                                               acceptor_structures=acceptor_structures).main(fp_radii, fp_bits)

    # Get datatypes of categorical features
    feature_types_file = min_dir / "feature_types.json"
    with feature_types_file.open("r") as f:
        feature_types: dict = json.load(f)
    dataset: pd.DataFrame = assign_datatypes(dataset, feature_types)

    # Specify paths for saving
    dataset_csv = min_dir / "cleaned_dataset.csv"
    dataset_pkl = min_dir / dataset_fname

    # Save the dataset
    dataset.to_pickle(dataset_pkl)
    readable: pd.DataFrame = get_readable_only(dataset)
    readable.to_csv(dataset_csv)


if __name__ == "__main__":
    fp_radii: list[int] = [3, 4, 5, 6]
    fp_bits: list[int] = [512, 1024, 2048, 4096]
    for solv_props_as_nan in [True, False]:
        print(f"Running with solv_props_as_nan={solv_props_as_nan}")
        pre_main(fp_radii=fp_radii, fp_bits=fp_bits, solv_props_as_nan=solv_props_as_nan)
