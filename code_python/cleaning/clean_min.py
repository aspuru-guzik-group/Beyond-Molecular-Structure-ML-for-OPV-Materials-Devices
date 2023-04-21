import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from code_python import DATASETS


class FeatureCleaner:
    def __init__(self,
                 dataset: pd.DataFrame,
                 duplicated_labels: Optional[pd.DataFrame],
                 solvent_properties: pd.DataFrame,
                 interlayer_properties: pd.DataFrame,
                 **kwargs
                 ) -> None:
        self.dataset: pd.DataFrame = dataset.drop(columns=["ref number from paper", "reference link"]
                                                  )
        self.duplicated_labels: dict[str, list[str]] = self.get_duplicate_labels(duplicated_labels)
        self.solvent_properties: pd.DataFrame = self.select_solvent_properties(solvent_properties, **kwargs)
        self.interlayer_properties: pd.DataFrame = self.select_interlayer_properties(interlayer_properties, **kwargs)

    def main(self) -> pd.DataFrame:
        # Replace duplicate labels in Donors and Acceptors
        for material in ["Donor", "Acceptor"]:
            self.dataset[material] = self.replace_duplicate_labels(self.dataset[material])

        # Clean up processing features

        # # Calculate hole:electron mobility ratio
        self.dataset["hole:electron mobility ratio"] = self.calculate_hole_electron_mobility_ratio()

        # # Add calculated PCE
        self.dataset["calculated PCE (%)"]: pd.Series = self.calculate_correct_pce()

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
            for material_property in self.solvent_properties.columns:
                self.dataset[f"{material} {material_property}"] = self.assign_properties(self.solvent_properties,
                                                                                         material_property,
                                                                                         material)

        # Add interlayer energy levels
        for interlayer in ["HTL", "ETL"]:
            self.dataset[f"{interlayer} energy level (eV)"] = self.assign_properties(self.interlayer_properties,
                                                                                     "Energy Level",
                                                                                     interlayer)

        return self.dataset

    def assign_properties(self, properties: pd.DataFrame, material_property: str, material: str) -> pd.Series:
        """
        Creates a Series corresponding to the properties from the DataFrame.

        Args:
            properties: DataFrame containing material properties
            material_property: Property for which to create Series
            material: Material for which to create Series

        Returns:
            pandas Series of the property
        """
        # Select the desired property column from the properties DataFrame
        property_col: pd.Series = properties[material_property]

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
        mob_ratio: pd.Series = self.dataset["hole mobility blend (cm^2 V^-1 s^-1)"] / self.dataset["electron mobility blend (cm^2 V^-1 s^-1)"]
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
    def select_interlayer_properties(full_properties: pd.DataFrame,
                                     selected_solvent_properties: Optional[list[str]] = None,
                                     selected_interlayer_properties: Optional[list[str]] = None,
                                     ) -> pd.DataFrame:
        """
        Down-select interlayer properties is selected_properties kwarg is provided.

        Args:
            full_properties: solvent properties DataFrame
            selected_solvent_properties: unused in this method
            selected_interlayer_properties: interlayer properties to include

        Returns:
            DataFrame of down-selected interlayer properties
        """
        if not selected_interlayer_properties:
            filtered_properties: pd.DataFrame = full_properties
        else:
            filtered_properties: pd.DataFrame = full_properties[selected_interlayer_properties]
        return filtered_properties

    @staticmethod
    def select_solvent_properties(full_properties: pd.DataFrame,
                                  selected_solvent_properties: Optional[list[str]] = None,
                                  selected_interlayer_properties: Optional[list[str]] = None,
                                  ) -> pd.DataFrame:
        """
        Down-select solvent properties is selected_properties kwarg is provided.

        Args:
            full_properties: solvent properties DataFrame
            selected_solvent_properties: properties to include
            selected_interlayer_properties: unused in this method

        Returns:
            DataFrame of down-selected solvent properties
        """
        if not selected_solvent_properties:
            filtered_properties: pd.DataFrame = full_properties
        else:
            filtered_properties: pd.DataFrame = full_properties[selected_solvent_properties]
        return filtered_properties


class StructureCleaner:
    structure_errors: set[str] = {
        "not in drive, not in literature",
        "error",
        "wrong structure",
        "same name different structure",
        "not in literature",
    }

    string_representations: list[str] = [
        "SMILES",
        "SMILES w/o R group replacement",
        "SMILES w/o R group",
        "Big SMILES",
        "SELFIES",
    ]

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset: pd.DataFrame = dataset

    def main(self) -> pd.DataFrame:
        ### Clean up structural features and generate structural representations
        # ATTN: For both Donor and Acceptor
        # Step 1
        # donors = DonorClean(MASTER_DONOR_CSV, OPV_DONOR_DATA)
        # donors.clean_donor(CLEAN_DONOR_CSV)

        # # # # # Step 1b
        # donors.replace_r_with_arbitrary(CLEAN_DONOR_CSV)
        # donors.replace_r(CLEAN_DONOR_CSV)

        # # # # # # # Step 1d - canonSMILES to remove %10-%100
        # donors.canon_smi(CLEAN_DONOR_CSV)

        # # Step 2 - ERROR CORRECTION (fill in missing D/A)
        # unique_opvs = UniqueOPVs(opv_min=OPV_MIN, opv_clean=OPV_CLEAN)
        # # concatenate for donors
        # unique_opvs.concat_missing_and_clean(MISSING_SMI_DONOR, CLEAN_DONOR, "D")
        # donors.canon_smi(CLEAN_DONOR_CSV)

        # # concatenate for acceptors
        # unique_opvs.concat_missing_and_clean(MISSING_SMI_ACCEPTOR, CLEAN_ACCEPTOR, "A")
        # acceptors.canon_smi(CLEAN_ACCEPTOR_CSV)
        # print("Finished Step 2")

        # Step 3 - smiles_to_bigsmiles.py & smiles_to_selfies.py
        # smile_to_bigsmile(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV) # DO NOT RUN, BigSMILES was partially automated and manually done.
        sanity_check_bigsmiles(CLEAN_DONOR_CSV)
        opv_smiles_to_selfies(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)
        print("Finished Step 3")

        # Step 4
        pairings = DAPairs(OPV_DATA, CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV, SOLVENT_DATA)
        pairings.create_master_csv(MASTER_ML_DATA)
        # pairings.create_master_csv(MASTER_ML_DATA_PLOT)
        return self.dataset


def assign_datatypes(dataset: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    """
    Assigns dtypes to columns in the dataset.
    """
    dataset: pd.DataFrame = dataset.infer_objects()
    for feature, dtype in feature_types.items():
        dataset[feature] = dataset[feature].astype(dtype)
    return dataset


if __name__ == "__main__":
    min_dir: Path = DATASETS / "Min_2020_n558"
    min_raw_dir: Path = min_dir / "raw"

    # Import raw dataset downloaded from Google Drive
    raw_dataset_file = min_raw_dir / "raw dataset.csv"
    raw_dataset: pd.DataFrame = pd.read_csv(raw_dataset_file, index_col="ref")

    # Get list of duplicated Donor and Acceptor labels
    duplicated_labels_file = min_raw_dir / "duplicate labels.csv"
    duplicated_labels: pd.DataFrame = pd.read_csv(duplicated_labels_file, index_col="Name0")

    # Get selected properties
    selected_properties_file = min_dir / "selected properties.json"
    with selected_properties_file.open("r") as f:
        selected_properties = json.load(f)

    # Get solvent and solvent additive properties
    solvent_properties_file = min_raw_dir / "solvent properties.csv"
    solvent_properties: pd.DataFrame = pd.read_csv(solvent_properties_file, index_col="Name")
    selected_solvent_properties: list[str] = selected_properties["solvent"]

    # Get interlayer properties
    interlayer_properties_file = min_raw_dir / "interlayer properties.csv"
    interlayer_properties: pd.DataFrame = pd.read_csv(interlayer_properties_file, index_col="Name")
    selected_interlayer_properties: list[str] = selected_properties["interlayer"]

    # Clean features in the dataset
    dataset: pd.DataFrame = FeatureCleaner(raw_dataset,
                                           duplicated_labels,
                                           solvent_properties,
                                           interlayer_properties,
                                           selected_solvent_properties=selected_solvent_properties,
                                           selected_interlayer_properties=selected_interlayer_properties
                                           ).main()

    # Save dataset to csv without structural representations
    dataset_csv = min_dir / "cleaned dataset.csv"
    # dataset.to_csv(dataset_csv)

    # # Add structural representations to the dataset
    # dataset: pd.DataFrame = StructureCleaner(dataset).main()
    #
    # # Get datatypes of categorical features
    # feature_types_file = DATASETS / "Min_2020_n558" / "feature_types.csv"
    # with feature_types_file.open("r") as f:
    #     feature_types: dict = json.load(f)
    # dataset: pd.DataFrame = assign_datatypes(dataset, feature_types)
    #
    # # Save dataset to pickle with structural representations
    # dataset_pkl = DATASETS / "Min_2020_n558" / "cleaned_dataset.pkl"
    # dataset.to_pickle(dataset_pkl)
