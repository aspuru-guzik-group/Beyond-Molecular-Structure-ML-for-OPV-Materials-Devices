import json
from pathlib import Path
from typing import Optional

import mordred
import mordred.descriptors
import numpy as np
import pandas as pd
import selfies
from mordred import Calculator
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from scipy.stats import norm

from code_python import DATASETS


class FeatureCleaner:
    def __init__(self,
                 dataset: pd.DataFrame,
                 duplicated_labels: Optional[pd.DataFrame],
                 solvent_properties: pd.DataFrame,
                 interlayer_properties: pd.DataFrame,
                 solvent_descriptors: list[str],
                 **kwargs
                 ) -> None:
        self.dataset: pd.DataFrame = dataset.drop(columns=["ref number from paper", "reference link"]
                                                  )
        self.duplicated_labels: dict[str, list[str]] = self.get_duplicate_labels(duplicated_labels)
        # self.solvent_properties: pd.DataFrame = self.select_solvent_properties(solvent_properties, **kwargs)
        # self.interlayer_properties: pd.DataFrame = self.select_interlayer_properties(interlayer_properties, **kwargs)
        self.solvent_properties: pd.DataFrame = self.select_descriptors(solvent_properties, selected_descriptors=solvent_descriptors)
        self.interlayer_properties: pd.DataFrame = self.select_descriptors(interlayer_properties, selected_descriptors=["Energy Level"])

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
            # for material_property in self.solvent_properties.columns:
            #     self.dataset[f"{material} {material_property}"] = self.assign_solvent_descriptors(material)
            self.dataset[f"{material} descriptors"] = self.get_solvent_descriptors(material)

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
    def select_descriptors(full_properties: pd.DataFrame, selected_descriptors: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Down-select interlayer properties is selected_properties kwarg is provided.

        Args:
            full_properties: solvent properties DataFrame
            selected_descriptors: properties to include

        Returns:
            DataFrame of down-selected interlayer properties
        """
        filtered_properties: pd.DataFrame = full_properties[selected_descriptors] if selected_descriptors else full_properties
        return filtered_properties

    # @staticmethod
    # def select_interlayer_properties(full_properties: pd.DataFrame,
    #                                  selected_solvent_properties: Optional[list[str]] = None,
    #                                  selected_interlayer_properties: Optional[list[str]] = None,
    #                                  ) -> pd.DataFrame:
    #     """
    #     Down-select interlayer properties is selected_properties kwarg is provided.
    #
    #     Args:
    #         full_properties: solvent properties DataFrame
    #         selected_solvent_properties: unused in this method
    #         selected_interlayer_properties: interlayer properties to include
    #
    #     Returns:
    #         DataFrame of down-selected interlayer properties
    #     """
    #     if not selected_interlayer_properties:
    #         filtered_properties: pd.DataFrame = full_properties
    #     else:
    #         filtered_properties: pd.DataFrame = full_properties[selected_interlayer_properties]
    #     return filtered_properties
    #
    # @staticmethod
    # def select_solvent_properties(full_properties: pd.DataFrame,
    #                               selected_solvent_properties: Optional[list[str]] = None,
    #                               selected_interlayer_properties: Optional[list[str]] = None,
    #                               ) -> pd.DataFrame:
    #     """
    #     Down-select solvent properties is selected_properties kwarg is provided.
    #
    #     Args:
    #         full_properties: solvent properties DataFrame
    #         selected_solvent_properties: properties to include
    #         selected_interlayer_properties: unused in this method
    #
    #     Returns:
    #         DataFrame of down-selected solvent properties
    #     """
    #     if not selected_solvent_properties:
    #         filtered_properties: pd.DataFrame = full_properties
    #     else:
    #         filtered_properties: pd.DataFrame = full_properties[selected_solvent_properties]
    #     return filtered_properties


class StructureCleaner:
    # string_representations: list[str] = [
    #     "SMILES",
    #     "SMILES w/o R group replacement",
    #     "SMILES w/o R group",
    #     "Big SMILES",
    #     "SELFIES",
    # ]

    def __init__(self, dataset: pd.DataFrame, donor_structures: pd.DataFrame, acceptor_structures: pd.DataFrame) -> None:
        self.dataset: pd.DataFrame = dataset
        donor_structures["SMILES"] = self.canonicalize(donor_structures["SMILES"])
        acceptor_structures["SMILES"] = self.canonicalize(acceptor_structures["SMILES"])
        self.material_smiles: dict[str, pd.DataFrame] = {"Donor": donor_structures.set_index("Donor"),
                                                         "Acceptor": acceptor_structures.set_index("Acceptor")
                                                         # "Donor": donor_structures["SMILES"].set_index("Donor"),
                                                         # "Acceptor": acceptor_structures["SMILES"].set_index("Acceptor")
                                                         }

    def main(self) -> pd.DataFrame:
        for material in ["Donor", "Acceptor"]:
            self.dataset[f"{material} SMILES"] = self.assign_smiles(material, self.material_smiles[material])
            # print(self.dataset[material][self.dataset[f"{material} SMILES"].isna()].unique())
            self.dataset[f"{material} SELFIES"] = self.assign_selfies(self.dataset[f"{material} SMILES"])
            self.dataset[f"{material} BigSMILES"] = self.assign_bigsmiles(material, self.dataset[f"{material} SMILES"])
            self.dataset[f"{material} Mol"] = self.assign_mol(self.dataset[f"{material} SMILES"])
            self.assign_fingerprints(material, radius=5, nbits=512)

        # # TODO: Generate mordred descriptors
        # self.mordred_descriptors: pd.DataFrame = self.generate_mordred_descriptors()
        # self.mordred_descriptors_used: pd.Series = pd.Series(self.mordred_descriptors.columns.tolist())
        # for material in ["Donor", "Acceptor"]:
        #     self.dataset[f"{material} mordred"] = self.assign_mordred(self.dataset[f"{material} Mol"])

        # TODO: Add tokenized string representations??? SMILES, SELFIES, BigSMILES

        # # # Clean up structural features and generate structural representations
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
        # sanity_check_bigsmiles(CLEAN_DONOR_CSV)
        # opv_smiles_to_selfies(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)
        # print("Finished Step 3")

        # Step 4
        # pairings = DAPairs(OPV_DATA, CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV, SOLVENT_DATA)
        # pairings.create_master_csv(MASTER_ML_DATA)
        # pairings.create_master_csv(MASTER_ML_DATA_PLOT)
        return self.dataset

    def assign_bigsmiles(self, material: str, smiles_series: pd.Series) -> pd.Series:
        """
        If material is donor, assign BigSMILES. If material is acceptor, assign SMILES.

        Args:
            material: Donor or Acceptor
            smiles_series: Series of SMILES strings

        Returns:
            Series of BigSMILES strings
        """
        if material == "Donor":
            bigsmiles_series: pd.Series = smiles_series.map(lambda smiles: self.convert_to_big_smiles(smiles))
        elif material == "Acceptor":
            bigsmiles_series: pd.Series = smiles_series
        print(f"Done assigning {material} BigSMILES.")
        return bigsmiles_series

    def assign_fingerprints(self, material: str, radius: int = 3, nbits: int = 1024) -> None:
        """
        Assigns ECFP fingerprints to the dataset.
        """
        self.dataset[f"{material} ECFP{2*radius}_{nbits}"] = self.dataset[f"{material} Mol"].map(lambda mol: self.generate_fingerprint(mol, radius, nbits))
        print(f"Done assigning {material} ECFP{2*radius} fingerprints with {nbits} bits.")

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

    def assign_mordred(self, labels: pd.Series) -> pd.Series:
        """
        Assigns Mordred descriptors to the dataset.
        """
        # BUG: This won't work..?
        mordred_series: pd.Series = labels.map(lambda mol: self.generate_mordred_descriptors(mol))
        print("Done assigning Mordred descriptors.")
        return mordred_series

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
        # BUG: Mapping returns nans! What's wrong?
        # print(self.dataset[self.dataset.duplicated()])
        # print(mapping[mapping.duplicated(subset="SMILES")])
        mapping: dict[str, str] = mapping["SMILES"].to_dict()
        # print(mapping["P2F-Ehp"])
        # mapped_smiles: pd.Series = self.dataset[material].map(mapping["SMILES"])
        mapped_smiles: pd.Series = self.dataset[material].map(mapping)
        print(f"Done assigning {material} SMILES.")
        return mapped_smiles

    @staticmethod
    def canonicalize(smiles_column: pd.Series) -> list[str]:
        """Canonicalize SMILES strings."""
        # TODO: This should work with either apply or map.
        return [Chem.CanonSmiles(smiles) for smiles in smiles_column]

    @staticmethod
    def convert_to_big_smiles(smiles: str) -> str:
        """Convert SMILES string to BigSMILES."""
        # BUG: How to do since only partially automated?
        return smiles

    @staticmethod
    def generate_fingerprint(mol: Mol, radius: int = 3, nbits: int = 1024) -> np.array:
        """
        Generate ECFP fingerprint.

        Args:
            mol: RDKit Mol object
            radius: Fingerprint radius
            nbits: Number of bits in fingerprint

        Returns:
            ECFP fingerprint as numpy array
        """
        fingerprint: np.array = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits))
        return fingerprint

    def generate_mordred_descriptors(self) -> pd.DataFrame:
        """
        Generate Mordred descriptors from "Donor Mol" and "Acceptor Mol" columns in the dataset.
        Remove all columns with nan values, and remove all columns with zero variance.

        Returns:
            DataFrame of filtered Mordred descriptors
        """
        donor_mols: pd.Series = self.material_smiles["Donor"]["SMILES"].map(lambda smiles: Chem.MolFromSmiles(smiles))
        acceptor_mols: pd.Series = self.material_smiles["Acceptor"]["SMILES"].map(lambda smiles: Chem.MolFromSmiles(smiles))
        all_mols: pd.Series = pd.concat([donor_mols, acceptor_mols])

        # BUG: Get numpy "RuntimeWarning: overflow encountered in reduce"
        # Generate Mordred descriptors
        print("Generating mordred descriptors...")
        calc: Calculator = Calculator(mordred.descriptors, ignore_3D=True)
        descriptors: pd.Series = all_mols.map(lambda mol: calc(mol))
        mordred_descriptors: pd.DataFrame = pd.DataFrame(descriptors.tolist(), index=all_mols.index)
        # Remove any columns with nan values
        mordred_descriptors.dropna(axis=1, how='any', inplace=True)
        # Remove any columns with zero variance
        mordred_descriptors = mordred_descriptors.loc[:, mordred_descriptors.var() != 0]
        print("Done generating Mordred descriptors.")
        return mordred_descriptors

    def get_mordred_descriptors(self, label: str) -> np.ndarray:
        """
        Get Mordred descriptors for a given label.

        Args:
            label: Donor or Acceptor

        Returns:
            Mordred descriptors as numpy array
        """
        descriptors: np.ndarray = self.mordred_descriptors.loc[label].to_numpy(dtype=float, copy=True)
        return descriptors


def assign_datatypes(dataset: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    # BUG: ETL/HTL thickness assigned object! Should be float64.
    """
    Assigns dtypes to columns in the dataset.
    """
    dataset: pd.DataFrame = dataset.infer_objects()
    # print(dataset.dtypes)
    for feature, dtype in feature_types.items():
        dataset[feature] = dataset[feature].astype(dtype)
    # print(dataset.dtypes)
    return dataset


def get_readable_only(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Excludes non-human readable columns from the dataset.

    Args:
        dataset: DataFrame of dataset

    Returns:
        DataFrame of readable columns
    """
    df = dataset.select_dtypes(exclude=["object"])
    return df


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
    selected_properties_file = min_dir / "selected_properties.json"
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
                                           solvent_descriptors=selected_solvent_properties
                                           ).main()

    # Load cleaned donor and acceptor structures
    donor_structures_file = min_dir / "cleaned donors.csv"
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures_file = min_dir / "cleaned acceptors.csv"
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)

    # Add structural representations to the dataset
    dataset: pd.DataFrame = StructureCleaner(dataset, donor_structures=donor_structures, acceptor_structures=acceptor_structures).main()
    # mordred_used: pd.Series = StructureCleaner.mordred_descriptors_used

    # Get datatypes of categorical features
    feature_types_file = min_dir / "feature_types.json"
    with feature_types_file.open("r") as f:
        feature_types: dict = json.load(f)
    dataset: pd.DataFrame = assign_datatypes(dataset, feature_types)

    # Specify paths for saving
    dataset_csv = min_dir / "cleaned_dataset.csv"
    dataset_pkl = min_dir / "cleaned_dataset.pkl"
    mordred_csv = min_dir / "mordred_descriptors.csv"

    # Save the dataset
    dataset.to_pickle(dataset_pkl)
    readable: pd.DataFrame = get_readable_only(dataset)
    readable.to_csv(dataset_csv)

    # Save mordred descriptors
    mordred_used.to_csv(mordred_csv)

# ATTN: When done, check the following:
#  - Check that all labels don't have Tanimoto similarity = 1
#  - All labels have SMILES
#  - All rows have solvent descriptors
#  - All rows that have solvent additives have solvent additive descriptors
#  - All rows that have interlayer have interlayer descriptors
