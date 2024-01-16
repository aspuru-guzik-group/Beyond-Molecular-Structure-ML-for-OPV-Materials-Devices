from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
hutchison = DATASETS / "Hutchison_2023_n1001"


# Import csv version
dataset_csv = hutchison / "Hutchison_filtered_dataset_pipeline.csv"
hutch = pd.read_csv(dataset_csv)

# # Import pkl version
dataset_pkl = hutchison / "Hutchison_filtered_dataset_pipeline.pkl"

# Create Molecule and fingerprint objects for pickle file
hutch["Acceptor SMILES"] = hutch["Acceptor SMILES"].apply(lambda x: Chem.CanonSmiles(x))
hutch["Donor SMILES"] = hutch["Donor SMILES"].apply(lambda x: Chem.CanonSmiles(x))
hutch["Acceptor Mol"] = hutch["Acceptor SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
hutch["Donor Mol"] = hutch["Donor SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
hutch["Acceptor ECFP10_2048"] = hutch["Acceptor Mol"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)))
hutch["Donor ECFP10_2048"] = hutch["Donor Mol"].apply(lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)))

# Get unique donor and acceptor SMILES
donors = hutch[["Donor SMILES", "Donor"]].drop_duplicates(ignore_index=True)
donors["SMILES"] = donors["Donor SMILES"]
acceptors = hutch[["Acceptor SMILES", "Acceptor"]].drop_duplicates(ignore_index=True)
acceptors["SMILES"] = acceptors["Acceptor SMILES"]

# Get experimental material properties of donors and acceptors
prop_cols: list[str] = ["HOMO", "LUMO", "optbg", "wavelength_film", "wavelength_soln"]
mat_props_file = hutchison / "molecule_exp_data.csv"
mat_props = pd.read_csv(mat_props_file)
donor_props = mat_props[mat_props["AorD"] == "D"].set_index("Molecule")
acceptor_props = mat_props[mat_props["AorD"] == "A"].set_index("Molecule")
for col in prop_cols:
    donors[col] = donors["Donor"].map(donor_props[col])
    acceptors[col] = acceptors["Acceptor"].map(acceptor_props[col])

# Apply to dataset
donors = donors.set_index("Donor")
acceptors = acceptors.set_index("Acceptor")
hutch["HOMO_D (eV)"] = hutch["Donor"].map(donors["HOMO"])
hutch["HOMO_A (eV)"] = hutch["Acceptor"].map(acceptors["HOMO"])
hutch["LUMO_D (eV)"] = hutch["Donor"].map(donors["LUMO"])
hutch["LUMO_A (eV)"] = hutch["Acceptor"].map(acceptors["LUMO"])
hutch["Ehl_D (eV)"] = hutch["LUMO_D (eV)"] - hutch["HOMO_D (eV)"]
hutch["Ehl_A (eV)"] = hutch["LUMO_A (eV)"] - hutch["HOMO_A (eV"]
hutch["Eg_D (eV)"] = hutch["Donor"].map(donors["optbg"])
hutch["Eg_A (eV)"] = hutch["Acceptor"].map(acceptors["optbg"])
hutch["wavelength_film_D (nm)"] = hutch["Donor"].map(donors["wavelength_film"])
hutch["wavelength_film_A (nm)"] = hutch["Acceptor"].map(acceptors["wavelength_film"])
hutch["wavelength_soln_D (nm)"] = hutch["Donor"].map(donors["wavelength_soln"])
hutch["wavelength_soln_A (nm)"] = hutch["Acceptor"].map(acceptors["wavelength_soln"])

# Save to file
hutch.to_pickle(dataset_pkl)
hutch.to_csv(dataset_csv, index=False)
hutch_donors = hutchison / "donors.csv"
hutch_acceptors = hutchison / "acceptors.csv"
donors.to_csv(hutch_donors)
acceptors.to_csv(hutch_acceptors)
