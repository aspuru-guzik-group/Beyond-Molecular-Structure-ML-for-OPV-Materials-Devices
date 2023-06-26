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

# Save to pkl
hutch.to_pickle(dataset_pkl)

# Get unique donor and acceptor SMILES
donors = hutch[["Donor SMILES", "Donor"]].drop_duplicates(ignore_index=True)
donors["SMILES"] = donors["Donor SMILES"]
acceptors = hutch[["Acceptor SMILES", "Acceptor"]].drop_duplicates(ignore_index=True)
acceptors["SMILES"] = acceptors["Acceptor SMILES"]

hutch_donors = hutchison / "donors.csv"
hutch_acceptors = hutchison / "acceptors.csv"
donors[["Donor", "SMILES"]].to_csv(hutch_donors, index=False)
acceptors[["Acceptor", "SMILES"]].to_csv(hutch_acceptors, index=False)
