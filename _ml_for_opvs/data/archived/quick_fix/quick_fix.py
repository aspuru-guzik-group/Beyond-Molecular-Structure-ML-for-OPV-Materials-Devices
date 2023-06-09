from os import error
import pkg_resources
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import csv
import pandas as pd

MASTER_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/process/quick_fix/master_ml_for_opvs_from_min.csv"
)

DATA_566 = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/process/quick_fix/master_ml_for_opvs_from_min_copy.csv"
)


master_data = pd.read_csv(MASTER_DATA)
data_566 = pd.read_csv(DATA_566)

master_donors = []
master_acceptors = []

donors_566 = []
acceptors_566 = []

for index, row in master_data.iterrows():
    if master_data.at[index, "Donor"] not in master_donors:
        master_donors.append(master_data.at[index, "Donor"])
    if master_data.at[index, "Acceptor"] not in master_acceptors:
        master_acceptors.append(master_data.at[index, "Acceptor"])

for index, row in data_566.iterrows():
    if data_566.at[index, "Donor"] not in donors_566:
        donors_566.append(data_566.at[index, "Donor"])
    if data_566.at[index, "Acceptor"] not in acceptors_566:
        acceptors_566.append(data_566.at[index, "Acceptor"])

print(list(set(donors_566) - set(master_donors)))
print(list(set(acceptors_566) - set(master_acceptors)))
