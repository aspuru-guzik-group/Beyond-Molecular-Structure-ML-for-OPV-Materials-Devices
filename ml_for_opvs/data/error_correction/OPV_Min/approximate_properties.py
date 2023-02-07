import copy
from ctypes import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from scipy.stats import norm
from math import nan, isnan
from collections import Counter

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

DUPLICATE_DONORS = pkg_resources.resource_filename("ml_for_opvs", "data/preprocess/OPV_Min/duplicate labels - donors.csv")

DUPLICATE_ACCEPTORS = pkg_resources.resource_filename("ml_for_opvs", "data/preprocess/OPV_Min/duplicate labels - acceptors.csv")

def approximate_value(data_path: str, duplicate_donors: str, duplicate_acceptors: str):
    """Approximates material properties (HOMO, LUMO, Eg) with Gaussian distribution. New columns are added back to the original DataFrame with approximated values.

    Args:
        data_path (str): Filepath to .csv
    """
    data: pd.DataFrame = pd.read_csv(data_path)
    duplicate_donors: pd.DataFrame = pd.read_csv(duplicate_donors)
    duplicate_acceptors: pd.DataFrame = pd.read_csv(duplicate_acceptors)
    old_data: pd.DataFrame = copy.copy(data)
    # curate dictionary with unique donor/acceptors and their corresponding HOMO/LUMO/Eg values
    # TODO: save homo, lumo, donor, acceptor in 4 different json files. With the replacement values we want.
    material_D_properties: list = ["HOMO_D_eV", "LUMO_D_eV", "Eg_D_eV"]
    material_A_properties: list = ["HOMO_A_eV", "LUMO_A_eV", "Eg_A_eV"]
    d_materials: list = []
    a_materials: list = []
    labels_D: list = data["Donor"]
    labels_A: list = data["Acceptor"]
    for material in material_D_properties:
        d_material: dict = {}
        for label in labels_D:
            filtered_data: list = list(data[material][data["Donor"]==label].dropna())
            d_material[label] = filtered_data
        d_materials.append(d_material)
    
    for material in material_A_properties:
        a_material: dict = {}
        for label in labels_A:
            filtered_data: list = list(data[material][data["Acceptor"]==label].dropna())
            a_material[label] = filtered_data
        a_materials.append(a_material)
    
    print(d_materials)
    print(len(d_materials[0]))
    print(len(a_materials[0]))

    # Combine dictionary values if they match the duplicate labels.
    for index, row in duplicate_donors.iterrows():
        row_length: int = len(row)
        combined_homo: list = []
        combined_lumo: list = []
        combined_eg: list = []
        # collect all values from each duplicate label
        for i in range(row_length):
            if str(row[i]) != "nan":
                if row[i] in d_materials[0]:
                    combined_homo.extend(d_materials[0][row[i]])
                if row[i] in d_materials[1]:
                    combined_lumo.extend(d_materials[1][row[i]])
                if row[i] in d_materials[2]:
                    combined_eg.extend(d_materials[2][row[i]])
        
        # assign same collection of values for each label
        for i in range(row_length):
            d_materials[0][row[i]] = combined_homo
            d_materials[1][row[i]] = combined_lumo
            d_materials[2][row[i]] = combined_eg
    
    print("finished_donors")

    for index, row in duplicate_acceptors.iterrows():
        row_length: int = len(row)
        combined_a_homo: list = []
        combined_a_lumo: list = []
        combined_a_eg: list = []
        # collect all values from each duplicate label
        for i in range(row_length):
            if str(row[i]) != "nan":
                if row[i] in a_materials[0]:
                    combined_a_homo.extend(a_materials[0][row[i]])
                if row[i] in a_materials[1]:
                    combined_a_lumo.extend(a_materials[1][row[i]])
                if row[i] in a_materials[2]:
                    combined_a_eg.extend(a_materials[2][row[i]])
                print(len(combined_a_homo))
        
        # assign same collection of values for each label
        for i in range(row_length):
            a_materials[0][row[i]] = combined_a_homo
            a_materials[1][row[i]] = combined_a_lumo
            a_materials[2][row[i]] = combined_a_eg


    property_dicts: list = d_materials
    property_dicts.extend(a_materials)
    # print(f"{property_dicts=}")

    # fit gaussian to each unique donor/acceptor with multiple data points
    for unique_dict in property_dicts:
        for mol_key in unique_dict:
            # get rid of NaN
            x_data: list = list(
                filter(lambda x: isnan(x) == False, unique_dict[mol_key])
            )
            # NOTE: if a value occurs more than 10 (arbitrary) times, only keep 1 for Gaussian Fit
            clean_x_data = []
            count = Counter(x_data)
            for count_key in count:
                if count[count_key] > 10:
                    clean_x_data.append(count_key)
                else:
                    for i in range(count[count_key]):
                        clean_x_data.append(count_key)
            # print(x_data, "CLEAN: ", clean_x_data)
            # convert to np.array
            if len(clean_x_data) == 0:
                continue
            else:
                x_data = np.array(clean_x_data)
                mu, std = norm.fit(x_data)
                # # Plot the histogram of homo/lumo data.
                # plt.hist(x_data, bins=50, density=True, alpha=0.6, color='g')
                # # Plot the PDF.
                # xmin, xmax = plt.xlim()
                # x = np.linspace(xmin, xmax, 100)
                # p = norm.pdf(x, mu, std)
                # plt.plot(x, p, 'k', linewidth=2)
                # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
                # plt.title(title)
                # plt.show()
                unique_dict[mol_key] = [mu, std]

    # update DataFrame with Gaussian Fit homo/lumo energies
    for index, row in data.iterrows():
        # if data.at[index, "Donor"] == "J71" and data.at[index, "Acceptor"] == "ITC6-IC":
        #     print(row)
        try:
            data.at[index, "HOMO_D_eV"] = d_materials[0][row["Donor"]][0]
        except:
            print("No value for this molecule", row["Donor"], "HOMO_D_eV")
        try:
            data.at[index, "LUMO_D_eV"] = d_materials[1][row["Donor"]][0]
        except:
            print("No value for this molecule", row["Donor"], "LUMO_D_eV")
        try:
            data.at[index, "HOMO_A_eV"] = a_materials[0][row["Acceptor"]][0]
        except:
            print("No value for this molecule", row["Acceptor"], "HOMO_A_eV")
        try:
            data.at[index, "LUMO_A_eV"] = a_materials[1][row["Acceptor"]][0]
        except:
            print("No value for this molecule", row["Acceptor"], "LUMO_A_eV")
        try:
            data.at[index, "Eg_D_eV"] = d_materials[2][row["Donor"]][0]
        except:
            print("No value for this molecule", row["Donor"], "Eg_D_eV")
        try:
            data.at[index, "Eg_A_eV"] = a_materials[2][row["Acceptor"]][0]
        except:
            print("No value for this molecule", row["Acceptor"], "Eg_A_eV")
    # print(d_unique_homo_dict, a_unique_homo_dict, d_unique_lumo_dict, a_unique_lumo_dict)
    # print(data)
    data.to_csv(data_path, index=False)


def find_missing_homo_lumo(data_path: str) -> list:
    """

    Args:
        data_path (str): Filepath to .csv

    Returns:
         missing_d: list of donors with missing homo or lumo
         missing_a: list of acceptors with missing homo or lumo
    """
    data: pd.DataFrame = pd.read_csv(data_path)
    bool_homo_d_series = pd.isnull(data["HOMO_D_eV"])
    bool_lumo_d_series = pd.isnull(data["LUMO_D_eV"])
    bool_homo_a_series = pd.isnull(data["HOMO_A_eV"])
    bool_lumo_a_series = pd.isnull(data["LUMO_A_eV"])
    missing_homo_d = data[bool_homo_d_series]["Donor"]
    missing_lumo_d = data[bool_lumo_d_series]["Donor"]
    missing_homo_a = data[bool_homo_a_series]["Acceptor"]
    missing_lumo_a = data[bool_lumo_a_series]["Acceptor"]
    missing_d = set(missing_homo_d) | set(missing_lumo_d)
    missing_a = set(missing_homo_a) | set(missing_lumo_a)
    print("missing_homo_d", missing_homo_d, "missing_lumo_d", missing_lumo_d)
    print("missing_homo_a", missing_homo_a, "missing_lumo_a", missing_lumo_a)
    print(missing_d)
    print(missing_a)
    print(len(missing_d), len(missing_a))
    return missing_d, missing_a

def calculate_homo_lumo_gap(data_path: str):
    """
    Args:
        data_path (str): Filepath to .csv

    Returns:
        Updates pd.Dataframe with calculated Ehl_D_eV and Ehl_A_eV and exports to data_path .csv
    """
    data: pd.DataFrame = pd.read_csv(data_path)
    for index, row in data.iterrows():
        data.at[index, "Ehl_D_eV"] = data.at[index, "HOMO_D_eV"] - data.at[index, "LUMO_D_eV"]
        data.at[index, "Ehl_A_eV"] = data.at[index, "HOMO_A_eV"] - data.at[index, "LUMO_A_eV"]
    
    data.to_csv(data_path, index=False)


if __name__ == "__main__":
    approximate_value(MASTER_ML_DATA, DUPLICATE_DONORS, DUPLICATE_ACCEPTORS)
    # find_missing_homo_lumo(MASTER_ML_DATA)
    pass
