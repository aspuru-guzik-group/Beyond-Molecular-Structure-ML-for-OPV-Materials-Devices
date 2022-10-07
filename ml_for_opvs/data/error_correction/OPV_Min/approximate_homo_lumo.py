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


def approximate_value(data_path: str):
    """Approximates value of HOMO/LUMO with Gaussian distribution. New columns are added back to the original DataFrame with approximated values.

    Args:
        data_path (str): Filepath to .csv
    """
    data: pd.DataFrame = pd.read_csv(data_path)
    old_data: pd.DataFrame = copy.copy(data)
    # curate dictionary with unique donor/acceptors and their corresponding HOMO/LUMO values
    # TODO: save homo, lumo, donor, acceptor in 4 different json files. With the replacement values we want.
    d_unique_homo_dict = {}
    d_unique_lumo_dict = {}
    a_unique_homo_dict = {}
    a_unique_lumo_dict = {}
    for index, row in data.iterrows():
        if row["Donor"] not in d_unique_homo_dict:
            d_unique_homo_dict[row["Donor"]] = [row["HOMO_D_eV"]]
        else:
            d_unique_homo_dict[row["Donor"]].append(row["HOMO_D_eV"])

        if row["Donor"] not in d_unique_lumo_dict:
            d_unique_lumo_dict[row["Donor"]] = [row["LUMO_D_eV"]]
        else:
            d_unique_lumo_dict[row["Donor"]].append(row["LUMO_D_eV"])

        if row["Acceptor"] not in a_unique_homo_dict:
            a_unique_homo_dict[row["Acceptor"]] = [row["HOMO_A_eV"]]
        else:
            a_unique_homo_dict[row["Acceptor"]].append(row["HOMO_A_eV"])

        if row["Acceptor"] not in a_unique_lumo_dict:
            a_unique_lumo_dict[row["Acceptor"]] = [row["LUMO_A_eV"]]
        else:
            a_unique_lumo_dict[row["Acceptor"]].append(row["LUMO_A_eV"])

    homo_lumo: list = [
        d_unique_homo_dict,
        d_unique_lumo_dict,
        a_unique_homo_dict,
        a_unique_lumo_dict,
    ]

    # fit gaussian to each unique donor/acceptor with multiple data points
    for unique_dict in homo_lumo:
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
            data.at[index, "HOMO_D_eV"] = d_unique_homo_dict[row["Donor"]][0]
        except:
            print("No value for this molecule", row["Donor"])
        try:
            data.at[index, "LUMO_D_eV"] = d_unique_lumo_dict[row["Donor"]][0]
        except:
            print("No value for this molecule", row["Donor"])
        try:
            data.at[index, "HOMO_A_eV"] = a_unique_homo_dict[row["Acceptor"]][0]
        except:
            print("No value for this molecule", row["Acceptor"])
        try:
            data.at[index, "LUMO_A_eV"] = a_unique_lumo_dict[row["Acceptor"]][0]
        except:
            print("No value for this molecule", row["Acceptor"])
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


if __name__ == "__main__":
    # approximate_value(MASTER_ML_DATA)
    find_missing_homo_lumo(MASTER_ML_DATA)
    pass
