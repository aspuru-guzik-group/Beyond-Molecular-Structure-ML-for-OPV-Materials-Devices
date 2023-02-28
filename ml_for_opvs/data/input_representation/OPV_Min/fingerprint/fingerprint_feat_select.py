import pandas as pd
import numpy as np
import pkg_resources
from cmath import nan

# OPV data after pre-processing
FP_DATA = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/input_representation/OPV_Min/fingerprint/master_fingerprint.csv",
)

DATA_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/fingerprint/"
)

# create sets of groups of selected features

output = {
    "PCE_percent",
    "calc_PCE_percent",
    "Voc_V",
    "Jsc_mA_cm_pow_neg2",
    "FF_percent",
}

molecules = {"Donor", "Acceptor", "DA_FP_radius_3_nbits_1024"}

properties = {
    "HOMO_D_eV",
    "LUMO_D_eV",
    "HOMO_A_eV",
    "LUMO_A_eV",
    "Eg_D_eV",
    "Ehl_D_eV",
    "Eg_A_eV",
    "Ehl_A_eV",
}

electrical = {
    "hole_mobility_blend_cm_2_V_neg1_s_neg1",
    "electron_mobility_blend_cm_2_V_neg1_s_neg1",
}

fabrication = {
    "D_A_ratio_m_m",
    "solvent",
    "total_solids_conc_mg_mL",
    "solvent_additive",
    "solvent_additive_conc_v_v_percent",
    "annealing_temperature",
}

fabrication_wo_solid = {
    "D_A_ratio_m_m",
    "solvent",
    "solvent_additive",
    "solvent_additive_conc_v_v_percent",
    "annealing_temperature",
}

device = {"active_layer_thickness_nm", "hole_contact_layer", "electron_contact_layer"}

device_wo_thickness = {"hole_contact_layer", "electron_contact_layer"}

solvent = {
    "solvent_BP",
    "solvent_MP",
    "solvent_density",
    "solvent_dielectric",
    "solvent_dipole",
    "solvent_log_pow",
    "solvent_disp",
    "solvent_h_bond",
    "solvent_polar",
}

solvent_additive = {
    "solvent_additive_BP",
    "solvent_additive_MP",
    "solvent_additive_density",
    "solvent_additive_dielectric",
    "solvent_additive_dipole",
    "solvent_additive_log_pow",
    "solvent_additive_disp",
    "solvent_additive_h_bond",
    "solvent_additive_polar",
}

# create a dictionary with feature group's name as the key and the feature list as the value
feature_dict = {
    "molecules_only": list(molecules | output),
    "molecules": list(molecules | properties | output),
    "fabrication": list(molecules | properties | fabrication | output),
    "device": list(molecules | properties | fabrication | device | output),
    "electrical": list(
        molecules | properties | fabrication | device | electrical | output
    ),
    "fabrication_wo_solid": list(
        molecules | properties | fabrication_wo_solid | output
    ),
    "device_wo_thickness": list(
        molecules | properties | fabrication_wo_solid | device_wo_thickness | output
    ),
    "full": list(
        molecules
        | properties
        | fabrication
        | device
        | electrical
        | output
        | solvent
        | solvent_additive
    ),
}


class FeatureSelection:
    """
    Class that contains functions to create new csv files with selected features for model training.
    The output csv files will have no null value.
    """

    def __init__(self, data):
        """
        Function that reads the csv data file into pandas dataframe for later data cleaning
        Args:
            file path of the original csv file
        Returns:
            pandas dataframe
        """
        self.data = pd.read_csv(data)

    def feat_select(self, feat_list):
        """
        Function that creates a csv files where (1) only selected features are included and (2) all null entries are excluded
        Args:
            the name of the group of feature
        Returns:
            csv file with only selected features and no empty entry
        """
        # create a list of all features' names
        feat_full = self.data.columns.tolist()
        # create a list of features that will not be included in training
        feat_to_drop = list(set(feat_full) - set(feature_dict[feat_list]))
        # create a dataframe where unselected features are dropped
        df_full = self.data.drop(columns=feat_to_drop)
        # delete all rows that have at least 1 empty entry
        df = df_full.dropna(how="any")
        # save the processed dataframe to a new csv file whose name includes the group of features that are included
        df.to_csv(
            DATA_PATH + "processed_fingerprint_{x}.csv".format(x=feat_list), index=False
        )


fs = FeatureSelection(FP_DATA)

fs.feat_select("molecules_only")
fs.feat_select("molecules")
fs.feat_select("fabrication_wo_solid")
fs.feat_select("device_wo_thickness")
fs.feat_select("full")
fs.feat_select("fabrication")
fs.feat_select("device")
fs.feat_select("electrical")
