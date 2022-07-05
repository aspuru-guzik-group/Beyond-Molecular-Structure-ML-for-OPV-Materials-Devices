import pandas as pd
import numpy as np
import pkg_resources
from cmath import nan

# OPV data after pre-processing
AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/aug_SMILES/train_aug_master5.csv")


# create lists of groups of selected features
none = ['Unnamed: 0', 'Donor', 'Acceptor', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'DA_pair_aug', 'DA_pair_tokenized_aug']
electronic = ['Unnamed: 0', 'Donor', 'Acceptor', 'HOMO_D (eV)', 'LUMO_D (eV)', 'HOMO_A (eV)', 'LUMO_A (eV)', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'DA_pair_aug', 'DA_pair_tokenized_aug']
electronic_only = ['Unnamed: 0', 'HOMO_D (eV)', 'LUMO_D (eV)', 'HOMO_A (eV)', 'LUMO_A (eV)', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)']
device = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'solvent', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'active layer thickness (nm)', 'annealing temperature', 'hole contact layer', 'electron contact layer', 'hole mobility blend (cm^2 V^-1 s^-1)', 'electron mobility blend (cm^2 V^-1 s^-1)', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'DA_pair_aug', 'DA_pair_tokenized_aug']
device_solvent = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'active layer thickness (nm)', 'annealing temperature', 'hole contact layer', 'electron contact layer', 'hole mobility blend (cm^2 V^-1 s^-1)', 'electron mobility blend (cm^2 V^-1 s^-1)', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'BP', 'MP', 'Density', 'Dielectric', 'Dipole', 'log Pow', 'Hansen Disp', 'Hansen H-Bond', 'Hansen Polar', 'DA_pair_aug', 'DA_pair_tokenized_aug']
fabrication = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'solvent', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'active layer thickness (nm)', 'annealing temperature', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'DA_pair_aug', 'DA_pair_tokenized_aug']
fabrication_solvent = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'active layer thickness (nm)', 'annealing temperature', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'BP', 'MP', 'Density', 'Dielectric', 'Dipole', 'log Pow', 'Hansen Disp', 'Hansen H-Bond', 'Hansen Polar', 'DA_pair_aug', 'DA_pair_tokenized_aug']
device_solvent_only = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'active layer thickness (nm)', 'annealing temperature', 'hole contact layer', 'electron contact layer', 'hole mobility blend (cm^2 V^-1 s^-1)', 'electron mobility blend (cm^2 V^-1 s^-1)', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'BP', 'MP', 'Density', 'Dielectric', 'Dipole', 'log Pow', 'Hansen Disp', 'Hansen H-Bond', 'Hansen Polar']
fabrication_solvent_only = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'active layer thickness (nm)', 'annealing temperature', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'BP', 'MP', 'Density', 'Dielectric', 'Dipole', 'log Pow', 'Hansen Disp', 'Hansen H-Bond', 'Hansen Polar']
fabrication_solvent_minus_active_layer = ['Unnamed: 0', 'Donor', 'Acceptor', 'D:A ratio (m/m)', 'total solids conc. (mg/mL)', 'solvent additive', 'solvent additive conc. (%v/v)', 'annealing temperature', 'PCE (%)', 'calc_PCE (%)', 'Voc (V)', 'Jsc (mA cm^-2)', 'FF (%)', 'BP', 'MP', 'Density', 'Dielectric', 'Dipole', 'log Pow', 'Hansen Disp', 'Hansen H-Bond', 'Hansen Polar', 'DA_pair_aug', 'DA_pair_tokenized_aug']

# create a dictionary with feature group's name as the key and the feature list as the value 
feature_dict = {'none' : none,'electronic': electronic,'electronic_only': electronic_only, 'device': device, 'device_solvent': device_solvent, 'fabrication': fabrication, 'fabrication_solvent': fabrication_solvent, 'device_solvent_only': device_solvent_only, 'fabrication_solvent_only': fabrication_solvent_only, 'fabrication_solvent_minus_active_layer': fabrication_solvent_minus_active_layer}

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
        feat_to_drop = list(set(feat_full ) - set(feature_dict[feat_list]))
        # create a dataframe where unselected features are dropped
        df_full = self.data.drop(columns = feat_to_drop)
        # delete all rows that have at least 1 empty entry
        df = df_full.dropna(how = 'any')
        # save the processed dataframe to a new csv file whose name includes the group of features that are included
        df.to_csv('processed_augment_{x}.csv'.format(x = feat_list), index = False)

fs = FeatureSelection(AUGMENT_SMILES_DATA)

fs.feat_select('electronic')

