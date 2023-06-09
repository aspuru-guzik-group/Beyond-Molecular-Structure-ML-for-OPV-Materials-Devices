import pkg_resources
import pandas as pd

DATA_DIR = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/input_representation/OPV_Min/master_ml_for_opvs_from_min_for_plotting.csv"
)

TRAIN_MASTER_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/input_representation/OPV_Min/hw_frag/train_frag_master.csv"
)

AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/input_representation/OPV_Min/augmentation/train_aug_master5.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/input_representation/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/input_representation/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "_ml_for_opvs", "data/input_representation/OPV_Min/fingerprint/opv_fingerprint.csv"
)

data = pd.read_csv(DATA_DIR)

data.drop("Unnamed: 0", inplace=True, axis=1)

data.to_csv(DATA_DIR, index=False)
