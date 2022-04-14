import pkg_resources
import pandas as pd

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

TRAIN_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/hw_frag/train_frag_master.csv"
)

AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/augmentation/train_aug_master5.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)

data = pd.read_csv(DATA_DIR)

data = data.drop_duplicates(
    subset=["Donor", "Acceptor"], keep="first", ignore_index=True
)
print(len(data))

data.to_csv(DATA_DIR, index=False)
