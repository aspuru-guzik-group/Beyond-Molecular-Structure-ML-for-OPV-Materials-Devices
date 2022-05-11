from ml_for_opvs.data.preprocess.OPV_Min.clean_donors_acceptors import (
    MASTER_DONOR_CSV,
    OPV_DONOR_DATA,
    CLEAN_DONOR_CSV,
    MASTER_ACCEPTOR_CSV,
    OPV_ACCEPTOR_DATA,
    CLEAN_ACCEPTOR_CSV,
    OPV_DATA,
    MASTER_ML_DATA,
    MASTER_ML_DATA_PLOT,
    DAPairs,
    DonorClean,
    AcceptorClean,
)

from ml_for_opvs.data.error_correction.OPV_Min.unique_opvs import (
    UniqueOPVs,
    OPV_MIN,
    OPV_CLEAN,
    MISSING_SMI_ACCEPTOR,
    MISSING_SMI_DONOR,
    CLEAN_ACCEPTOR,
    CLEAN_DONOR,
)

from ml_for_opvs.data.error_correction.OPV_Min.remove_anomaly import Anomaly

from ml_for_opvs.data.preprocess.OPV_Min.smiles_to_bigsmiles import smile_to_bigsmile
from ml_for_opvs.data.preprocess.OPV_Min.smiles_to_selfies import opv_smiles_to_selfies

from ml_for_opvs.data.exploration.OPV_Min.correlation import (
    Correlation,
    PARAMETER_INVENTORY,
)

from ml_for_opvs.data.postprocess.OPV_Min.augmentation.augment import (
    Augment,
    AUGMENT_SMILES_DATA,
)
from ml_for_opvs.data.postprocess.OPV_Min.BRICS.brics_frag import BRIC_FRAGS
from ml_for_opvs.data.postprocess.OPV_Min.manual_frag.manual_frag import (
    manual_frag,
    MANUAL_ACCEPTOR_CSV,
    MANUAL_DONOR_CSV,
    MASTER_MANUAL_DATA,
)
from ml_for_opvs.data.postprocess.OPV_Min.fingerprint.morgan_fingerprints import (
    fp_data,
    FP_DATA,
)

# Step 1
donors = DonorClean(MASTER_DONOR_CSV, OPV_DONOR_DATA)
donors.clean_donor(CLEAN_DONOR_CSV)

# # # # Step 1b
donors.replace_r(CLEAN_DONOR_CSV)

# # # # # # Step 1d - canonSMILES to remove %10-%100
donors.canon_smi(CLEAN_DONOR_CSV)

# # # # # Step 1
acceptors = AcceptorClean(MASTER_ACCEPTOR_CSV, OPV_ACCEPTOR_DATA)
acceptors.clean_acceptor(CLEAN_ACCEPTOR_CSV)

# # Step 1b
acceptors.replace_r(CLEAN_ACCEPTOR_CSV)

# # # # # Step 1d - canonSMILES to remove %10-%100
acceptors.canon_smi(CLEAN_ACCEPTOR_CSV)

# Step 2 - ERROR CORRECTION (fill in missing D/A)
unique_opvs = UniqueOPVs(opv_min=OPV_MIN, opv_clean=OPV_CLEAN)
# concatenate for donors
unique_opvs.concat_missing_and_clean(MISSING_SMI_DONOR, CLEAN_DONOR, "D")

# concatenate for acceptors
unique_opvs.concat_missing_and_clean(MISSING_SMI_ACCEPTOR, CLEAN_ACCEPTOR, "A")

# Step 3 - smiles_to_bigsmiles.py & smiles_to_selfies.py
smile_to_bigsmile(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)
opv_smiles_to_selfies(CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)

# Step 4
pairings = DAPairs(OPV_DATA, CLEAN_DONOR_CSV, CLEAN_ACCEPTOR_CSV)
pairings.create_master_csv(MASTER_ML_DATA)
pairings.create_master_csv(MASTER_ML_DATA_PLOT)

# # # # Step 4b - Convert STR -> FLOAT
pairings.convert_str_to_float(MASTER_ML_DATA)
pairings.convert_str_to_float(MASTER_ML_DATA_PLOT)

# Step 4c - Remove anomalies!
# Go to ml_for_opvs > data > error_correction > remove_anomaly.py
anomaly = Anomaly(MASTER_ML_DATA)
anomaly.remove_anomaly(MASTER_ML_DATA)
anomaly.remove_anomaly(MASTER_ML_DATA_PLOT)
anomaly.correct_anomaly(MASTER_ML_DATA)
anomaly.correct_anomaly(MASTER_ML_DATA_PLOT)

# # Step 4c - Fill empty values for Thermal Annealing, and Solvent Additives
pairings.fill_empty_values(MASTER_ML_DATA)

# Step 5
# Add solvents
corr_plot = Correlation(MASTER_ML_DATA_PLOT)
corr_plot.solvent_correlation(PARAMETER_INVENTORY, MASTER_ML_DATA_PLOT)
corr_plot = Correlation(MASTER_ML_DATA)
corr_plot.solvent_correlation(PARAMETER_INVENTORY, MASTER_ML_DATA)

# Step 6
# Postprocessing!!!
b_frag = BRIC_FRAGS(MASTER_ML_DATA)
frag_dict = b_frag.bric_frag()

augmenter = Augment(MASTER_ML_DATA)
augmenter.aug_smi_doRandom(AUGMENT_SMILES_DATA, 4)
augmenter.aug_smi_tokenize(AUGMENT_SMILES_DATA)

fp_main = fp_data(MASTER_ML_DATA)
fp_main.create_master_fp(FP_DATA, 3, 512)

manual = manual_frag(MASTER_ML_DATA, MANUAL_DONOR_CSV, MANUAL_ACCEPTOR_CSV)
frag_dict = manual.return_frag_dict()
manual.bigsmiles_from_frag(MANUAL_DONOR_CSV, MANUAL_ACCEPTOR_CSV)
manual.create_manual_csv(frag_dict, MASTER_MANUAL_DATA)
