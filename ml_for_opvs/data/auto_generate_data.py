"""File that contains all functions to go from .csv, to preprocessed data and input_representation data ready for training.
"""

from ml_for_opvs.data.preprocess.clean_donors_acceptors import (
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
from ml_for_opvs.data.error_correction.OPV_Min.approximate_homo_lumo import approximate_value

from ml_for_opvs.data.input_representation.OPV_Min.aug_SMILES.augment import aug_smi_doRandom, aug_smi_tokenize
from ml_for_opvs.data.input_representation.OPV_Min.fingerprint.morgan_fingerprints import create_master_fp
from ml_for_opvs.data.input_representation.OPV_Min.BRICS.brics_frag import bric_frag
from ml_for_opvs.data.input_representation.OPV_Min.manual_frag.manual_frag import export_manual_frag, fragment_files
from ml_for_opvs.data.preprocess.OPV_Min.smiles_to_bigsmiles import smile_to_bigsmile
from ml_for_opvs.data.preprocess.OPV_Min.smiles_to_selfies import opv_smiles_to_selfies

from ml_for_opvs.data.exploration.OPV_Min.correlation import (
    Correlation,
    PARAMETER_INVENTORY,
)

from ml_for_opvs.data.input_representation.OPV_Min.aug_SMILES.augment import (
    AUGMENT_SMILES_DATA,
)
from ml_for_opvs.data.input_representation.OPV_Min.manual_frag.manual_frag import (
    manual_frag,
    MASTER_MANUAL_DATA,
)
from ml_for_opvs.data.input_representation.OPV_Min.fingerprint.morgan_fingerprints import (
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

# # Step 4c - Fill empty values for Thermal Annealing, and solvent_additives
pairings.fill_empty_values(MASTER_ML_DATA)

# Add HOMO/LUMO approximation.
approximate_value(MASTER_ML_DATA)

# Step 5
# Add solvents
corr_plot = Correlation(MASTER_ML_DATA_PLOT)
corr_plot.solvent_correlation(PARAMETER_INVENTORY, MASTER_ML_DATA_PLOT)
corr_plot = Correlation(MASTER_ML_DATA)
corr_plot.solvent_correlation(PARAMETER_INVENTORY, MASTER_ML_DATA)

# Step 6
# input_representationing!!!
frag_dict = bric_frag(MASTER_ML_DATA)

aug_smi_doRandom(MASTER_ML_DATA, AUGMENT_SMILES_DATA, 5)
aug_smi_tokenize(AUGMENT_SMILES_DATA)

create_master_fp(MASTER_ML_DATA, FP_DATA, 3, 512)

export_manual_frag()
