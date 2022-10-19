# python ../train.py --train_path ../../../data/input_representation/OPV_Min/aug_SMILES/processed_augment_full/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/aug_SMILES/processed_augment_full/KFold/input_test_[0-9].csv --input_representation DA_pair_aug --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/aug_SMILES/processed_augment_full --random_state 22

# python ../train.py --train_path ../../../data/input_representation/OPV_Min/BRICS/processed_brics_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/BRICS/processed_brics_frag/KFold/input_test_[0-9].csv --input_representation DA_pair_BRICS --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/BRICS/processed_brics_frag --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_molecules_only/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_molecules_only/KFold/input_test_[0-9].csv --input_representation fingerprint --feature_names DA_FP_radius_3_nbits_1024 --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_molecules_only --random_state 22

# python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/processed_manual_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/manual_frag/processed_manual_frag/KFold/input_test_[0-9].csv --input_representation DA_manual --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/manual_frag/processed_manual_frag --random_state 22

# python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_test_[0-9].csv --input_representation DA_manual_aug --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/manual_frag_aug --random_state 22

# python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_test_[0-9].csv --input_representation DA_SMILES --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/SMILES --random_state 22

# python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_test_[0-9].csv --input_representation DA_SELFIES --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/SELFIES --random_state 22

# python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_test_[0-9].csv --input_representation DA_BigSMILES --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/BigSMILES --random_state 22


# generate bash files for training across all (input rep) + (feature set) + (model type). need to add one for .format{} to the code to indicate model type