python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024  --target_name calc_PCE_percent --model_type MLR --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_device_wo_thickness --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024 --feature_names HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,total_solids_conc_mg_mL,solvent_additive,annealing_temperature,hole_contact_layer,electron_contact_layer --target_name calc_PCE_percent --model_type MLR --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_device_wo_thickness --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024 --feature_names HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV --target_name calc_PCE_percent --model_type MLR --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_device_wo_thickness --random_state 22

ntation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024  --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_fabrication_wo_solid --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024 --feature_names HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_fabrication_wo_solid --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_fabrication_wo_solid/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024 --feature_names HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_fabrication_wo_solid --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024  --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_device_wo_thickness --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024 --feature_names HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,total_solids_conc_mg_mL,solvent_additive,annealing_temperature,hole_contact_layer,electron_contact_layer --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_device_wo_thickness --random_state 22

python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_train_4.csv  --test_path ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_0.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_1.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_2.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_3.csv ../../../data/input_representation/OPV_Min/fingerprint/processed_fingerprint_device_wo_thickness/KFold/input_test_4.csv  --input_representation DA_FP_radius_3_nbits_1024 --feature_names HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV --target_name calc_PCE_percent --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint/result_fingerprint_device_wo_thickness --random_state 22

# 2
feat_select_group=("fabrication_wo_solid" "device_wo_thickness")

# 3
declare -a input_rep_features

# 4
declare -a feature_name_dict

# 5
target_name=("calc_PCE_percent")

# 6
model_type=("RF" "XGBoost" "KRR" "MLR" "SVM")

for ir in ${input_rep[@]}; do
    for fsg in ${feat_select_group[@]}; do
        case "$fsg" in
            "fabrication_wo_solid") feature_name_dict=("" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature")
            ;;
            "device_wo_thickness") feature_name_dict=("" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,hole_contact_layer,electron_contact_layer")
            ;;
        esac
        for fnd in ${feature_name_dict[@]}; do
            for tn in ${target_name[@]}; do
                for mt in ${model_type[@]}; do
                    case "$ir" in
                        "fingerprint") input_rep_features=("DA_FP_radius_3_nbits_1024")
                        ;;
                        "BRICS") input_rep_features=("DA_tokenized_BRICS")
                        ;;
                        "manual_frag") input_rep_features=("DA_manual_tokenized")
                        ;;
                        "aug_SMILES") input_rep_features=("DA_pair_aug" "DA_pair_tokenized_aug")
                        ;;
                        "smiles") input_rep_features=("DA_SMILES" "DA_SELFIES" "DA_BigSMILES")
                        ;;
                    esac
                    # initialize train and test data paths as empty strings
                    train_path=""
                    test_path=""
                    for fold in {0..4}; do
                        train_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg/KFold/input_train_$fold.csv"
                        test_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg/KFold/input_test_$fold.csv"
                    done
                    feature_clause=""
                    if [ fnd != "" ]
                    then
                        feature_clause="--feature_names $fnd"
                    fi
                    for irf in ${input_rep_features[@]}; do
                        python ../train.py --train_path $train_path  --test_path $test_path --input_representation $irf $feature_clause --target_name $tn --model_type $mt --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg --random_state 22
                    done
                done
            done
        done
    done
done

# generate bash files for training across all (input rep) + (feature set) + (model type). need to add one for .format{} to the code to indicate model type

# 1
input_rep=("fingerprint" "BRICS" "manual_frag" "aug_SMILES" "smiles")

# 1.1
declare -a input_rep_filename_dict

input_rep_filename_dict=(["fingerprint"]="fingerprint" ["BRICS"]="brics_frag" ["manual_frag"]="manual_frag" ["aug_SMILES"]="augment" ["smiles"]="smiles")

# 2
feat_select_group=("fabrication_wo_solid" "device_wo_thickness")

# 3
declare -a input_rep_features

# 4
declare -a feature_name_dict

# 5
target_name=("calc_PCE_percent")

# 6
model_type=("RF" "XGBoost" "KRR" "MLR" "SVM")

for ir in ${input_rep[@]}; do
    for fsg in ${feat_select_group[@]}; do
        case "$fsg" in
            "fabrication_wo_solid") feature_name_dict=("" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature")
            ;;
            "device_wo_thickness") feature_name_dict=("" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,hole_contact_layer,electron_contact_layer")
            ;;
        esac
        for fnd in ${feature_name_dict[@]}; do
            for tn in ${target_name[@]}; do
                for mt in ${model_type[@]}; do
                    case "$ir" in
                        "fingerprint") input_rep_features=("DA_FP_radius_3_nbits_1024")
                        ;;
                        "BRICS") input_rep_features=("DA_tokenized_BRICS")
                        ;;
                        "manual_frag") input_rep_features=("DA_manual_tokenized")
                        ;;
                        "aug_SMILES") input_rep_features=("DA_pair_aug" "DA_pair_tokenized_aug")
                        ;;
                        "smiles") input_rep_features=("DA_SMILES" "DA_SELFIES" "DA_BigSMILES")
                        ;;
                    esac
                    # initialize train and test data paths as empty strings
                    train_path=""
                    test_path=""
                    for fold in {0..4}; do
                        train_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg/KFold/input_train_$fold.csv"
                        test_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg/KFold/input_test_$fold.csv"
                    done
                    feature_clause=""
                    if [ fnd != "" ]
                    then
                        feature_clause="--feature_names $fnd"
                    fi
                    for irf in ${input_rep_features[@]}; do
                        python ../train.py --train_path $train_path  --test_path $test_path --input_representation $irf $feature_clause --target_name $tn --model_type $mt --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg --random_state 22
                    done
                done
            done
        done
    done
done