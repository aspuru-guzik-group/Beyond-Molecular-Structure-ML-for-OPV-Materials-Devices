#!/bin/bash
# 1
input_rep=("mordred" "mordred_pca") # "fingerprint" "BRICS" "smiles"

# 1.1
declare -A input_rep_filename_dict

input_rep_filename_dict=(["fingerprint"]="fingerprint" ["BRICS"]="brics_frag" ["smiles"]="smiles" ["mordred"]="mordred" ["mordred_pca"]="mordred_pca")

# 2
feat_select_group=("molecules_only") # "fabrication_wo_solid" "device_wo_thickness"

# 3
declare -a input_rep_features

# 4
declare -a feature_name_dict

# 5
target_name=("calc_PCE_percent")

# 6
model_type=("SVM") # "RF" "XGBoost" "KRR" "MLR" "SVM" "Lasso" "KNN"

for ir in ${input_rep[@]}; do
    for fsg in ${feat_select_group[@]}; do
        case "$fsg" in
            "molecules_only") feature_name_dict=("''")
            ;;
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
                        "smiles") input_rep_features=("DA_SMILES" "DA_SELFIES" "DA_BigSMILES")
                        ;;
                        "mordred") input_rep_features=("DA_mordred")
                        ;;
                        "mordred_pca") input_rep_features=("DA_mordred_pca")
                        ;;
                    esac
                    # initialize train and test data paths as empty strings
                    train_path=""
                    test_path=""
                    for fold in {0..4}; do
                        train_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg/KFold/input_train_$fold.csv"
                        test_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_$fsg/KFold/input_test_$fold.csv"
                    done
                    for irf in ${input_rep_features[@]}; do
                        feature_clause="$irf"
                        if [ $fnd != "''" ]
                        then
                            feature_clause+=",$fnd"
                        fi
                        python ../train.py --train_path $train_path  --test_path $test_path --input_representation $irf --feature_names $feature_clause --target_name $tn --model_type $mt --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/$ir/result_$fsg --random_state 22
                        # sbatch opv_train_submit.sh $train_path  $test_path $feature_clause $tn $mt $ir $irf ${input_rep_filename_dict[$ir]} $fsg
                    done
                done
            done
        done
    done
done