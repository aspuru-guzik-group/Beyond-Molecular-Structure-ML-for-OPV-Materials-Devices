#!/bin/bash
# 1
input_rep=("fingerprint" "ohe" "mordred") # "fingerprint" "BRICS" "smiles"

# 1.1
declare -A input_rep_filename_dict

input_rep_filename_dict=(["fingerprint"]="fingerprint" ["BRICS"]="brics_frag" ["smiles"]="smiles" ["mordred"]="mordred" ["mordred_pca"]="mordred_pca" ["graphembed"]="graphembed" ["ohe"]="ohe")

# 2
feat_select_group=("molecules_only") # "fabrication_wo_solid" "device_wo_thickness"

# 3
declare -a input_rep_features

# 4
declare -a feature_name_dict

# 5
target_name=("calc_PCE_percent") #"calc_PCE_percent" "FF_percent" "Jsc_mA_cm_pow_neg2" "Voc_V"

# 6
model_type=("RF" "XGBoost" "KRR" "MLR" "SVM" "Lasso" "KNN") # "RF" "XGBoost" "KRR" "MLR" "SVM" "Lasso" "KNN"

#7
multi_output_type=("''") # "ensemble" or "multi"

for ir in ${input_rep[@]}; do
    for fsg in ${feat_select_group[@]}; do
        case "$fsg" in
            "molecules_only") feature_name_dict=("''")
            ;;
            "materials_only") feature_name_dict=("''" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV")
            ;;
            "fabrication_wo_solid") feature_name_dict=("HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,solvent_additive_conc_v_v_percent,solvent_BPt,solvent_density,solvent_dD,solvent_RI,solvent_MW,solvent_dP,solvent_dH,solvent_log_kow,solvent_MPt,solvent_dHDon,solvent_dHAcc,solvent_trouton,solvent_dipole,solvent_RER,solvent_ParachorGA,solvent_RD,solvent_DCp,solvent_log_n,solvent_SurfTen,solvent_additive_BPt,solvent_additive_density,solvent_additive_dD,solvent_additive_RI,solvent_additive_MW,solvent_additive_dP,solvent_additive_dH,solvent_additive_log_kow,solvent_additive_MPt,solvent_additive_dHDon,solvent_additive_dHAcc,solvent_additive_trouton,solvent_additive_dipole,solvent_additive_RER,solvent_additive_ParachorGA,solvent_additive_RD,solvent_additive_DCp,solvent_additive_log_n,solvent_additive_SurfTen") # "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,Eg_D_eV,Ehl_D_eV,Eg_A_eV,Ehl_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,solvent_additive_conc_v_v_percent"
            ;;
            "device_wo_thickness") feature_name_dict=("HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,solvent_additive_conc_v_v_percent,hole_contact_layer,electron_contact_layer,solvent_BPt,solvent_density,solvent_dD,solvent_RI,solvent_MW,solvent_dP,solvent_dH,solvent_log_kow,solvent_MPt,solvent_dHDon,solvent_dHAcc,solvent_trouton,solvent_dipole,solvent_RER,solvent_ParachorGA,solvent_RD,solvent_DCp,solvent_log_n,solvent_SurfTen,solvent_additive_BPt,solvent_additive_density,solvent_additive_dD,solvent_additive_RI,solvent_additive_MW,solvent_additive_dP,solvent_additive_dH,solvent_additive_log_kow,solvent_additive_MPt,solvent_additive_dHDon,solvent_additive_dHAcc,solvent_additive_trouton,solvent_additive_dipole,solvent_additive_RER,solvent_additive_ParachorGA,solvent_additive_RD,solvent_additive_DCp,solvent_additive_log_n,solvent_additive_SurfTen") # "''" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV,D_A_ratio_m_m,solvent,solvent_additive,annealing_temperature,solvent_additive_conc_v_v_percent,hole_contact_layer,electron_contact_layer"
            ;;
        esac
        for fnd in ${feature_name_dict[@]}; do
            for tn in ${target_name[@]}; do
                for mt in ${model_type[@]}; do
                    for mot in ${multi_output_type[@]}; do
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
                            "graphembed") input_rep_features=("DA_graphembed")
                            ;;
                            "ohe") input_rep_features=("DA_ohe")
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
                            python ../train.py --train_path $train_path  --test_path $test_path --input_representation $irf --feature_names $feature_clause --feature_set $fsg --target_name $tn --model_type $mt --multi_output_type $mot --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/$ir/result_$fsg --random_state 22
                            # sbatch opv_train_submit.sh $train_path  $test_path $feature_clause $tn $mt $ir $irf ${input_rep_filename_dict[$ir]} $fsg
                            # --feature_set $fsg",solvent_properties"
                        done
                    done
                done
            done
        done
    done
done