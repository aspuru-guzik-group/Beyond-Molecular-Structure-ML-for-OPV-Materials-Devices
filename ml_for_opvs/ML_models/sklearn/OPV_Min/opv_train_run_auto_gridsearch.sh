#!/bin/bash
# runs molecules only
# 1
input_rep=("graphembed")

# 1.1
declare -A input_rep_filename_dict

input_rep_filename_dict=(["fingerprint"]="fingerprint" ["BRICS"]="brics_frag" ["smiles"]="smiles" ["mordred"]="mordred" ["mordred_pca"]="mordred_pca" ["graphembed"]="graphembed")

# 2
feat_select_group=("molecules_only") # "fabrication_wo_solid" "device_wo_thickness"

# 3
declare -a input_rep_features

# 5
target_name=("calc_PCE_percent")

# 6
model_type=("RF" "XGBoost" "KRR" "MLR" "SVM" "Lasso" "KNN")

for ir in ${input_rep[@]}; do
    for tn in ${target_name[@]}; do
        for mt in ${model_type[@]}; do
            case "$ir" in
                "fingerprint") input_rep_features=("DA_FP_radius_3_nbits_1024")
                ;;
                "BRICS") input_rep_features=("DA_tokenized_BRICS")
                ;;
                "smiles") input_rep_features=("DA_SMILES" "DA_SELFIES" "DA_BigSMILES" "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV")
                ;;
                "mordred") input_rep_features=("DA_mordred")
                ;;
                "mordred_pca") input_rep_features=("DA_mordred_pca")
                ;;
                "graphembed") input_rep_features=("DA_graphembed")
                ;;
            esac
            # initialize train and test data paths as empty strings
            train_path=""
            test_path=""
            for fold in {0..4}; do
                train_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_molecules_only/KFold/input_train_$fold.csv"
                test_path+=" ../../../data/input_representation/OPV_Min/$ir/processed_${input_rep_filename_dict[$ir]}_molecules_only/KFold/input_test_$fold.csv"
            done
            for irf in ${input_rep_features[@]}; do
                feature_clause="$irf"
                if [ $fnd != "''" ]
                then
                    feature_clause+=",$fnd"
                fi
                python ../train.py --train_path $train_path  --test_path $test_path --input_representation $irf $feature_clause --target_name $tn --model_type $mt --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/$ir/result_molecules_only --random_state 22
                # sbatch opv_train_submit.sh $train_path  $test_path $feature_clause $tn $mt $ir $irf ${input_rep_filename_dict[$ir]} $fsg
            done
        done
    done
done