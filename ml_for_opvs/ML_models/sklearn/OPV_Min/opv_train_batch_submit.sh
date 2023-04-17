# generate bash files for training across all (input rep) + (feature set) + (model type). need to add one for .format{} to the python_code to indicate model type
#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --output=/project/6033559/stanlo/ml_for_opvs/ml_for_opvs/ML_models/pytorch/OPV_Min/slurm.out
#SBATCH --error=/project/6033559/stanlo/ml_for_opvs/ml_for_opvs/ML_models/pytorch/OPV_Min/slurm.err
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate

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
model_type=("RF" "XGBoost" "kNN" "MLR" "SVM")

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