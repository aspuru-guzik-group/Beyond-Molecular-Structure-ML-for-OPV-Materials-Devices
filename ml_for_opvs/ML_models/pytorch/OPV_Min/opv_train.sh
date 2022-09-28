model_types=('LSTM') # 'LSTM'
input=('BRICS' 'SMILES' 'SMILES') # data folder name
# 'aug_SMILES' 'BRICS' 'fingerprint' 'manual_frag' 'SMILES' 
input2=('BRICS' 'SELFIES' 'BigSMILES') # result folder name
# ('aug_SMILES' 'BRICS' ' fingerprint' 'manual_frag' 'SMILES' 'SELFIES' 'BigSMILES')
input3=('DA_pair_BRICS' 'DA_SELFIES' 'DA_BigSMILES' ) # input feature name
# ('DA_pair_aug' 'DA_pair_BRICS' 'DA_FP_radius_3_nbits_512' 'DA_manual_tokenized' 'DA_SMILES')
input4=('processed_brics_frag' 'processed_smiles' 'processed_smiles') # feature folder name
# 'processed_augment' 'processed_brics_frag' 'processed_fingerprint' 'processed_manual_frag' 'processed_smiles'
input5=('device' 'device_wo_thickness' 'electrical' 'fabrication_wo_solid' 'fabrication' 'full' 'molecules_only' 'molecules') # specific feature name

# TODO: feature dictionary of different configs.
# declare -A feature_dict


for model in "${model_types[@]}"
do
    for i in "${!input[@]}"; do
        printf "%s is in %s\n" "${input[i]}" "${input4[i]}"
        python ../train.py --train_path ../../../data/input_representation/OPV_Min/"${input[i]}"/"${input4[i]}"_full/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/OPV_Min/"${input[i]}"/"${input4[i]}"_full/KFold/input_test_[0-9].csv --input_representation "${input2[i]}" --feature_names "${input3[i]}" --target_name calc_PCE_percent --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/OPV_Min/"${input2[i]}"/"${input4[i]}"_full
    done
done