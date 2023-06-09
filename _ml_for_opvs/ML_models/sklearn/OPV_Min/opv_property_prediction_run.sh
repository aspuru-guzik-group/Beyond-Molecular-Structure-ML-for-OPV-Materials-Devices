#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/_ml_for_opvs/_ml_for_opvs/ML_models/sklearn/OPV_Min/slurm.out
#SBATCH --error=/project/6033559/stanlo/_ml_for_opvs/_ml_for_opvs/ML_models/sklearn/OPV_Min/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python train.py --train_path ../../data/input_representation/OPV_Min/BRICS/KFold/input_train_[0-9].csv --validation_path ../../data/input_representation/OPV_Min/BRICS/KFold/input_valid_[0-9].csv --train_params_path ./OPV_Min/opv_train_params.json --model_type RF --hyperparameter_opt True --hyperparameter_space_path ./OPV_Min/opv_hpo_space.json --results_path ../../training/OPV_Min --random_state 22