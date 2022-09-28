#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --output=/project/6033559/stanlo/ml_for_opvs/ml_for_opvs/ML_models/pytorch/OPV_Min/slurm.out
#SBATCH --error=/project/6033559/stanlo/ml_for_opvs/ml_for_opvs/ML_models/pytorch/OPV_Min/slurm.err
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G

module load python/3.9.6
source /project/6025683/stanlo/opv_project/bin/activate
python ../train.py --train_path $1 --test_path $2 --input_representation $7 --feature_name $3 --target_name $4 --model_type $5 --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/$6/result_$8_$9 --random_state 22