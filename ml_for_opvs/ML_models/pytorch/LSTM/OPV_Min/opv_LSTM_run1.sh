#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH --output=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/pytorch/LSTM/slurm.out
#SBATCH --error=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/pytorch/LSTM/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python opv_LSTM.py --n_hidden 256 --n_embedding 256 --drop_prob 0.3 --learning_rate 1e-2 --train_batch_size 256