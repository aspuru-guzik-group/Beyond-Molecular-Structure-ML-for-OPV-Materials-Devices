#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --output=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/sklearn/RF/OPV_Min/slurm.out
#SBATCH --error=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/sklearn/RF/OPV_Min/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python opv_RF_cv.py