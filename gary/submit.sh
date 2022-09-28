#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=64        # CPU cores/threads
#SBATCH --mem=92000M              # memory per node
#SBATCH --time=0-02:00            # time (DD-HH:MM)
#SBATCH -J ml_opv

module load python/3.8
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.03.3

source ~/env/opv/bin/activate

# featurizing
# echo Working with Min et al. FP feature
# time python featurizer.py --num_workers=64 --dataset=min --feature=fp
# echo Working with Min et al. Mordred feature
# time python featurizer.py --num_workers=64 --dataset=min --feature=mordred

# note that the arguments
# $1 = model
# $2 = feature
python train.py --model=$1 --feature=$2



deactivate




