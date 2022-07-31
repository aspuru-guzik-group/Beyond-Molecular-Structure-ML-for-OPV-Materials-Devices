#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --cpus-per-task=40        # CPU cores/threads
#SBATCH --mem=92000M              # memory per node
#SBATCH --time=0-04:00            # time (DD-HH:MM)

module load python/3.8
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.03.3

source ~/env/opv/bin/activate

time python main.py 

deactivate




