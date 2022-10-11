#!/bin/bash

ntrials=150
script="submit_training.sh"
# script="submit_cpu_optimize.sh"

for model in gp ngboost  
do
	for feature in pca_mordred #fp mordred 
	do
		echo Running for $model on $feature
	 	sbatch $script $model $feature $ntrials
	done
done

# graph feature
# for model in gnn
# do
# 	for feature in graph
# 	do
# 		echo Running for $model on $feature for $ntrials
# 	 	sbatch $script $model $feature $ntrials
# 	done
# done

