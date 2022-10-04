#!/bin/bash

ntrials=200

for model in gp ngboost  
do
	for feature in fp mordred
	do
		echo Running for $model on $feature
	 	sbatch submit_cpu_optimize.sh $model $feature $ntrials
	done
done

# graph feature
for model in gnn
do
	for feature in graph
	do
		echo Running for $model on $feature for $ntrials
	 	sbatch submit_cpu_optimize.sh $model $feature $ntrials
	done
done