#!/bin/bash

for model in gp ngboost  
do
	for feature in fp mordred
	do
		echo Running for $model on $feature
	 	sbatch submit_optimize.sh $model $feature
	done
done

# graph feature
for model in gnn
do
	for feature in graph
	do
		echo Running for $model on $feature
	 	sbatch submit_optimize.sh $model $feature
	done
done