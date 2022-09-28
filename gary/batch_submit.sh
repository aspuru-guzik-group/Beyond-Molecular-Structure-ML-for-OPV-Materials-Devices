#!/bin/bash

for model in RF XGBoost GP
do
	for feature in fp mordred
	do
		# arguments
		# model type
		# feature
		# blah blah
		echo Running for $model on $feature
	 	sbatch submit.sh $model $feature
	done
done

