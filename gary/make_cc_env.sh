#!/bin/bash
# for use on the cluster
module load python/3.8 
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.03.3

virtualenv --no-download ~/env/opv
source ~/env/opv/bin/activate

pip install --no-index --upgrade pip
# pip install tensorflow tensorflow-probability
pip install scipy pandas matplotlib
pip install seaborn scikit-learn
pip install pandarallel
pip install mordred

# pip install dm-sonnet ml_collections graph_nets tf-models-official
# pip install tensorflow==2.9.0
# pip install ngboost gpflow
# pip install mordred dask
# pip install scikit-multilearn 
# pip install ipython hdbscan umap-learn cairosvg

source deactivate

