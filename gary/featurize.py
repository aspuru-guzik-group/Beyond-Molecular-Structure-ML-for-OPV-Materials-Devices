import numpy as np
import pandas as pd

import rdkit.Chem.AllChem as Chem
from pandarallel import pandarallel

import pickle
from argparse import ArgumentParser

from ml_for_opvs.utils import get_features
from ml_for_opvs.graphs import get_onehot_encoder, from_smiles

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--feature", action="store", type=str, default="fp", help="Feature. Defaults fp.")
    parser.add_argument("--dataset", action="store", type=str, default="min", help="Dataset of choice.")

    FLAGS = parser.parse_args()
    assert FLAGS.dataset in ["min", "min_fab_nosolid"], "Invalid dataset flag. Pick `min`, `min_fab_nosolid`"
    assert FLAGS.feature in ["fp", "mordred", "pca_mordred", "graph", "simple_graph"], "Invalid feature."

    pandarallel.initialize(progress_bar=False, nb_workers=FLAGS.num_workers)
    df = pd.read_csv(f'data/{FLAGS.dataset}.csv')
    if 'DA_SMILES' in df.keys():
        df['d_smi'] = df['DA_SMILES'].str.split('.').str.get(0)
        df['a_smi'] = df['DA_SMILES'].str.split('.').str.get(1)
    elif 'Donor_SMILES' in df.keys():
        df['d_smi'] = df['Donor_SMILES']
        df['a_smi'] = df['Acceptor_SMILES']
    else:
        raise ValueError("Issue with csv columns.")

    feat = FLAGS.feature
    print(f'Looking at {feat}')

    if feat == 'simple_graph':
        smi_list = df['d_smi'].tolist()
        smi_list.extend(df['a_smi'].tolist())
        encoder = get_onehot_encoder(smi_list)
        d_feat = df['d_smi'].parallel_apply(lambda x: from_smiles(x, encoder))
        a_feat = df['a_smi'].parallel_apply(lambda x: from_smiles(x, encoder))
    else:
        d_feat = df['d_smi'].parallel_apply(lambda x: get_features(x, feat))
        print('     Donor features made.')
        a_feat = df['a_smi'].parallel_apply(lambda x: get_features(x, feat))
        print('     Acceptor features made.')

    d_feat = d_feat.to_list()
    a_feat = a_feat.to_list()

    if feat not in ['graph', 'simple_graph']:
        # turn lists into arrays
        features = {'donor': np.array(d_feat), 'acceptor': np.array(a_feat)}
    else:
        features = {'donor': d_feat, 'acceptor': a_feat}
    # features = np.concatenate((d_feat, a_feat), axis=-1)
    
    # save features
    pickle.dump(features, open(f'data/{FLAGS.dataset}_{feat}.pkl', 'wb'))
    # np.savez(f'data/{FLAGS.dataset}_{feat}.npz', feature=features, d_feature=d_feat, a_feature=a_feat)

