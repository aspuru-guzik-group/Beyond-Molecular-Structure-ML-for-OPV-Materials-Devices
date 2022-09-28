import numpy as np
import pandas as pd

import rdkit.Chem.AllChem as Chem
from pandarallel import pandarallel

import pickle
from argparse import ArgumentParser

from ml_for_opvs.utils import get_features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--feature", action="store", type=str, default="fp", help="Feature. Defaults fp.")
    parser.add_argument("--dataset", action="store", type=str, default="min", help="Dataset of choice.")

    FLAGS = parser.parse_args()
    assert FLAGS.dataset in ["min", "min_fab_nosolid"], "Invalid dataset flag. Pick `min`, `min_fab_nosolid`"
    assert FLAGS.feature in ["fp", "mordred", "graph"], "Invalid feature. Pick `fp`, `mordred`, `graph`."

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
    d_feat = df['d_smi'].parallel_apply(lambda x: get_features(x, feat))
    print('     Donor features made.')
    a_feat = df['a_smi'].parallel_apply(lambda x: get_features(x, feat))
    print('     Acceptor features made.')

    d_feat = d_feat.to_list()
    a_feat = a_feat.to_list()

    if feat != "graph":
        # turn lists into arrays
        features = {'donor': np.array(d_feat), 'acceptor': np.array(a_feat)}
    else:
        features = {'donor': d_feat, 'acceptor': a_feat}
    # features = np.concatenate((d_feat, a_feat), axis=-1)
    
    # save features
    pickle.dump(features, open(f'data/{FLAGS.dataset}_{feat}.pkl', 'wb'))
    # np.savez(f'data/{FLAGS.dataset}_{feat}.npz', feature=features, d_feature=d_feat, a_feature=a_feat)

'''
    for feat in ['fp', 'mordred']:
        print('Predicting model on ' + feat)
        y = df['PCE_percent'].to_numpy()
        splits = get_cv_splits(features[feat], y)

        r2 = []
        r = []
        for ((x_train, y_train), (x_test, y_test)) in zip(splits['train'], splits['test']):
            m = RandomForestRegressor(n_estimators=100)
            m.fit(x_train, y_train)
            y_pred = m.predict(x_test)
            r2.append(r2_score(y_test, y_pred))
            r.append(r_score(y_test, y_pred))

        print(f'Mean R2: {np.mean(r2)}')
        print(f'Std R2: {np.std(r2)}')

        print(f'Mean R: {np.mean(r)}')
        print(f'Std R: {np.std(r)}')

        # visualize predictions on test set
        maxval = max(y_test.max(), y_pred.max())
        fig, ax = plt.subplots()
        ax.plot([0,maxval], [0, maxval], 'k--')
        ax.scatter(y_test, y_pred)
        ax.plot(0,0, label=f'$R^2$: {np.mean(r2):.3f} $\pm$ {np.std(r2):.3f}', linestyle=None)
        ax.plot(0,0, label=f'R: {np.mean(r):.3f} $\pm$ {np.std(r):.3f}', linestyle=None)
        ax.legend(handlelength=0, handletextpad=0)
        ax.set_xlim([0, maxval])
        ax.set_ylim([0, maxval])
        plt.savefig(f'{feat}_rf.png')
'''
