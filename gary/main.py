import pandas as pd
import numpy as np
import seaborn as sns

import mordred
import mordred.descriptors
import rdkit.Chem.AllChem as Chem

import multiprocessing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=40)

import sklearn 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import pickle

datapath = '../ml_for_opvs/data/input_representation/OPV_Min/smiles/processed_smiles_fabrication_wo_solid.csv'

def get_features(smi, feature_type = 'fp'):
    # get desired feature
    mol = Chem.MolFromSmiles(smi)
    if feature_type == 'fp':
        feat = np.array(Chem.GetMorganFingerprintAsBitVect(mol, radius=3))
    elif feature_type == 'mordred':
        calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
        vals = calc(mol)._values
        feat = np.array([float(v) for v in vals])
    else:
        raise NotImplementedError('No such feature.')
    return feat

def get_cv_splits(x, y):
    results = {'split': [], 'train': [], 'test': []}
    indices = range(len(x))
    splitter = KFold(n_splits=5)
    # print(split_ind)
    for i, (train_ind, test_ind) in enumerate(splitter.split(indices)):
        results['split'].append(i)
        results['train'].append((x[train_ind], y[train_ind]))
        results['test'].append((x[test_ind], y[test_ind]))
    return results

def r_score(x, y):
    return np.corrcoef(x,y)[0,1]


if __name__ == '__main__':
    df = pd.read_csv(datapath)
    df['d_smi'] = df['DA_SMILES'].str.split('.').str.get(0)
    df['a_smi'] = df['DA_SMILES'].str.split('.').str.get(1)

    features = {}
    for feat in ['fp', 'mordred']:
        print(f'Looking at {feat}')
        d_feat = df['d_smi'].parallel_apply(lambda x: get_features(x, feat))
        print('     Donor features made.')
        a_feat = df['a_smi'].parallel_apply(lambda x: get_features(x, feat))
        print('     Acceptor features made.')

        features[feat] = np.concatenate((
            np.array(d_feat.to_list()), 
            np.array(a_feat.to_list())), axis=-1
        )
    
    # save features
    pickle.dump(features, open('features.pkl', 'wb'))

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


