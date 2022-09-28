import numpy as np
import random

import rdkit.Chem.AllChem as Chem
import mordred
import mordred.descriptors

import torch
from torch_geometric.utils import from_smiles

import sklearn
from sklearn.decomposition import PCA 
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor

def get_features(smi, feature_type = 'fp'):
    # get desired feature from smiles
    mol = Chem.MolFromSmiles(smi)
    if feature_type == 'fp':
        feat = np.array(Chem.GetMorganFingerprintAsBitVect(mol, radius=3))
    elif feature_type == 'mordred':
        calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
        vals = calc(mol)._values
        feat = np.array([float(v) for v in vals])
    elif feature_type == 'graph':
        feat = from_smiles(Chem.MolToSmiles(mol))
    else:
        raise NotImplementedError('No such feature.')
    return feat

def set_seed(seed = 22):
    # set random seed for all used modules
    print(f'Random seed set to {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def get_cv_splits(x, n_splits=5, val_split = 0.2):
    # return dictionary with the indices of tvt splits
    test, train = [], []
    indices = range(len(x))
    splitter = KFold(n_splits=n_splits)
    for i, (train_ind, test_ind) in enumerate(splitter.split(indices)):
        test.append(test_ind)      
        train.append(train_ind)
    return train, test

def r_score(x, y):
    return np.corrcoef(x,y)[0,1]

def remove_zero_variance(features):
    # return features without 0 variance columns
    features = np.delete(features, np.isnan(features).any(axis=0), axis=1)
    var =  np.var(np.array(features).astype(float), axis=0) 
    red_feature = np.array(features, dtype=float)[:, var > 0]
    return red_feature

def pca_features(features, threshold = 0.99999):
    # return pca reduced features with enough dimensions to account 
    # for threshold of variance in data
    pca = PCA()
    pca.fit(features)
    count = 0
    for i, vals in enumerate(pca.explained_variance_ratio_):
        count += vals
        if count >= threshold:
            break
    red_features = pca.transform(features)
    red_features = red_features[:, :i+1]
    return red_features
