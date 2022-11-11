import glob
import numpy as np
import random
import scipy.stats

import rdkit.Chem.AllChem as Chem
import mordred
import mordred.descriptors

import torch
# from torch_geometric.utils import from_smiles

from .graphs_tf import MolTensorizer

import sklearn
from sklearn.decomposition import PCA 
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

def get_features(smi, feature_type = 'fp'):
    # get desired feature from smiles
    mol = Chem.MolFromSmiles(smi)
    if feature_type == 'fp':
        feat = np.array(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512))
    elif feature_type in ['mordred', 'pca_mordred']:
        calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
        vals = calc(mol)._values
        feat = np.array([float(v) for v in vals])
    # elif feature_type == 'graph':
    #     # feat = from_smiles(Chem.MolToSmiles(mol))

    #     feat = 
    else:
        raise NotImplementedError('No such feature.')
    return feat

def set_seed(seed = 22):
    # set random seed for all used modules
    # print(f'Random seed set to {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def get_cv_splits(x, n_splits=5, val_split = 0.2):
    # return dictionary with the indices of tvt splits
    test, train, val = [], [], []
    indices = range(len(x))
    splitter = KFold(n_splits=n_splits)
    for i, (train_ind, test_ind) in enumerate(splitter.split(indices)):
        test.append(test_ind)      
        train_ind, val_ind = train_test_split(train_ind, test_size=0.2)
        train.append(train_ind)
        val.append(val_ind)
    return train, val, test

def r_score(x, y):
    # return np.corrcoef(x,y)[0,1]
    pearson_r = scipy.stats.pearsonr(x, y)[0]
    return pearson_r

def spearman_score(x, y):
    spearman_r = scipy.stats.spearmanr(x, y)[0]
    return spearman_r

def remove_nan(features):
    features = np.delete(features, np.isnan(features).any(axis=0), axis=1)  # remove the invalid features
    return features

def remove_zero_variance(features):
    # return features without 0 variance columns
    var =  np.var(np.array(features, dtype=float), axis=0) 
    red_feature = np.array(features, dtype=float)[:, var > 0]
    return red_feature

def pca_features(features, num_dims = 128, threshold = 0.99999):
    # return pca reduced features with enough dimensions to account 
    # for threshold of variance in data
    pca = PCA()
    features = np.array(features)
    pca.fit(features)
    # count = 0
    # for i, vals in enumerate(pca.explained_variance_ratio_):
    #     count += vals
    #     if count >= threshold:
    #         break
    red_features = pca.transform(features)
    red_features = red_features[:, :num_dims]
    return red_features


def read_split_files(dataset, data_dir = 'data'):
    fnames = glob.glob(f'{data_dir}/{dataset}_splits*.npz')
    splits = []
    for f in fnames:
        splits.append(np.load(f))
        
    return splits


def calculate_metric(metric, y_pred, y_true):
    if metric == 'rmse':
        return np.sqrt(mse(y_true.ravel(), y_pred.ravel()))
    elif metric == 'r':
        return r_score(y_true.ravel(), y_pred.ravel())
    elif metric == 'r2':
        return r2_score(y_true.ravel(), y_pred.ravel())
    elif metric == 'spearman':
        return spearman_score(y_true.ravel(), y_pred.ravel())
    elif metric == 'mse':
        return mse(y_true.ravel(), y_pred.ravel())
    elif metric == 'mae':
        return np.mean(np.abs(y_true.ravel() - y_pred.ravel()))
    else:
        raise ValueError('Invalid metric')
