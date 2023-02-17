import pickle
import copy
import pandas as pd
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import StandardScaler

from ml_for_opvs import utils
from ml_for_opvs.graphs import PairDataset, pair_collate, get_graphs
from ml_for_opvs.models import GNNEmbedder, GNNPredictor

import matplotlib.pyplot as plt

import os, sys
import pickle
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import optuna
from optuna.visualization import plot_contour, plot_param_importances

from ngboost import NGBRegressor
import gpytorch
import torch
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, FunctionTransformer

from ml_for_opvs import utils
from ml_for_opvs.models import GPRegressor, GNNEmbedder, GNNPredictor
from ml_for_opvs.graphs import PairDataset, pair_collate, get_graphs

ALL_METRICS = ['rmse', 'r', 'r2', 'spearman', 'mse', 'mae']


def run_training(model, out_dir='trained_results', n_splits=5):

    # collectors for training information
    val_metric = []
    test_metrics = {n: [] for n in ALL_METRICS}
    results = {'y_pred': [], 'y_true': [], 'split': []}
    
    for i in range(n_splits):
        data = pickle.load(open(f'{out_dir}/graphembed_split{i}.pkl', 'rb'))
        
        train, valid, test = data['train'], data['valid'], data['test']

        x_donor = np.array(train['donor'])
        x_acceptor = np.array(train['acceptor'])
        x_train = np.concatenate((x_donor, x_acceptor), axis=-1)
        y_train = np.array(train['target'])
        
        x_donor = np.array(valid['donor'])
        x_acceptor = np.array(valid['acceptor'])
        x_val = np.concatenate((x_donor, x_acceptor), axis=-1)
        y_val = np.array(valid['target'])

        x_donor = np.array(test['donor'])
        x_acceptor = np.array(test['acceptor'])
        x_test = np.concatenate((x_donor, x_acceptor), axis=-1)
        y_test = np.array(test['target'])

        # optimize depending on model
        if model == 'ngboost':
            hp = {
                'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 2000
            }


            # train return loss (minimize)
            m = NGBRegressor(
                Base = DecisionTreeRegressor(
                    criterion='friedman_mse', 
                    max_depth=hp['max_depth'],
                    min_samples_leaf=hp['min_samples_leaf'],
                    min_samples_split=hp['min_samples_split']
                ), 
                n_estimators=hp['n_estimators'],
                verbose=True,
            )
            m.fit(x_train, y_train.ravel(), x_val, y_val.ravel(), early_stopping_rounds=100)
            y_pred = m.predict(x_val)
            val_metric.append(mse(y_pred, y_val.ravel()))

            y_pred = m.predict(x_test)
            results['y_pred'].extend(y_pred.ravel().tolist())
            results['y_true'].extend(y_test.ravel().tolist())
            results['split'].extend([i]*len(y_test))
            for metric in test_metrics.keys():
                test_metrics[metric].append(utils.calculate_metric(metric, y_pred, y_test))


        elif model == 'gp':
            # use tanimoto if bit fingerprints
            hp = {
                # 'kernel': 'tanimoto' if feature == 'fp' else 'rbf',
                'kernel': 'rbf',
                'lengthscale': 1.0,
                'lr': 0.01
            }
            n_epoch = 2000

            # fit the scaler on training set only
            x_train, y_train = x_train.astype(float), y_train.astype(float)
            # x_scaler = QuantileTransformer(n_quantiles=int(x_train.shape[0]/2.0)) if feature == 'mordred' else FunctionTransformer()
            x_scaler = FunctionTransformer()
            y_scaler = MinMaxScaler()

            # transform the sets
            x_train = torch.tensor(x_scaler.fit_transform(x_train))
            y_train = torch.tensor(y_scaler.fit_transform(y_train))
            x_val = torch.tensor(x_scaler.transform(x_val.astype(float)))
            y_val = torch.tensor(y_scaler.transform(y_val.astype(float)))
            x_test = torch.tensor(x_scaler.transform(x_test.astype(float)))
            y_test = torch.tensor(y_scaler.transform(y_test.astype(float)))

            # train return loss (minimize)
            ll = gpytorch.likelihoods.GaussianLikelihood()
            m = GPRegressor(x_train, y_train.ravel(), ll, **hp)
            optimizer = torch.optim.Adam(m.parameters(), lr=hp['lr'])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(ll, m)

            m.train()
            ll.train()
            for _ in range(n_epoch):
                optimizer.zero_grad()
                y_pred = m(x_train)
                loss = -mll(y_pred, y_train.ravel())
                print(f'Epoch {_} | NLL: {loss.item()}')
                loss.backward()
                optimizer.step()

            m.eval()
            ll.eval()
            with torch.no_grad():
                y_pred = ll(m(x_val)).mean.numpy()
                loss = mse(y_pred, y_val.numpy().ravel()) 
                val_metric.append(loss)

                y_test = y_test.numpy()
                y_pred = y_scaler.inverse_transform(ll(m(x_test)).mean.numpy().reshape(y_test.shape))
                y_test = y_scaler.inverse_transform(y_test)
                results['y_pred'].extend(y_pred.ravel().tolist())
                results['y_true'].extend(y_test.ravel().tolist())
                results['split'].extend([i]*len(y_test))
                for metric in test_metrics.keys():
                    test_metrics[metric].append(utils.calculate_metric(metric, y_pred, y_test))

        else:
            raise ValueError('Invalid model.')

    return pd.DataFrame(test_metrics)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, help="Name of model.")
    parser.add_argument("--dataset", action="store", type=str, default="min", help="Dataset of choice.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults to 1.")

    FLAGS = parser.parse_args()
    
    # df = pd.read_csv(f'data/{FLAGS.dataset}.csv')
    # with open(f'data/{FLAGS.dataset}_graphembed.pkl', 'rb') as f:
    #     data = pickle.load(f)

    study_name=f'{FLAGS.dataset}_{FLAGS.model}_graphembed'
    os.makedirs(f'trained_results/' , exist_ok=True)


    # perform training and get statistics on training set
    metrics_df = run_training(FLAGS.model)
    metrics_df.to_csv(f'trained_results/{study_name}.csv', index=False)

    vmap = {
        'min': 'OPV_Min',
        'fp': 'DA_FP_radius_3_nbits_512',
        'brics': 'DA_tokenized_BRICS',
        'selfies': 'DA_SELFIES',
        'smiles': 'DA_SMILES',
        'bigsmiles': 'DA_BigSMILES',
        'graph': 'DA_gnn',
        'homolumo': 'HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV',
        'mordred': 'mordred',
        'pca_mordred': 'pca_mordred'
    }
    
    summary_df = pd.DataFrame(
        {
            'Dataset': vmap[FLAGS.dataset],
            'num_of_folds': 5,
            'Features': 'graphembed',
            'Targets Model': 'calc_PCE_percent',
            'r_mean': metrics_df['r'].mean(),
            'r_std': metrics_df['r'].std(),
            'r2_mean': metrics_df['r2'].mean(),
            'r2_std': metrics_df['r2'].std(),
            'rmse_mean': metrics_df['rmse'].mean(),
            'rmse_std': metrics_df['rmse'].std(),
            'mae_mean': metrics_df['mae'].mean(),
            'mae_std': metrics_df['mae'].std(),
            'num_of_data': len(metrics_df)
        }, index=[0]
    )
    
    summary_df.to_csv(f'trained_results/{study_name}_summary.csv', index=False)