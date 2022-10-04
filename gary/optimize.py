import pickle
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
from sklearn.preprocessing import StandardScaler

from ml_for_opvs import utils
from ml_for_opvs.models import GPRegressor, GNNEmbedder, GNNPredictor
from ml_for_opvs.graphs import PairDataset, pair_collate, get_graphs


def objective(trial, x, y, model, feature):
    # wrapper to optimize hyperparameters
    # split the data with CV
    utils.set_seed()
    if feature == 'graph':
        train, val, test = utils.get_cv_splits(x[0])
    else:
        train, val, test = utils.get_cv_splits(x)   # 64%/16%/20%   # just the indices
    
    # optimize depending on model
    if model == 'ngboost':
        hp = {
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 4),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1500, step=50)
        }
        metric = []
        for tr_, va_, _ in zip(train, val, test):
            x_train, y_train = x[tr_], y[tr_]
            x_val, y_val = x[va_], y[va_]

            # train return loss (minimize)
            m = NGBRegressor(
                Base = DecisionTreeRegressor(
                    criterion='friedman_mse', 
                    max_depth=hp['max_depth'],
                    min_samples_leaf=hp['min_samples_leaf'],
                    min_samples_split=hp['min_samples_split']
                ), 
                n_estimators=hp['n_estimators'],
                verbose=False,
            )
            m.fit(x_train, y_train.ravel(), x_val, y_val.ravel(), early_stopping_rounds=50)
            y_pred = m.predict(x_val)
            metric.append(mse(y_pred, y_val.ravel()))

    elif model == 'gp':
        # use tanimoto if bit fingerprints
        hp = {
            'kernel': 'tanimoto' if feature == 'fp' else 'rbf', # trial.suggest_categorical('kernel', ['rbf', 'cosine', 'matern', 'rff']),
            'lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True)
        }

        # set priors
        if hp['kernel'] == 'rbf':
            hp.update({'lengthscale': trial.suggest_float('lengthscale', 0.05, 2.5)})
        elif hp['kernel'] == 'cosine':
            hp.update({'period_length': trial.suggest_float('period_length', 0.1, 3.0)})
        elif hp['kernel'] == 'matern':
            hp.update({
                'nu': trial.suggest_categorical('nu', [0.5, 1.5, 2.5]),
                'lengthscale': trial.suggest_float('lengthscale', 0.05, 2.5)
            })
        elif hp['kernel'] == 'rff':
            hp.update({'num_samples': trial.suggest_int('num_samples', 10, 100, step=5)})

        metric = []
        for tr_, va_, _ in zip(train, val, test):
            x_train, y_train = torch.tensor(x[tr_].astype(float)), torch.tensor(y[tr_].astype(float))
            x_val, y_val = torch.tensor(x[va_].astype(float)), torch.tensor(y[va_].astype(float))

            # train return loss (minimize)
            ll = gpytorch.likelihoods.GaussianLikelihood()
            m = GPRegressor(x_train, y_train.ravel(), ll, **hp)
            optimizer = torch.optim.Adam(m.parameters(), lr=hp['lr'])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(ll, m)

            m.train()
            ll.train()
            for i in range(1000):
                optimizer.zero_grad()
                y_pred = m(x_train)
                loss = -mll(y_pred, y_train.ravel())
                loss.backward()
                optimizer.step()
                # print(loss.item())

            m.eval()
            ll.eval()
            with torch.no_grad():
                y_pred = ll(m(x_val))
                loss = mse(y_pred.mean.numpy(), y_val.numpy().ravel()) 
                rscore = utils.r_score(y_pred.mean.numpy(), y_val.numpy().ravel())
                metric.append(loss)

    elif model == 'gnn':
        hp = {
            'latent_dim': trial.suggest_int('latent_dim', 10, 200),
            'embed_dim': trial.suggest_int('embed_dim', 10, 200),   
            'batch_size': trial.suggest_int('batch_size', 3, 7),     # exponent of 2
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        }
        batch_size = 2**hp['batch_size']

        # model settings
        n_epoch = 2000
        patience = 100
        num_node_features = x[0][0].x.shape[-1]
        num_edge_features = x[0][0].edge_attr.shape[-1]
        output_dim = y.shape[-1]

        metric = []
        for i, tr, va, te in zip(range(len(train)), train, val, test):
            d_tr, a_tr = get_graphs(x, tr)
            d_va, a_va = get_graphs(x, va)
            y = torch.tensor(y, dtype=torch.float)
            # d_te, a_te = get_graphs(x, te)

            train_dl = DataLoader(PairDataset(d_tr, a_tr, y[tr]),
                batch_size=batch_size, shuffle=True, collate_fn=pair_collate)
            valid_dl = DataLoader(PairDataset(d_va, a_va, y[va]),
                batch_size=batch_size, collate_fn=pair_collate)

            # make the models
            d_gnn = GNNEmbedder(num_node_features, num_edge_features, hp['latent_dim'], hp['embed_dim'])
            a_gnn = GNNEmbedder(num_node_features, num_edge_features, hp['latent_dim'], hp['embed_dim'])
            net = GNNPredictor(d_gnn, a_gnn, hp['embed_dim'], output_dim)

            optimizer = torch.optim.Adam(net.parameters(), lr=hp['lr'])
            criterion = torch.nn.MSELoss()

            # early stopping criteria
            best_loss = np.inf
            best_model = None
            best_epoch = 0
            count = 0 
            for epoch in range(n_epoch):
                epoch_loss = 0
                net.train()
                for d_graph, a_graph, target in train_dl:
                    output = net(d_graph, a_graph)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    epoch_loss += loss.item()
                    optimizer.step()
                train_avg_loss = epoch_loss / len(train_dl)

                # validation
                epoch_loss = 0
                with torch.no_grad():
                    net.eval()
                    for d_graph, a_graph, target in valid_dl:
                        output = net(d_graph, a_graph)
                        loss = criterion(output, target)
                        epoch_loss += loss.item()
                val_avg_loss = epoch_loss / len(valid_dl)

                # check early stopping
                if val_avg_loss < best_loss:
                    best_loss = val_avg_loss
                    best_model = copy.deepcopy(net)
                    best_epoch = epoch
                    count = 0
                else:
                    count += 1

                trial.report(val_avg_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if count >= patience:
                    break

            metric.append(val_avg_loss)
    else:
        raise ValueError('Invalid model.')
    

    return np.mean(metric)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, help="Name of model.")
    parser.add_argument("--n_trials", action="store", type=int, default=1, help="Number of optimization trials.")
    parser.add_argument("--feature", action="store", type=str, default='fp', help="Name of feature.")
    parser.add_argument("--dataset", action="store", type=str, default="min", help="Dataset of choice.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults to 1.")

    FLAGS = parser.parse_args()

    print(f'Running for {FLAGS.feature}, on {FLAGS.model}, on {FLAGS.dataset}')

    # read in the data
    df = pd.read_csv(f'data/{FLAGS.dataset}.csv')
    with open(f'data/{FLAGS.dataset}_{FLAGS.feature}.pkl', 'rb') as f:
        data = pickle.load(f)

    # if FLAGS.feature == 'mordred':
    #     x_donor = utils.pca_features(x_donor)
    #     x_acceptor = utils.pca_features(x_acceptor)

    if FLAGS.feature in ['mordred', 'fp']:
        # remove zero variance and concatentate if vector features
        x_donor = utils.remove_zero_variance(data['donor'])
        x_acceptor = utils.remove_zero_variance(data['acceptor'])
        x = np.concatenate((x_donor, x_acceptor), axis=-1)
    elif FLAGS.feature in ['graph']:
        x = [data['donor'], data['acceptor']]
    else:
        raise ValueError('No such feature')
    y = df[['calc_PCE_percent']].to_numpy()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, x, y, FLAGS.model, FLAGS.feature), n_trials=FLAGS.n_trials, n_jobs=FLAGS.num_workers)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    plot_contour(study)
    plt.savefig(f'opt_{FLAGS.dataset}_{FLAGS.model}_{FLAGS.feature}.png')

    # TODO ... 
    # run the training


    

