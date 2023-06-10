import os, sys

sys.path.append("../_ml_for_opvs")
from ML_models.sklearn.tokenizer import Tokenizer

import pickle
import time
import copy
import ast
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

import tensorflow as tf

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    MinMaxScaler,
    FunctionTransformer,
)

import selfies as sf

from ml_for_opvs import utils
from ml_for_opvs.models import GPRegressor  # , GNNEmbedder, GNNPredictor
from ml_for_opvs.gnn import default_hp, GNNEmbedder, GNNPredictor
from ml_for_opvs.graphs_tf import get_graphs

# from _ml_for_opvs.graphs import PairDataset, pair_collate, get_graphs

ALL_METRICS = ["rmse", "r", "r2", "spearman", "mse", "mae"]


def run_training(x, y, model, feature, out_dir="trained_results"):
    # wrapper to optimize hyperparameters
    # split the data with CV
    utils.set_seed()
    if feature in ["graph", "simple_graph"]:
        train, val, test = utils.get_cv_splits(x[0].globals.numpy())
    else:
        train, val, test = utils.get_cv_splits(x)  # 64%/16%/20%   # just the indices

    # collectors for training information
    val_metric = []
    test_metrics = {n: [] for n in ALL_METRICS}
    results = {"y_pred": [], "y_true": [], "split": []}

    # optimize depending on model
    if model == "ngboost":
        # hp = {
        #     'max_depth': trial.suggest_int('max_depth', 3, 7),
        #     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
        #     'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        #     'n_estimators': trial.suggest_int('n_estimators', 1000, 2000, step=50)
        # }
        hp = {
            "max_depth": 6,
            "min_samples_leaf": 3,
            "min_samples_split": 3,
            "n_estimators": 2000,
        }

        for i, tr_, va_, te_ in zip(range(len(train)), train, val, test):
            x_train, y_train = x[tr_], y[tr_]
            x_val, y_val = x[va_], y[va_]
            x_test, y_test = x[te_], y[te_]

            # train return loss (minimize)
            m = NGBRegressor(
                # Base = DecisionTreeRegressor(
                #     criterion='friedman_mse',
                #     max_depth=hp['max_depth'],
                #     min_samples_leaf=hp['min_samples_leaf'],
                #     min_samples_split=hp['min_samples_split']
                # ),
                n_estimators=hp["n_estimators"],
                verbose=True,
            )
            m.fit(
                x_train,
                y_train.ravel(),
                x_val,
                y_val.ravel(),
                early_stopping_rounds=100,
            )
            y_pred = m.predict(x_val)
            val_metric.append(mse(y_pred, y_val.ravel()))

            y_pred = m.predict(x_test)
            results["y_pred"].extend(y_pred.ravel().tolist())
            results["y_true"].extend(y_test.ravel().tolist())
            results["split"].extend([i] * len(y_test))
            for metric in test_metrics.keys():
                test_metrics[metric].append(
                    utils.calculate_metric(metric, y_pred, y_test)
                )

    elif model == "gp":
        # use tanimoto if bit fingerprints
        hp = {
            "kernel": "tanimoto" if feature == "fp" else "rbf",
            "lengthscale": 1.0,
            "lr": 0.05,
        }
        n_epoch = 2000

        # set priors
        # if hp['kernel'] == 'rbf':
        #     hp.update({'lengthscale': trial.suggest_float('lengthscale', 0.05, 3.0)})
        # elif hp['kernel'] == 'cosine':
        #     hp.update({'period_length': trial.suggest_float('period_length', 0.1, 3.0)})
        # elif hp['kernel'] == 'matern':
        #     hp.update({
        #         'nu': trial.suggest_categorical('nu', [0.5, 1.5, 2.5]),
        #         'lengthscale': trial.suggest_float('lengthscale', 0.05, 3.0)
        #     })
        # elif hp['kernel'] == 'rff':
        #     hp.update({'num_samples': trial.suggest_int('num_samples', 10, 100, step=5)})

        for i, tr_, va_, te_ in zip(range(len(train)), train, val, test):
            # fit the scaler on training set only
            x_train, y_train = x[tr_].astype(float), y[tr_].astype(float)
            x_scaler = (
                QuantileTransformer(n_quantiles=int(x_train.shape[0] / 2.0))
                if feature in ["mordred", "pca_mordred"]
                else FunctionTransformer()
            )
            y_scaler = MinMaxScaler()

            # transform the sets
            x_train = torch.tensor(x_scaler.fit_transform(x_train))
            y_train = torch.tensor(y_scaler.fit_transform(y_train))
            x_val = torch.tensor(x_scaler.transform(x[va_].astype(float)))
            y_val = torch.tensor(y_scaler.transform(y[va_].astype(float)))
            x_test = torch.tensor(x_scaler.transform(x[te_].astype(float)))
            y_test = torch.tensor(y_scaler.transform(y[te_].astype(float)))

            # train return loss (minimize)
            ll = gpytorch.likelihoods.GaussianLikelihood()
            m = GPRegressor(x_train, y_train.ravel(), ll, **hp)
            optimizer = torch.optim.Adam(m.parameters(), lr=hp["lr"])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(ll, m)

            m.train()
            ll.train()
            for _ in range(n_epoch):
                optimizer.zero_grad()
                y_pred = m(x_train)
                loss = -mll(y_pred, y_train.ravel())
                print(f"Epoch {_} | NLL: {loss.item()}")
                loss.backward()
                optimizer.step()

            m.eval()
            ll.eval()
            with torch.no_grad():
                y_pred = ll(m(x_val)).mean.numpy()
                loss = mse(y_pred, y_val.numpy().ravel())
                val_metric.append(loss)

                y_test = y_test.numpy()
                y_pred = y_scaler.inverse_transform(
                    ll(m(x_test)).mean.numpy().reshape(y_test.shape)
                )
                y_test = y_scaler.inverse_transform(y_test)
                results["y_pred"].extend(y_pred.ravel().tolist())
                results["y_true"].extend(y_test.ravel().tolist())
                results["split"].extend([i] * len(y_test))
                for metric in test_metrics.keys():
                    test_metrics[metric].append(
                        utils.calculate_metric(metric, y_pred, y_test)
                    )

    elif model == "gnn":
        hp = default_hp(output_dim=1)

        # batch_size = 32

        # model settings
        n_epoch = 1000
        patience = 100
        # num_node_features = x[0][0].x.shape[-1]
        # num_edge_features = x[0][0].edge_attr.shape[-1]
        output_dim = y.shape[-1]

        for i, tr_, va_, te_ in zip(range(len(train)), train, val, test):
            # split into groups
            d_tr = get_graphs(x[0], tr_)
            a_tr = get_graphs(x[1], tr_)
            d_va = get_graphs(x[0], va_)
            a_va = get_graphs(x[1], va_)
            d_te = get_graphs(x[0], te_)
            a_te = get_graphs(x[1], te_)

            # scale the targets
            y_scaler = MinMaxScaler()

            # transform the sets
            y_train = torch.tensor(
                y_scaler.fit_transform(y[tr_].astype(float)), dtype=torch.float
            )
            y_val = torch.tensor(
                y_scaler.transform(y[va_].astype(float)), dtype=torch.float
            )
            y_test = torch.tensor(
                y_scaler.transform(y[te_].astype(float)), dtype=torch.float
            )

            # train_dl = DataLoader(PairDataset(d_tr, a_tr, y_train),
            #     batch_size=batch_size, shuffle=True, collate_fn=pair_collate)
            # valid_dl = DataLoader(PairDataset(d_va, a_va, y_val),
            #     batch_size=batch_size, shuffle=False, collate_fn=pair_collate)

            # make the models
            # d_gnn = GNNEmbedder(num_layers, num_edge_features, hp['latent_dim'], hp['embed_dim'])
            # a_gnn = GNNEmbedder(num_node_features, num_edge_features, hp['latent_dim'], hp['embed_dim'])
            d_gnn = GNNEmbedder.from_hparams(hp)
            a_gnn = GNNEmbedder.from_hparams(hp)
            net = GNNPredictor(d_gnn, a_gnn, output_dim=hp["output_dim"])

            optimizer = tf.optimizers.Adam(1e-4)
            # criterion = torch.nn.MSELoss()
            criterion = tf.keras.losses.MeanSquaredError()

            # early stopping criteria
            best_loss = np.inf
            best_model = None
            count = 0
            for epoch in range(n_epoch):
                s_time = time.time()
                epoch_loss = 0
                # net.train()
                # for d_graph, a_graph, target in train_dl:
                d_graph, a_graph, target = d_tr, a_tr, y_train

                with tf.GradientTape() as tape:
                    output = net(d_graph, a_graph)
                    loss = criterion(output, target)
                grads = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))

                epoch_loss += loss.numpy()
                train_avg_loss = epoch_loss

                # validation
                epoch_loss = 0
                # with torch.no_grad():
                # net.eval()
                # for d_graph, a_graph, target in valid_dl:
                d_graph, a_graph, target = d_va, a_va, y_val
                output = net(d_graph, a_graph)
                loss = criterion(output, target)
                epoch_loss += loss.numpy()
                val_avg_loss = epoch_loss

                # check early stopping
                if val_avg_loss < best_loss:
                    best_loss = val_avg_loss
                    best_model = copy.deepcopy(net)
                    count = 0
                else:
                    count += 1

                print(
                    f"Epoch {epoch:<4} Loss: {val_avg_loss} Best loss: {best_loss} Time elapsed: {time.time()-s_time}"
                )

                if count >= patience:
                    print("Early stopping reached.\n")
                    break

            val_metric.append(best_loss)

            # also get the embeddings
            # with torch.no_grad():
            # non shuffle the training set
            # train_dl = DataLoader(PairDataset(d_tr, a_tr, y_train),
            #     batch_size=batch_size, shuffle=False, collate_fn=pair_collate)
            # test_dl = DataLoader(PairDataset(d_te, a_te, y_test),
            #     batch_size=len(y_test), shuffle=False, collate_fn=pair_collate)

            # best_model.eval()
            embeds = {
                key: {"acceptor": [], "donor": [], "target": []}
                for key in ["train", "valid", "test"]
            }
            # for d_graph, a_graph, target in train_dl:
            d_graph, a_graph = d_tr, a_tr
            embeds["train"]["acceptor"].extend(
                best_model.embed_acceptor(a_graph).numpy()
            )
            embeds["train"]["donor"].extend(best_model.embed_donor(d_graph).numpy())
            embeds["train"]["target"].extend(y_scaler.inverse_transform(y_train))

            d_graph, a_graph = d_va, a_va
            embeds["valid"]["acceptor"].extend(
                best_model.embed_acceptor(a_graph).numpy()
            )
            embeds["valid"]["donor"].extend(best_model.embed_donor(d_graph).numpy())
            embeds["valid"]["target"].extend(y_scaler.inverse_transform(y_val))

            d_graph, a_graph = d_te, a_te
            y_test = y_scaler.inverse_transform(y_test)
            embeds["test"]["acceptor"].extend(
                best_model.embed_acceptor(a_graph).numpy()
            )
            embeds["test"]["donor"].extend(best_model.embed_donor(d_graph).numpy())
            embeds["test"]["target"].extend(y_test)
            y_pred = y_scaler.inverse_transform(
                best_model(d_graph, a_graph).numpy()
            )  # not batched

            # save test results
            results["y_pred"].extend(y_pred.ravel().tolist())
            results["y_true"].extend(y_test.ravel().tolist())
            results["split"].extend([i] * len(y_test))

            pickle.dump(embeds, open(f"{out_dir}/graphembed_split{i}.pkl", "wb"))

            for metric in test_metrics.keys():
                test_metrics[metric].append(
                    utils.calculate_metric(metric, y_pred, y_test)
                )
    else:
        raise ValueError("Invalid model.")

    return pd.DataFrame(test_metrics)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, help="Name of model.")
    parser.add_argument(
        "--feature", action="store", type=str, default="fp", help="Name of feature."
    )
    parser.add_argument(
        "--dataset", action="store", type=str, default="min", help="Dataset of choice."
    )
    parser.add_argument(
        "--num_workers",
        action="store",
        type=int,
        default=1,
        help="Number of workers, defaults to 1.",
    )

    FLAGS = parser.parse_args()

    df = pd.read_csv(f"data/{FLAGS.dataset}.csv")

    study_name = f"{FLAGS.dataset}_{FLAGS.model}_{FLAGS.feature}"
    os.makedirs(f"trained_results/", exist_ok=True)

    # get the features
    if FLAGS.feature in ["mordred", "fp", "pca_mordred"]:
        # remove zero variance and concatentate if vector features
        with open(f"data/{FLAGS.dataset}_{FLAGS.feature}.pkl", "rb") as f:
            data = pickle.load(f)
        x_donor = data["donor"]
        x_acceptor = data["acceptor"]
        x = np.concatenate((x_donor, x_acceptor), axis=-1)

    elif FLAGS.feature == "selfies":
        f_df = pd.read_csv(
            "../_ml_for_opvs/data/input_representation/OPV_Min/smiles/master_smiles.csv"
        )
        tk = Tokenizer()
        token2idx, max_len = tk.tokenize_selfies(f_df["DA_SELFIES"])
        x = (
            f_df["DA_SELFIES"]
            .apply(lambda r: tk.tokenize_from_dict(token2idx, r))
            .tolist()
        )
        x = np.array(tk.pad_input(x, max_len))

    elif FLAGS.feature == "smiles":
        f_df = pd.read_csv(
            "../_ml_for_opvs/data/input_representation/OPV_Min/smiles/master_smiles.csv"
        )
        x, max_length, vocab_length, token2idx = Tokenizer().tokenize_data(
            f_df["DA_SMILES"]
        )
        x = np.array(x)

    elif FLAGS.feature == "bigsmiles":
        f_df = pd.read_csv(
            "../_ml_for_opvs/data/input_representation/OPV_Min/smiles/master_smiles.csv"
        )
        x, max_length, vocab_length, token2idx = Tokenizer().tokenize_data(
            f_df["DA_BigSMILES"]
        )
        x = np.array(x)

    elif FLAGS.feature == "brics":  # label encoding of brics fragments
        f_df = pd.read_csv(
            "../_ml_for_opvs/data/input_representation/OPV_Min/BRICS/master_brics_frag.csv"
        )
        x = np.array(f_df["DA_tokenized_BRICS"].apply(ast.literal_eval).tolist())

    elif FLAGS.feature == "homolumo":
        f_df = pd.read_csv(
            "../_ml_for_opvs/data/input_representation/OPV_Min/smiles/master_smiles.csv"
        )
        x = f_df[
            [
                "HOMO_D_eV",
                "LUMO_D_eV",
                "HOMO_A_eV",
                "LUMO_A_eV",
                "Eg_D_eV",
                "Ehl_D_eV",
                "Eg_A_eV",
                "Ehl_A_eV",
            ]
        ].to_numpy()

    elif FLAGS.feature in ["graph", "simple_graph"]:
        with open(f"data/{FLAGS.dataset}_{FLAGS.feature}.pkl", "rb") as f:
            data = pickle.load(f)
        x = [data["donor"], data["acceptor"]]

    else:
        raise ValueError("No such feature")

    # get the targets
    y = df[["calc_PCE_percent"]].to_numpy()

    # remove invalids
    if FLAGS.feature not in ["graph", "simple_graph"]:
        valid_idx = ~np.isnan(x).any(axis=1)[0]
        x = x[valid_idx]
        y = y[valid_idx]

    # perform training and get statistics on training set
    metrics_df = run_training(x, y, FLAGS.model, FLAGS.feature)
    metrics_df.to_csv(f"trained_results/{study_name}.csv", index=False)

    vmap = {
        "min": "OPV_Min",
        "fp": "DA_FP_radius_3_nbits_512",
        "brics": "DA_tokenized_BRICS",
        "selfies": "DA_SELFIES",
        "smiles": "DA_SMILES",
        "bigsmiles": "DA_BigSMILES",
        "graph": "DA_gnn",
        "mordred": "mordred",
        "pca_mordred": "pca_mordred",
        "homolumo": "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV",
    }

    summary_df = pd.DataFrame(
        {
            "Dataset": vmap[FLAGS.dataset],
            "num_of_folds": 5,
            "Features": vmap[FLAGS.feature],
            "Targets Model": "calc_PCE_percent",
            "r_mean": metrics_df["r"].mean(),
            "r_std": metrics_df["r"].std(),
            "r2_mean": metrics_df["r2"].mean(),
            "r2_std": metrics_df["r2"].std(),
            "rmse_mean": metrics_df["rmse"].mean(),
            "rmse_std": metrics_df["rmse"].std(),
            "mae_mean": metrics_df["mae"].mean(),
            "mae_std": metrics_df["mae"].std(),
            "num_of_data": len(metrics_df),
        },
        index=[0],
    )

    summary_df.to_csv(f"trained_results/{study_name}_summary.csv", index=False)
