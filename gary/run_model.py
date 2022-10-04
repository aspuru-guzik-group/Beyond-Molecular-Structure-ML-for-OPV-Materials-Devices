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

# load the dataset
data = pickle.load(open('data/min_graph.pkl', 'rb'))
d_donor, d_acceptor = data['donor'], data['acceptor']
df = pd.read_csv('data/min.csv')

# get the targets
y = df[['calc_PCE_percent']].to_numpy()
# scaler = StandardScaler().fit(y)
# y = scaler.transform(y)
y = torch.tensor(y, dtype=torch.float)

num_node_features = d_donor[0].x.shape[-1]
num_edge_features = d_donor[0].edge_attr.shape[-1]
output_dim = y.shape[-1]
latent_dim = 32
embed_dim = 100
patience = 200
n_epoch = 5000
batch_size = 32

# get the splits
utils.set_seed()
train, val, test = utils.get_cv_splits(y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get splits
metrics = {f'r': [], 'loss': []}
for i, tr, va, te in zip(range(len(train)), train, val, test):

    d_tr, a_tr = get_graphs([d_donor, d_acceptor], tr)
    d_va, a_va = get_graphs([d_donor, d_acceptor], va)
    d_te, a_te = get_graphs([d_donor, d_acceptor], te)

    train_dl = DataLoader(PairDataset(d_tr, a_tr, y[tr]),
        batch_size=batch_size, shuffle=True, collate_fn=pair_collate)
    valid_dl = DataLoader(PairDataset(d_va, a_va, y[va]),
        batch_size=batch_size, collate_fn=pair_collate)
    test_dl = DataLoader(PairDataset(d_te, a_te, y[te]),
        batch_size=len(te), collate_fn=pair_collate)


    # build models
    num_node_features = d_donor[0].x.shape[-1]
    num_edge_features = d_donor[0].edge_attr.shape[-1]
    output_dim = y.shape[-1]
    d_gnn = GNNEmbedder(num_node_features, num_edge_features, latent_dim, embed_dim)
    a_gnn = GNNEmbedder(num_node_features, num_edge_features, latent_dim, embed_dim)
    net = GNNPredictor(d_gnn, a_gnn, embed_dim, output_dim)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
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

        print(f'Epoch: {epoch} | train: {train_avg_loss} | validation: {val_avg_loss} ')

        if count >= patience:
            print(f'Early stopping hit! \nBest model at {best_epoch} with loss {best_loss}')
            break

    # prediction on test set
    best_model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for d_graph, a_graph, target in test_dl:
            output = best_model(d_graph, a_graph)
            loss = criterion(output, target)
            epoch_loss += loss.item()

    fig, ax = plt.subplots()
    ax.scatter(target.numpy().ravel(), output.numpy().ravel())
    ax.set_xlabel('Test values')
    ax.set_ylabel('Predicted values')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.savefig(f'{i}_results.png')
    plt.close()


    test_avg_loss = epoch_loss / len(test_dl)
    rscore = utils.r_score(output.numpy().ravel(), target.numpy().ravel())
    # print(f'Test loss: {test_avg_loss}')
    # print(f'r score:  {rscore}')
    metrics[f'r'].append(rscore)
    metrics[f'loss'].append(test_avg_loss)

    # save the graph_embeddings
    # TODO

metrics = pd.DataFrame(metrics)
metrics.to_csv('results.csv', index=False)

