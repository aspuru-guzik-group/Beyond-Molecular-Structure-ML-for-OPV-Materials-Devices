import sys
from pathlib import Path
import torch
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

sys.path.append("../training/")
from pytorch_mpnn import smiles2data, DMPNNEncoder, RevIndexedData

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, mol, y=None):
        self.mol = mol
        self.y = y

    def __len__(self):
        return len(self.mol)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.y is None:
            return self.mol[idx]
        return self.mol[idx], self.y[idx] 
    
class GraphEmbedder(torch.nn.Module):
    def __init__(self, hidden_size, num_node_features, num_edge_features, depth, out_dim, embed_dim):
        super(GraphEmbedder, self).__init__()
        self.mpnn = DMPNNEncoder(hidden_size, num_node_features, num_edge_features, depth)
        self.embedder = nn.Linear(hidden_size,  embed_dim)
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim),
        )

    def embed(self, graph):
        return self.embedder(self.mpnn(graph))
    
    def forward(self, graph):
        x = self.embed(graph)
        y_pred = self.predictor(x)
        return y_pred


class GNNEmbedder():
    def __init__(self,
                 hidden_size=55,
                 depth=2,
                 embed_dim = 100,
                 lr=1e-4):
        self.hidden_size = hidden_size
        self.depth = depth
        self.embed_dim = embed_dim
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_data(self, x_train, y_train = None):
        graphs = [RevIndexedData(smiles2data(s)) for s in x_train]
        
        if y_train is not None:
            scaler = StandardScaler()
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if type(y_train) is not np.ndarray:
                y_train = y_train.to_numpy()
            y_train = scaler.fit_transform(y_train)
            
            y_train = torch.tensor(y_train, dtype=torch.float)
            self.dataset = GraphDataset(graphs, y_train)
        else:
            self.dataset = GraphDataset(graphs)

        self.num_node_features = graphs[0].x.shape[-1]
        self.num_edge_features = graphs[0].edge_attr.shape[-1]
        if y_train is not None:
            self.out_dim = y_train.shape[-1]

    def fit(self, x_train, y_train):
        # prepare dataloaders
        self.create_data(x_train, y_train)
        train_ind, val_ind = train_test_split(list(range(len(self.dataset))), test_size=0.15)
        train_loader = DataLoader(torch.utils.data.Subset(self.dataset, train_ind), batch_size=64, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(self.dataset, val_ind), batch_size=64, shuffle=False)

        # make the model
        self.model = GraphEmbedder(
            self.hidden_size, 
            self.num_node_features,
            self.num_edge_features, 
            self.depth,
            self.out_dim,
            self.embed_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # early stopping
        patience = 500
        count = 0 
        best_loss = np.inf

        n_epoch = 1500
        for _ in range(n_epoch):
            self.model.train()
            train_loss = 0
            for dg, y in train_loader:
                dg, y = dg.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(dg)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
            # print(f'Loss: {train_loss / len(train_loader)}')

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for dg, y in val_loader:
                    dg, y = dg.to(self.device), y.to(self.device)
                    y_pred = self.model(dg)
                    loss = loss_fn(y_pred, y)
                    val_loss += loss.cpu().item()
            val_loss /= len(val_loader)
            # print(f'Val loss: {val_loss}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()
                count = 0
            else:
                count += 1

            if count >= patience:
                print(f'Epoch: {_}. Early stopping reached. Best loss: {best_loss}')
                self.model.load_state_dict(best_model)
                break

        print(f'Epoch: {_}. Best loss: {best_loss}')

    def get_embeddings(self, x):
        self.create_data(x)
        loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
        embeddings = []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for dg in loader:
                dg = dg.to(self.device)
                emb = self.model.embed(dg)
                embeddings.append(emb)
        
        embeddings = torch.concat(embeddings, axis=0)
        return embeddings.cpu().numpy()
        

if __name__ == '__main__':
    in_file: Path = DATASETS / 'Min_2020_n558' / 'cleaned_dataset.pkl'
    out_file: Path = DATASETS / 'Min_2020_n558' / 'cleaned_dataset_gnnembeddings.pkl'
    df = pd.read_pickle(in_file)

    luts = {}
    for tag in ['Donor', 'Acceptor']:
        sub_df = df[[f'{tag} SMILES', f'HOMO_{tag[0]} (eV)', f'LUMO_{tag[0]} (eV)', f'Eg_{tag[0]} (eV)']].groupby(f'{tag} SMILES').mean().reset_index()
        x = sub_df[f'{tag} SMILES']
        y = sub_df[[f'HOMO_{tag[0]} (eV)', f'LUMO_{tag[0]} (eV)', f'Eg_{tag[0]} (eV)']]
        
        embedder = GNNEmbedder()
        embedder.fit(x,y)
        embeddings = embedder.get_embeddings(x)
        luts[tag] = pd.DataFrame(embeddings, index=x, columns=[f'{i}_{tag[0]}' for i in range(embeddings.shape[-1])])

    new_df = df[['Donor SMILES', 'Acceptor SMILES']]

    collector = []
    for i, row in new_df.iterrows():
        emb = luts['Donor'].loc[row['Donor SMILES']]
        emb = pd.concat([emb, luts['Acceptor'].loc[row['Acceptor SMILES']]])
        collector.append(emb)

    final_embeddings = pd.concat(collector, axis=1).T
    pickle.dump(final_embeddings, open(out_file, 'wb'))

    
