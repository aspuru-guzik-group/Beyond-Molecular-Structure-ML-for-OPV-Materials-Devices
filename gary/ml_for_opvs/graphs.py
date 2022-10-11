import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Batch, Data

from sklearn.preprocessing import OneHotEncoder
import numpy as np

from . import utils
import rdkit.Chem as Chem


e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ]
}

def get_onehot_encoder(smiles_list):
    # return a onehot encoder fitted for the list of smiles
    pt = Chem.GetPeriodicTable()

    vocab = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        for atom in mol.GetAtoms():
            anum = pt.GetElementSymbol(atom.GetAtomicNum())
            if anum not in vocab:
                vocab.append(anum)

    enc = OneHotEncoder(sparse=False)
    enc.fit(np.array([vocab]).T)
    return enc


def from_smiles(smiles, enc):
    ''' Get a PyG graph using simple one-hot encoding of the nodes
    and edges, rather than the built in molecule encoder.
    Requires a vocabulray (use get_atom_types)
    '''
    pt = Chem.GetPeriodicTable()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles('')

    x = []
    for atom in mol.GetAtoms():
        x.append(enc.transform([[pt.GetElementSymbol(atom.GetAtomicNum())]])[0].tolist())
    x = torch.tensor(x, dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 1)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attrs = edge_index[:, perm], edge_attrs[perm]

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attrs, smiles=smiles)
    return graph
    


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, donor, acceptor, y):
        self.donor = donor
        self.acceptor = acceptor
        self.y = y

    def __len__(self):
        return len(self.donor)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.donor[idx], self.acceptor[idx], self.y[idx]

def pair_collate(self, data_list):
    # gather batches with targets for dataloader
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    target = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, target

def get_graphs(graph_list, indices):
    # get specific graphs using indices from a list of graphs
    collect = []
    for g in graph_list:
        graphs = []
        for i in indices:
            graphs.append(g[i])
        collect.append(graphs)

    return collect
