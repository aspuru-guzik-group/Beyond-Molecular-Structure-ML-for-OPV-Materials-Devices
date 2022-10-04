import torch
from torch_geometric.data import Batch

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
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    target = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, target

def get_graphs(graph_list, indices):
    collect = []
    for g in graph_list:
        graphs = []
        for i in indices:
            graphs.append(g[i])
        collect.append(graphs)

    return collect
