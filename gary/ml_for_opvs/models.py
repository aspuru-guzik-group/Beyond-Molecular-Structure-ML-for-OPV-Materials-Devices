import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_add_pool

import gpytorch

### GP regressor ###

class GPRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel = 'rbf'):
        super(GPRegressor, self).__init__(train_x, train_y, likelihood)
        if kernel == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
            )
        elif kernel == ''
            pass
        else:
            raise ValueError('Invalid kernel')

    def forwards(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


### NGBoost ###

# class NGB()

### GNN ###

class GNNEmbedder(torch.nn.Module):
    def __init__(
            self, 
            num_node_features, 
            num_edge_features, 
            output_dim,
            latent_dim,
            embed_dim
        ):
        super().__init__()
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, latent_1),
            nn.ReLU(),
            nn.Linear(latent_1, num_node_features*latent_1))
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, latent_1),
            nn.ReLU(),
            nn.Linear(latent_1, latent_1*latent_2))
        self.conv1 = NNConv(num_node_features, latent_1, conv1_net)
        self.conv2 = NNConv(latent_1, latent_2, conv2_net)
        self.fc = nn.Linear(latent_2, embed_dim)
        self.out = nn.Linear(embed_dim, output_dim)

    def embed(self, data):
        batch, x, edge_index, edge_attr = (
            data.batch, data.x, data.edge_index, data.edge_attr)

        # First graph conv layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # Second graph conv layer
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # pool and run through embedding layer
        x = global_add_pool(x,batch)
        x = self.fc(x)
        return x

    def forward(self, data):
        x = self.embed(data)
        output = self.out(F.relu(x))
        return output

