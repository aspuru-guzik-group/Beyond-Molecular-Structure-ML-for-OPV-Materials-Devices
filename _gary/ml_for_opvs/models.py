import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_add_pool, global_mean_pool
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.aggr import GraphMultisetTransformer

import gpytorch
from gpytorch.kernels.kernel import default_postprocess_script

### GP regressor ###

# tanimoto distance kernel
def batch_tanimoto_sim(
    x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Tanimoto between two batched tensors, across last 2 dimensions.
    eps argument ensures numerical stability if all zero tensors are added.
    """
    # Tanimoto distance is proportional to (<x, y>) / (||x||^2 + ||y||^2 - <x, y>) where x and y are bit vectors
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2**2, dim=-1, keepdims=True)
    return (dot_prod + eps) / (
        eps + x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod
    )

class BitDistance(torch.nn.Module):
    r"""
    Distance module for bit vector test_kernels.
    """

    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sim(self, x1, x2, postprocess, x1_eq_x2=False, metric="tanimoto"):
        r"""
        Computes the similarity between x1 and x2
        Args:
            :attr: `x1`: (Tensor `n x d` or `b x n x d`):
                First set of data where b is a batch dimension
            :attr: `x2`: (Tensor `m x d` or `b x m x d`):
                Second set of data where b is a batch dimension
            :attr: `postprocess` (bool):
                Whether to apply a postprocess script (default is none)
            :attr: `x1_eq_x2` (bool):
                Is x1 equal to x2
            :attr: `metric` (str):
                String specifying the similarity metric. One of ['tanimoto']
        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the similarity matrix between `x1` and `x2`
        """

        # Branch for Tanimoto metric
        if metric == "tanimoto":
            res = batch_tanimoto_sim(x1, x2)
            res.clamp_min_(0)  # zero out negative values
            return self._postprocess(res) if postprocess else res
        else:
            raise RuntimeError(
                "Similarity metric not supported. Available options are 'tanimoto'"
            )


class TanimotoKernel(gpytorch.kernels.Kernel):
    ''' Tanimoto kernel from FlowMO and GAUCHE
    (https://github.com/leojklarner/gauche/blob/main/gprotorch/kernels/fingerprint_kernels/tanimoto_kernel.py)
    '''
    def __init__(self, metric="tanimoto", **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = metric

    def covar_dist(
        self,
        x1,
        x2,
        last_dim_is_batch=False,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params,
    ):
        r"""
        This is a helper method for computing the bit vector similarity between
        all pairs of points in x1 and x2.
        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?
        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if (
            not self.distance_module
            or self.distance_module._postprocess != dist_postprocess_func
        ):
            self.distance_module = BitDistance(dist_postprocess_func)

        res = self.distance_module._sim(
            x1, x2, postprocess, x1_eq_x2, self.metric
        )

        return res

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)


class GPRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kwargs):
        super(GPRegressor, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if kwargs['kernel'] == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
            )
            self.covar_module.base_kernel.lengthscale = kwargs['lengthscale']
        elif kwargs['kernel'] == 'cosine':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
            self.covar_module.base_kernel.period_length = kwargs['period_length']
        elif kwargs['kernel'] == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[-1], nu=kwargs['nu'])
            )
            self.covar_module.base_kernel.lengthscale = kwargs['lengthscale']
        elif kwargs['kernel'] == 'rff':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RFFKernel(kwargs['num_samples'])
            )
        elif kwargs['kernel'] == 'tanimoto':
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        else:
            raise ValueError('Invalid kernel')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



### GNN ###

class GNNEmbedder(torch.nn.Module):
    def __init__(
            self,
            num_layers,
            gnn_hidden_dim, 
            pool_hidden_dim,
            embed_dim
        ):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_hidden_dim = gnn_hidden_dim
        self.pool_hidden_dim = pool_hidden_dim
        self.embed_dim = embed_dim

        self.gnn = GraphSAGE(
            in_channels = -1,
            hidden_channels = gnn_hidden_dim,
            num_layers = num_layers,
        ) 
        self.pool = GraphMultisetTransformer(
            in_channels = gnn_hidden_dim,
            hidden_channels = pool_hidden_dim, 
            out_channels = embed_dim
        )

    def forward(self, data):
        batch, x, edge_index, edge_attr = (
            data.batch, data.x, data.edge_index, data.edge_attr)

        x = x.to(torch.float)
        res = self.gnn(x, edge_index)
        res = self.pool(res, batch, edge_index=edge_index)  # batch x embed_dim

        return res
    

class GNNEmbedder_Conv(torch.nn.Module):
    def __init__(
            self, 
            num_node_features, 
            num_edge_features, 
            latent_dim,
            embed_dim
        ):
        super().__init__()
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_node_features*latent_dim))
        # conv2_net = nn.Sequential(
        #     nn.Linear(num_edge_features, latent_1),
        #     nn.ReLU(),
        #     nn.Linear(latent_1, latent_1*latent_2))
        self.conv1 = NNConv(num_node_features, latent_dim, conv1_net)
        # self.conv2 = NNConv(latent_1, latent_2, conv2_net)
        self.fc = nn.Linear(latent_dim, embed_dim)

    def forward(self, data):
        batch, x, edge_index, edge_attr = (
            data.batch, data.x, data.edge_index, data.edge_attr)

        x = x.to(torch.float)
        edge_attr = edge_attr.to(torch.float)

        # First graph conv layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # Second graph conv layer
        # x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # pool and run through embedding layer
        x = global_mean_pool(x, batch)
        x = self.fc(F.relu(x))
        return x

class GNNPredictor(torch.nn.Module):
    def __init__(
            self, 
            gnn_donor,
            gnn_acceptor,
            embed_dim,
            output_dim
        ):
        super().__init__()
        self.gnn_donor = gnn_donor
        self.gnn_acceptor = gnn_acceptor
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        self.fc = nn.Linear(embed_dim*2, output_dim)

    def embed_donor(self, data):
        return self.gnn_donor(data)

    def embed_acceptor(self, data):
        return self.gnn_acceptor(data)
    
    def forward(self, donor, acceptor):
        donor = self.embed_donor(donor)
        acceptor = self.embed_acceptor(acceptor)
        combined = torch.cat((donor, acceptor), -1)
        output = self.fc(combined)
        return output



