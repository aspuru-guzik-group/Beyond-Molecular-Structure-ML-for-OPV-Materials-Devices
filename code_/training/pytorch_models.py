import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch, Data

import gpytorch
from sklearn.base import BaseEstimator
from pytorch_mpnn import smiles2data, DMPNNEncoder


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

    def __init__(self, postprocess_script=lambda x: x):
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
        dist_postprocess_func=lambda x: x,
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


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kwargs):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if kwargs['kernel'] == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
            )
            # self.covar_module.base_kernel.lengthscale = kwargs['lengthscale']
        elif kwargs['kernel'] == 'tanimoto':
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        else:
            raise ValueError('Invalid kernel')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPRegressor(BaseEstimator):

    def __init__(
        self, 
        kernel='rbf',
        lr = 5e-2,

    ):
        self.ll = gpytorch.likelihoods.GaussianLikelihood()
        self.kernel = kernel
        self.lr = lr

    # def set_params(self, **params):
    #     for key, val in params.items():
    #         setattr(self, key, val)

    def fit(self, X_train, Y_train):
    
        n_epoch = 100

        X_train = torch.tensor(X_train, dtype=torch.float)
        Y_train = torch.tensor(Y_train.ravel(), dtype=torch.float)

        self.model = GP(X_train, Y_train.ravel(), self.ll, kernel=self.kernel)

        # train return loss (minimize)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.ll, self.model)

        self.model.train()
        self.ll.train()
        for _ in range(n_epoch):
            optimizer.zero_grad()
            y_pred = self.model(X_train)
            loss = -mll(y_pred, Y_train.ravel())
            print(f'LOSS: {loss.item()}', end='\r')
            loss.backward()
            optimizer.step()

    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float)

        self.model.eval()
        self.ll.eval()
        with torch.no_grad():
            y_pred = self.ll(self.model(X_test)).mean.numpy()

        return y_pred

### GNN ###

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


class GNNPredictor(BaseEstimator):
    def __init__(self,
                 hidden_size=100,
                 depth=2,
                 lr=1e-3):
        
        self.hidden_size = hidden_size
        self.depth = depth
        self.lr = lr

    # def set_params(self, **params):
    #     for key, val in params.items():
    #         setattr(self, key, val)

    def create_data(self, x_donor, x_acceptor, y_train):
        d_graphs = [smiles2data(s) for s in x_donor]
        a_graphs = [smiles2data(s) for s in x_acceptor]
        y_train = torch.tensor(y_train, dtype=torch.float)
        
        dataset = PairDataset(d_graphs, a_graphs, y_train)
        
        self.num_node_features = d_graphs[0].x.shape[-1]
        self.num_edge_features = d_graphs[0].edge_attr.shape[-1]
        self.out_dim = y_train.shape[-1]

    def fit(self, x_train, y_train):
        import pdb; pdb.set_trace()
        self.create_data(x_train, y_train)
        self.model = nn.Sequential(
            DMPNNEncoder(
                self.hidden_size,
                self.num_node_features,
                self.num_edge_features,
                self.depth
            ),
            nn.Linear(self.hidden_size,  self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_dim),
        )

        return
    
    def predict(self, x_test):
        return

