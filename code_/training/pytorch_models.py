import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from pytorch_mpnn import DMPNNPredictor, RevIndexedData, smiles2data


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
    x1_sum = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2 ** 2, dim=-1, keepdims=True)
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
            lr=1e-2,
    ):
        self.ll = gpytorch.likelihoods.GaussianLikelihood()
        self.kernel = kernel
        self.lr = lr

    def fit(self, X_train, Y_train):

        n_epoch = 100

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()

        X_train = torch.tensor(X_train, dtype=torch.float)
        Y_train = torch.tensor(Y_train.ravel(), dtype=torch.float)

        self.model = GP(X_train, Y_train.ravel(), self.ll, kernel=self.kernel)

        # train return loss (minimize)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.ll, self.model)

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            self.model = self.model.cuda()
            mll = mll.cuda()

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
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()

        X_test = torch.tensor(X_test, dtype=torch.float)

        if torch.cuda.is_available():
            X_test = X_test.cuda()
            self.model = self.model.cuda()
            self.ll = self.ll.cuda()

        self.model.eval()
        self.ll.eval()
        with torch.no_grad():
            y_pred = self.ll(self.model(X_test)).mean.cpu().numpy()

        return y_pred


### GNN ###

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, donor, acceptor, y=None):
        self.donor = donor
        self.acceptor = acceptor
        self.y = y

    def __len__(self):
        return len(self.donor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.y is None:
            return self.donor[idx], self.acceptor[idx]
        return self.donor[idx], self.acceptor[idx], self.y[idx]


def pair_collate(self, data_list):
    # gather batches with targets for dataloader
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    target = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, target


class GNNPredictor(BaseEstimator):
    def __init__(self,
                 hidden_size=55,
                 depth=2,
                 lr=1e-3):

        self.hidden_size = hidden_size
        self.depth = depth
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_data(self, x_train, y_train=None):
        d_graphs = [RevIndexedData(smiles2data(s)) for s in x_train.iloc[:, 0]]
        a_graphs = [RevIndexedData(smiles2data(s)) for s in x_train.iloc[:, 1]]

        if y_train is not None:
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if type(y_train) is not np.ndarray:
                y_train = y_train.to_numpy()
            y_train = torch.tensor(y_train, dtype=torch.float)
            self.dataset = PairDataset(d_graphs, a_graphs, y_train)
        else:
            self.dataset = PairDataset(d_graphs, a_graphs)

        self.num_node_features = d_graphs[0].x.shape[-1]
        self.num_edge_features = d_graphs[0].edge_attr.shape[-1]
        if y_train is not None:
            self.out_dim = y_train.shape[-1]

    def fit(self, x_train, y_train):
        # prepare dataloaders
        self.create_data(x_train, y_train)
        train_ind, val_ind = train_test_split(list(range(len(self.dataset))), test_size=0.1)
        train_loader = DataLoader(torch.utils.data.Subset(self.dataset, train_ind), batch_size=64, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(self.dataset, val_ind), batch_size=64, shuffle=False)

        # make the model
        self.model = DMPNNPredictor(
            self.hidden_size,
            self.num_node_features,
            self.num_edge_features,
            self.depth,
            self.out_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # early stopping
        patience = 6
        count = 0
        best_loss = np.inf

        n_epoch = 100
        for _ in range(n_epoch):
            self.model.train()
            train_loss = 0
            for dg, ag, y in train_loader:
                dg, ag, y = dg.to(self.device), ag.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(dg, ag)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
            print(f'Loss: {train_loss / len(train_loader)}')

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for dg, ag, y in val_loader:
                    dg, ag, y = dg.to(self.device), ag.to(self.device), y.to(self.device)
                    y_pred = self.model(dg, ag)
                    loss = loss_fn(y_pred, y)
                    val_loss += loss.cpu().item()
            val_loss /= len(val_loader)
            print(f'Val loss: {val_loss}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()
                count = 0
            else:
                count += 1

            if count >= patience:
                print(f'Early stopping reached. Best loss: {best_loss}')
                self.model.load_state_dict(best_model)
                break

    def predict(self, x_test):
        self.create_data(x_test)
        loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
        y_collect = []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for dg, ag in loader:
                dg, ag = dg.to(self.device), ag.to(self.device)
                y_pred = self.model(dg, ag)
                y_collect.append(y_pred)

        y_collect = torch.concat(y_collect, axis=0)
        return y_collect.cpu().numpy().reshape(-1)


class OrthoLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class XavierLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class NNModel(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 embedding_size=1024,
                 hidden_size=2048,
                 activation=nn.ReLU,
                 # output_size=1,
                 n_layers=3
                 ):
        """Instantiates NN linear model with arguments from

        Args:
            config (args): Model Configuration parameters.
        """
        super(NNModel, self).__init__()
        self.embeds: nn.Sequential = nn.Sequential(  # Defines first two layers: input and embedding
            nn.Linear(input_size, embedding_size),
            activation(),
            OrthoLinear(embedding_size, hidden_size),
            activation(),
        )
        self.linearlayers: nn.ModuleList = nn.ModuleList(  # Add n hidden layers same size as embedding layer
            [nn.Sequential(OrthoLinear(hidden_size, hidden_size), activation()) for _ in range(n_layers)]
        )

        self.output: nn.Linear = nn.Linear(hidden_size, output_size)  # Final output layer

    def forward(self, x: torch.tensor, **kwargs):
        """
        Args:
            x (torch.tensor): Shape[batch_size, input_size]

        Returns:
            _type_: _description_
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)  # Convert to a PyTorch tensor if it's a NumPy array
        x = x.type(torch.float32)
        embeds: torch.tensor = self.embeds(x)
        for i, layer in enumerate(self.linearlayers):
            embeds: torch.tensor = layer(embeds)
        output: torch.tensor = self.output(embeds)
        return output
