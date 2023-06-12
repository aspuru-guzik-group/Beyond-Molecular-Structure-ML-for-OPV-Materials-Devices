import numpy as np
from ngboost import NGBRegressor
from scipy.spatial.distance import jaccard
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from skopt.space import Categorical, Integer, Real
from xgboost import XGBRegressor


def tanimoto_distance(fp1: np.array, fp2: np.array, **kwargs) -> float:
    """
    Computes the Tanimoto distance between two bit vectors (fp1 and fp2).

    Args:
        fp1, fp2: Numpy 1D array of the fingerprint.

    Returns:
        float: Tanimoto similarity
    """
    return 1 - jaccard(fp1, fp2)


# class TanimotoKernelRidge(KernelRidge):
#     def __init__(self):
#         super().__init__(kernel=tanimoto_distance)
#
#
# class TanimotoGPRegressor(GaussianProcessRegressor):
#     def __init__(self):
#         super().__init__(kernel=tanimoto_distance)


# class OrthoLinear(torch.nn.Linear):
#     def reset_parameters(self):
#         torch.nn.init.orthogonal_(self.weight)
#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)
#
#
# class XavierLinear(torch.nn.Linear):
#     def reset_parameters(self):
#         torch.nn.init.xavier_normal_(self.weight)
#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)
#
#
# class NNModel(nn.Module):
#     def __init__(self, config):
#         """Instantiates NN linear model with arguments from
#
#         Args:
#             config (args): Model Configuration parameters.
#         """
#         super(NNModel, self).__init__()
#         self.embeds: nn.Sequential = nn.Sequential(  # ATTN: Defines first two layers: input and embedding
#             nn.Linear(input_size, embedding_size),
#             nn.ReLU(),
#             OrthoLinear(embedding_size, hidden_size),
#             nn.ReLU(),
#         )
#         self.linearlayers: nn.ModuleList = nn.ModuleList(  # NOTE: Add n hidden layers same size as embedding layer
#             [
#                 nn.Sequential(
#                     OrthoLinear(hidden_size, hidden_size), nn.ReLU()
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#
#         self.output: nn.Linear = nn.Linear(hidden_size, output_size)  # NOTE: Final output layer
#
#     def forward(self, x: torch.tensor, **kwargs):
#         """
#         Args:
#             x (torch.tensor): Shape[batch_size, input_size]
#
#         Returns:
#             _type_: _description_
#         """
#         embeds: torch.tensor = self.embeds(x)
#         for i, layer in enumerate(self.linearlayers):
#             embeds: torch.tensor = layer(embeds)
#         output: torch.tensor = self.output(embeds)
#         return output
#
#
# class NNRegressor(NeuralNetRegressor):
#     def __init__(self, *args, **kwargs):
#         module = NNModel
#         super().__init__(module, *args, **kwargs)


class NNRegressor(MLPRegressor):
    def __init__(self,
                 n_layers=1,
                 n_neurons=100,
                 activation="relu",
                 *,
                 solver="adam",
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 ):
        hidden_layer_sizes = (n_neurons,) * n_layers
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         solver=solver,
                         alpha=alpha,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init,
                         power_t=power_t,
                         max_iter=max_iter,
                         shuffle=shuffle,
                         random_state=random_state,
                         tol=tol,
                         verbose=verbose,
                         warm_start=warm_start,
                         momentum=momentum,
                         nesterovs_momentum=nesterovs_momentum,
                         early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         n_iter_no_change=n_iter_no_change,
                         max_fun=max_fun,
                         )
        self.n_layers = n_layers
        self.n_neurons = n_neurons


regressor_factory: dict[str, type] = {
    "MLR":   LinearRegression,
    "Lasso": Lasso,
    "KRR":   KernelRidge,
    # "KRR":   TanimotoKernelRidge,
    "KNN":   KNeighborsRegressor,
    "SVR":   SVR,
    "RF":    RandomForestRegressor,
    "XGB":   XGBRegressor,
    "HGB":   HistGradientBoostingRegressor,
    "NGB":   NGBRegressor,
    # "GP":    GaussianProcessRegressor,  # ATTN: Don't use this one for multi-output?
    # "NN":    MLPRegressor,  # ATTN: Not actually this one?
    "NN":    NNRegressor,
}


def model_dropna(model_type: str) -> bool:
    """Returns whether the model_type requires dropping NaNs from data."""
    if model_type == "HGB":
        return False
    else:
        return True


regressor_search_space: dict[str, dict] = {
    "MLR":   { "regressor__regressor__fit_intercept": [True, False]},
    "Lasso": { "regressor__regressor__alpha":         Real(1e-3, 1e3, prior="log-uniform"),
               "regressor__regressor__fit_intercept": [True, False],
               "regressor__regressor__selection":     Categorical(["cyclic", "random"]),
              },
    "KRR":   { "regressor__regressor__alpha":  Real(1e-3, 1e3, prior="log-uniform"),
               "regressor__regressor__kernel": Categorical(["rbf", tanimoto_distance]),
               "regressor__regressor__gamma":  Real(1e-3, 1e3, prior="log-uniform"),
              },
    "KNN":   { "regressor__regressor__n_neighbors": Integer(1, 100),
               "regressor__regressor__weights":     Categorical(["uniform", "distance"]),
               "regressor__regressor__algorithm":   Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
               "regressor__regressor__leaf_size":   Integer(1, 100),
               "regressor__regressor__p":           Integer(1, 5),
              },
    "SVR":   { "regressor__regressor__kernel": Categorical(["linear", "rbf", "poly", "sigmoid"]),
               "regressor__regressor__gamma":  Real(1e-3, 1e3, prior="log-uniform"),
              },
    "RF":    { "regressor__regressor__n_estimators":      Integer(50, 2000),
               "regressor__regressor__max_depth":         Integer(10, 10000),
               "regressor__regressor__min_samples_split": Real(0.005, 1),
               "regressor__regressor__min_samples_leaf":  Real(0.005, 1),
               "regressor__regressor__max_features":      Real(0.5, 1.0),
              },
    "XGB":   { "regressor__regressor__n_estimators":      Integer(50, 2000),
               "regressor__regressor__max_depth":         Integer(10, 10000),
               "regressor__regressor__learning_rate":     Real(0.005, 1),
               "regressor__regressor__subsample":         Real(0.5, 1.0),
               "regressor__regressor__colsample_bytree":  Real(0.5, 1.0),
               "regressor__regressor__colsample_bylevel": Real(0.5, 1.0),
              },
    "HGB":   { "regressor__regressor__loss":             Categorical(["absolute_error", "squared_error", "quantile", "poisson"]),
               "regressor__regressor__learning_rate":    Real(0.005, 1),
               "regressor__regressor__max_depth":        Integer(10, 10000),
               "regressor__regressor__min_samples_leaf": Integer(1, 600),
               "regressor__regressor__max_iter":         Integer(50, 2000),
              #  "regressor__regressor__l2_regularization": Real(0.005, 1),
               "regressor__regressor__max_bins":         Integer(10, 255),
               "regressor__regressor__early_stopping":   [True],
               "regressor__regressor__n_iter_no_change": Integer(1, 10),
              },
    "NGB":   { "regressor__regressor__n_estimators":     Integer(50, 2000),
               "regressor__regressor__learning_rate":    Real(0.005, 1),
               "regressor__regressor__minibatch_frac":   Real(0.1, 1),
               "regressor__regressor__minibatch_size":   Integer(1, 100),
              #  "regressor__regressor__Base":             Categorical(["DecisionTreeRegressor", "Ridge", "Lasso",
              #                                            "KernelRidge", "SVR"]),
               "regressor__regressor__natural_gradient": [True, False],
               "regressor__regressor__verbose":          [False],
              },
    # "GP":  { "regressor__regressor__kernel": Categorical([tanimoto_distance]),
    #          "regressor__regressor__n_restarts_optimizer": Integer(0, 5),
    #          "regressor__regressor__normalize_y":         [True],
    #         },
    "NN":    { "regressor__regressor__n_layers":       Integer(1, 20),
               "regressor__regressor__n_neurons":      Integer(1, 100),
               "regressor__regressor__activation":     Categorical(["logistic", "tanh", "relu"]),
              #  "regressor__regressor__solver":             Categorical("lbfgs", "sgd", "adam"),
               "regressor__regressor__alpha":          Real(1e-5, 1e-3, prior="log-uniform"),
               "regressor__regressor__learning_rate":  Categorical(["constant", "invscaling", "adaptive"]),
              #  "regressor__regressor__learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
              #  "regressor__regressor__max_iter":           Integer(50, 500),
               "regressor__regressor__early_stopping": [True, False],
              #  "regressor__regressor__validation_fraction":Real(0.005, 0.5),
              #  "regressor__regressor__beta_1":             Real(0.005, 0.5),
              #  "regressor__regressor__beta_2":             Real(0.005, 0.5),
              #  "regressor__regressor__epsilon":            Real(1e-9, 1e-7),
              },
}
