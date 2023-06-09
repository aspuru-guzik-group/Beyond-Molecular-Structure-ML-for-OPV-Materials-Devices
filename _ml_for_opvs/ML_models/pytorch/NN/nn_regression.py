from typing import OrderedDict
import torch
from torch import nn
import numpy as np
import tensorboard
import torch.nn.functional as F


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
    def __init__(self, config):
        """Instantiates NN linear model with arguments from

        Args:
            config (args): Model Configuration parameters.
        """
        super(NNModel, self).__init__()
        self.embeds: nn.Sequential = nn.Sequential(  # ATTN: Defines first two layers: input and embedding
            nn.Linear(config["input_size"], config["embedding_size"]),
            nn.ReLU(),
            OrthoLinear(config["embedding_size"], config["hidden_size"]),
            nn.ReLU(),
        )
        self.linearlayers: nn.ModuleList = nn.ModuleList(  # NOTE: Add n hidden layers same size as embedding layer
            [
                nn.Sequential(
                    OrthoLinear(config["hidden_size"], config["hidden_size"]), nn.ReLU()
                )
                for _ in range(config["n_layers"])
            ]
        )

        self.output: nn.Linear = nn.Linear(config["hidden_size"], config["output_size"])  # NOTE: Final output layer

    def forward(self, x: torch.tensor):
        """
        Args:
            x (torch.tensor): Shape[batch_size, input_size]

        Returns:
            _type_: _description_
        """
        embeds: torch.tensor = self.embeds(x)
        for i, layer in enumerate(self.linearlayers):
            embeds: torch.tensor = layer(embeds)
        output: torch.tensor = self.output(embeds)
        return output
