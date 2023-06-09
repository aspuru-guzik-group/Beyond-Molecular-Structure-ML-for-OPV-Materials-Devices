from typing import OrderedDict, Tuple
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


class LSTMModel(nn.Module):
    def __init__(self, config):
        """Instantiates NN linear model with arguments from

        Args:
            config (args): Model Configuration parameters.
        """
        super(LSTMModel, self).__init__()
        self.device: torch.device = torch.device("cuda:0")
        self.embeds: nn.Sequential = nn.Sequential(
            nn.Embedding(config["vocab_size"], config["embedding_size"]),
            nn.Dropout(config["dropout"]),
        )
        self.num_layers: int = config["num_lstm_layers"]
        self.hidden_size: int = config["hidden_size"]
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_lstm_layers"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.linearlayers: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    OrthoLinear(config["hidden_size"], config["hidden_size"]), nn.ReLU()
                )
                for _ in range(config["n_layers"])
            ]
        )

        self.output: nn.Linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: torch.tensor):
        """
        Args:
            x (torch.tensor): Shape[batch_size, input_size]

        Returns:
            _type_: _description_
        """
        CUDA_LAUNCH_BLOCKING = 1
        x: torch.long = x.long()
        embeds: torch.tensor = self.embeds(x)  # [input_size, embedding_size]
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        lstm_output, (hidden_state, cell_state) = self.lstm(embeds, (h_0, c_0))
        for i, layer in enumerate(self.linearlayers):
            hidden_state: torch.tensor = layer(hidden_state)
        output: torch.tensor = self.output(hidden_state)
        output: torch.tensor = torch.squeeze(output, dim=2)
        return output
