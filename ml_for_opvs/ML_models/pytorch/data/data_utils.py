import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PolymerDataset(Dataset):
    def __init__(
        self,
        data_array: np.ndarray,
        target_array: np.ndarray,
        random_state: int,
    ):
        """_summary_

        Args:
            data_path (str): _description_
            feature_names (str): _description_
            target_names (str): _description_
            random_state (int): _description_
        """
        self.data: torch.tensor = torch.from_numpy(data_array)
        self.target: torch.tensor = torch.from_numpy(target_array)
        self.random_state: int = random_state

    def __len__(self):
        """
        Returns the length of the dataset (i.e. the number of molecules).

        Returns:
            The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        Args:
            idx (int): An index or a slice object.
        Returns:
            A PolymerDatapoint if an int is provided or a list of PolymerDatapoint if a slice is provided.
        """
        return self.data[idx], self.target[idx]
