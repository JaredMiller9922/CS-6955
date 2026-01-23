import numpy as np
import torch
from torch.utils.data import Dataset

class BCDataset(Dataset):
    # Create the dataset
    def __init__(self, obs, actions):
        self.obs = torch.tensor(obs)
        self.actions = torch.tensor(actions)

    # Other required methods for inheritance
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, index):
        return self.obs[index], self.actions[index]
    
class ChunkedBCDataset(Dataset):
    def __init__(self, obs, actions, chunk_size):
        self.chunk_size = chunk_size

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        # last valid t is len(actions) - chunk_size - 1
        self.obs = obs[:-chunk_size]
        self.actions = actions  # keep full; we’ll slice in __getitem__

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        # input: obs at time t
        x = self.obs[index]
        # target: action chunk from t to t+k-1  (shape: [k, act_dim])
        y = self.actions[index:index + self.chunk_size]
        return x, y