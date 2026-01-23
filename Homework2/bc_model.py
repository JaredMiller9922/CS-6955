import torch
import torch.nn as nn

# This class provides the framework to train the standard behavioral cloning policy
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    # To run the forward pass we simply pass in the observations
    def forward(self,obs):
        return self.net(obs)

# This class provides the framework to train the action chunking behavioral cloning policy
class ChunkedBCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_size, hidden_dim=256):
        super().__init__()
        # Allows us to predict multiple steps in the future
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim * chunk_size),
        )

    def forward(self, obs):
        out = self.net(obs)
        # Reshape so that we know that each element is a time step
        return out.view(-1, self.chunk_size, 1)
