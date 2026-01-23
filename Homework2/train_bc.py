import numpy as np
import torch

from torch.utils.data import DataLoader
from bc_utils import BCDataset
from bc_model import BCPolicy

# Load the data
obs = np.load("obs.npy")
actions = np.load("actions.npy")

# Create a Dataset to train the policy network on
dataset = BCDataset(obs,actions)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Use cuda if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model, optimizer, and loss function
model = BCPolicy(obs_dim=3, act_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
loss_fn = torch.nn.MSELoss()

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_obs, batch_actions in loader:
        # Convert to cuda
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)

        # Forward pass through network to calculate loss
        pred = model(batch_obs)
        loss = loss_fn(pred, batch_actions)
        
        # Back propegation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.6f}")

torch.save(model.state_dict(), "bc_policy_k1.pt")