import numpy as np
import torch
from torch.utils.data import DataLoader
from bc_utils import ChunkedBCDataset
from bc_model import ChunkedBCPolicy

CHUNK_SIZE = 5

# Load the data
obs = np.load("obs.npy")
acts = np.load("actions.npy")

# Create the dataset
dataset = ChunkedBCDataset(obs, acts, CHUNK_SIZE)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model, decide how to optimize, and decide loss function
model = ChunkedBCPolicy(obs_dim=3, act_dim=1, chunk_size=CHUNK_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

EPOCHS = 20

# Train the policy
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_obs, batch_acts in loader:
        batch_obs = batch_obs.to(device)
        batch_acts = batch_acts.to(device)

        # Forward pass through network
        pred = model(batch_obs)
        loss = loss_fn(pred, batch_acts)

        # Back propegation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.6f}")

torch.save(model.state_dict(), f"bc_policy_k{CHUNK_SIZE}.pt")