import gymnasium as gym
import torch
from gymnasium.wrappers import TimeLimit
from bc_model import BCPolicy

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("Pendulum-v1", render_mode="human", max_episode_steps=1000)

# Create the model and load in our expert policy
model = BCPolicy(obs_dim=3, act_dim=1).to(device)
model.load_state_dict(torch.load("bc_policy_k1.pt", map_location=device))
model.eval()

obs, info = env.reset()

# Run the expert policy for 300 iterations
for _ in range(1000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        action = model(obs_tensor).cpu().numpy()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()