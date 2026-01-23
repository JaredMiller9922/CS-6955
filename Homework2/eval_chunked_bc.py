import gymnasium as gym
import torch
from bc_model import ChunkedBCPolicy

CHUNK_SIZE = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("Pendulum-v1", render_mode="human", max_episode_steps=1000)

model = ChunkedBCPolicy(obs_dim=3, act_dim=1, chunk_size=CHUNK_SIZE).to(device)
model.load_state_dict(torch.load(f"bc_policy_k{CHUNK_SIZE}.pt", map_location=device))
model.eval()

obs, info = env.reset()

action_buffer = []

for _ in range(1000):
    if len(action_buffer) == 0:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_seq = model(obs_tensor)[0].cpu().numpy()
        action_buffer = list(action_seq)

    action = action_buffer.pop(0)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        action_buffer = []

env.close()
