# This class if for testing how good our expert is in the environment
import gymnasium as gym
from stable_baselines3 import PPO

ENV_NAME = "Pendulum-v1"

# Create the environment and load the expert
env = gym.make(ENV_NAME, render_mode="human")
model = PPO.load("ppo_pendulum_expert.zip")

obs, info = env.reset()

# Let the expert run for 300 timesteps
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()