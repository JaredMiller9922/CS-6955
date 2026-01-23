import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

ENV_NAME = "Pendulum-v1"
NUM_EPISODES = 50

# Create your environment and model
env = gym.make(ENV_NAME)
model = PPO.load("ppo_pendulum_expert.zip")

observations = []
actions = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False

    while not done:
        # What action would the expert take
        action, _ = model.predict(obs, deterministic=True)

        # Update our observation list and action list
        observations.append(obs)
        actions.append(action)

        # Actually take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()

# Convert our action list and observation lists into np array
observations = np.array(observations)
actions = np.array(actions)

# Save our demos to be used for later
np.save("obs.npy", observations)
np.save("actions.npy", actions)

print("States Shape: " + str(observations.shape))
print("Actions Shape: " + str(actions.shape))

