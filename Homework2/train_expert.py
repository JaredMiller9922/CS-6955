import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_NAME = "Pendulum-v1"


# Create 4 different environments for faster training
env = make_vec_env(ENV_NAME, n_envs = 4)

model = PPO(
    # Use a simple multi-layer perceptron
    "MlpPolicy",
    env,
    verbose = 1,
    learning_rate = 3e-4,
    n_steps = 2048,
    batch_size = 64,
    gamma = 0.99,
)

print("Training the expert...")
model.learn(total_timesteps=200_000)

model.save("ppo_pendulum_expert")
env.close()