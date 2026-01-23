import gymnasium as gym

ENV_NAME = "Pendulum-v1"

env = gym.make(ENV_NAME, render_mode="human")
obs, info = env.reset()
print(env.observation_space)
print(env.action_space)

for _ in range(200):
    # Sample a random action
    action = env.action_space.sample()

    # Take that random action
    obs, reward, terminated, truncated, info = env.step(action)

    # Terminated: agent reached terminal state | Trunacated: agent ran out of time
    if terminated or truncated:
        obs, info = env.reset()

env.close()