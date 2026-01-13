import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import gym
env = gym.make('MountainCar-v0')

env.reset()

done = False
while not done:
    # For each time step policy is a random action
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    
env.close()
