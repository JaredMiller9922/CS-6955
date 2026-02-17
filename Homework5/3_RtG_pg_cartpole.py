# Resolves version error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from core import MLPActorCritic

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    env = gym.make(env_name)

    # replace logits_net with actor-critic (we only train ac.pi for now)
    ac = MLPActorCritic(env.observation_space, env.action_space,
                        hidden_sizes=hidden_sizes)

    # action sampling works for both Discrete and Box
    def get_action(obs):
        pi = ac.pi._distribution(obs)
        a = pi.sample()
        if isinstance(env.action_space, Discrete):
            return int(a.item())
        else:
            return a.numpy()

    # generic logp computation works for both Discrete and Box
    def compute_loss(obs, act, weights):
        _, logp = ac.pi(obs, act)
        return -(logp * weights).mean()

    optimizer = Adam(ac.pi.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs, batch_acts, batch_weights = [], [], []
        batch_rets, batch_lens = [], []

        obs, info = env.reset()
        ep_rews = []
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += list(reward_to_go(ep_rews))

                obs, info = env.reset()
                ep_rews = []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()

        act_dtype = torch.int64 if isinstance(env.action_space, Discrete) else torch.float32

        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=act_dtype),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )

        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # render env (optional)
    render_env = gym.make(env_name, render_mode='human')

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()

        # render 1 episode
        obs, info = render_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            act = ac.act(obs_t)  # numpy, no_grad inside core.py
            if isinstance(render_env.action_space, Discrete):
                act = int(act) if np.isscalar(act) else int(act.item())
            obs, rew, terminated, truncated, info = render_env.step(act)

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
