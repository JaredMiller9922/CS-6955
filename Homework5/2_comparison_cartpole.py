import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews, dtype=np.float32)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, seed=0,
          use_reward_to_go=True, do_render_episode=False):

    # seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.reset(seed=seed)

    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_net = mlp([obs_dim] + hidden_sizes + [n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    # OPTIONAL: render env (separate)
    render_env = gym.make(env_name, render_mode="human") if do_render_episode else None

    total_steps = 0
    log_timesteps = []
    log_returns = []

    def train_one_epoch():
        batch_obs, batch_acts, batch_weights = [], [], []
        batch_rets, batch_lens = [], []

        obs, info = env.reset()
        terminated = truncated = False
        ep_rews = []

        while True:
            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)

            done = terminated or truncated
            if done:
                ep_ret = float(sum(ep_rews))
                ep_len = len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                if use_reward_to_go:
                    batch_weights += list(reward_to_go(ep_rews))
                else:
                    # baseline method: same total return for every timestep in episode
                    batch_weights += [ep_ret] * ep_len

                obs, info = env.reset()
                terminated = truncated = False
                ep_rews = []

                if len(batch_obs) >= batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
            act=torch.as_tensor(np.array(batch_acts), dtype=torch.int64),
            weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item(), batch_rets, batch_lens, len(batch_obs)

    for epoch in range(epochs):
        loss, batch_rets, batch_lens, steps_this_epoch = train_one_epoch()
        total_steps += steps_this_epoch

        avg_ret = float(np.mean(batch_rets))
        log_timesteps.append(total_steps)
        log_returns.append(avg_ret)

        # Render exactly 1 episode after training epoch (optional)
        if render_env is not None:
            obs, info = render_env.reset(seed=seed + epoch)
            terminated = truncated = False
            while not (terminated or truncated):
                with torch.no_grad():
                    act = get_action(torch.as_tensor(obs, dtype=torch.float32))
                obs, rew, terminated, truncated, info = render_env.step(act)

        print(f"epoch: {epoch:3d}\t loss: {loss:.3f}\t return: {avg_ret:.3f}\t ep_len: {np.mean(batch_lens):.1f}")

    env.close()
    if render_env is not None:
        render_env.close()

    return np.array(log_timesteps), np.array(log_returns)


import numpy as np
import matplotlib.pyplot as plt

def run_experiments(env_name="CartPole-v1", epochs=50, batch_size=5000, lr=1e-2, seeds=range(5)):
    # Store runs as list of (timesteps, returns)
    rtg_runs = []
    base_runs = []

    for s in seeds:
        # reward-to-go
        t_rtg, r_rtg = train(env_name=env_name, epochs=epochs, batch_size=batch_size, lr=lr,
                             seed=s, use_reward_to_go=True, do_render_episode=False)
        rtg_runs.append((t_rtg, r_rtg))

        # baseline (full return per timestep)
        t_base, r_base = train(env_name=env_name, epochs=epochs, batch_size=batch_size, lr=lr,
                               seed=s, use_reward_to_go=False, do_render_episode=False)
        base_runs.append((t_base, r_base))

    # Convert to arrays [num_runs, num_epochs]
    # Assumes timesteps align (same batch_size/epochs) — they will here.
    timesteps = rtg_runs[0][0]
    rtg_returns = np.stack([r for (_, r) in rtg_runs], axis=0)
    base_returns = np.stack([r for (_, r) in base_runs], axis=0)

    # Save to disk (so you can include in report / reload later)
    np.savez(
        "pg_compare_runs.npz",
        timesteps=timesteps,
        rtg_returns=rtg_returns,
        base_returns=base_returns,
        seeds=np.array(list(seeds)),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Compute mean and std across seeds
    rtg_mean = rtg_returns.mean(axis=0)
    rtg_std  = rtg_returns.std(axis=0)

    base_mean = base_returns.mean(axis=0)
    base_std  = base_returns.std(axis=0)

    # Plot (no seaborn, no explicit colors)
    plt.figure()
    plt.plot(timesteps, rtg_mean, label="Reward-to-Go")
    plt.fill_between(timesteps, rtg_mean - rtg_std, rtg_mean + rtg_std, alpha=0.2)

    plt.plot(timesteps, base_mean, label="Full Return (no RTG)")
    plt.fill_between(timesteps, base_mean - base_std, base_mean + base_std, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Average Return (per epoch batch)")
    plt.title("Policy Gradient Learning Curves (mean ± std over 5 seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

run_experiments()
