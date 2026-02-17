# Resolves version error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from core import MLPActorCritic


def discount_cumsum(x, discount):
    """Compute discounted cumulative sums of vectors."""
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        y[i] = running
    return y


def gae_advantages(rews, vals, gamma=1, lam=0.97):
    """
    GAE-Lambda advantage estimates and discounted returns (targets for V).

    rews: length T
    vals: length T+1 (bootstrap value appended)
    """
    rews = np.asarray(rews, dtype=np.float32)
    vals = np.asarray(vals, dtype=np.float32)

    deltas = rews + gamma * vals[1:] - vals[:-1]          # length T
    adv = discount_cumsum(deltas, gamma * lam)            # length T
    rtg = discount_cumsum(rews, gamma)                    # length T
    return adv, rtg


def train(env_name="CartPole-v1",
          hidden_sizes=(64, 64),
          lr=3e-4,
          epochs=50,
          batch_size=5000,
          gamma=0.99,
          lam=0.97,
          train_v_iters=80,
          render=False,
          seed=0):

    # --- envs ---
    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)
    render_env = gym.make(env_name, render_mode="human") if render else None

    # --- actor-critic (pi + v) ---
    ac = MLPActorCritic(env.observation_space, env.action_space,
                        hidden_sizes=hidden_sizes)

    # two optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=lr)

    # action sampler (works for discrete + continuous)
    def get_action(obs_t):
        pi = ac.pi._distribution(obs_t)
        a_t = pi.sample()
        if isinstance(env.action_space, Discrete):
            return int(a_t.item())
        else:
            return a_t.numpy()

    # losses
    def compute_loss_pi(obs, act, adv):
        _, logp = ac.pi(obs, act)
        return -(logp * adv).mean()

    def compute_loss_v(obs, rtg):
        v = ac.v(obs)
        return ((v - rtg) ** 2).mean()

    def train_one_epoch():
        batch_obs, batch_acts = [], []
        batch_advs, batch_rtgs = [], []
        batch_rets, batch_lens = [], []

        obs, info = env.reset()
        ep_rews, ep_vals = [], []

        while True:
            batch_obs.append(obs.copy())

            obs_t = torch.as_tensor(obs, dtype=torch.float32)

            # value prediction for GAE (no grad needed during collection)
            with torch.no_grad():
                v = ac.v(obs_t).item()
            ep_vals.append(v)

            # sample action
            act = get_action(obs_t)

            # step
            next_obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            obs = next_obs

            if done:
                # bootstrap value if truncated (time-limit), else 0 at terminal
                if truncated:
                    with torch.no_grad():
                        last_val = ac.v(torch.as_tensor(obs, dtype=torch.float32)).item()
                else:
                    last_val = 0.0

                ep_vals.append(last_val)  # now length T+1

                adv, rtg = gae_advantages(ep_rews, ep_vals, gamma=gamma, lam=lam)
                batch_advs += list(adv)
                batch_rtgs += list(rtg)

                batch_rets.append(float(sum(ep_rews)))
                batch_lens.append(len(ep_rews))

                obs, info = env.reset()
                ep_rews, ep_vals = [], []

                if len(batch_obs) >= batch_size:
                    break

        # tensors
        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        adv_tensor = torch.as_tensor(np.array(batch_advs), dtype=torch.float32)
        rtg_tensor = torch.as_tensor(np.array(batch_rtgs), dtype=torch.float32)

        # actions tensor: dtype depends on space
        if isinstance(env.action_space, Discrete):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.int64)
        else:
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.float32)

        # normalize advantages (recommended)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # --- policy update ---
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(obs_tensor, act_tensor, adv_tensor)
        loss_pi.backward()
        pi_optimizer.step()

        # --- value function updates ---
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(obs_tensor, rtg_tensor)
            loss_v.backward()
            vf_optimizer.step()

        return float(loss_pi.item()), float(loss_v.item()), batch_rets, batch_lens

    for epoch in range(epochs):
        loss_pi, loss_v, rets, lens = train_one_epoch()

        # render exactly 1 episode after each epoch (optional)
        if render_env is not None:
            obs, info = render_env.reset(seed=seed + epoch)
            terminated = truncated = False
            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                act = ac.act(obs_t)  # numpy; core.py uses no_grad inside

                if isinstance(render_env.action_space, Discrete):
                    act = int(act) if np.isscalar(act) else int(act.item())

                obs, rew, terminated, truncated, info = render_env.step(act)

        print(f"epoch: {epoch:3d}\t loss_pi: {loss_pi:8.3f}\t loss_v: {loss_v:8.3f}\t"
              f" return: {np.mean(rets):7.2f}\t ep_len: {np.mean(lens):6.1f}")

    env.close()
    if render_env is not None:
        render_env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(env_name=args.env_name,
          lr=args.lr,
          epochs=args.epochs,
          batch_size=args.batch_size,
          render=args.render,
          seed=args.seed)
