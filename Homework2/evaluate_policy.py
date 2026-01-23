import argparse
import numpy as np
import gymnasium as gym
import torch

from bc_model import BCPolicy, ChunkedBCPolicy  # make sure ChunkedBCPolicy exists if you use chunked

def eval_bc_k1(policy_path, episodes=20, max_steps=1000, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("Pendulum-v1", max_episode_steps=max_steps)
    env.reset(seed=seed)

    model = BCPolicy(obs_dim=3, act_dim=1).to(device)
    model.load_state_dict(torch.load(policy_path, map_location=device))
    model.eval()

    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = model(obs_t).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)

    env.close()
    returns = np.array(returns)
    return returns.mean(), returns.std()


def eval_bc_chunked(policy_path, chunk_size, episodes=20, max_steps=1000, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("Pendulum-v1", max_episode_steps=max_steps)
    env.reset(seed=seed)

    model = ChunkedBCPolicy(obs_dim=3, act_dim=1, chunk_size=chunk_size).to(device)
    model.load_state_dict(torch.load(policy_path, map_location=device))
    model.eval()

    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0

        action_buffer = []

        while not done:
            if len(action_buffer) == 0:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_seq = model(obs_t)[0].cpu().numpy()  # shape (k, 1)
                action_buffer = list(action_seq)

            action = action_buffer.pop(0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated

        returns.append(ep_ret)

    env.close()
    returns = np.array(returns)
    return returns.mean(), returns.std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["k1", "chunked", "both"],
        default="both",
        help="Which policy to evaluate (default: both)"
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to BC k=1 policy")
    parser.add_argument("--chunked_policy", type=str, default=None,
                        help="Path to chunked BC policy (required if mode=chunked or both)")
    parser.add_argument("--k", type=int, default=5,
                        help="Chunk size for chunked BC")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    if args.mode in ["k1", "both"]:
        mean_k1, std_k1 = eval_bc_k1(
            args.policy, args.episodes, args.max_steps
        )
        print(f"BC k=1     | mean return: {mean_k1:.2f} ± {std_k1:.2f}")

    if args.mode in ["chunked", "both"]:
        if args.chunked_policy is None:
            raise ValueError("Must provide --chunked_policy for chunked evaluation")

        mean_k, std_k = eval_bc_chunked(
            args.chunked_policy, args.k, args.episodes, args.max_steps
        )
        print(f"BC k={args.k:<3} | mean return: {mean_k:.2f} ± {std_k:.2f}")

    if args.mode == "both":
        print("\nComparison:")
        delta = mean_k - mean_k1
        print(f"  Δ return (k={args.k} − k=1): {delta:+.2f}")
