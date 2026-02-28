import cv2
import gymnasium as gym
import numpy as np

from collections import deque
import time
from tetris_gymnasium.envs.tetris import Tetris

def aggregate_phi_last_k(history, k=20):
    if len(history) == 0:
        return None
    phis = [phi for (phi, _) in list(history)[-k:]]
    return np.mean(phis, axis=0)

def tamer_update(w, phi_bar, h, alpha):
    h_hat = float(np.dot(w, phi_bar))
    err = h - h_hat
    w += alpha * err * phi_bar

    # safety: prevent explosion
    np.clip(w, -1e6, 1e6, out=w)
    return h_hat, err

def compute_column_heights(occ: np.ndarray) -> np.ndarray:
    """
    occ: (H, W) binary occupancy (0/1). Row 0 is top, row H-1 is bottom.
    returns: (W,) column heights in [0, H]
    """
    H, W = occ.shape
    heights = np.zeros(W, dtype=np.int32)

    for c in range(W):
        col = occ[:, c]
        filled = np.flatnonzero(col)  # indices of filled cells from top
        if filled.size == 0:
            heights[c] = 0
        else:
            top_r = filled[0]
            heights[c] = H - top_r

    return heights

def count_holes(occ: np.ndarray) -> int:
    """
    occ: (H, W) binary occupancy (0/1). Row 0 is top, row H-1 is bottom.
    returns: total number of holes.
    """
    H, W = occ.shape
    holes = 0

    for c in range(W):
        col = occ[:, c]
        filled = np.flatnonzero(col)
        if filled.size == 0:
            continue
        top_r = filled[0]
        # holes are empty cells below the topmost filled cell
        holes += int(np.sum(col[top_r:] == 0))

    return holes

def extract_board_features(occ):
    heights = compute_column_heights(occ)
    agg_height = np.sum(heights)
    max_height = np.max(heights)
    bumpiness = np.sum(np.abs(np.diff(heights)))
    holes = count_holes(occ)

    features = np.concatenate([
        heights,                 # 18 dims
        [agg_height],
        [max_height],
        [bumpiness],
        [holes],
    ])

    return features.astype(np.float32)

# Create a one hot encoding of the action index [0,0,1,0,0,0,0,0]
def one_hot_action(a: int, n_actions: int = 8) -> np.ndarray: 
    v = np.zeros(n_actions, dtype=np.float32)
    v[a] = 1.0 
    return v


# Get board features and one hot encoding action and append them
def featurize(obs, action: int) -> np.ndarray:
    board = obs["board"]
    active = obs["active_tetromino_mask"]
    # Occupancy is defined as whether the falling piece occupies the space or a static piece
    occ = ((board > 0) | (active > 0)).astype(np.int32)

    board_feats = extract_board_features(occ)     # shape (d_board,)
    a_feats = one_hot_action(action, 8)           # shape (8,)
    return np.concatenate([board_feats, a_feats]).astype(np.float32)

def select_action(w, obs, epsilon=0.05, n_actions=8):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)

    scores = np.array([np.dot(w, featurize(obs, a)) for a in range(n_actions)], dtype=np.float32)

    # Debug + safety
    if not np.all(np.isfinite(scores)):
        print("WARNING: non-finite scores:", scores)
        print("w finite?", np.all(np.isfinite(w)), "||w||", np.linalg.norm(w))
        # fall back to random action
        return np.random.randint(n_actions)

    m = scores.max()
    best = np.flatnonzero(scores == m)
    if best.size == 0:
        # should never happen if finite, but safe anyway
        return int(np.argmax(scores))
    return int(np.random.choice(best))

def aggregate_phi(history, tf, tmin=0.2, tmax=2.0):
    phis = []
    for phi, t in history:
        dt = tf - t
        if tmin <= dt <= tmax:
            phis.append(phi)
    if not phis:
        return None
    return np.mean(phis, axis=0)
    

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")

    obs, info = env.reset()
    phi_dim = featurize(obs, 0).shape[0]
    w = np.zeros(phi_dim, dtype=np.float32)

    alpha = 0.02          # I recommend lowering to start
    epsilon = 0.20        # higher exploration early
    episodes = 50

    TMAX_KEEP = 2000       # max history length (steps), since we use last-k

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        history = deque(maxlen=TMAX_KEEP)

        done = False
        episode_steps = 0
        ep_reward = 0.0

        while not done:
            a = select_action(w, obs, epsilon=epsilon, n_actions=8)
            phi_taken = featurize(obs, a)

            next_obs, env_r, term, trunc, info = env.step(a)
            env.render()

            done = term or trunc
            episode_steps += 1
            ep_reward += float(env_r)

            history.append((phi_taken, time.time()))

            # feedback every N steps
            if episode_steps % 10 == 0:
                fb = input("feedback [1=good,0=bad,enter=skip]: ").strip()
                if fb in {"1", "0"}:
                    h = 1.0 if fb == "1" else -1.0
                    phi_bar = aggregate_phi_last_k(history, k=10)
                    if phi_bar is not None:
                        h_hat, err = tamer_update(w, phi_bar, h, alpha)
                        print(f"ep={ep} step={episode_steps} h={h:+.0f} pred={h_hat:+.3f} err={err:+.3f} ||w||={np.linalg.norm(w):.3f}")

            obs = next_obs
            cv2.waitKey(1)  # doesn't control gym render, but harmless

        print(f"Episode {ep} finished: steps={episode_steps}, env_return={ep_reward:.2f}, ||w||={np.linalg.norm(w):.3f}")

        # Optional: anneal epsilon
        epsilon = max(0.05, epsilon * 0.95)

    env.close()