import time
import cv2
import gymnasium as gym
import numpy as np

# IMPORTANT: registers env
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401


# ---------------------------
# Feature engineering helpers
# ---------------------------

def compute_column_heights(occ: np.ndarray) -> np.ndarray:
    H, W = occ.shape
    heights = np.zeros(W, dtype=np.int32)
    for c in range(W):
        filled = np.flatnonzero(occ[:, c])
        heights[c] = 0 if filled.size == 0 else (H - filled[0])
    return heights


def count_holes(occ: np.ndarray) -> int:
    holes = 0
    _, W = occ.shape
    for c in range(W):
        col = occ[:, c]
        filled = np.flatnonzero(col)
        if filled.size == 0:
            continue
        top_r = filled[0]
        holes += int(np.sum(col[top_r:] == 0))
    return holes


def extract_board_features(occ: np.ndarray) -> np.ndarray:
    heights = compute_column_heights(occ).astype(np.float32)
    agg_height = float(np.sum(heights))
    max_height = float(np.max(heights)) if heights.size else 0.0
    bumpiness = float(np.sum(np.abs(np.diff(heights)))) if heights.size >= 2 else 0.0
    holes = float(count_holes(occ))
    tail = np.array([agg_height, max_height, bumpiness, holes], dtype=np.float32)
    return np.concatenate([heights, tail], dtype=np.float32)


def one_hot_action(a: int, n_actions: int) -> np.ndarray:
    v = np.zeros(n_actions, dtype=np.float32)
    v[a] = 1.0
    return v


def featurize(obs, action: int, n_actions: int) -> np.ndarray:
    board = obs["board"]
    active = obs["active_tetromino_mask"]
    occ = ((board > 0) | (active > 0)).astype(np.int32)
    board_feats = extract_board_features(occ)
    a_feats = one_hot_action(action, n_actions)
    return np.concatenate([board_feats, a_feats]).astype(np.float32)


# ---------------------------
# TAMER model + policy
# ---------------------------

def select_action(w: np.ndarray, obs, epsilon: float, n_actions: int) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))

    scores = np.empty(n_actions, dtype=np.float32)
    for a in range(n_actions):
        scores[a] = float(np.dot(w, featurize(obs, a, n_actions)))

    if not np.all(np.isfinite(scores)):
        return int(np.random.randint(n_actions))

    best = np.flatnonzero(scores == scores.max())
    return int(np.random.choice(best)) if best.size else int(np.argmax(scores))


def tamer_update(w: np.ndarray, phi_bar: np.ndarray, h: float, alpha: float):
    h_hat = float(np.dot(w, phi_bar))
    err = float(h - h_hat)
    err = float(np.clip(err, -5.0, 5.0))
    w += alpha * err * phi_bar
    np.clip(w, -1e6, 1e6, out=w)
    return h_hat, err


# ---------------------------
# Rendering + critique (OpenCV)
# ---------------------------

# During pause:
#   g => +1 (good placement)
#   b => -1 (bad placement)
#   space/enter => skip
#   q => quit
KEY_TO_H = {ord("g"): +1.0, ord("b"): -1.0}

def render_and_get_key(env, window="Tetris", delay_ms=1) -> int:
    frame = env.render()  # RGB uint8 HxWx3
    if frame is not None:
        cv2.imshow(window, frame[..., ::-1])  # RGB -> BGR
    return cv2.waitKey(delay_ms) & 0xFF


def wait_for_placement_feedback(env, window="Tetris") -> str:
    """Blocks only at placement time."""
    while True:
        key = render_and_get_key(env, window=window, delay_ms=30)
        if key == ord("q"):
            return "quit"
        if key == ord("g"):
            return "good"
        if key == ord("b"):
            return "bad"
        if key in (32, 13):  # space or enter
            return "skip"


def locked_mask(obs) -> np.ndarray:
    """
    Returns a boolean mask for *locked* cells only.
    Key idea: board includes active piece, so remove it using active mask.
    """
    board = obs["board"]
    active = obs["active_tetromino_mask"]
    return ((board > 0) & (active == 0))


# ---------------------------
# Main
# ---------------------------

def main():
    ENV_ID = "tetris_gymnasium/Tetris"
    N_ACTIONS = 8

    alpha = 0.02
    epsilon = 0.30
    epsilon_decay = 0.95
    epsilon_min = 0.05
    episodes = 50

    STEP_SLEEP = 0.03  # normal falling speed (no pausing here)

    env = gym.make(ENV_ID, render_mode="rgb_array")

    print("Controls (CLICK the Tetris window so it has focus):")
    print("  When a piece LOCKS (pause happens):")
    print("    g = good, b = bad, space/enter = skip, q = quit\n")

    obs, info = env.reset()
    phi_dim = featurize(obs, 0, N_ACTIONS).shape[0]
    w = np.zeros(phi_dim, dtype=np.float32)

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            done = False

            # track locked board only
            prev_locked = locked_mask(obs).copy()

            piece_traj = []
            placements = 0
            updates = 0
            ep_steps = 0
            ep_env_return = 0.0

            while not done:
                # render + allow quit anytime (DO NOT PAUSE here)
                key = render_and_get_key(env, delay_ms=1)
                if key == ord("q"):
                    print("Quitting.")
                    return

                # act
                a = select_action(w, obs, epsilon, N_ACTIONS)
                phi_taken = featurize(obs, a, N_ACTIONS)

                next_obs, env_r, term, trunc, info = env.step(a)
                done = bool(term or trunc)
                ep_steps += 1
                ep_env_return += float(env_r)

                # collect micro-actions for this piece
                piece_traj.append(phi_taken)

                # placement detection: locked mask changed (piece locked or lines cleared)
                next_locked = locked_mask(next_obs)
                placed = not np.array_equal(next_locked, prev_locked)
                prev_locked = next_locked.copy()

                if placed:
                    placements += 1

                    phi_bar = np.mean(np.stack(piece_traj, axis=0), axis=0) if piece_traj else None
                    piece_traj = []

                    verdict = wait_for_placement_feedback(env)
                    if verdict == "quit":
                        print("Quitting.")
                        return

                    if verdict != "skip" and phi_bar is not None:
                        h = +1.0 if verdict == "good" else -1.0
                        h_hat, err = tamer_update(w, phi_bar, h, alpha)
                        updates += 1
                        print(
                            f"ep={ep} place={placements} step={ep_steps} "
                            f"verdict={verdict} h={h:+.0f} pred={h_hat:+.3f} err={err:+.3f} "
                            f"||w||={np.linalg.norm(w):.3f}"
                        )
                    else:
                        print(f"ep={ep} place={placements} step={ep_steps} verdict=skip")

                obs = next_obs
                time.sleep(STEP_SLEEP)

            print(
                f"Episode {ep} done | steps={ep_steps} env_return={ep_env_return:.2f} "
                f"placements={placements} updates={updates} epsilon={epsilon:.3f} ||w||={np.linalg.norm(w):.3f}"
            )
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()