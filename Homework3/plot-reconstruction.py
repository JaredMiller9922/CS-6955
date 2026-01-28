import numpy as np
import matplotlib.pyplot as plt

def run_bandit_testbed(
    k=10,
    steps=500,
    runs=2000,
    epsilons=(0.0, 0.01, 0.1),
    seed=0,
):
    rng = np.random.default_rng(seed)
    avg_rewards = {eps: np.zeros(steps, dtype=float) for eps in epsilons}

    for eps in epsilons:
        rewards_sum = np.zeros(steps, dtype=float)

        for _ in range(runs):
            # True action values Q*(a) ~ N(0,1)
            q_star = rng.normal(loc=0.0, scale=1.0, size=k)

            # Estimated value of action
            q_hat = np.zeros(k, dtype=float)
            # Number of times the action has been taken
            n = np.zeros(k, dtype=int)

            for t in range(steps):
                # Epsilon-greedy action selection with random tie-breaking
                if rng.random() < eps:
                    # Take a random action 
                    a = rng.integers(k)
                else:
                    # Take the maximum of our current estimates
                    max_q = np.max(q_hat)
                    # Break ties arbitrarily
                    candidates = np.flatnonzero(q_hat == max_q)
                    a = rng.choice(candidates)

                # Reward R ~ N(Q*(a), 1)  (variance 1 => std 1)
                r = rng.normal(loc=q_star[a], scale=1.0)

                # Sample-average update
                n[a] += 1
                q_hat[a] += (r - q_hat[a]) / n[a]

                rewards_sum[t] += r

        avg_rewards[eps] = rewards_sum / runs

    return avg_rewards

if __name__ == "__main__":
    steps = 500
    runs = 500
    epsilons = (0.0, 0.01, 0.1)

    avg_rewards = run_bandit_testbed(steps=steps, runs=runs, epsilons=epsilons, seed=42)

    x = np.arange(1, steps + 1)
    plt.figure()
    for eps in epsilons:
        label = "greedy (ε=0)" if eps == 0.0 else f"ε-greedy (ε={eps})"
        plt.plot(x, avg_rewards[eps], label=label)

    plt.xlabel("Plays (time steps)")
    plt.ylabel("Average reward")
    plt.title(f"10-Armed Testbed: Average Reward over {steps} Plays (averaged over {runs} runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
