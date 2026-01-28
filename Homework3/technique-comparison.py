import numpy as np
import matplotlib.pyplot as plt

def bernoulli_reward(rng, p):
    # returns 1 with prob p, else 0
    return 1 if rng.random() < p else 0

def select_eps_greedy(rng, Q, eps):
    k = len(Q)
    if rng.random() < eps:
        return rng.integers(k)
    maxQ = np.max(Q)
    candidates = np.flatnonzero(Q == maxQ)
    return rng.choice(candidates)

def select_softmax(rng, Q, tau):
    # numerically stable softmax
    z = (Q - np.max(Q)) / max(tau, 1e-12)
    probs = np.exp(z)
    probs = probs / probs.sum()
    return rng.choice(len(Q), p=probs)

def select_ucb1(rng, Q, N, t, c=0.5):
    # UCB1: choose any untried arm first
    k = len(Q)
    untried = np.flatnonzero(N == 0)
    if len(untried) > 0:
        return rng.choice(untried)

    # t is 1-indexed time step here (>=1)
    bonus = c * np.sqrt(np.log(t) / N)
    ucb = Q + bonus
    maxU = np.max(ucb)
    candidates = np.flatnonzero(ucb == maxU)
    return rng.choice(candidates)

def run_one_problem(p_arms, steps, algo, rng, eps=0.1, tau=0.1, c=0.5):
    k = len(p_arms)
    Q = np.zeros(k, dtype=float)   # estimated mean reward per arm
    N = np.zeros(k, dtype=int)     # pulls per arm

    rewards = np.zeros(steps, dtype=float)
    optimal = np.zeros(steps, dtype=float)

    a_star = int(np.argmax(p_arms))  # true best arm for this problem

    for t in range(1, steps + 1):  # 1-indexed for log(t)
        if algo == "eps":
            a = select_eps_greedy(rng, Q, eps)
        elif algo == "softmax":
            a = select_softmax(rng, Q, tau)
        elif algo == "ucb1":
            a = select_ucb1(rng, Q, N, t, c=c)
        else:
            raise ValueError("Unknown algo")

        r = bernoulli_reward(rng, p_arms[a])

        # sample-average update
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t - 1] = r
        optimal[t - 1] = 1.0 if a == a_star else 0.0

    return rewards, optimal

def experiment(
    num_problems=100,
    k=5,
    steps=500,
    seed=0,
    eps=0.1,
    tau=0.1,
    c=0.5
):
    rng = np.random.default_rng(seed)

    algos = ["eps", "softmax", "ucb1"]
    avg_reward = {a: np.zeros(steps) for a in algos}
    avg_optimal = {a: np.zeros(steps) for a in algos}

    for i in range(num_problems):
        # Each problem instance: p_a ~ Uniform[0,1]
        p_arms = rng.uniform(0.0, 1.0, size=k)

        # Use separate RNG streams per algorithm so results aren’t identical,
        # but are still reproducible.
        for algo in algos:
            local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            r, opt = run_one_problem(p_arms, steps, algo, local_rng, eps=eps, tau=tau, c=c)
            avg_reward[algo] += r
            avg_optimal[algo] += opt

    for algo in algos:
        avg_reward[algo] /= num_problems
        avg_optimal[algo] /= num_problems

    return avg_reward, avg_optimal

if __name__ == "__main__":
    steps = 500
    avg_reward, avg_optimal = experiment(
        num_problems=100,
        k=5,
        steps=steps,
        seed=42,
        eps=0.1,   # choose ONE epsilon here
        tau=0.1,   # softmax temperature (try 0.05, 0.1, 0.2)
        c=0.5      # UCB exploration constant (try 1.0, 2.0)
    )

    x = np.arange(1, steps + 1)

    plt.figure()
    plt.plot(x, np.cumsum(avg_reward["eps"]) / x, label="ε-greedy (ε=0.1)")
    plt.plot(x, np.cumsum(avg_reward["softmax"]) / x, label="Softmax (τ=0.1)")
    plt.plot(x, np.cumsum(avg_reward["ucb1"]) / x, label="UCB1 (c=0.5)")
    plt.xlabel("Time step")
    plt.ylabel("Average reward (running mean)")
    plt.title("5-Armed Bernoulli Bandit: Reward (avg over 100 problems)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(x, 100 * (np.cumsum(avg_optimal["eps"]) / x), label="ε-greedy (ε=0.1)")
    plt.plot(x, 100 * (np.cumsum(avg_optimal["softmax"]) / x), label="Softmax (τ=0.1)")
    plt.plot(x, 100 * (np.cumsum(avg_optimal["ucb1"]) / x), label="UCB1 (c=0.5)")
    plt.xlabel("Time step")
    plt.ylabel("% Optimal action (running mean)")
    plt.title("5-Armed Bernoulli Bandit: % Optimal (avg over 100 problems)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
