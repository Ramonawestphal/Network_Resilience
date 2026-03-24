import numpy as np
from graph_utils import make_ba_graph
from cascade_env import CascadeRecoveryEnv

def estimate_B_star(policy_fn, G, tau=0.8, B_range=range(1, 11),
                    n_trials=200, alpha=0.2, pfail=0.1):
    results = {}
    for B in B_range:
        ancs = []
        for seed in range(n_trials):
            np.random.seed(seed)
            env = CascadeRecoveryEnv(G, alpha=alpha, pfail=pfail, budget=B)
            state = env.reset()
            done = False
            while not done:
                action = policy_fn(state)
                state, _, done, _ = env.step(action)
            from cascade_env import compute_anc
            ancs.append(compute_anc(env.G, env.active))
        results[B] = (np.mean(ancs), np.std(ancs) / np.sqrt(n_trials))  # mean ± SE
    B_star = min((B for B, (m, _) in results.items() if m >= tau), default=None)
    return B_star, results
