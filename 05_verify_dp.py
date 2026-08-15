
import numpy as np
import itertools
import sys
from importlib import import_module

import importlib.util


N = 3
ALPHABET_TEST = ['a', 'b', 'c']

def train_markov_test(train_seqs, k, alpha):
    from collections import defaultdict
    counts = defaultdict(lambda: np.zeros(N))
    for seq in train_seqs:
        for t in range(k, len(seq)):
            context = tuple(seq[t-k:t])
            counts[context][seq[t]] += 1
    P = {}
    for context, c in counts.items():
        P[context] = (c + alpha) / (c.sum() + N * alpha)
    return P

def log_prob_test(P, context, symbol, alpha):
    if context in P:
        return np.log(P[context][symbol])
    return np.log(1.0/N)

def exact_reconstruct_test(seq, mask_positions, P, k, alpha):
    T = len(seq)
    mask_set = set(mask_positions)
    domains = [[seq[t]] if t not in mask_set else list(range(N)) for t in range(T)]
    if T < k:
        return [d[0] for d in domains]
    def admissible_prefixes():
        prefixes = [()]
        for t in range(k):
            new_prefixes = []
            for p in prefixes:
                for a in domains[t]:
                    new_prefixes.append(p + (a,))
            prefixes = new_prefixes
        return prefixes
    V = {}
    backptr = {}
    init_prefixes = admissible_prefixes()
    layer = {p: 0.0 for p in init_prefixes}
    V[k-1] = layer
    for t in range(k, T):
        prev_layer = V[t-1]
        new_layer, bp = {}, {}
        for state, score in prev_layer.items():
            for a in domains[t]:
                new_state = state[1:] + (a,)
                s = score + log_prob_test(P, state, a, alpha)
                if new_state not in new_layer or s > new_layer[new_state]:
                    new_layer[new_state] = s
                    bp[new_state] = (state, a)
        V[t] = new_layer
        backptr[t] = bp
    final_layer = V[T-1]
    best_state = max(final_layer, key=lambda s: final_layer[s])
    recon = [None]*T
    state = best_state
    for t in range(T-1, k-2, -1):
        if t == k-1:
            for i, sym in enumerate(state):
                recon[t-(k-1)+i] = sym
            break
        prev_state, a = backptr[t][state]
        recon[t] = a
        state = prev_state
    return recon

def brute_force_reconstruct(seq, mask_positions, P, k, alpha):
    T = len(seq)
    mask_set = set(mask_positions)
    domains = [[seq[t]] if t not in mask_set else list(range(N)) for t in range(T)]
    best_score = -np.inf
    best_seq = None
    for candidate in itertools.product(*domains):
        score = 0.0
        for t in range(k, T):
            context = tuple(candidate[t-k:t])
            score += log_prob_test(P, context, candidate[t], alpha)
        if score > best_score:
            best_score = score
            best_seq = candidate
    return list(best_seq), best_score

def dp_score(seq, recon, P, k, alpha):
    score = 0.0
    for t in range(k, len(seq)):
        context = tuple(recon[t-k:t])
        score += log_prob_test(P, context, recon[t], alpha)
    return score

rng = np.random.RandomState(42)
n_tests = 30
n_pass = 0
for test_i in range(n_tests):
    T = rng.randint(6, 10)
    k = rng.choice([1, 2])
    seq = rng.randint(0, N, size=T).tolist()
    train_seqs = [rng.randint(0, N, size=rng.randint(8,15)).tolist() for _ in range(5)]
    P = train_markov_test(train_seqs, k, 0.5)
    n_missing = rng.randint(1, max(2, T - k - 1))
    mask_positions = rng.choice(T, size=min(n_missing, T), replace=False).tolist()

    dp_recon = exact_reconstruct_test(seq, mask_positions, P, k, 0.5)
    dp_sc = dp_score(seq, dp_recon, P, k, 0.5)

    bf_recon, bf_sc = brute_force_reconstruct(seq, mask_positions, P, k, 0.5)
    match_score = abs(dp_sc - bf_sc) < 1e-9
    if match_score:
        n_pass += 1
    else:
        print(f"MISMATCH test {test_i}: T={T} k={k} mask={mask_positions}")
        print(f"  DP score={dp_sc:.6f} recon={dp_recon}")
        print(f"  BF score={bf_sc:.6f} recon={bf_recon}")

print(f"\n{n_pass}/{n_tests} tests passed (DP optimal score == brute-force optimal score)")
