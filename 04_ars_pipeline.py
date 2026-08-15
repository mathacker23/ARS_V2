
import numpy as np
import json
import random
from collections import defaultdict

ALPHABET = ['S', 'R', 'G', 'm', 'P', 'D', 'N']
N = len(ALPHABET)
SYM2IDX = {s: i for i, s in enumerate(ALPHABET)}
ALPHA = 0.5         
MASTER_SEED = 20260731  
ORDERS = [1, 2, 3]
MISSING_RATES = [0.10, 0.30, 0.50]
N_MASK_REPEATS = 20   

with open(SEQ_PATH) as f:
    sequences_raw = json.load(f)
sequences = {k: [SYM2IDX[s] for s in v] for k, v in sequences_raw.items() if len(v) >= 15}
names = sorted(sequences.keys())
print(f"Using {len(names)} sequences (>=15 symbols) out of {len(sequences_raw)} extracted")

rng = random.Random(MASTER_SEED)
np_rng = np.random.RandomState(MASTER_SEED)

shuffled = names[:]
rng.shuffle(shuffled)
n_test = max(1, int(round(0.2 * len(shuffled))))
test_names = shuffled[:n_test]
train_names = shuffled[n_test:]
print(f"Train: {len(train_names)} sequences, Test: {len(test_names)} sequences")


def train_markov(train_seqs, k, alpha):
    """Dirichlet posterior mean estimator (paper eq 1)."""
    counts = defaultdict(lambda: np.zeros(N))
    for seq in train_seqs:
        for t in range(k, len(seq)):
            context = tuple(seq[t-k:t])
            counts[context][seq[t]] += 1
    init_counts = defaultdict(lambda: np.zeros(N))
    for seq in train_seqs:
        if len(seq) >= k:
            init_counts[tuple(seq[:k])][0] += 0 

    P = {}
    for context, c in counts.items():
        P[context] = (c + alpha) / (c.sum() + N * alpha)
    return P


def log_prob(P, context, symbol, alpha):
    if context in P:
        return np.log(P[context][symbol])
    else:
        return np.log(1.0 / N)


def exact_reconstruct(seq, mask_positions, P, k, alpha):

    T = len(seq)
    mask_set = set(mask_positions)
    domains = [[seq[t]] if t not in mask_set else list(range(N)) for t in range(T)]

    if T < k:

        return [d[0] for d in domains]

    V = {}
    backptr = {} 

    def admissible_prefixes():
        prefixes = [()]
        for t in range(k):
            new_prefixes = []
            for p in prefixes:
                for a in domains[t]:
                    new_prefixes.append(p + (a,))
            prefixes = new_prefixes
        return prefixes

    init_prefixes = admissible_prefixes()
    layer = {}
    for prefix in init_prefixes:
        layer[prefix] = 0.0
    V[k-1] = layer 

    for t in range(k, T):
        prev_layer = V[t-1]
        new_layer = {}
        bp = {}
        for state, score in prev_layer.items():
            for a in domains[t]:
                new_state = state[1:] + (a,)
                s = score + log_prob(P, state, a, alpha)
                if new_state not in new_layer or s > new_layer[new_state]:
                    new_layer[new_state] = s
                    bp[new_state] = (state, a)
        V[t] = new_layer
        backptr[t] = bp
    final_layer = V[T-1]
    best_state = max(final_layer, key=lambda s: final_layer[s])
    recon = [None] * T
    state = best_state
    for t in range(T-1, k-2, -1):
        if t == k-1:
            for i, sym in enumerate(state):
                recon[t - (k-1) + i] = sym
            break
        prev_state, a = backptr[t][state]
        recon[t] = a
        state = prev_state
    return recon


def missing_symbol_accuracy(true_seq, recon_seq, mask_positions):
    correct = sum(1 for t in mask_positions if recon_seq[t] == true_seq[t])
    return correct / len(mask_positions) if mask_positions else None

results = []
train_seqs = [sequences[n] for n in train_names]

for k in ORDERS:
    P = train_markov(train_seqs, k, ALPHA)
    n_contexts_seen = len(P)
    n_contexts_possible = N ** k
    for miss_rate in MISSING_RATES:
        accs = []
        for test_name in test_names:
            seq = sequences[test_name]
            T = len(seq)
            n_missing = max(1, int(round(miss_rate * T)))
            if n_missing >= T - k:  
                n_missing = max(1, T - k - 1)
            for rep in range(N_MASK_REPEATS):
                mask_positions = np_rng.choice(T, size=n_missing, replace=False).tolist()
                recon = exact_reconstruct(seq, mask_positions, P, k, ALPHA)
                acc = missing_symbol_accuracy(seq, recon, mask_positions)
                if acc is not None:
                    accs.append(acc)
        mean_acc = float(np.mean(accs))
        sd_acc = float(np.std(accs))
        results.append({
            "order": k,
            "missing_rate": miss_rate,
            "mean_accuracy": mean_acc,
            "sd": sd_acc,
            "n_evaluations": len(accs),
            "n_test_sequences": len(test_names),
            "n_contexts_seen": n_contexts_seen,
            "n_contexts_possible": n_contexts_possible,
        })
        print(f"k={k}, missing={miss_rate:.0%}: acc={mean_acc:.3f} (sd={sd_acc:.3f}, n={len(accs)}) "
              f"[contexts seen: {n_contexts_seen}/{n_contexts_possible}]")

uniform_baseline = 1.0 / N
print(f"\nUniform-guess baseline: {uniform_baseline:.3f}")

output = {
    "config": {
        "alphabet": ALPHABET,
        "alpha": ALPHA,
        "master_seed": MASTER_SEED,
        "n_mask_repeats": N_MASK_REPEATS,
        "orders": ORDERS,
        "missing_rates": MISSING_RATES,
        "train_sequences": train_names,
        "test_sequences": test_names,
        "uniform_baseline": uniform_baseline,
    },
    "results": results
}
with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {RESULTS_PATH}")
