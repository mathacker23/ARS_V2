"""
Step 4: Run the actual ARS reconstruction pipeline (as specified in the
paper, Sections 3-5, 8, 10) on the real extracted Yaman symbol sequences.

- Order-k Markov model, symmetric Dirichlet(alpha) prior -> posterior mean (eq 1)
- Exact constrained MAP reconstruction via dynamic programming (Section 4, Algorithm 2)
- No grammar constraints applied in this first pass (F(O,R) = all sequences
  consistent with observation only) -- grammar-constrained pass can be added later
- Missing-symbol accuracy (eq 13) as primary metric
- Train/test split at the sequence level; masks generated independently per
  test sequence, uniformly at random (same MAR mechanism as paper's synthetic
  experiment, Section 8.1) at rates 10/30/50%
- Fixed master seed for reproducibility, matching paper's protocol
"""
import numpy as np
import json
import random
from collections import defaultdict

SEQ_PATH = "/home/claude/ars_eval/symbols/sequences.json"
RESULTS_PATH = "/home/claude/ars_eval/results/real_data_results.json"

ALPHABET = ['S', 'R', 'G', 'm', 'P', 'D', 'N']
N = len(ALPHABET)
SYM2IDX = {s: i for i, s in enumerate(ALPHABET)}
ALPHA = 0.5           # Dirichlet concentration, matches paper's synthetic setup
MASTER_SEED = 20260731  # same master seed as paper for continuity
ORDERS = [1, 2, 3]
MISSING_RATES = [0.10, 0.30, 0.50]
N_MASK_REPEATS = 20    # repeated random masks per sequence per condition, for CI

with open(SEQ_PATH) as f:
    sequences_raw = json.load(f)

# Keep only sequences long enough to be useful for held-out testing (need
# room for context + several missing positions even at high order)
sequences = {k: [SYM2IDX[s] for s in v] for k, v in sequences_raw.items() if len(v) >= 15}
names = sorted(sequences.keys())
print(f"Using {len(names)} sequences (>=15 symbols) out of {len(sequences_raw)} extracted")

rng = random.Random(MASTER_SEED)
np_rng = np.random.RandomState(MASTER_SEED)

# ---- Train/test split at sequence level (80/20, matches paper's spirit of
# held-out test sequences distinct from training) ----
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
    # also collect initial-context distribution (paper Appendix E.1) —
    # empirical distribution over the first k symbols, Dirichlet-smoothed
    init_counts = defaultdict(lambda: np.zeros(N))
    for seq in train_seqs:
        if len(seq) >= k:
            init_counts[tuple(seq[:k])][0] += 0  # placeholder; using uniform init below

    P = {}
    for context, c in counts.items():
        P[context] = (c + alpha) / (c.sum() + N * alpha)
    return P


def log_prob(P, context, symbol, alpha):
    if context in P:
        return np.log(P[context][symbol])
    else:
        # unseen context -> pure Dirichlet prior (uniform, since alpha symmetric)
        return np.log(1.0 / N)


def exact_reconstruct(seq, mask_positions, P, k, alpha):
    """
    Exact DP reconstruction (paper Section 4.2, Algorithm 2), no hard grammar
    (F(O,R) = observation-consistent sequences only). Returns reconstructed
    full sequence (list of symbol indices).
    """
    T = len(seq)
    mask_set = set(mask_positions)
    # Domains: D_t = {observed symbol} if t not masked, else full alphabet
    domains = [[seq[t]] if t not in mask_set else list(range(N)) for t in range(T)]

    # DP over contexts of length k. State at time t (0-indexed, after placing
    # symbol t) = tuple of last k symbols. V[state] = best score so far.
    # Initialize: enumerate all admissible length-k prefixes.
    if T < k:
        # sequence shorter than order -- nothing to reconstruct meaningfully
        return [d[0] for d in domains]

    V = {}
    backptr = {}  # backptr[(t, state)] = (prev_state, chosen_symbol)

    # layer for t = k-1 (0-indexed): build all admissible prefixes of length k
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
        # uniform initial-context score (paper allows this; log pi = const,
        # doesn't affect argmax across sequences of fixed length as long as
        # pi is fixed -- use uniform here since no init distribution trained)
        layer[prefix] = 0.0
    V[k-1] = layer  # keyed by t index representing "state after placing symbol at t"

    # propagate forward
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

    # backtrack from best terminal state
    final_layer = V[T-1]
    best_state = max(final_layer, key=lambda s: final_layer[s])
    # reconstruct symbol sequence
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


# ---- Main experiment loop ----
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
            if n_missing >= T - k:  # keep at least k observed anchors
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
