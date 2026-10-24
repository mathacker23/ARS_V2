"""ARS evaluation: order-k Dirichlet Markov model + exact constrained MAP decoder.

Implements the paper's core:
  * posterior-mean estimator  P[c,a] = (C(c,a) + alpha) / (C(c) + N*alpha)      (eq. 1)
  * score  l(x) = log pi(x_1..k) + sum_t log P[x_{t-k..t-1}, x_t]               (eq. 4)
  * exact DP over contexts with observation and grammar constraints          (Thm 4.1)
  * BIC(k) = -2 loglik + N^k (N-1) log L                                     (eq. 11)
Contexts are encoded base-N: c = x_{t-k}*N^(k-1) + ... + x_{t-1}.
"""
import itertools
import numpy as np

NEG = -np.inf


# ======================================================================= model
class MarkovK:
    def __init__(self, k, n, alpha=0.5):
        self.k, self.n, self.alpha = k, n, alpha
        self.counts = np.zeros((n ** k, n))
        self.init_counts = np.zeros(n ** k)
        # row c = the k symbols of context c (most significant = oldest)
        self.digits = np.array(list(itertools.product(range(n), repeat=k)))

    def fit(self, seqs):
        """seqs: list of int arrays. Transitions counted within each sequence;
        initial-context distribution uses every k-gram (windows start mid-piece)."""
        k, n = self.k, self.n
        w = n ** np.arange(k - 1, -1, -1)
        for s in seqs:
            s = np.asarray(s)
            if len(s) <= k:
                continue
            ctx = np.lib.stride_tricks.sliding_window_view(s, k) @ w
            np.add.at(self.counts, (ctx[:-1], s[k:]), 1)
            np.add.at(self.init_counts, ctx, 1)
        a = self.alpha
        self.logP = np.log((self.counts + a) / (self.counts.sum(1, keepdims=True) + n * a))
        self.logpi = np.log((self.init_counts + a) / (self.init_counts.sum() + (n ** k) * a))
        return self

    @property
    def n_transitions(self):
        return int(self.counts.sum())

    def contexts_seen(self):
        return int((self.counts.sum(1) > 0).sum())

    def loglik(self, seqs):
        k, n = self.k, self.n
        w = n ** np.arange(k - 1, -1, -1)
        ll = 0.0
        for s in seqs:
            s = np.asarray(s)
            if len(s) <= k:
                continue
            ctx = np.lib.stride_tricks.sliding_window_view(s, k) @ w
            ll += self.logP[ctx[:-1], s[k:]].sum()
        return ll

    def bic(self, seqs):
        L = max(self.n_transitions, 1)
        d = self.n ** self.k * (self.n - 1)
        return -2.0 * self.loglik(seqs) + d * np.log(L)

    def score(self, x):
        """Exact score l(x) of a full int sequence (used by the brute-force check)."""
        k, n = self.k, self.n
        c = 0
        for v in x[:k]:
            c = c * n + v
        s = self.logpi[c]
        for t in range(k, len(x)):
            s += self.logP[c, x[t]]
            c = (c * n + x[t]) % (n ** k)
        return s


# ===================================================================== decoder
def decode(model, obs, allowed=None, trans_mask=None):
    """Exact constrained MAP reconstruction.
    obs:        int array, -1 at missing positions.
    allowed:    optional bool array (n,) -- symbols permitted at MISSING positions
                (e.g. the raga's svaras). Observed symbols are always kept.
    trans_mask: optional bool array (n,n) of permitted first-order transitions
                (hard grammar). None = no grammar.
    Returns int array, or None if the feasible set is empty."""
    k, n = model.k, model.n
    T = len(obs)
    nk = n ** k
    logP = model.logP
    if trans_mask is not None:
        last = np.arange(nk) % n
        logP = np.where(trans_mask[last], logP, NEG)

    def dom(t):
        m = np.zeros(n, bool)
        if obs[t] >= 0:
            m[obs[t]] = True
        else:
            m[:] = True if allowed is None else allowed
        return m

    if T < k:
        raise ValueError('sequence shorter than model order')

    # initial layer: all contexts consistent with the first k positions
    V = model.logpi.copy()
    digits = model.digits
    ok = np.ones(nk, bool)
    for j in range(k):
        ok &= dom(j)[digits[:, j]]
        if trans_mask is not None and j > 0:
            ok &= trans_mask[digits[:, j - 1], digits[:, j]]
    V[~ok] = NEG
    back = []
    nk1 = n ** (k - 1)
    for t in range(k, T):
        d = dom(t)
        cand = V[:, None] + np.where(d[None, :], logP, NEG)        # (nk, n)
        cand = cand.reshape(n, nk1, n)                               # (oldest, mid, a)
        bp = cand.argmax(0)                                          # (mid, a)
        V = np.take_along_axis(cand, bp[None], 0)[0].reshape(nk)     # new ctx = mid*n + a
        back.append(bp.reshape(nk))
    if not np.isfinite(V).any():
        return None
    c = int(V.argmax())
    out = np.empty(T, int)
    for t in range(T - 1, k - 1, -1):
        out[t] = c % n
        old = back[t - k][c]
        c = old * nk1 + c // n
    for j in range(k - 1, -1, -1):
        out[j] = c % n
        c //= n
    return out


def brute_force(model, obs, allowed=None, trans_mask=None):
    """Exhaustive search over the feasible set (small cases only)."""
    n = model.n
    doms = []
    for t in range(len(obs)):
        if obs[t] >= 0:
            doms.append([obs[t]])
        else:
            doms.append([a for a in range(n) if allowed is None or allowed[a]])
    best, arg = NEG, None
    for x in itertools.product(*doms):
        if trans_mask is not None and not all(trans_mask[x[i], x[i + 1]] for i in range(len(x) - 1)):
            continue
        s = model.score(x)
        if s > best:
            best, arg = s, np.array(x)
    return arg, best


# ======================================================================= masks
def make_mask(T, rate, kind, rng):
    """Boolean mask (True = missing). Exactly round(rate*T) positions, >=1.
    uniform:    positions sampled without replacement (paper's MAR mechanism)
    contiguous: one gap
    burst:      several short gaps of length 2-4"""
    m = max(1, int(round(rate * T)))
    mask = np.zeros(T, bool)
    if kind == 'uniform':
        mask[rng.choice(T, m, replace=False)] = True
    elif kind == 'contiguous':
        s = rng.integers(0, T - m + 1)
        mask[s:s + m] = True
    elif kind == 'burst':
        left = m
        while left > 0:
            L = int(min(left, rng.integers(2, 5)))
            free = [s for s in range(T - L + 1) if not mask[max(0, s - 1):s + L + 1].any()]
            if not free:                      # fall back to any unmasked positions
                idx = np.flatnonzero(~mask)
                mask[rng.choice(idx, left, replace=False)] = True
                break
            s = rng.choice(free)
            mask[s:s + L] = True
            left -= L
    else:
        raise ValueError(kind)
    return mask


def windows(seq, L=32, max_windows=5, min_len=15):
    """Test windows from one recording: up to `max_windows` evenly spaced,
    non-overlapping windows of length L; a sequence shorter than L but >= min_len
    is used whole (the paper's >=15-symbol rule)."""
    n = len(seq)
    if n < min_len:
        return []
    if n <= L:
        return [seq]
    starts = np.linspace(0, n - L, min(max_windows, n // L)).round().astype(int)
    return [seq[s:s + L] for s in np.unique(starts)]


# ======================================================================= stats
def cluster_bootstrap_ci(values, groups, n_boot=2000, seed=0, level=0.95):
    """CI for the mean of per-group means, resampling groups (recordings)."""
    df_means = {}
    for v, g in zip(values, groups):
        df_means.setdefault(g, []).append(v)
    gm = np.array([np.mean(v) for v in df_means.values()])
    rng = np.random.default_rng(seed)
    boots = gm[rng.integers(0, len(gm), (n_boot, len(gm)))].mean(1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return gm.mean(), lo, hi, len(gm)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


# ================================================================ experiments
def encode(seq_str, alphabet):
    idx = {s: i for i, s in enumerate(alphabet)}
    return np.array([idx[s] for s in str(seq_str).split() if s in idx], int)


def mask_rng(seed, *keys):
    return np.random.default_rng([int(seed)] + [int(k) for k in keys])


def evaluate_fold(train_seqs, test_items, n, cfg, fold, allowed=None, trans_mask=None,
                  orders=(1, 2, 3), alpha=None, extra=None):
    """Fit k = orders on train, evaluate every test window under every mask.
    test_items: list of dicts with keys rec_id, rec_idx, group, dataset, raga, windows.
    Masks depend only on (seed, fold, recording, window, rate, kind, rep), so every
    model sees the same masks (paired comparison, paper §7.4).
    Returns (rows, fold_info)."""
    alpha = cfg['alpha'] if alpha is None else alpha
    models = {k: MarkovK(k, n, alpha).fit(train_seqs) for k in orders}
    bics = {k: m.bic(train_seqs) for k, m in models.items()}
    k_bic = min(bics, key=bics.get)
    # unigram baseline: most frequent training symbol (among allowed symbols if constrained)
    freq = np.bincount(np.concatenate(train_seqs), minlength=n).astype(float)
    if allowed is not None:
        freq[~allowed] = -1
    uni = int(freq.argmax())
    rows = []
    for it in test_items:
        for w, win in enumerate(it['windows']):
            T = len(win)
            for ri, rate in enumerate(cfg['rates']):
                for ki, kind in enumerate(cfg['mask_kinds']):
                    for rep in range(cfg['masks_per_window']):
                        mask = make_mask(T, rate, kind, mask_rng(cfg['seed'], fold, it['rec_idx'], w, ri, ki, rep))
                        obs = np.where(mask, -1, win)
                        truth = win[mask]
                        base = dict(rec_id=it['rec_id'], group=it['group'], dataset=it['dataset'],
                                    raga=it['raga'], fold=fold, window=w, rate=rate, mask=kind, rep=rep,
                                    n_missing=int(mask.sum()))
                        if allowed is not None:
                            base['truth_in_allowed'] = float(allowed[truth].mean())
                        if extra:
                            base.update(extra)
                        for k, m in models.items():
                            if T <= k:
                                continue
                            x = decode(m, obs, allowed, trans_mask)
                            acc = np.nan if x is None else float((x[mask] == truth).mean())
                            rows.append(dict(base, model=f'k={k}', feasible=x is not None, acc=acc))
                            if k == k_bic:
                                rows.append(dict(base, model='BIC-selected', feasible=x is not None, acc=acc))
                        rows.append(dict(base, model='unigram', feasible=True, acc=float((truth == uni).mean())))
    info = dict(fold=fold, k_bic=k_bic, n_train=len(train_seqs),
                transitions=models[orders[0]].n_transitions,
                **{f'k{k}_contexts_seen': models[k].contexts_seen() for k in orders},
                **{f'k{k}_contexts_possible': n ** k for k in orders},
                **{f'bic_k{k}': bics[k] for k in orders})
    if extra:
        info.update(extra)
    return rows, info


def run_cv(df, seq_col, alphabet, cfg, label, allowed=None, trans_mask=None, orders=(1, 2, 3),
           alpha=None, extra=None):
    """Artist-disjoint GroupKFold cross-validation on one corpus."""
    from sklearn.model_selection import GroupKFold
    n = len(alphabet)
    df = df.reset_index(drop=True)
    seqs = [encode(s, alphabet) for s in df[seq_col]]
    keep = [i for i, s in enumerate(seqs) if len(s) >= cfg['min_events']]
    df, seqs = df.iloc[keep].reset_index(drop=True), [seqs[i] for i in keep]
    groups = df['group'].astype(str).values
    n_groups = len(set(groups))
    if n_groups < 2:
        return [], []
    gkf = GroupKFold(n_splits=min(cfg['n_folds'], n_groups))
    all_rows, infos = [], []
    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups)):
        test_items = []
        for i in te:
            wins = windows(seqs[i], cfg['win_len'], cfg['max_windows'], cfg['min_eval_len'])
            if wins:
                test_items.append(dict(rec_id=df.rec_id[i], rec_idx=i, group=groups[i],
                                       dataset=df.dataset[i], raga=df.raga[i], windows=wins))
        rows, info = evaluate_fold([seqs[i] for i in tr], test_items, n, cfg, fold,
                                   allowed, trans_mask, orders, alpha, extra)
        for r in rows:
            r['experiment'] = label
        info.update(experiment=label, n_test_recordings=len(test_items))
        all_rows += rows
        infos.append(info)
    return all_rows, infos


def summarize(res, by, n_boot=2000, seed=0):
    """Mean of per-recording mean accuracy with a recording-level bootstrap CI."""
    out = []
    for key, sub in res.dropna(subset=['acc']).groupby(by):
        mean, lo, hi, n_rec = cluster_bootstrap_ci(sub.acc.values, sub.rec_id.values, n_boot, seed)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(acc=round(mean, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                   sd_eval=round(sub.acc.std(), 4), recordings=n_rec, evaluations=len(sub),
                   feasible_rate=round(sub.feasible.mean(), 4) if 'feasible' in sub else np.nan)
        out.append(row)
    import pandas as pd
    return pd.DataFrame(out)


def paired_tests(res, by, comparisons):
    """Wilcoxon signed-rank on per-recording mean accuracy for pre-specified model
    pairs, Holm-corrected within each `by` group."""
    import pandas as pd
    from scipy.stats import wilcoxon
    out = []
    for key, sub in res.dropna(subset=['acc']).groupby(by):
        per = sub.groupby(['rec_id', 'model']).acc.mean().unstack()
        block = []
        for a, b in comparisons:
            if a not in per or b not in per:
                continue
            d = (per[a] - per[b]).dropna()
            if len(d) < 6 or np.allclose(d, 0):
                p = np.nan
            else:
                p = wilcoxon(d).pvalue
            block.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                              comparison=f'{a} vs {b}', mean_diff=round(d.mean(), 4),
                              recordings=len(d), share_a_better=round((d > 0).mean(), 3), p=p))
        ps = [r['p'] for r in block]
        valid = [i for i, p in enumerate(ps) if not np.isnan(p)]
        if valid:
            adj = holm([ps[i] for i in valid])
            for j, i in enumerate(valid):
                block[i]['p_holm'] = adj[j]
        out += block
    return pd.DataFrame(out)


def run_by_raga(df, seq_col, cfg, label, alphabet_fn, allowed_fn=None, orders=(1, 2, 3),
                alpha=None, min_recordings=3, log=print):
    """Raga-conditioned evaluation (the paper's setting): a separate model per raga,
    artist-disjoint folds inside each raga. alphabet_fn(raga) -> list of symbols;
    allowed_fn(raga, alphabet) -> bool mask for missing positions, or None."""
    rows, infos, skipped = [], [], []
    for raga, sub in df.groupby('raga'):
        if len(sub) < min_recordings or sub['group'].nunique() < 2:
            skipped.append((raga, len(sub), sub['group'].nunique()))
            continue
        alph = alphabet_fn(raga)
        allowed = allowed_fn(raga, alph) if allowed_fn else None
        r, i = run_cv(sub, seq_col, alph, cfg, label, allowed=allowed, orders=orders,
                      alpha=alpha, extra=dict(raga_model=raga, N=len(alph)))
        rows += r
        infos += i
    log(f'{label}: {len(set(x["raga_model"] for x in infos))} ragas evaluated, '
        f'{len(skipped)} skipped (fewer than {min_recordings} recordings or only one artist)')
    return rows, infos, skipped
