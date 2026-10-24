"""ARS data preparation: pitch track -> symbolic svara sequence.

Mirrors the ARS paper's audio-to-symbol pipeline (Sec. 9.3):
  median filter -> change-point segmentation (100 cents, min 4 frames at the
  paper's 11.61 ms hop) -> segment median in cents re tonic -> quantize ->
  collapse repeats.
Pitch tracks from any source are first resampled to the paper's hop so the
frame-based parameters mean the same thing on every dataset.
"""
import re
import unicodedata
import numpy as np

# ---------------------------------------------------------------- config
PAPER_SR, PAPER_HOP = 22050, 256
PAPER_HOP_S = PAPER_HOP / PAPER_SR          # 11.61 ms
CFG = dict(
    hop_s=PAPER_HOP_S,
    median_kernel=5,        # frames; set to the value used in ARS_V2
    cp_threshold_cents=100,
    min_seg_frames=4,       # ~46 ms
    min_events=10,          # paper: drop sequences with <10 symbols
    min_eval_len=15,        # paper: >=15 symbols for held-out reconstruction
    tonic_conf_threshold=0.08,
)

# Saraga solfege convention: lowercase = komal, m = shuddha Ma, M = tivra Ma.
SVARA12 = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
SVARA_CENTS = {s: 100 * i for i, s in enumerate(SVARA12)}

# Svara sets for the six ragas in the paper (Saraga convention).
RAGA_SVARAS = {
    'yaman':      'S R G M P D N',
    'bhairav':    'S r G m P d N',
    'todi':       'S r g M P d N',   # Miyan ki Todi
    'marwa':      'S r G M D N',
    'purvi':      'S r G m M P d N',
    'bhimpalasi': 'S R g m P D n',
}
RAGA_ALIASES = {
    'yaman': ['yaman', 'iman', 'eman'],   # not Yaman Kalyan (uses shuddha Ma)
    'bhairav': ['bhairav', 'bhairava'],
    'todi': ['todi', 'miyankitodi', 'miyantodi'],
    'marwa': ['marwa', 'marva', 'maarva'],
    'purvi': ['purvi', 'poorvi'],
    'bhimpalasi': ['bhimpalasi', 'bhimpalasri', 'bheempalasi', 'bhimpalasa'],
}


def norm_name(name):
    """'Bilāsakhānī tōḍī' -> 'bilasakhanitodi'."""
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z]', '', s.lower())


def raga_key(name):
    """Map a dataset raga label to one of the paper's six ragas, else None."""
    n = norm_name(name)
    for key, aliases in RAGA_ALIASES.items():
        if n in aliases:
            return key
    return None


def raga_alphabet(name):
    k = raga_key(name)
    return RAGA_SVARAS[k].split() if k else None


# The paper writes Yaman's tivra Ma as 'm' (600 cents); Saraga writes it 'M'.
PAPER_TO_SARAGA = {'yaman': {'m': 'M'}}


def paper_to_saraga(seq, raga):
    """Convert a sequence in the paper's symbols to the Saraga convention."""
    table = PAPER_TO_SARAGA.get(raga_key(raga) or raga, {})
    return [table.get(s, s) for s in seq]


# ---------------------------------------------------------------- pitch
def resample_track(times, f0, hop_s=PAPER_HOP_S):
    """Nearest-frame resample of an f0 track onto a uniform grid at hop_s."""
    times = np.asarray(times, float)
    f0 = np.asarray(f0, float)
    if len(times) < 2:
        return np.array([]), np.array([])
    grid = np.arange(times[0], times[-1], hop_s)
    idx = np.searchsorted(times, grid)
    idx = np.clip(idx, 1, len(times) - 1)
    left_closer = (grid - times[idx - 1]) < (times[idx] - grid)
    idx = np.where(left_closer, idx - 1, idx)
    return grid, f0[idx]


def hz_to_cents(f0, tonic_hz):
    f0 = np.asarray(f0, float)
    out = np.full_like(f0, np.nan)
    v = f0 > 0
    out[v] = 1200.0 * np.log2(f0[v] / tonic_hz)
    return out


def median_filter_voiced(cents, k):
    """Median filter applied within voiced runs only (NaN = unvoiced)."""
    if k <= 1:
        return cents.copy()
    from scipy.ndimage import median_filter
    out = cents.copy()
    for s, e in voiced_runs(cents):
        if e - s >= k:
            out[s:e] = median_filter(cents[s:e], size=k, mode='nearest')
    return out


def voiced_runs(cents):
    v = ~np.isnan(cents)
    d = np.diff(np.concatenate([[0], v.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def segment(cents, threshold=100.0, min_frames=4):
    """Change-point segmentation: start a new segment when a frame deviates
    more than `threshold` cents from the running mean of the current segment,
    or when voicing breaks. Returns list of (start, end, median_cents)."""
    segs = []
    for s, e in voiced_runs(cents):
        x = cents[s:e]
        start, total, n = 0, x[0], 1
        for i in range(1, len(x)):
            if abs(x[i] - total / n) > threshold:
                if i - start >= min_frames:
                    segs.append((s + start, s + i, float(np.median(x[start:i]))))
                start, total, n = i, x[i], 1
            else:
                total += x[i]
                n += 1
        if len(x) - start >= min_frames:
            segs.append((s + start, e, float(np.median(x[start:]))))
    return segs


def quantize(cents_list, alphabet=None):
    """Octave-fold and snap to the nearest svara in `alphabet` (default: 12)."""
    alphabet = alphabet or SVARA12
    ref = np.array([SVARA_CENTS[a] for a in alphabet], float)
    out = []
    for c in cents_list:
        pc = c % 1200.0
        d = np.abs(((pc - ref) + 600.0) % 1200.0 - 600.0)   # circular distance
        out.append(alphabet[int(np.argmin(d))])
    return out


def collapse(seq):
    return [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]


def pitch_to_symbols(times, f0, tonic_hz, alphabet=None, cfg=CFG):
    """Full ARS pipeline for one pitch track. Returns (symbols, seg_times)."""
    grid, f = resample_track(times, f0, cfg['hop_s'])
    if len(grid) == 0:
        return [], []
    c = median_filter_voiced(hz_to_cents(f, tonic_hz), cfg['median_kernel'])
    segs = segment(c, cfg['cp_threshold_cents'], cfg['min_seg_frames'])
    syms = quantize([m for _, _, m in segs], alphabet)
    seg_t = [(grid[s], grid[min(e, len(grid) - 1)]) for s, e, _ in segs]
    # collapse repeats, keeping the first segment's start and last segment's end
    out_s, out_t = [], []
    for sym, t in zip(syms, seg_t):
        if out_s and out_s[-1] == sym:
            out_t[-1] = (out_t[-1][0], t[1])
        else:
            out_s.append(sym)
            out_t.append(t)
    return out_s, out_t


# ---------------------------------------------------------------- tonic
def estimate_tonic(f0, bins_per_semitone=10):
    """Histogram-mode tonic estimate on octave-folded voiced pitch, with
    heuristic octave correction to a typical vocal Sa range (100-300 Hz).
    confidence = share of voiced frames within +/-50 cents of the mode.
    Approximates the paper's estimator; replace with the ARS_V2 function
    for exact parity."""
    f0 = np.asarray(f0, float)
    f = f0[f0 > 0]
    if len(f) < 50:
        return np.nan, 0.0
    ref = 55.0
    pc = (1200.0 * np.log2(f / ref)) % 1200.0
    nb = 12 * bins_per_semitone
    h, edges = np.histogram(pc, bins=nb, range=(0, 1200))
    h = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, 'same')[2:-2]
    mode_c = (edges[np.argmax(h)] + edges[np.argmax(h) + 1]) / 2
    dist = np.abs(((pc - mode_c) + 600) % 1200 - 600)
    conf = float(np.mean(dist <= 50))
    tonic = ref * 2 ** (mode_c / 1200.0)
    while tonic < 100:
        tonic *= 2
    while tonic > 300:
        tonic /= 2
    return float(tonic), conf


def cents_error(est_hz, ref_hz, octave_invariant=True):
    c = 1200.0 * np.log2(est_hz / ref_hz)
    return float(abs(((c + 600) % 1200) - 600)) if octave_invariant else float(abs(c))


# ---------------------------------------------------------------- expert phrases
PHRASE_TOKEN = re.compile(r'[SrRgGmMPdDnN]')


BASE7 = {'S': 'S', 'r': 'R', 'R': 'R', 'g': 'G', 'G': 'G', 'm': 'M', 'M': 'M',
         'P': 'P', 'd': 'D', 'D': 'D', 'n': 'N', 'N': 'N'}


def to_base7(seq):
    """Drop komal/tivra distinctions: 12-svara symbols -> 7 base letters.
    Used for Carnatic phrase labels, which are written lowercase without the
    Hindustani komal/shuddha marking (e.g. 'ndmdnsndn')."""
    out = []
    for s in seq:
        b = BASE7.get(s) or BASE7.get(s.upper()) or BASE7.get(s.lower())
        if b:
            out.append(b)
    return collapse(out)


def phrase_tokens(label):
    """Saraga phrase label (e.g. 'rg', 'S R g m') -> svara tokens.
    Returns (tokens, leftover_chars) so unexpected markup can be audited."""
    label = str(label or '')
    toks = PHRASE_TOKEN.findall(label)
    leftover = PHRASE_TOKEN.sub('', label).replace(' ', '')
    return toks, leftover


def edit_distance(a, b):
    """Levenshtein distance between two symbol lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def lcs_len(a, b):
    """Length of the longest common subsequence (order kept, gaps allowed)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def symbols_in_span(symbols, seg_times, start, end):
    """Pipeline symbols whose segment midpoint falls inside [start, end]."""
    out = [s for s, (a, b) in zip(symbols, seg_times) if start <= (a + b) / 2 <= end]
    return collapse(out)


# ---------------------------------------------------------------- stats
def context_coverage(seqs, k, n_symbols):
    seen = set()
    trans = 0
    for s in seqs:
        for i in range(k, len(s)):
            seen.add(tuple(s[i - k:i]))
            trans += 1
    return len(seen), n_symbols ** k, trans
