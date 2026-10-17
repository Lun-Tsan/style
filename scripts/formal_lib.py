"""formal_lib.py -- formal stylometric distance measures for the S-vs-T single-turn attribution test.

Pure Python + numpy (+ stdlib zlib/lzma). Data come from style_nb.build_turns() (read-only), with the missing
analyze_chat parser replaced by formal_chatshim (validated: it reproduces every style_nb baseline exactly).

Features: character n-grams, n = 1..3, over the raw text (newlines kept, no normalisation), = style_nb.grams.
Vocabulary: top-N n-grams by frequency in the pooled TRAIN turn texts (ties broken by the n-gram string), so
top-100 is a prefix of top-300 is a prefix of top-1000 and every N is a slice of the 1000-dim count vector.
Relative frequency f_i = count_i / (total number of 1-3-grams in the bag, in- or out-of-vocabulary).
A bag of several turns is "\n".join(turn texts); a profile is the bag of all of one person's TRAIN turns.
Signed score convention everywhere: s(x) = student-likeness; s > thr -> predict S, else T (ties -> T, the
train majority).  For a distance d: s = d(x, T-profile) - d(x, S-profile).
"""
import math
import os
import sys
import time
import lzma
import zlib
from collections import Counter

import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
if SCRATCH not in sys.path:
    sys.path.insert(0, SCRATCH)
import formal_chatshim  # noqa: E402

formal_chatshim.install()
import style_nb as M  # noqa: E402
import formal_addon_defs  # noqa: E402

ACK_RE, PUNCT, core, boot_pair_ci = formal_addon_defs.load()

S, T = M.S, M.T
NS = (100, 300, 1000)
NMAX = 1000
NMAIN = 300
CODECS = ("zlib", "lzma")
ORDERS = ("xy", "yx")
KS = (1, 3, 5, 10, 25, 50)
COUNT_MEAS = ("M1", "M2", "M2b", "M3", "M4")
MAIN = ["M1_N300", "M2_N300", "M2b_N300", "M3_N300", "M4_N300", "M5", "M6_zlib_xy", "M7_N1000"]
LABEL = {
    "M1": "M1 Burrows Delta", "M2": "M2 Cosine Delta", "M2b": "M2b cosine on raw f", "M3": "M3 Min-max (Ruzicka)",
    "M4": "M4 Jensen-Shannon", "M5": "M5 author-LM LLR (NB)", "M6": "M6 NCD", "M7": "M7 Craig's Zeta",
}


def label(key):
    base = key.split("_")[0]
    rest = key[len(base) + 1:]
    return LABEL[base] + (f" [{rest}]" if rest else "")


# ----------------------------------------------------------------------------- data
def load_pools():
    msgs, n_sys, turns = M.build_turns()
    tr_b = [t for t in turns if t["split"] == "train" and t["text_b"]]
    te_b = [t for t in turns if t["split"] == "test" and t["text_b"]]
    te_c = [t for t in te_b if t["short"]]

    def cnt(pool):
        return sum(t["who"] == S for t in pool), sum(t["who"] == T for t in pool)
    assert cnt(tr_b) == (785, 862), cnt(tr_b)
    assert cnt(te_b) == (134, 149), cnt(te_b)
    assert cnt(te_c) == (51, 35), cnt(te_c)
    for t in tr_b + te_b:          # addon.py's agree/acknowledge definition, verbatim
        tb = t["text_b"]
        t["ack"] = bool(tb) and t["short"] and bool(core(tb)) and bool(ACK_RE.match(core(tb)))
    return msgs, n_sys, turns, tr_b, te_b, te_c


def date_blocks(turns, K=5):
    """K contiguous chronological blocks of ~equal size; a calendar date is never split across blocks."""
    n = len(turns)
    cuts = [0]
    for f in range(1, K):
        c = round(n * f / K)
        while c < n and turns[c]["date"] == turns[c - 1]["date"]:
            c += 1
        cuts.append(c)
    cuts.append(n)
    return [list(range(cuts[i], cuts[i + 1])) for i in range(K)]


# ----------------------------------------------------------------------------- features
grams = M.grams    # all char n-grams n = 1..3 (newline is an ordinary character)


def total_grams(L):
    return sum(max(0, L - n + 1) for n in (1, 2, 3))


def top_vocab(texts, N):
    c = Counter()
    for x in texts:
        c.update(grams(x))
    return [g for g, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:N]]


# ----------------------------------------------------------------------------- distances (rows of A vs rows of B)
def _z(F, mu, sd):
    return (F - mu) / sd


def burrows_delta(ZA, ZB):
    # M1 Burrows' Delta (Burrows 2002): Delta(A,B) = (1/n) * sum_i |z_i(A) - z_i(B)|,  z_i = (f_i - mu_i) / sigma_i
    return np.abs(ZA - ZB).mean(1)


def cosine_dist(A, B):
    # M2 Cosine Delta (Smith & Aldridge 2011; Evert et al. 2017): 1 - cos(z(A), z(B)) when A,B are z-vectors.
    # M2b: the same formula on raw relative frequencies.  cos(a,b) = <a,b> / (|a| |b|); a zero vector -> distance 1.
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return 1.0 - cos


def minmax_dist(FA, FB):
    # M3 Min-max / Ruzicka (Koppel & Winter 2014; Kestemont et al. 2016): 1 - sum_i min(f_i(A), f_i(B)) / sum_i max(...)
    mn = np.minimum(FA, FB).sum(1)
    mx = np.maximum(FA, FB).sum(1)
    return np.where(mx > 0, 1.0 - np.divide(mn, mx, out=np.zeros_like(mn), where=mx > 0), 1.0)


def jsd(XA, XB):
    # M4 Jensen-Shannon divergence (Lin 1991), base 2 so JSD in [0, 1]:
    #   P_i = (c_i + 0.5) / (sum_j c_j + 0.5 N) over the N-gram vocabulary (add-0.5 smoothing), same for Q,
    #   JSD(P,Q) = 1/2 KL(P||M) + 1/2 KL(Q||M),  M = (P + Q) / 2,  KL(P||M) = sum_i P_i log2(P_i / M_i)
    N = XA.shape[1]
    P = (XA + 0.5) / (XA.sum(1, keepdims=True) + 0.5 * N)
    Q = (XB + 0.5) / (XB.sum(1, keepdims=True) + 0.5 * N)
    Mx = 0.5 * (P + Q)
    return 0.5 * (P * np.log2(P / Mx)).sum(1) + 0.5 * (Q * np.log2(Q / Mx)).sum(1)


# ----------------------------------------------------------------------------- compression (M6)
def clen(b, codec):
    if codec == "zlib":
        return len(zlib.compress(b, 9))
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 6}]))


def ncd(x, y, cx, cy, codec, order="xy"):
    # M6 NCD (Li et al. 2004; Cilibrasi & Vitanyi 2005): NCD(x,y) = (C(xy) - min(C(x),C(y))) / max(C(x),C(y))
    # order 'xy' = test text first (as specified); 'yx' = reference first (ablation).
    cxy = clen(x + y if order == "xy" else y + x, codec)
    return (cxy - min(cx, cy)) / max(cx, cy)


class NCDRef:
    """10 reference bags per person of >= 3,000 characters, each an independent random sample (without replacement
    within the bag) of that person's TRAIN turns joined by '\n' (seed 0)."""

    def __init__(self, train_turns, seed=0, nbags=10, target=3000, codecs=CODECS):
        rng = np.random.default_rng(seed)
        self.refs = {}
        for P in (S, T):
            pool = [t["text_b"] for t in train_turns if t["who"] == P]
            bags = []
            for _ in range(nbags):
                acc, L = [], -1
                for i in rng.permutation(len(pool)):
                    acc.append(pool[i])
                    L += len(pool[i]) + 1
                    if L >= target:
                        break
                bags.append("\n".join(acc).encode("utf-8"))
            self.refs[P] = bags
        self.cy = {(P, c): [clen(b, c) for b in self.refs[P]] for P in (S, T) for c in codecs}
        self.ref_chars = {P: [len(b.decode("utf-8")) for b in self.refs[P]] for P in (S, T)}

    def dists(self, text, codec, order):
        x = text.encode("utf-8")
        cx = clen(x, codec)
        return {P: float(np.mean([ncd(x, y, cx, cy, codec, order) for y, cy in zip(self.refs[P], self.cy[(P, codec)])]))
                for P in (S, T)}

    def score(self, texts, codec, order):
        out = np.empty(len(texts))
        for i, x in enumerate(texts):
            d = self.dists(x, codec, order)
            out[i] = d[T] - d[S]
        return out


# ----------------------------------------------------------------------------- the fitted model
class Model:
    """Everything fitted on a list of TRAIN turns: vocabulary, profiles, z-reference, NB, Zeta."""

    def __init__(self, train_turns, seed=0, timer=None):
        tm = timer if timer is not None else {}
        t0 = time.perf_counter()
        texts = [t["text_b"] for t in train_turns]
        ys = [t["who"] for t in train_turns]
        self.vocab = top_vocab(texts, NMAX)
        self.vidx = {g: i for i, g in enumerate(self.vocab)}
        self.prof_X, self.prof_tot = {}, {}
        for P in (S, T):
            x, tot = self.count_vec("\n".join(t["text_b"] for t in train_turns if t["who"] == P))
            self.prof_X[P], self.prof_tot[P] = x, tot
        tm["fit: vocabulary + profiles"] = tm.get("fit: vocabulary + profiles", 0) + time.perf_counter() - t0

        # z-score reference population (M1, M2). DEVIATION from canonical Delta: with only two authors the
        # reference is 400 random TRAIN bags of 25 turns (200 per person, seed 0), not a multi-author corpus.
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        R = []
        for P in (S, T):
            pool = [t["text_b"] for t in train_turns if t["who"] == P]
            for _ in range(200):
                ids = rng.choice(len(pool), 25, replace=False)
                x, tot = self.count_vec("\n".join(pool[i] for i in ids))
                R.append(x / tot)
        R = np.array(R)
        self.mu = R.mean(0)
        self.sd = R.std(0, ddof=1)
        tm["fit: z reference (M1/M2)"] = tm.get("fit: z reference (M1/M2)", 0) + time.perf_counter() - t0

        # M5: style_nb.NB as-is; llr table only to score long bags fast (checked equal to NB.margin)
        t0 = time.perf_counter()
        self.nb = M.NB(texts, ys)
        self.prior = self.nb.prior[S] - self.nb.prior[T]
        self.llr = {g: self.nb.loglik(S, g) - self.nb.loglik(T, g) for g in self.nb.vocab}
        tm["fit: NB (M5)"] = tm.get("fit: NB (M5)", 0) + time.perf_counter() - t0

        # M7 Craig's Zeta (Craig & Kinney 2009; Burrows 2007): segments = non-overlapping runs of 10 consecutive
        # TRAIN turns per person (chronological, remainder dropped); p_P(g) = share of P's segments containing g;
        # Z(g) = p_T(g) - p_S(g); score(x) = sum over distinct vocabulary n-grams g of x of Z(g).
        t0 = time.perf_counter()
        self.pseg, self.nseg = {}, {}
        for P in (S, T):
            pool = [t["text_b"] for t in train_turns if t["who"] == P]
            segs = [pool[i:i + 10] for i in range(0, len(pool) - len(pool) % 10, 10)]
            pres = np.array([self.count_vec("\n".join(sg))[0] > 0 for sg in segs], float)
            self.pseg[P] = pres.mean(0)
            self.nseg[P] = len(segs)
        self.Z = self.pseg[T] - self.pseg[S]
        tm["fit: Zeta (M7)"] = tm.get("fit: Zeta (M7)", 0) + time.perf_counter() - t0

    # ------------------------------------------------------------------ counting
    def count_vec(self, text):
        x = np.zeros(NMAX)
        for g, k in Counter(grams(text)).items():
            i = self.vidx.get(g)
            if i is not None:
                x[i] += k
        return x, total_grams(len(text))

    def margin_fast(self, text):
        z = self.prior
        for g, k in Counter(grams(text)).items():
            l = self.llr.get(g)
            if l is not None:
                z += k * l
        return z

    # ------------------------------------------------------------------ distances
    def dist(self, meas, N, XA, tA, XB, tB):
        """Distance between rows of XA and rows of XB (or one row broadcast), vocabulary = top-N."""
        FA = XA[:, :N] / tA[:, None]
        FB = XB[:, :N] / tB[:, None]
        if meas in ("M1", "M2"):
            mask = self.sd[:N] > 0
            ZA = _z(FA[:, mask], self.mu[:N][mask], self.sd[:N][mask])
            ZB = _z(FB[:, mask], self.mu[:N][mask], self.sd[:N][mask])
            return burrows_delta(ZA, ZB) if meas == "M1" else cosine_dist(ZA, ZB)
        if meas == "M2b":
            return cosine_dist(FA, FB)
        if meas == "M3":
            return minmax_dist(FA, FB)
        if meas == "M4":
            return jsd(XA[:, :N], XB[:, :N])
        raise ValueError(meas)

    def scores_from_counts(self, X, tot, margin, timer=None):
        """Student-likeness s for every count-based measure / N variant + M5 + M7 (dict key -> array)."""
        tm = timer if timer is not None else {}
        out = {}
        for meas in COUNT_MEAS:
            t0 = time.perf_counter()
            for N in NS:
                d = {P: self.dist(meas, N, X, tot, self.prof_X[P][None, :], np.array([self.prof_tot[P]])) for P in (S, T)}
                out[f"{meas}_N{N}"] = d[T] - d[S]
            tm[meas] = tm.get(meas, 0) + time.perf_counter() - t0
        out["M5"] = np.asarray(margin, float)
        t0 = time.perf_counter()
        for N in NS:
            out[f"M7_N{N}"] = -((X[:, :N] > 0).astype(float) @ self.Z[:N])
        tm["M7"] = tm.get("M7", 0) + time.perf_counter() - t0
        return out


class Store:
    """Per-turn sufficient statistics so that bags ("\n".join of turns) can be featurised exactly and fast:
    bag n-grams = each turn's internal n-grams + the n-grams that cover a '\n' separator."""

    def __init__(self, model, texts):
        self.m = model
        self.texts = list(texts)
        self.idx, self.cnt = [], []
        self.marg = np.zeros(len(self.texts))
        for j, x in enumerate(self.texts):
            ii, cc, mm = [], [], 0.0
            for g, k in Counter(grams(x)).items():
                i = model.vidx.get(g)
                if i is not None:
                    ii.append(i)
                    cc.append(k)
                l = model.llr.get(g)
                if l is not None:
                    mm += k * l
            self.idx.append(np.array(ii, np.int64))
            self.cnt.append(np.array(cc, float))
            self.marg[j] = mm

    def features(self, bags):
        """bags: list of int sequences (indices into self.texts). Returns X (n, NMAX), tot (n,), NB margin (n,), joined."""
        nb = len(bags)
        rows, idxs, ws = [], [], []
        tot = np.empty(nb)
        mar = np.full(nb, self.m.prior)
        joined = []
        vidx, llr = self.m.vidx, self.m.llr
        for b, ids in enumerate(bags):
            ids = list(ids)
            parts = [self.texts[i] for i in ids]
            s = "\n".join(parts)
            joined.append(s)
            L = len(s)
            tot[b] = total_grams(L)
            for i in ids:
                idxs.append(self.idx[i])
                ws.append(self.cnt[i])
                rows.append(np.full(len(self.idx[i]), b, np.int64))
                mar[b] += self.marg[i]
            if len(parts) > 1:
                seen, pos = set(), 0
                for x in parts[:-1]:
                    pos += len(x)
                    p = pos              # separator position
                    for n in (1, 2, 3):
                        for i0 in range(p - n + 1, p + 1):
                            if i0 >= 0 and i0 + n <= L:
                                seen.add((i0, n))
                    pos += 1
                ci, cw, cr = [], [], []
                for i0, n in seen:
                    g = s[i0:i0 + n]
                    vi = vidx.get(g)
                    if vi is not None:
                        ci.append(vi)
                        cw.append(1.0)
                        cr.append(b)
                    l = llr.get(g)
                    if l is not None:
                        mar[b] += l
                if ci:
                    idxs.append(np.array(ci, np.int64))
                    ws.append(np.array(cw))
                    rows.append(np.array(cr, np.int64))
        r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        ii = np.concatenate(idxs) if idxs else np.zeros(0, np.int64)
        w = np.concatenate(ws) if ws else np.zeros(0)
        X = np.bincount(r * NMAX + ii, weights=w, minlength=nb * NMAX).reshape(nb, NMAX)
        return X, tot, mar, joined


# ----------------------------------------------------------------------------- decision helpers
def predict(s, thr=0.0):
    return np.where(np.asarray(s) > thr, 1, 0)      # 1 = S, 0 = T  (ties -> T)


def tune_threshold(s, y):
    """Threshold maximising balanced accuracy of 'S if s > thr' on (out-of-fold) scores; ties -> higher accuracy,
    then smaller |thr|."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    v = np.unique(s)
    cands = np.concatenate([[v[0] - 1.0], (v[:-1] + v[1:]) / 2.0, [v[-1] + 1.0]])
    P = s[None, :] > cands[:, None]
    nS, nT = (y == 1).sum(), (y == 0).sum()
    rS = (P & (y == 1)).sum(1) / nS
    rT = (~P & (y == 0)).sum(1) / nT
    bal = (rS + rT) / 2
    acc = ((P & (y == 1)).sum(1) + (~P & (y == 0)).sum(1)) / len(y)
    order = np.lexsort((np.abs(cands), -acc, -bal))
    j = order[0]
    return float(cands[j]), float(bal[j]), float(acc[j])


def metrics01(y, p):
    ys = [S if v else T for v in y]
    ps = [S if v else T for v in p]
    return M.metrics(ys, ps)
