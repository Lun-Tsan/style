"""Frequency profiles, z-reference, NB and the formal distances (port of the validated formal_lib.py).

Score convention everywhere (as validated): s(x) = partner-likeness; for a distance d, s = d(x, T) - d(x, S);
for NB, s = margin = log P(S|x) - log P(T|x).  Larger s -> more like S (partner); smaller -> more like T (target).
Keys: M1 Burrows Delta, M2 Cosine Delta, M2b cosine on raw f, M3 Min-max, M4 JSD, M5 NB LLR (M6 NCD lives in ncd.py).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .. import config as C
from .author_lm import NB
from .ngrams import grams, top_vocab, total_grams

S, T = C.PARTNER, C.TARGET
NMAX = C.VOCAB_MAX
NMAIN = C.VOCAB_N
COUNT_MEAS = ("M1", "M2", "M2b", "M3", "M4")


def _z(F, mu, sd):
    return (F - mu) / sd


def burrows_delta(ZA, ZB):
    return np.abs(ZA - ZB).mean(1)


def cosine_dist(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return 1.0 - cos


def minmax_dist(FA, FB):
    mn = np.minimum(FA, FB).sum(1)
    mx = np.maximum(FA, FB).sum(1)
    return np.where(mx > 0, 1.0 - np.divide(mn, mx, out=np.zeros_like(mn), where=mx > 0), 1.0)


def jsd(XA, XB):
    N = XA.shape[1]
    P = (XA + 0.5) / (XA.sum(1, keepdims=True) + 0.5 * N)
    Q = (XB + 0.5) / (XB.sum(1, keepdims=True) + 0.5 * N)
    Mx = 0.5 * (P + Q)
    return 0.5 * (P * np.log2(P / Mx)).sum(1) + 0.5 * (Q * np.log2(Q / Mx)).sum(1)


class Model:
    """Everything fitted on a list of training texts with speaker labels: vocabulary, profiles, z-reference, NB, Zeta."""

    def __init__(self, texts, whos, seed: int = 0):
        texts, whos = list(texts), list(whos)
        self.vocab = top_vocab(texts, NMAX)
        self.vidx = {g: i for i, g in enumerate(self.vocab)}
        self.prof_X, self.prof_tot = {}, {}
        pools = {P: [x for x, w in zip(texts, whos) if w == P] for P in (S, T)}
        for P in (S, T):
            self.prof_X[P], self.prof_tot[P] = self.count_vec("\n".join(pools[P]))
        # z-reference: 200 random training bags of 25 turns per person (two-author deviation from canonical Delta)
        rng = np.random.default_rng(seed)
        R = []
        for P in (S, T):
            pool = pools[P]
            for _ in range(C.Z_REF_BAGS):
                ids = rng.choice(len(pool), C.Z_REF_K, replace=False)
                x, tot = self.count_vec("\n".join(pool[i] for i in ids))
                R.append(x / tot)
        R = np.array(R)
        self.mu = R.mean(0)
        self.sd = R.std(0, ddof=1)
        self.nb = NB(texts, whos, pos=S, neg=T)
        self.prior = self.nb.prior_diff
        self.llr = self.nb.llr
        # Craig's Zeta marker n-grams (descriptive only): segments of 10 consecutive turns per person
        self.pseg = {}
        for P in (S, T):
            pool = pools[P]
            segs = [pool[i:i + 10] for i in range(0, len(pool) - len(pool) % 10, 10)]
            pres = np.array([self.count_vec("\n".join(sg))[0] > 0 for sg in segs], float)
            self.pseg[P] = pres.mean(0) if len(segs) else np.zeros(NMAX)
        self.Z = self.pseg[T] - self.pseg[S]

    def count_vec(self, text):
        x = np.zeros(NMAX)
        for g, k in Counter(grams(text)).items():
            i = self.vidx.get(g)
            if i is not None:
                x[i] += k
        return x, total_grams(len(text))

    def margin(self, text):
        return self.nb.margin(text)

    def dist(self, meas, N, XA, tA, XB, tB):
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

    def scores_from_counts(self, X, tot, margin, N: int = NMAIN):
        """Partner-likeness for every count-based measure + NB (dict key -> array)."""
        out = {}
        for meas in COUNT_MEAS:
            d = {P: self.dist(meas, N, X, tot, self.prof_X[P][None, :], np.array([self.prof_tot[P]])) for P in (S, T)}
            out[meas] = d[T] - d[S]
        out["M5"] = np.asarray(margin, float)
        return out

    def markers(self, k: int = 15):
        """Top Zeta marker n-grams per person: (ngram, Z) with Z = p_T - p_S; descriptive use only."""
        order = np.argsort(self.Z, kind="stable")
        top_T = [(self.vocab[i], float(self.Z[i])) for i in order[::-1][:k]]
        top_S = [(self.vocab[i], float(self.Z[i])) for i in order[:k]]
        return {"T": top_T, "S": top_S}


class Store:
    """Per-text sufficient statistics so that bags ('\\n'.join of texts) are featurised exactly and fast."""

    def __init__(self, model: Model, texts):
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
        """bags: sequences of indices into self.texts -> X (n, NMAX), tot (n,), NB margin (n,), joined texts."""
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
                    p = pos
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
