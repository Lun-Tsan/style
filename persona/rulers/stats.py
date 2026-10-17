"""Statistics helpers: AUC, threshold tuning, exact binomial CI, isotonic (PAV) calibration, Cllr, Brier."""
from __future__ import annotations

import math

import numpy as np


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(s, y):
    """P(score of a random positive > score of a random negative), ties 1/2 (Mann-Whitney). y in {0,1}."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    a, b = s[y == 1], s[y == 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = _rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def tune_threshold(s, y):
    """Threshold maximising balanced accuracy of 'positive if s > thr'; ties -> higher accuracy, then smaller |thr|."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    v = np.unique(s)
    cands = np.concatenate([[v[0] - 1.0], (v[:-1] + v[1:]) / 2.0, [v[-1] + 1.0]])
    P = s[None, :] > cands[:, None]
    nP, nN = (y == 1).sum(), (y == 0).sum()
    rP = (P & (y == 1)).sum(1) / nP
    rN = (~P & (y == 0)).sum(1) / nN
    bal = (rP + rN) / 2
    acc = ((P & (y == 1)).sum(1) + (~P & (y == 0)).sum(1)) / len(y)
    j = np.lexsort((np.abs(cands), -acc, -bal))[0]
    return float(cands[j]), float(bal[j]), float(acc[j])


def acc_parts(s, y, thr):
    p = (np.asarray(s, float) > thr).astype(int)
    y = np.asarray(y, int)
    return float((p == y).mean()), float((p[y == 1] == 1).mean()), float((p[y == 0] == 0).mean())


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    s, lp, lq = 0.0, math.log(p), math.log1p(-p)
    for i in range(k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(s, 1.0)


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return (float("nan"), float("nan"))

    def bisect(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else bisect(lambda p: 1 - binom_cdf(k - 1, n, p) >= a / 2)
    hi = 1.0 if k == n else bisect(lambda p: binom_cdf(k, n, p) <= a / 2)
    return lo, hi


def pct_ci(values, lo=2.5, hi=97.5):
    v = np.asarray(values, float)
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else None


class Isotonic:
    """Pool-adjacent-violators calibration score -> P(y = 1). Prediction is a step function with linear
    interpolation between block centres and clipping to [eps, 1 - eps]."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.x = self.p = None

    def fit(self, s, y):
        s = np.asarray(s, float)
        y = np.asarray(y, float)
        order = np.argsort(s, kind="mergesort")
        s, y = s[order], y[order]
        # blocks: (sum_y, n, x_lo, x_hi)
        blocks = [[y[i], 1.0, s[i], s[i]] for i in range(len(s))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
                a, b = blocks[i], blocks[i + 1]
                blocks[i:i + 2] = [[a[0] + b[0], a[1] + b[1], a[2], b[3]]]
                i = max(i - 1, 0)
            else:
                i += 1
        self.x = np.array([(b[2] + b[3]) / 2 for b in blocks])
        self.p = np.clip(np.array([b[0] / b[1] for b in blocks]), self.eps, 1 - self.eps)
        return self

    def prob(self, s):
        s = np.asarray(s, float)
        if len(self.x) == 1:
            return np.full(len(s), self.p[0])
        return np.interp(s, self.x, self.p)

    def logodds(self, s):
        p = self.prob(s)
        return np.log(p / (1 - p))


def cllr(llr, y):
    """Log-likelihood-ratio cost (Brummer & du Preez 2006): 0 = perfect, 1 = uninformative reference."""
    llr = np.asarray(llr, float)
    y = np.asarray(y, int)
    a = np.log2(1 + np.exp(-llr[y == 1])).mean()
    b = np.log2(1 + np.exp(llr[y == 0])).mean()
    return float(0.5 * (a + b))


def brier(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(((p - y) ** 2).mean())
