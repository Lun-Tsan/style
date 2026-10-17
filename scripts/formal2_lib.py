"""formal2_lib.py -- shared helpers for the post-review re-analysis (formal2_*.py).

Re-uses formal_lib.py unchanged (measures M1-M7, Model, Store, NCDRef, date_blocks, tune_threshold).
New here:
  * Zeta variants (review item: the summed Zeta score drifts with text length):
      M7m = length-normalised Zeta: s(x) = - mean_{g in distinct V1000 n-grams of x} Z(g)          (0 if none)
      M7p = marker-share Zeta (Craig & Kinney 2009 style marker lists): markers_T = the 100 n-grams with the
            largest Z, markers_S = the 100 with the smallest Z (V1000);  s(x) = (n_S - n_T) / (n_S + n_T),
            n_P = number of P-markers present in x                                                    (0 if none)
  * bag draws for the OUT-OF-FOLD TRAIN calibration (5 contiguous date blocks, every component refit per fold)
  * AUC, exact McNemar, a process pool for NCD (zlib only).
Score convention as in formal_lib: s > thr -> predict S (1), else T (0).
"""
import os
import sys
import time

import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
if SCRATCH not in sys.path:
    sys.path.insert(0, SCRATCH)
import formal_lib as L  # noqa: E402
from formal_lib import S, T, M  # noqa: E402,F401

KS = L.KS
NBAG = 300          # test bags per person per k (as in formal_run E3)
NOOF = 60           # OOF train bags per person per date block per k  (5 blocks -> 300 per person)
NMARK = 100
MAIN = ["M1_N300", "M2_N300", "M2b_N300", "M3_N300", "M4_N300", "M5", "M6_zlib_xy", "M7_N1000"]
ZETA_X = ["M7m_N1000", "M7p_N1000"]
BOOTKEYS = MAIN + ZETA_X
NAME = {"M1_N300": "M1 Burrows Delta", "M2_N300": "M2 Cosine Delta", "M2b_N300": "M2b cosine, raw f",
        "M3_N300": "M3 Min-max", "M4_N300": "M4 JSD", "M5": "M5 NB author-LM LLR", "M6_zlib_xy": "M6 NCD zlib",
        "M7_N1000": "M7 Zeta, summed (spec)", "M7m_N1000": "M7m Zeta, mean-Z (length-normalised)",
        "M7p_N1000": "M7p Zeta, marker share (100+100)"}


# ----------------------------------------------------------------------------- scoring
def zeta_variants(model, X):
    P = (X > 0).astype(float)
    Z = model.Z
    n = P.sum(1)
    sz = P @ Z
    m7m = -np.divide(sz, n, out=np.zeros_like(sz), where=n > 0)
    order = np.argsort(Z, kind="stable")
    iS, iT = order[:NMARK], order[::-1][:NMARK]
    nS, nT = P[:, iS].sum(1), P[:, iT].sum(1)
    m7p = np.divide(nS - nT, nS + nT, out=np.zeros_like(nS), where=(nS + nT) > 0)
    return {"M7m_N1000": m7m, "M7p_N1000": m7p}


def score_counts(model, X, tot, mar):
    out = model.scores_from_counts(X, tot, mar)
    out.update(zeta_variants(model, X))
    return out


def auc(s, y):
    """P(score of a random S item > score of a random T item), ties count 1/2 (Mann-Whitney)."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    a, b = s[y == 1], s[y == 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    r = _rankdata(allv)
    ra = r[:len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


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


def mcnemar_exact(c1, c2):
    """Two-sided exact McNemar on paired correctness vectors: b = c1 right & c2 wrong, c = c1 wrong & c2 right."""
    c1, c2 = np.asarray(c1, bool), np.asarray(c2, bool)
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * M.binom_cdf(min(b, c), n, 0.5))
    return b, c, p


def acc_parts(s, y, thr):
    p = L.predict(s, thr)
    y = np.asarray(y, int)
    return float((p == y).mean()), float((p[y == 1] == 1).mean()), float((p[y == 0] == 0).mean())


# ----------------------------------------------------------------------------- NCD pool (zlib only)
_W = {}


def _winit():
    _, _, _, tr_b, _, _ = L.load_pools()
    _W["tr_b"] = tr_b
    _W["blocks"] = L.date_blocks(tr_b)
    _W["refs"] = {}


def _wref(key):
    if key not in _W["refs"]:
        tr_b = _W["tr_b"]
        if key == "full":
            train = tr_b
        else:
            held = set(_W["blocks"][key])
            train = [t for i, t in enumerate(tr_b) if i not in held]
        _W["refs"][key] = L.NCDRef(train, codecs=("zlib",))
    return _W["refs"][key]


def _wncd(task):
    key, order, texts = task
    t0 = time.perf_counter()
    return _wref(key).score(texts, "zlib", order), time.perf_counter() - t0


def ncd_scores(pool, key, texts, orders=("xy",), chunk=25, timer=None):
    tasks, where = [], []
    for o in orders:
        for a in range(0, len(texts), chunk):
            tasks.append((key, o, texts[a:a + chunk]))
            where.append((o, a))
    res = pool.map(_wncd, tasks)
    out = {f"M6_zlib_{o}": np.empty(len(texts)) for o in orders}
    for (o, a), (s, el) in zip(where, res):
        out[f"M6_zlib_{o}"][a:a + len(s)] = s
        if timer is not None:
            timer[f"M6_zlib_{o}"] = timer.get(f"M6_zlib_{o}", 0.0) + el
    return out


# ----------------------------------------------------------------------------- bag draws
def test_bag_draws(iS, iT):
    """Exactly the E3 draws of formal_run.py: one default_rng(0), for k in KS: 300 S bags then 300 T bags,
    each k distinct turns (without replacement within a bag, with replacement across bags)."""
    rng = np.random.default_rng(0)
    out = {}
    for K in KS:
        bS = [iS[rng.choice(len(iS), K, replace=False)] for _ in range(NBAG)]
        bT = [iT[rng.choice(len(iT), K, replace=False)] for _ in range(NBAG)]
        out[K] = (bS, bT)
    return out


def fold_models(tr_b, seed=0):
    blocks = L.date_blocks(tr_b)
    res = []
    for fi, held in enumerate(blocks):
        hs = set(held)
        train = [t for i, t in enumerate(tr_b) if i not in hs]
        res.append((fi, held, L.Model(train, seed=seed)))
    return blocks, res


def oof_bag_draws(tr_b, blocks):
    """OOF TRAIN bags: for each date block, NOOF bags of k distinct held-out turns per person (default_rng(0))."""
    rng = np.random.default_rng(0)
    out = {}
    for fi, held in enumerate(blocks):
        hS = np.array([j for j, i in enumerate(held) if tr_b[i]["who"] == S])
        hT = np.array([j for j, i in enumerate(held) if tr_b[i]["who"] == T])
        for K in KS:
            bS = [hS[rng.choice(len(hS), K, replace=False)] for _ in range(NOOF)]
            bT = [hT[rng.choice(len(hT), K, replace=False)] for _ in range(NOOF)]
            out[(fi, K)] = (bS, bT)
    return out
