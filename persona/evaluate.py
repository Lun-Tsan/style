"""persona.evaluate -- acceptance rulers (PIPELINE_SPEC §6.1). The training ruler is NOT used here.

A "system output" is a list aligned with the test samples: [{sid, msgs: [str, ...]}]. Controls:
  C-real  = the real replies, C-shuf = real replies reassigned to other contexts (perfect style, wrong content),
  C-S     = the partner's real test turns (style negative control; discriminator only).
Metrics that need a model (NLL, MMI) take `model, tok`; everything else is pure Python/numpy (+ bge-m3 for OnTopic).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from . import config as C
from .generate import CopyChecker, leak_chars, reply_text
from .normalize import PH_LINE
from .rulers import features as FE
from .rulers.author_lm import NB
from .rulers.ncd import NCDRef
from .rulers.stats import auc
from .turns import Turn

REAL, GEN = "real", "gen"


# ----------------------------------------------------------------------------- helpers
def real_outputs(samples):
    return [{"sid": s.sid, "msgs": [PH_LINE.get(m.strip(), m) for m in s.target.msgs]} for s in samples]


def shuffled_outputs(samples, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(samples))
    while any(perm == np.arange(len(samples))) and len(samples) > 1:     # no reply stays with its own context
        perm = rng.permutation(len(samples))
    real = real_outputs(samples)
    return [{"sid": s.sid, "msgs": real[j]["msgs"]} for s, j in zip(samples, perm)]


def pseudo_turn(msgs, date="2026-09-01"):
    return Turn(C.TARGET, date, 0, 0, [m if m not in PH_LINE.values() else {v: k for k, v in PH_LINE.items()}[m] for m in msgs])


def texts_of(outputs):
    return [reply_text(o["msgs"]) for o in outputs]


# ----------------------------------------------------------------------------- independent discriminator
def discriminator(real_texts, gen_texts, k: int = C.K_ATTR, folds: int = 5, nbags: int = 150, seed: int = 0):
    """Char n-gram NB trained real-vs-generated on the held-out period, out-of-fold. Folds are assigned per text
    STRING (seeded hash), so an identical string can never be in a training fold and a test fold at the same time
    (the shuffled-real control exposed that per-context folds let the NB memorise strings). Returns single-turn
    AUC, bag AUC (k, bags of out-of-fold scores) and n. Empty texts (sticker-only replies) are skipped."""
    import zlib
    real = [t for t in real_texts if t]
    gen = [t for t in gen_texts if t]
    n = min(len(real), len(gen))
    if n < 2 * folds:
        return {"auc1": math.nan, f"auc{k}": math.nan, "n": n}
    fold_of = lambda t: (zlib.crc32((str(seed) + t).encode("utf-8")) & 0xFFFFFFFF) % folds
    fr = np.array([fold_of(t) for t in real])
    fg = np.array([fold_of(t) for t in gen])
    s_real, s_gen = np.zeros(len(real)), np.zeros(len(gen))
    for f in range(folds):
        tr_r = [t for t, ff in zip(real, fr) if ff != f]
        tr_g = [t for t, ff in zip(gen, fg) if ff != f]
        if not tr_r or not tr_g or not ((fr == f).any() or (fg == f).any()):
            continue
        nb = NB(tr_r + tr_g, [REAL] * len(tr_r) + [GEN] * len(tr_g), pos=GEN, neg=REAL)
        for i in np.where(fr == f)[0]:
            s_real[i] = nb.margin(real[i])
        for i in np.where(fg == f)[0]:
            s_gen[i] = nb.margin(gen[i])
    a1 = auc(np.concatenate([s_gen, s_real]), np.array([1] * len(gen) + [0] * len(real)))
    rng = np.random.default_rng(seed)
    kk = min(k, len(real), len(gen))
    bg = [s_gen[rng.choice(len(gen), kk, replace=False)].sum() for _ in range(nbags)]
    br = [s_real[rng.choice(len(real), kk, replace=False)].sum() for _ in range(nbags)]
    ak = auc(np.array(bg + br), np.array([1] * nbags + [0] * nbags))
    return {"auc1": a1, f"auc{k}": ak, "n": n, "n_real": len(real), "n_gen": len(gen)}


def discriminator_floor(real_texts, k: int = C.K_ATTR, reps: int = 10, seed: int = 0):
    """Real vs real: split the real replies into two random halves `reps` times; mean AUC (expected 0.5)."""
    rng = np.random.default_rng(seed)
    texts = [t for t in real_texts if t]
    a1, ak = [], []
    for r in range(reps):
        perm = rng.permutation(len(texts))
        h = len(texts) // 2
        d = discriminator([texts[i] for i in perm[:h]], [texts[i] for i in perm[h:2 * h]], k=k, seed=seed + r)
        a1.append(d["auc1"])
        ak.append(d[f"auc{k}"])
    return {"auc1": float(np.mean(a1)), f"auc{k}": float(np.mean(ak)), "auc1_sd": float(np.std(a1)), "reps": reps}


# ----------------------------------------------------------------------------- other automatic measures
def ncd_delta(texts, ref: NCDRef, k: int = 25, nbags: int = 150, seed: int = 0):
    """Mean partner-likeness NCD score of k-bags (T-like -> negative); compare with the real replies' value."""
    texts = [t for t in texts if t]
    if len(texts) < k:
        return math.nan
    rng = np.random.default_rng(seed)
    bags = ["\n".join(texts[i] for i in rng.choice(len(texts), k, replace=False)) for _ in range(nbags)]
    return float(ref.score(bags).mean())


def band_metrics(outputs, bands, k: int = C.K_VALID, nbags: int = 200, seed: int = 0):
    turns = [pseudo_turn(o["msgs"]) for o in outputs if o["msgs"]]
    rng = np.random.default_rng(seed)
    kk = min(k, len(turns))
    inside, defined = [], []
    for _ in range(nbags):
        bag = [turns[i] for i in rng.choice(len(turns), kk, replace=False)]
        chk = FE.band_check(FE.bag_features(bag), bands)
        inside.append(chk["n_inside"])
        defined.append(chk["n_defined"])
    return {"band_inside_mean": float(np.mean(inside)), "band_defined_mean": float(np.mean(defined)), "k": kk}


def feature_jsd(outputs, real_outputs_, bins: int = 10):
    """Mean over features of the JSD between the turn-level feature histograms of system and real replies."""
    A = [FE.turn_features(pseudo_turn(o["msgs"])) for o in outputs if o["msgs"]]
    B = [FE.turn_features(pseudo_turn(o["msgs"])) for o in real_outputs_ if o["msgs"]]
    vals = []
    for f in FE.FEATURES:
        a = np.array([r[f] for r in A], float)
        b = np.array([r[f] for r in B], float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 5 or len(b) < 5:
            continue
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if hi <= lo:
            vals.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p = np.histogram(a, edges)[0] / len(a)
        q = np.histogram(b, edges)[0] / len(b)
        m = 0.5 * (p + q)
        kl = lambda x: float(np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(m > 0, m, 1)), 0.0)))
        vals.append(0.5 * kl(p) + 0.5 * kl(q))
    return float(np.mean(vals)) if vals else math.nan


def ontopic_rate(gen_texts, real_texts, bge, thr: float):
    idx = [i for i, (g, r) in enumerate(zip(gen_texts, real_texts)) if g and r]
    if not idx:
        return math.nan
    c = bge.cos([gen_texts[i] for i in idx], [real_texts[i] for i in idx])
    return float((c >= thr).mean())


def copy_rate(gen_texts, checker: CopyChecker):
    t = [g for g in gen_texts if g]
    return float(np.mean([checker.is_copy(g) for g in t])) if t else math.nan


def leak_rate(gen_texts):
    t = [g for g in gen_texts if g]
    vals = [leak_chars(g) for g in t]
    if not t or vals[0] is None:
        return None
    return float(100.0 * sum(vals) / max(1, sum(len(g) for g in t)))


def diversity(outputs, n_rarefy: int | None = None, seed: int = 0):
    """distinct-2 over character bigrams of all replies; Hill number of order 1 (exp Shannon entropy) of reply
    strings; both after rarefaction to n_rarefy replies."""
    rng = np.random.default_rng(seed)
    reps = ["\n".join(o["msgs"]) for o in outputs if o["msgs"]]
    if n_rarefy and len(reps) > n_rarefy:
        reps = [reps[i] for i in rng.choice(len(reps), n_rarefy, replace=False)]
    big = Counter(r[i:i + 2] for r in reps for i in range(len(r) - 1))
    d2 = len(big) / max(1, sum(big.values()))
    cnt = np.array(list(Counter(reps).values()), float)
    p = cnt / cnt.sum()
    hill1 = float(np.exp(-(p * np.log(p)).sum()))
    return {"distinct2": float(d2), "hill1": hill1, "n": len(reps), "unique_share": len(cnt) / max(1, len(reps))}


def model_nll(model, tok, samples, outputs, batch_size: int = 4):
    """Mean per-token NLL of the given replies under the model (teacher-forced with the shared prompt builder)."""
    from .data_sft import encode_samples, mean_nll, nll_per_item
    from .samples import Sample
    pseudo = [Sample(s.sid, s.ctx, pseudo_turn(o["msgs"], s.target.date)) for s, o in zip(samples, outputs) if o["msgs"]]
    items = encode_samples(tok, pseudo)
    return mean_nll(nll_per_item(model, items, tok.pad_token_id, batch_size))


def mmi(model, tok, samples, outputs, batch_size: int = 4):
    """Coherence: mean over replies of log p(y|c) - log p(y) per token under `model` (the base model in the paper)."""
    from .data_sft import encode_samples, nll_per_item
    from .samples import Sample
    keep = [(s, o) for s, o in zip(samples, outputs) if o["msgs"]]
    with_ctx = [Sample(s.sid, s.ctx, pseudo_turn(o["msgs"], s.target.date)) for s, o in keep]
    no_ctx = [Sample(s.sid, [], pseudo_turn(o["msgs"], s.target.date)) for s, o in keep]
    a = nll_per_item(model, encode_samples(tok, with_ctx), tok.pad_token_id, batch_size)
    b = nll_per_item(model, encode_samples(tok, no_ctx), tok.pad_token_id, batch_size)
    vals = [(y[0] - x[0]) / x[1] for x, y in zip(a, b) if x[1]]
    return float(np.mean(vals)) if vals else math.nan


# ----------------------------------------------------------------------------- the evaluation bundle
class Evaluator:
    """Holds everything that is fixed across systems: test samples, real replies, partner texts, NCD reference,
    feature bands, copy checker, bge threshold."""

    def __init__(self, test_samples, train_turns, test_partner_turns, bands, bge=None, ontopic_thr=None):
        self.samples = test_samples
        self.real = real_outputs(test_samples)
        self.real_texts = texts_of(self.real)
        self.partner_texts = [t.ruler_text for t in test_partner_turns if t.ruler_text and not t.symbol]
        tr = [t for t in train_turns if t.ruler_text and not t.symbol]
        self.ncd_ref = NCDRef([t.ruler_text for t in tr], [t.who for t in tr])
        self.bands = bands
        self.copy = CopyChecker([t.ruler_text for t in tr if t.who == C.TARGET])
        self.bge, self.thr = bge, ontopic_thr
        self._real_ncd = ncd_delta(self.real_texts, self.ncd_ref)
        self._sens = discriminator(self.real_texts, self.partner_texts[:len(self.real_texts)])
        self._floor = discriminator_floor(self.real_texts)

    def evaluate(self, outputs, model=None, tok=None, base_model=None, n_rarefy=None):
        gen_texts = texts_of(outputs)
        d = discriminator(self.real_texts, gen_texts)
        m = {"n": len(outputs), "d_auc1": d["auc1"], f"d_auc{C.K_ATTR}": d[f"auc{C.K_ATTR}"], "d_n": d["n"],
             "floor_auc1": self._floor["auc1"], f"floor_auc{C.K_ATTR}": self._floor[f"auc{C.K_ATTR}"],
             "sens_auc1": self._sens["auc1"], f"sens_auc{C.K_ATTR}": self._sens[f"auc{C.K_ATTR}"],
             "ncd_delta": ncd_delta(gen_texts, self.ncd_ref) - self._real_ncd,
             "feature_jsd": feature_jsd(outputs, self.real),
             "copy_rate": copy_rate(gen_texts, self.copy), "leak_per_100": leak_rate(gen_texts),
             "sticker_only_share": float(np.mean([not g for g in gen_texts])) if gen_texts else math.nan}
        m.update(band_metrics(outputs, self.bands))
        m.update({"div_" + k: v for k, v in diversity(outputs, n_rarefy).items()})
        if self.bge is not None and self.thr is not None:
            m["ontopic"] = ontopic_rate(gen_texts, self.real_texts, self.bge, self.thr)
        if model is not None and tok is not None:
            m["nll"] = model_nll(model, tok, self.samples, outputs)
        if base_model is not None and tok is not None:
            m["mmi"] = mmi(base_model, tok, self.samples, outputs)
        return m
