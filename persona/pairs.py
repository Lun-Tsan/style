"""persona.pairs -- ruler-ranked DPO pairs inside behaviour cells (PIPELINE_SPEC §5.3).

chosen   = highest calibrated score among candidates that pass the constraints (on-topic, no verbatim copy, no
           leakage) and exceed +delta, taken from the REAL reply's behaviour cell; if that cell has no qualifying
           candidate, from the neighbouring cells (same sticker flag, length bucket +-1), then from any cell
           (the level is recorded; the smoke test showed that an exact cell match is rare with 8 candidates);
rejected = lowest score among all candidates below -delta;
no qualifying chosen -> the real reply is used (DITTO-style), at most `ditto_max` of all pairs;
no qualifying rejected -> the context yields no pair.
Scores are calibrated log-odds of 'written by T' (isotonic on OOF training LLRs), capped at the cap_q quantile
of the real training replies' scores (typicality band).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import config as C
from .generate import CopyChecker, leak_chars, reply_text
from .normalize import PH, PH_LINE
from .rulers import features as FE
from .rulers.calibrate import Calibrator
from .rulers.profiles import Model

WS = re.compile(r"\s")
PH_LINES = set(PH_LINE.values())
PARTICLES = FE.PARTICLE_T | FE.PARTICLE_S


def cell(msgs) -> tuple:
    """(length bucket, has sticker, burst class, ends with particle) of a reply given as message list."""
    text_msgs = [m for m in msgs if m.strip() not in PH_LINES and m.strip() not in PH]
    n = sum(len(WS.sub("", m)) for m in text_msgs)
    length = 0 if n <= 5 else 1 if n <= 15 else 2 if n <= 40 else 3
    sticker = any(m.strip() in PH_LINES or m.strip() in PH for m in msgs)
    burst = min(len(msgs), 3)
    particle = any((e := FE.TRAIL.sub("", m)) and e[-1] in PARTICLES for m in text_msgs)
    return (length, sticker, burst, particle)


def cell_level(c: tuple, real: tuple) -> int:
    """0 = same cell; 1 = same sticker flag and length bucket within +-1; 2 = anything else."""
    if c == real:
        return 0
    if c[1] == real[1] and abs(c[0] - real[0]) <= 1:
        return 1
    return 2


class Scorer:
    """Calibrated, capped target-likeness of a reply (0 for replies without text)."""

    def __init__(self, model: Model, cal: Calibrator, cap: float):
        self.model, self.cal, self.cap = model, cal, cap

    def raw(self, msgs) -> float:
        text = reply_text(msgs)
        if not text:
            return 0.0
        return float(self.cal.logodds([-self.model.margin(text)])[0])

    def score(self, msgs) -> float:
        return min(self.raw(msgs), self.cap)


def fit_scorer(train_turns, oof_margin, oof_y_is_S, cap_q: float = C.DPO["cap_q"]):
    """train_turns: non-symbol training turns with ruler_text (both speakers); oof_* from validate.run()."""
    model = Model([t.ruler_text for t in train_turns], [t.who for t in train_turns])
    cal = Calibrator().fit(-np.asarray(oof_margin), 1 - np.asarray(oof_y_is_S))
    sc = Scorer(model, cal, cap=float("inf"))
    real = [sc.raw(t.msgs) for t in train_turns if t.who == C.TARGET]
    sc.cap = float(np.quantile(real, cap_q))
    return sc


def build_pairs(samples, cands, scorer: Scorer, checker: CopyChecker, bge=None, thr=None,
                delta: float = C.DPO["delta"], ditto_max: float = C.DPO["ditto_max"], seed: int = 0):
    """samples: training samples; cands: per sample a list of candidate message lists. Returns (pairs, stats)."""
    rng = np.random.default_rng(seed)
    pairs, st = [], Counter()
    cell_hist = Counter()
    for s, cs in zip(samples, cands):
        real_msgs = [PH_LINE.get(m.strip(), m) for m in s.target.msgs]
        real_cell = cell(real_msgs)
        cell_hist[real_cell] += 1
        real_text = reply_text(real_msgs)
        uniq = list({tuple(c): c for c in cs if c}.values())
        st["candidates"] += len(cs)
        st["candidates_unique"] += len(uniq)
        rows = []
        for c in uniq:
            txt = reply_text(c)
            sc = scorer.score(c)
            ok = True
            if txt and checker.is_copy(txt):
                ok = False
                st["rejected_copy"] += 1
            lk = leak_chars(txt) if txt else None
            if lk:
                ok = False
                st["rejected_leak"] += 1
            if ok and bge is not None and thr is not None and txt and real_text:
                if float(bge.cos([txt], [real_text])[0]) < thr:
                    ok = False
                    st["rejected_offtopic"] += 1
            rows.append({"msgs": c, "score": sc, "cell": cell(c), "ok": ok})
        for r in rows:
            r["level"] = cell_level(r["cell"], real_cell)
        st["contexts"] += 1
        st["contexts_with_cell_match"] += any(r["level"] == 0 for r in rows)
        ch = [r for r in rows if r["ok"] and r["score"] > delta]
        rj = [r for r in rows if r["score"] < -delta]
        if not rj:
            st["contexts_no_rejected"] += 1
            continue
        rejected = min(rj, key=lambda r: r["score"])
        if ch:
            best_level = min(r["level"] for r in ch)
            chosen = max((r for r in ch if r["level"] == best_level), key=lambda r: r["score"])
            st[f"chosen_level_{best_level}"] += 1
            pairs.append({"sid": s.sid, "chosen": chosen["msgs"], "rejected": rejected["msgs"], "ditto": False,
                          "level": best_level, "score_chosen": chosen["score"], "score_rejected": rejected["score"],
                          "cell": list(real_cell)})
        else:
            pairs.append({"sid": s.sid, "chosen": real_msgs, "rejected": rejected["msgs"], "ditto": True, "level": -1,
                          "score_chosen": scorer.score(real_msgs), "score_rejected": rejected["score"], "cell": list(real_cell)})
    n_ditto = sum(p["ditto"] for p in pairs)
    if pairs and n_ditto > ditto_max * len(pairs):
        keep = int(ditto_max * (len(pairs) - n_ditto) / (1 - ditto_max))         # ditto / total <= ditto_max
        di = [i for i, p in enumerate(pairs) if p["ditto"]]
        drop = set(rng.choice(di, len(di) - keep, replace=False).tolist()) if len(di) > keep else set()
        pairs = [p for i, p in enumerate(pairs) if i not in drop]
        st["ditto_dropped"] = len(drop)
    stats = dict(st)
    stats["n_pairs"] = len(pairs)
    stats["n_ditto"] = sum(p["ditto"] for p in pairs)
    stats["ditto_share"] = stats["n_ditto"] / len(pairs) if pairs else None
    stats["mean_score_chosen"] = float(np.mean([p["score_chosen"] for p in pairs])) if pairs else None
    stats["mean_score_rejected"] = float(np.mean([p["score_rejected"] for p in pairs])) if pairs else None
    stats["real_cells"] = {str(k): v for k, v in cell_hist.most_common()}
    stats["delta"] = delta
    stats["cap"] = scorer.cap
    return pairs, stats


def pairs_feature_jsd(pairs, samples_by_sid):
    """Check (PIPELINE_SPEC §5.3): feature distribution of the chosen set vs the real replies of the same contexts."""
    from .evaluate import feature_jsd
    chosen = [{"sid": p["sid"], "msgs": p["chosen"]} for p in pairs]
    real = [{"sid": p["sid"], "msgs": [PH_LINE.get(m.strip(), m) for m in samples_by_sid[p["sid"]].target.msgs]} for p in pairs]
    return feature_jsd(chosen, real)
