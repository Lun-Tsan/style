"""persona.pairs -- ruler-ranked DPO pairs inside behaviour cells (PIPELINE_SPEC §5.3).

chosen   = highest score among candidates that pass the constraints (on-topic, no verbatim copy, no
           leakage) and exceed +delta, taken from the REAL reply's behaviour cell; if that cell has no qualifying
           candidate, from the neighbouring cells (same sticker flag, length bucket +-1), then from any cell
           (the level is recorded; the smoke test showed that an exact cell match is rare with 8 candidates);
rejected = lowest score among all candidates below -delta;
symbol-like candidates and unfinished ones (no EOS before max_new_tokens) are excluded from both roles;
no qualifying chosen -> the real reply is used (DITTO-style), at most `ditto_max` of all pairs;
no qualifying rejected -> the context yields no pair.
Scores are calibrated log-odds of 'written by T' (isotonic on OOF training LLRs) of the raw NB target-LLR capped at
the cap_q quantile of the real T training turns' OOF LLRs (typicality band); ties are broken with the seeded rng.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import config as C
from .flag_symbols import is_symbol_turn
from .text import CopyChecker, leak_chars, reply_text
from .normalize import PH, PH_LINE
from .rulers import features as FE
from .rulers.calibrate import Calibrator
from .rulers.profiles import Model

WS = re.compile(r"\s")
PH_LINES = set(PH_LINE.values())
PARTICLES = FE.PARTICLE_T | FE.PARTICLE_S
TIE = 1e-9


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
    """raw(msgs) = NB target-LLR (-margin) of the reply's ruler view, None without text; score(msgs) = calibrated
    log-odds of min(raw, cap_raw), 0 without text."""

    def __init__(self, model: Model, cal: Calibrator, cap_raw: float):
        self.model, self.cal, self.cap_raw = model, cal, cap_raw
        self.fit_stats = {"cap_raw": cap_raw}

    def raw(self, msgs):
        text = reply_text(msgs)
        return -float(self.model.margin(text)) if text else None

    def score(self, msgs) -> float:
        r = self.raw(msgs)
        return 0.0 if r is None else float(self.cal.logodds([min(r, self.cap_raw)])[0])


def fit_scorer(train_turns, oof_margin, oof_y_is_S, cap_q: float = C.DPO["cap_q"]):
    """train_turns: non-symbol training turns with ruler_text, both speakers, chronological: the population of
    rulers.validate.run(), so the OOF arrays of reports/ruler_validation.json align with it position by position."""
    margin, y_S = np.asarray(oof_margin, float), np.asarray(oof_y_is_S, int)
    if not len(margin) == len(y_S) == len(train_turns) or any((t.who == C.PARTNER) != bool(y)
                                                               for t, y in zip(train_turns, y_S)):
        raise ValueError(f"OOF arrays ({len(margin)} turns) do not line up with the {len(train_turns)} training turns; "
                         f"re-run python -m persona.stage1")
    model = Model([t.ruler_text for t in train_turns], [t.who for t in train_turns])
    cal = Calibrator().fit(-margin, 1 - y_S)
    # the cap sits on the raw LLR: the isotonic fit is clipped at p = 0.999, and 9.6% of the T OOF turns of reports/
    # (2026-10-17) sit at that ceiling, so the cap_q quantile of calibrated scores was the ceiling and capped nothing
    llr_T = -margin[y_S == 0]
    cap_raw = float(np.quantile(llr_T, cap_q))
    sc = Scorer(model, cal, cap_raw)
    sc.fit_stats = {"cap_q": cap_q, "cap_raw": cap_raw, "cap": float(cal.logodds([cap_raw])[0]),
                    "cap_T_oof_above": float(np.mean(llr_T > cap_raw))}
    return sc


def _pick(rows, sign: int, rng):
    """Row with the highest (sign 1) or lowest (sign -1) rank; rows within TIE of it are drawn from at random, so
    the generation order does not decide."""
    best = max(sign * r["rank"] for r in rows)
    tied = [r for r in rows if sign * r["rank"] >= best - TIE]
    return tied[int(rng.integers(len(tied)))] if len(tied) > 1 else tied[0]


def build_pairs(samples, cands, scorer: Scorer, checker: CopyChecker, bge=None, thr=None,
                delta: float = C.DPO["delta"], ditto_max: float = C.DPO["ditto_max"], seed: int = 0, finished=None):
    """samples: training samples; cands: per sample a list of candidate message lists; finished: per sample a list of
    bools aligned with cands (True = EOS before max_new_tokens), a missing list counting as all True.
    Returns (pairs, stats)."""
    rng = np.random.default_rng(seed)
    pairs, st = [], Counter(dict.fromkeys(("cand_unfinished", "cand_symbol", "cand_eligible", "cand_capped"), 0))
    cell_hist = Counter()
    fins = finished if finished is not None else [None] * len(samples)
    for s, cs, fin in zip(samples, cands, fins, strict=True):
        real_msgs = [PH_LINE.get(m.strip(), m) for m in s.target.msgs]
        real_cell = cell(real_msgs)
        cell_hist[real_cell] += 1
        real_text = reply_text(real_msgs)
        uniq = {}                                    # candidate -> finished in at least one of its draws
        for c, f in zip(cs, fin if fin is not None else [True] * len(cs), strict=True):
            if c:
                uniq[tuple(c)] = uniq.get(tuple(c), False) or bool(f)
        st["candidates"] += len(cs)
        st["candidates_unique"] += len(uniq)
        rows = []
        for c, f in uniq.items():
            c = list(c)
            txt = reply_text(c)
            sym = is_symbol_turn(txt)                # contract (d): the rulers do not score symbol-like turns
            st["cand_unfinished"] += not f
            st["cand_symbol"] += sym
            if not f or sym:                         # an unfinished one would teach DPO a cut-off ending (#16)
                continue
            st["cand_eligible"] += 1
            raw = scorer.raw(c)
            st["cand_capped"] += raw is not None and raw > scorer.cap_raw
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
            # rank = capped raw LLR. The score is a non-decreasing function of it, but flat where the calibrator is
            # (its 0.999 ceiling starts below cap_raw in reports/), and ranking on those flat steps would leave the
            # cap without effect; ties within TIE (e.g. several capped candidates) are drawn at random in _pick
            rows.append({"msgs": c, "score": scorer.score(c), "rank": 0.0 if raw is None else min(raw, scorer.cap_raw),
                         "cell": cell(c), "ok": ok})
        for r in rows:
            r["level"] = cell_level(r["cell"], real_cell)
        st["contexts"] += 1
        st["contexts_with_cell_match"] += any(r["level"] == 0 for r in rows)
        ch = [r for r in rows if r["ok"] and r["score"] > delta]
        rj = [r for r in rows if r["score"] < -delta]
        if not rj:
            st["contexts_no_rejected"] += 1
            continue
        rejected = _pick(rj, -1, rng)
        if ch:
            best_level = min(r["level"] for r in ch)
            chosen = _pick([r for r in ch if r["level"] == best_level], 1, rng)
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
    stats["symbol_share"] = st["cand_symbol"] / st["candidates_unique"] if st["candidates_unique"] else None
    stats["cand_capped_share"] = st["cand_capped"] / st["cand_eligible"] if st["cand_eligible"] else None
    # a DITTO chosen is the real reply, scored in-sample: the §5.3 checks use the non-DITTO pairs (#10)
    nd = [p for p in pairs if not p["ditto"]]
    stats["mean_score_chosen"] = float(np.mean([p["score_chosen"] for p in nd])) if nd else None
    stats["mean_score_chosen_all"] = float(np.mean([p["score_chosen"] for p in pairs])) if pairs else None
    stats["mean_score_rejected"] = float(np.mean([p["score_rejected"] for p in pairs])) if pairs else None
    stats["real_cells"] = {str(k): v for k, v in cell_hist.most_common()}
    stats["delta"] = delta
    stats.update(scorer.fit_stats)
    return pairs, stats


def pairs_feature_jsd(pairs, samples_by_sid, include_ditto: bool = False):
    """Check (PIPELINE_SPEC §5.3): feature distribution of the chosen set vs the real replies of the same contexts,
    on the non-DITTO pairs unless include_ditto (a DITTO chosen is the real reply itself). None without pairs."""
    from .evaluate import feature_jsd
    ps = [p for p in pairs if include_ditto or not p["ditto"]]
    if not ps:
        return None
    chosen = [{"sid": p["sid"], "msgs": p["chosen"]} for p in ps]
    real = [{"sid": p["sid"], "msgs": [PH_LINE.get(m.strip(), m) for m in samples_by_sid[p["sid"]].target.msgs]} for p in ps]
    return feature_jsd(chosen, real)
