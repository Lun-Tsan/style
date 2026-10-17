"""persona.samples -- (context -> target turn) samples and aggregate statistics (no message text)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config as C
from .normalize import deid


@dataclass
class Sample:
    sid: int
    ctx: list          # preceding turns of the same segment (oldest first), possibly empty
    target: object     # Turn by the target speaker, never symbol-flagged

    @property
    def split(self):
        return self.target.split

    @property
    def date(self):
        return self.target.date


def build_samples(segments, target: str = C.TARGET, ctx_turns: int = C.CTX_TURNS):
    out = []
    for sg in segments:
        for j, tr in enumerate(sg):
            if tr.who != target or tr.symbol:
                continue
            out.append(Sample(len(out), sg[max(0, j - ctx_turns):j], tr))
    return out


def _q(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = np.asarray(xs, float)
    return {f"q{int(q * 100)}": float(np.quantile(xs, q)) for q in qs} if len(xs) else {}


def stats(samples, turns, segments):
    """Aggregate statistics for the data table of the paper."""
    by_split = {sp: [s for s in samples if s.split == sp] for sp in ("train", "test")}
    out = {"n_turns": len(turns), "n_segments": len(segments),
           "n_turns_by_split_speaker": {sp: dict(Counter(C.CODE[t.who] for t in turns if t.split == sp)) for sp in ("train", "test")},
           "n_symbol_turns_by_speaker": dict(Counter(C.CODE[t.who] for t in turns if t.symbol)),
           "segment_turns": _q([len(sg) for sg in segments])}
    deid_hits = Counter()
    for t in turns:
        for b in t.msgs:
            deid_hits += deid(b)[1]
    out["deid_hits_all_messages"] = dict(deid_hits)
    for sp, ss in by_split.items():
        if not ss:
            continue
        out[sp] = {
            "n_samples": len(ss),
            "ctx_turns": dict(Counter(len(s.ctx) for s in ss)),
            "ctx_empty_share": float(np.mean([len(s.ctx) == 0 for s in ss])),
            "target_chars": _q([s.target.len_b for s in ss]),
            "target_msgs": dict(Counter(s.target.n_msgs for s in ss)),
            "target_sticker_only_share": float(np.mean([s.target.text_b is None for s in ss])),
            "target_has_sticker_share": float(np.mean([s.target.has_sticker for s in ss])),
            "per_month": dict(sorted(Counter(s.target.month for s in ss).items())),
            "ctx_symbol_placeholder_share": float(np.mean([any(t.symbol for t in s.ctx) for s in ss])),
        }
    return out
