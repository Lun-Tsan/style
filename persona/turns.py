"""persona.turns -- messages -> turns -> segments.

turn    = maximal run of same-speaker messages on the same day with gaps <= GAP_MIN (identical to style_nb)
segment = maximal run of turns with gaps <= SEG_GAP_H hours (absolute time, so a chat across midnight stays together)
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from . import config as C
from .normalize import PH, emoticon_placeholders

_WS = re.compile(r"\s")


def to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def abs_min(date: str, t: int) -> int:
    y, mo, d = map(int, date.split("-"))
    return dt.date(y, mo, d).toordinal() * 1440 + t


@dataclass
class Turn:
    who: str
    date: str
    t_start: int
    t_end: int
    msgs: list = field(default_factory=list)
    idx: int = -1
    seg: int = -1
    split: str = ""
    block: int = -1
    symbol: bool = False        # set by stage1 from flag_symbols.is_symbol_turn(ruler_text)

    def __post_init__(self):
        self.derive()

    def derive(self):
        bodies = self.msgs
        self.n_msgs = len(bodies)
        self.kinds = ["placeholder" if b.strip() in PH else "text" for b in bodies]
        self.has_sticker = any(k == "placeholder" for k in self.kinds)
        self.text_a = "\n".join(PH.get(b.strip(), b) for b in bodies)
        kept = [b for b in bodies if b.strip() not in PH]
        self.n_text = len(kept)
        self.text_b = "\n".join(kept) if kept else None
        self.len_b = len(_WS.sub("", self.text_b)) if self.text_b else 0
        self.len_a = len(_WS.sub("", self.text_a))
        self.ruler_text = emoticon_placeholders(self.text_b) if self.text_b else None

    @property
    def abs_start(self) -> int:
        return abs_min(self.date, self.t_start)

    @property
    def abs_end(self) -> int:
        return abs_min(self.date, self.t_end)

    @property
    def month(self) -> str:
        return self.date[:7]


def build_turns(msgs, gap_min: int = C.GAP_MIN, split_date: str = C.SPLIT_DATE):
    turns, cur = [], None
    for m in msgs:
        t = to_min(m.time)
        if cur is not None and cur.who == m.who and cur.date == m.date and t - cur.t_end <= gap_min:
            cur.msgs.append(m.text)
            cur.t_end = t
        else:
            cur = Turn(m.who, m.date, t, t, [m.text])
            turns.append(cur)
    for i, tr in enumerate(turns):
        tr.idx = i
        tr.derive()
        tr.split = "test" if tr.date >= split_date else "train"
    return turns


def assign_segments(turns, gap_h: float = C.SEG_GAP_H):
    """Group consecutive turns into segments; sets turn.seg. Returns the list of segments (lists of Turn)."""
    segs, cur, last_end = [], [], None
    gap = gap_h * 60.0
    for tr in turns:
        if cur and tr.abs_start - last_end > gap:
            segs.append(cur)
            cur = []
        cur.append(tr)
        last_end = tr.abs_end
    if cur:
        segs.append(cur)
    for si, sg in enumerate(segs):
        for tr in sg:
            tr.seg = si
    return segs


def counts(turns, text_only=True):
    """(S, T) counts per split, for text turns (text_b) or all turns."""
    out = {}
    for split in ("train", "test"):
        pool = [t for t in turns if t.split == split and (t.text_b or not text_only)]
        out[split] = (sum(t.who == C.PARTNER for t in pool), sum(t.who == C.TARGET for t in pool))
    return out
