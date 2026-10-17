"""Normalised compression distance (zlib level 9), test text first; 10 reference bags of ~3,000 chars per person."""
from __future__ import annotations

import zlib

import numpy as np

from .. import config as C

S, T = C.PARTNER, C.TARGET


def clen(b: bytes) -> int:
    return len(zlib.compress(b, 9))


def ncd(x: bytes, y: bytes, cx: int, cy: int) -> float:
    cxy = clen(x + y)
    return (cxy - min(cx, cy)) / max(cx, cy)


def ncd_pair(a: str, b: str) -> float:
    x, y = a.encode("utf-8"), b.encode("utf-8")
    return ncd(x, y, clen(x), clen(y))


class NCDRef:
    def __init__(self, texts, whos, seed: int = 0, nbags: int = C.NCD_BAGS, target: int = C.NCD_CHARS):
        rng = np.random.default_rng(seed)
        self.refs = {}
        for P in (S, T):
            pool = [x for x, w in zip(texts, whos) if w == P]
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
        self.cy = {P: [clen(b) for b in self.refs[P]] for P in (S, T)}

    def dists(self, text: str):
        x = text.encode("utf-8")
        cx = clen(x)
        return {P: float(np.mean([ncd(x, y, cx, cy) for y, cy in zip(self.refs[P], self.cy[P])])) for P in (S, T)}

    def score(self, texts):
        """Partner-likeness: NCD to T references minus NCD to S references."""
        out = np.empty(len(texts))
        for i, x in enumerate(texts):
            d = self.dists(x)
            out[i] = d[T] - d[S]
        return out
