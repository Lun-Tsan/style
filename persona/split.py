"""persona.split -- time split, dev cut, date blocks, nested time-uniform subsets."""
from __future__ import annotations

import math
from collections import Counter

from . import config as C


def date_blocks(items, K: int = C.N_BLOCKS):
    """K contiguous chronological blocks of ~equal size; a calendar date is never split across blocks.
    items must be chronological and have .date. Returns lists of positional indices."""
    n = len(items)
    cuts = [0]
    for f in range(1, K):
        c = round(n * f / K)
        while c < n and items[c].date == items[c - 1].date:
            c += 1
        cuts.append(c)
    cuts.append(n)
    return [list(range(cuts[i], cuts[i + 1])) for i in range(K)]


def assign_blocks(train_turns, K: int = C.N_BLOCKS):
    blocks = date_blocks(train_turns, K)
    for bi, idx in enumerate(blocks):
        for i in idx:
            train_turns[i].block = bi
    return blocks


def dev_cut(train_samples, frac: float = C.DEV_FRAC):
    """Last `frac` of the training samples by time -> dev, moving the cut forward to a segment boundary
    so that no segment contributes to both parts."""
    ss = sorted(train_samples, key=lambda s: (s.target.date, s.target.idx))
    cut = round(len(ss) * (1.0 - frac))
    while 0 < cut < len(ss) and ss[cut].target.seg == ss[cut - 1].target.seg:
        cut += 1
    return ss[:cut], ss[cut:]


def nested_subsets(samples, sizes=C.SUBSET_SIZES):
    """Nested, time-uniform subsets: the first size takes equally spaced samples; each larger size adds equally
    spaced samples from the ones not yet chosen. None = all. Returns {size: list of samples (time order)}."""
    ss = sorted(samples, key=lambda s: (s.target.date, s.target.idx))
    chosen = set()
    out = {}
    for size in sizes:
        if size is None or size >= len(ss):
            out[size] = list(ss)
            continue
        need = size - len(chosen)
        if need > 0:
            rest = [i for i in range(len(ss)) if i not in chosen]
            for j in range(need):
                chosen.add(rest[int((j + 0.5) * len(rest) / need)])
        out[size] = [ss[i] for i in sorted(chosen)]
    return out


def month_hist(samples) -> Counter:
    return Counter(s.target.month for s in samples)


def jsd_hist(c1: Counter, c2: Counter) -> float:
    """Jensen-Shannon divergence (base 2) between two count histograms over the union of keys."""
    keys = set(c1) | set(c2)
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2:
        return float("nan")
    tot = 0.0
    for k in keys:
        p, q = c1.get(k, 0) / n1, c2.get(k, 0) / n2
        m = 0.5 * (p + q)
        if p:
            tot += 0.5 * p * math.log2(p / m)
        if q:
            tot += 0.5 * q * math.log2(q / m)
    return tot
