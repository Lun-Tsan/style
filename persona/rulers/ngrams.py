"""Character n-grams (n = 1..3; newline and placeholder characters are ordinary characters)."""
from __future__ import annotations

from collections import Counter


def grams(text: str, nmax: int = 3):
    out = []
    for n in range(1, nmax + 1):
        out.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return out


def total_grams(L: int) -> int:
    return sum(max(0, L - n + 1) for n in (1, 2, 3))


def top_vocab(texts, N: int):
    """Top-N n-grams by pooled frequency, ties by string, so top-100 is a prefix of top-300 is a prefix of top-1000."""
    c = Counter()
    for x in texts:
        c.update(grams(x))
    return [g for g, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:N]]
