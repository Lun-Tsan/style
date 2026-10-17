"""Author language model: multinomial Naive Bayes over char 1-3-grams, add-one smoothing, class prior.
margin(x) = log P(pos|x) - log P(neg|x) up to the shared normaliser (pos = partner S in the validated runs).
llr_target(x) = -margin(x) when pos = S, neg = T: positive = more like the target person."""
from __future__ import annotations

import math
from collections import Counter

from .ngrams import grams


class NB:
    def __init__(self, texts, labels, pos, neg, alpha: float = 1.0):
        self.pos, self.neg, self.alpha = pos, neg, alpha
        self.cnt = {pos: Counter(), neg: Counter()}
        ny = Counter(labels)
        for x, y in zip(texts, labels):
            self.cnt[y].update(grams(x))
        self.vocab = set(self.cnt[pos]) | set(self.cnt[neg])
        V = len(self.vocab)
        self.tot = {c: sum(self.cnt[c].values()) for c in (pos, neg)}
        self.den = {c: math.log(self.tot[c] + alpha * V) for c in (pos, neg)}
        self.prior = {c: math.log(ny[c] / len(labels)) for c in (pos, neg)}
        self.llr = {g: self.loglik(pos, g) - self.loglik(neg, g) for g in self.vocab}
        self.prior_diff = self.prior[pos] - self.prior[neg]

    def loglik(self, c, g):
        return math.log(self.cnt[c][g] + self.alpha) - self.den[c]

    def margin(self, text: str) -> float:
        z = self.prior_diff
        for g, k in Counter(grams(text)).items():
            l = self.llr.get(g)
            if l is not None:
                z += k * l
        return z

    def predict(self, text: str):
        return self.pos if self.margin(text) > 0 else self.neg

    def top_ngrams(self, k: int = 15, min_count: int = 5):
        rows = []
        for g in self.vocab:
            if self.cnt[self.pos][g] + self.cnt[self.neg][g] < min_count or not g.strip():
                continue
            rows.append((self.llr[g], g, self.cnt[self.pos][g], self.cnt[self.neg][g]))
        rows.sort()
        return rows[::-1][:k], rows[:k]
