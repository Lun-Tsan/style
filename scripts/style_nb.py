"""Speaker separability probe: multinomial NB over char 1-3-grams (no installs, no GPU).

Train = turns dated before 2026-07-20, test = 2026-07-20 or later.
Turn = maximal run of same-speaker messages; breaks on speaker change, date change, or gap > 30 min.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import math
import random
import re
from collections import Counter
from pathlib import Path

from analyze_chat import detect_speakers, parse  # read-only import

from _local import S, T, CHAT  # noqa: E402  (local, git-ignored)
SPLIT = "2026-07-20"
GAP_MIN = 30
# S = student (positive class), T = tutor -- names come from _local
PH = {"貼圖": "\ue000", "圖片": "\ue001", "影片": "\ue002"}   # placeholder -> single private-use char
PH_SHOW = {v: f"[{k}]" for k, v in PH.items()}

CODE_RE = re.compile(r"[;{}()=<>#\[\]_\\/*+&|%^$@`]|https?:|\b(cout|cin|int|endl|return|include|for|while|if|else|main|"
                     r"vector|string|char|bool|std|void|long|double|printf|scanf)\b", re.I)
DIGIT_RE = re.compile(r"[0-9０-９]")


# ----------------------------------------------------------------------------- data
def to_min(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def build_turns():
    lines = CHAT.read_text(encoding="utf-8-sig").splitlines()
    speakers = detect_speakers(lines)
    assert set(speakers) == {S, T}, speakers
    msgs, n_sys = parse(lines, speakers)
    turns, cur = [], None
    for m in msgs:
        t = to_min(m.time)
        if cur and cur["who"] == m.who and cur["date"] == m.date and t - cur["last"] <= GAP_MIN:
            cur["msgs"].append(m.text)
            cur["last"] = t
        else:
            cur = {"who": m.who, "date": m.date, "last": t, "msgs": [m.text]}
            turns.append(cur)
    for tr in turns:
        bodies = tr["msgs"]
        tr["text_a"] = "\n".join(PH.get(b.strip(), b) for b in bodies)                    # placeholders as tokens
        kept = [b for b in bodies if b.strip() not in PH]
        tr["text_b"] = "\n".join(kept) if kept else None                                   # placeholders removed
        tr["split"] = "test" if tr["date"] >= SPLIT else "train"
        tb = tr["text_b"]
        tr["len_b"] = len(re.sub(r"\s", "", tb)) if tb else 0
        tr["len_a"] = len(re.sub(r"\s", "", tr["text_a"]))
        tr["short"] = bool(tb) and tr["len_b"] <= 10 and not CODE_RE.search(tb) and not DIGIT_RE.search(tb)
    return msgs, n_sys, turns


# ----------------------------------------------------------------------------- model
def grams(text, nmax=3):
    out = []
    for n in range(1, nmax + 1):
        out.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return out


class NB:
    def __init__(self, texts, ys, alpha=1.0):
        self.alpha = alpha
        self.cnt = {S: Counter(), T: Counter()}
        ny = Counter(ys)
        for x, y in zip(texts, ys):
            self.cnt[y].update(grams(x))
        self.vocab = set(self.cnt[S]) | set(self.cnt[T])
        V = len(self.vocab)
        self.tot = {c: sum(self.cnt[c].values()) for c in (S, T)}
        self.den = {c: math.log(self.tot[c] + alpha * V) for c in (S, T)}
        self.prior = {c: math.log(ny[c] / len(ys)) for c in (S, T)}

    def loglik(self, c, g):
        return math.log(self.cnt[c][g] + self.alpha) - self.den[c]

    def margin(self, text):
        """log P(S|x) - log P(T|x) (up to the shared normaliser)."""
        z = self.prior[S] - self.prior[T]
        for g in grams(text):
            if g in self.vocab:
                z += self.loglik(S, g) - self.loglik(T, g)
        return z

    def predict(self, text):
        return S if self.margin(text) > 0 else T

    def top_ngrams(self, k=15, min_count=5):
        rows = []
        for g in self.vocab:
            if self.cnt[S][g] + self.cnt[T][g] < min_count or not g.strip():
                continue
            rows.append((self.loglik(S, g) - self.loglik(T, g), g, self.cnt[S][g], self.cnt[T][g]))
        rows.sort()
        return rows[::-1][:k], rows[:k]


# ----------------------------------------------------------------------------- metrics
def metrics(ys, ps):
    cm = {(a, b): 0 for a in (S, T) for b in (S, T)}
    for y, p in zip(ys, ps):
        cm[(y, p)] += 1
    nS = cm[(S, S)] + cm[(S, T)]
    nT = cm[(T, S)] + cm[(T, T)]
    rS = cm[(S, S)] / nS if nS else float("nan")
    rT = cm[(T, T)] / nT if nT else float("nan")
    acc = (cm[(S, S)] + cm[(T, T)]) / len(ys)
    return {"n": len(ys), "nS": nS, "nT": nT, "acc": acc, "bal": (rS + rT) / 2, "rS": rS, "rT": rT,
            "cm": f"[[{cm[(S, S)]},{cm[(S, T)]}],[{cm[(T, S)]},{cm[(T, T)]}]]",
            "ci": clopper_pearson(cm[(S, S)] + cm[(T, T)], len(ys))}


def fit_length_rule(lens, ys):
    """Threshold + direction on turn length maximising train balanced accuracy."""
    best = None
    for th in sorted(set(lens)):
        for short_is in (S, T):
            ps = [short_is if L <= th else (T if short_is == S else S) for L in lens]
            m = metrics(ys, ps)
            key = (m["bal"], m["acc"])
            if best is None or key > best[0]:
                best = (key, th, short_is)
    _, th, short_is = best
    other = T if short_is == S else S
    return (lambda L: short_is if L <= th else other), th, short_is


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    s = 0.0
    lp, lq = math.log(p), math.log1p(-p)
    for i in range(k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(s, 1.0)


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return (float("nan"), float("nan"))

    def bisect(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else bisect(lambda p: 1 - binom_cdf(k - 1, n, p) >= a / 2)   # P(X>=k) rises with p
    hi = 1.0 if k == n else bisect(lambda p: binom_cdf(k, n, p) <= a / 2)            # P(X<=k) falls with p
    return lo, hi


# ----------------------------------------------------------------------------- pairwise
def sample_pairs(student, tutor, lenkey, n, rng):
    """One student turn + one tutor turn whose length is within +-30% of the student turn's."""
    pairs, seen, tries = [], set(), 0
    by_len = {}
    for j, t in enumerate(tutor):
        by_len.setdefault(t[lenkey], []).append(j)
    while len(pairs) < n and tries < 200000:
        tries += 1
        i = rng.randrange(len(student))
        L = student[i][lenkey]
        cands = [j for l2, js in by_len.items() if 0.7 * L <= l2 <= 1.3 * L for j in js]
        if not cands:
            continue
        j = rng.choice(cands)
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((student[i], tutor[j]))
    n_possible = sum(1 for s in student for t in tutor if 0.7 * s[lenkey] <= t[lenkey] <= 1.3 * s[lenkey])
    return pairs, n_possible


def pair_acc(pairs, score, rng):
    """score(turn) higher = more student-like; ties broken by coin flip."""
    k = 0
    for s, t in pairs:
        a, b = score(s), score(t)
        k += 1 if a > b else (0 if a < b else rng.random() < 0.5)
    return k, len(pairs)


# ----------------------------------------------------------------------------- main
def fmt(m):
    return (f"{m['n']} ({m['nS']}/{m['nT']}) | {m['acc']:.3f} | {m['bal']:.3f} | {m['rS']:.3f} | {m['rT']:.3f} | "
            f"{m['cm']} | [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")


def run_variant(name, train, test, tkey, lkey, rows):
    ytr = [t["who"] for t in train]
    yte = [t["who"] for t in test]
    nb = NB([t[tkey] for t in train], ytr)
    maj = Counter(ytr).most_common(1)[0][0]
    rule, th, short_is = fit_length_rule([t[lkey] for t in train], ytr)
    rows.append((name, "majority (" + ("S" if maj == S else "T") + ")", metrics(yte, [maj] * len(yte))))
    rows.append((name, f"length rule (len<={th} -> {'S' if short_is == S else 'T'})",
                 metrics(yte, [rule(t[lkey]) for t in test])))
    rows.append((name, "NB char 1-3g", metrics(yte, [nb.predict(t[tkey]) for t in test])))
    return nb, rule, short_is


def main():
    msgs, n_sys, turns = build_turns()
    tr_all = [t for t in turns if t["split"] == "train"]
    te_all = [t for t in turns if t["split"] == "test"]
    print(f"messages {len(msgs)} (system lines skipped {n_sys}); turns {len(turns)}: "
          f"train {len(tr_all)} {dict(Counter(t['who'] for t in tr_all))}, test {len(te_all)} {dict(Counter(t['who'] for t in te_all))}")
    print("test date range", min(t['date'] for t in te_all), max(t['date'] for t in te_all))

    # placeholder usage per person (message level)
    ph_cnt = Counter((m.who, m.text.strip()) for m in msgs if m.text.strip() in PH)
    per = Counter(m.who for m in msgs)
    for who in (S, T):
        print(f"placeholder msgs {who}: " + ", ".join(f"{k} {ph_cnt[(who, k)]}" for k in PH) + f" / {per[who]} msgs")

    rows = []
    # (a) all turns, placeholders as tokens
    nb_a, _, _ = run_variant("(a) all, placeholders as tokens", tr_all, te_all, "text_a", "len_a", rows)
    # (b) placeholders removed
    tr_b = [t for t in tr_all if t["text_b"]]
    te_b = [t for t in te_all if t["text_b"]]
    nb_b, rule_b, short_is_b = run_variant("(b) placeholders removed", tr_b, te_b, "text_b", "len_b", rows)
    # (c) short generic test subset, model (b)
    te_c = [t for t in te_b if t["short"]]
    tr_c = [t for t in tr_b if t["short"]]
    yte_c = [t["who"] for t in te_c]
    ytr_b = [t["who"] for t in tr_b]
    maj_b = Counter(ytr_b).most_common(1)[0][0]
    name_c = "(c) short generic test, model (b)"
    rows.append((name_c, "majority (" + ("S" if maj_b == S else "T") + ")", metrics(yte_c, [maj_b] * len(yte_c))))
    rows.append((name_c, "length rule from (b)", metrics(yte_c, [rule_b(t["len_b"]) for t in te_c])))
    rows.append((name_c, "NB char 1-3g", metrics(yte_c, [nb_b.predict(t["text_b"]) for t in te_c])))
    # (c') short generic both sides
    nb_c, rule_c, short_is_c = run_variant("(c') short generic train+test", tr_c, te_c, "text_b", "len_b", rows)

    print("\n| variant | method | n test (S/T) | acc | bal acc | recall S | recall T | CM [[SS,ST],[TS,TT]] | acc 95% CI |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v, meth, m in rows:
        print(f"| {v} | {meth} | {fmt(m)} |")

    # ------------------------------------------------------------- indicative n-grams
    print("\nTop n-grams, model (b) [count S / count T], LLR = log P(g|S) - log P(g|T):")
    topS, topT = nb_b.top_ngrams(15, 5)
    show = lambda g: repr(g.replace("\n", "⏎"))
    print("S:", "  ".join(f"{show(g)} {llr:+.2f} ({cs}/{ct})" for llr, g, cs, ct in topS))
    print("T:", "  ".join(f"{show(g)} {llr:+.2f} ({cs}/{ct})" for llr, g, cs, ct in topT))
    print("\nTop n-grams, short-generic-only model (c') min count 5:")
    topS, topT = nb_c.top_ngrams(15, 5)
    print("S:", "  ".join(f"{show(g)} {llr:+.2f} ({cs}/{ct})" for llr, g, cs, ct in topS))
    print("T:", "  ".join(f"{show(g)} {llr:+.2f} ({cs}/{ct})" for llr, g, cs, ct in topT))
    print("\nTop n-grams, model (a) (placeholders as tokens):")
    topS, topT = nb_a.top_ngrams(8, 5)
    show2 = lambda g: repr("".join(PH_SHOW.get(ch, ch) for ch in g).replace("\n", "⏎"))
    print("S:", "  ".join(f"{show2(g)} {llr:+.2f} ({cs}/{ct})" for llr, g, cs, ct in topS))
    print("T:", "  ".join(f"{show2(g)} {llr:+.2f} ({cs}/{ct})" for llr, g, cs, ct in topT))

    # ------------------------------------------------------------- pairwise
    print("\nPairwise (200 sampled test pairs, seed 0; pick the more student-like turn):")
    print("| pair pool | scorer | n pairs | possible pairs | acc | 95% CI (Clopper-Pearson) |")
    print("|---|---|---|---|---|---|")
    configs = [
        ("(b) all non-placeholder test turns", te_b, "len_b", [("NB (b)", lambda t: nb_b.margin(t["text_b"])),
                                                              ("length only", lambda t: (-1 if short_is_b == T else 1) * -t["len_b"])]),
        ("(a) all test turns, placeholders as tokens", te_all, "len_a", [("NB (a)", lambda t: nb_a.margin(t["text_a"]))]),
        ("(c) short generic test turns", te_c, "len_b", [("NB (b)", lambda t: nb_b.margin(t["text_b"])),
                                                       ("NB (c')", lambda t: nb_c.margin(t["text_b"])),
                                                       ("length only", lambda t: (-1 if short_is_b == T else 1) * -t["len_b"])]),
    ]
    for pool_name, pool, lk, scorers in configs:
        stu = [t for t in pool if t["who"] == S]
        tut = [t for t in pool if t["who"] == T]
        pairs, n_possible = sample_pairs(stu, tut, lk, 200, random.Random(0))
        for sname, sc in scorers:
            k, n = pair_acc(pairs, sc, random.Random(0))
            lo, hi = clopper_pearson(int(k), n)
            print(f"| {pool_name} | {sname} | {n} | {n_possible} | {k / n:.3f} | [{lo:.3f}, {hi:.3f}] |")
        # exhaustive (all eligible pairs, not independent) for the NB scorer
        sc = scorers[0][1]
        ms = {id(t): sc(t) for t in stu + tut}
        tot = win = 0.0
        for s in stu:
            for t in tut:
                if 0.7 * s[lk] <= t[lk] <= 1.3 * s[lk]:
                    tot += 1
                    win += 1 if ms[id(s)] > ms[id(t)] else (0.5 if ms[id(s)] == ms[id(t)] else 0)
        print(f"|   exhaustive check | {scorers[0][0]} | all {int(tot)} | - | {win / tot:.3f} | (pairs share turns; no CI) |")

    # lengths summary
    for who in (S, T):
        L = sorted(t["len_b"] for t in te_b if t["who"] == who)
        print(f"test median len_b {who}: {L[len(L) // 2]}  short-generic share: "
              f"{sum(1 for t in te_c if t['who'] == who)}/{len(L)}")


if __name__ == "__main__":
    main()
