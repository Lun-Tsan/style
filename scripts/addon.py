import sys
sys.stdout.reconfigure(encoding="utf-8")
import re, random
import numpy as np
from collections import Counter
import style_nb as M

ACK_RE = re.compile(r"^(好|嗯|恩|對|可以|OK|Ok|ok|okay|喔|哦|噢|沒錯|確實|是|了解|收到|沒問題|行|當然|"
                    r"啊|呀|的|吧|耶|欸|呢|啦|了|哈|唷|喲|囉|哇)+$")
PUNCT = re.compile(r"[\s!！～~。，,.、?？…:：）)(（]+")

def core(t):
    return PUNCT.sub("", t)

def boot_pair_ci(stu, tut, lk, score, B=2000, seed=0):
    s = np.array([score(t) for t in stu]); u = np.array([score(t) for t in tut])
    ls = np.array([t[lk] for t in stu], float); lu = np.array([t[lk] for t in tut], float)
    elig = (lu[None, :] >= 0.7 * ls[:, None]) & (lu[None, :] <= 1.3 * ls[:, None])
    win = np.where(s[:, None] > u[None, :], 1.0, np.where(s[:, None] == u[None, :], 0.5, 0.0))
    point = (win * elig).sum() / elig.sum()
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(stu), len(stu)); j = rng.integers(0, len(tut), len(tut))
        e = elig[np.ix_(i, j)]; w = win[np.ix_(i, j)]
        if e.sum(): bs.append((w * e).sum() / e.sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, lo, hi, int(elig.sum())

msgs, n_sys, turns = M.build_turns()
tr = [t for t in turns if t["split"] == "train" and t["text_b"]]
te = [t for t in turns if t["split"] == "test" and t["text_b"]]
nb_b = M.NB([t["text_b"] for t in tr], [t["who"] for t in tr])
tr_c = [t for t in tr if t["short"]]
nb_c = M.NB([t["text_b"] for t in tr_c], [t["who"] for t in tr_c])
print("train short-generic turns", dict(Counter(t["who"] for t in tr_c)))
for t in turns:
    t["ack"] = bool(t["text_b"]) and t["short"] and bool(core(t["text_b"])) and bool(ACK_RE.match(core(t["text_b"])))
te_ack = [t for t in te if t["ack"]]; tr_ack = [t for t in tr if t["ack"]]
print("ack turns train", dict(Counter(t["who"] for t in tr_ack)), "test", dict(Counter(t["who"] for t in te_ack)))
m = M.metrics([t["who"] for t in te_ack], [nb_b.predict(t["text_b"]) for t in te_ack])
print("single-turn NB(b) on ack test:", M.fmt(m))
sc_b = lambda t: nb_b.margin(t["text_b"]); sc_c = lambda t: nb_c.margin(t["text_b"])
for name, pool in [("(b) all", te), ("(c) short generic", [t for t in te if t["short"]]), ("ack-only", te_ack)]:
    stu = [t for t in pool if t["who"] == M.S]; tut = [t for t in pool if t["who"] == M.T]
    for sn, sc in [("NB(b)", sc_b), ("NB(c')", sc_c)]:
        p, lo, hi, ne = boot_pair_ci(stu, tut, "len_b", sc)
        print(f"exhaustive pairwise {name} {sn}: distinct S {len(stu)} T {len(tut)}, eligible pairs {ne}, acc {p:.3f}, 2-way bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    # sampled 200 on ack pool
    if name == "ack-only":
        pairs, npos = M.sample_pairs(stu, tut, "len_b", 200, random.Random(0))
        k, n = M.pair_acc(pairs, sc_b, random.Random(0))
        lo, hi = M.clopper_pearson(int(k), n)
        print(f"sampled ack pairs n={n} possible={npos} acc {k/n:.3f} CP CI [{lo:.3f},{hi:.3f}]")
# what the ack turns look like in aggregate: top core forms per person (short strings <=8 chars)
for who in (M.S, M.T):
    print(who, Counter(core(t["text_b"]) for t in tr_ack + te_ack if t["who"] == who).most_common(8))
# tilde / particle rates on all non-placeholder turns
for who in (M.S, M.T):
    ts = [t["text_b"] for t in tr + te if t["who"] == who]
    rate = lambda pat: sum(1 for x in ts if re.search(pat, x)) / len(ts)
    print(who, "turns", len(ts), "has ~/～ %.3f" % rate("[~～]"), "has 喔 %.3f" % rate("喔"), "has 嗯 %.3f" % rate("嗯"), "has ! %.3f" % rate("[!！]"))
