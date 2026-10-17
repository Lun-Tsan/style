import sys
sys.stdout.reconfigure(encoding="utf-8")
import math, random
from collections import Counter
import numpy as np
import style_nb as M
from addon import ACK_RE, core, boot_pair_ci, tr, te, nb_b

def margin_loo(nb, text, who, ntr):
    g_self = Counter(M.grams(text))
    cnt = {c: nb.cnt[c] for c in (M.S, M.T)}
    removed = {g for g, k in g_self.items() if cnt[M.S][g] + cnt[M.T][g] - k == 0}
    V = len(nb.vocab) - len(removed)
    tot = dict(nb.tot); tot[who] -= sum(g_self.values())
    ny = Counter(t["who"] for t in tr); ny[who] -= 1
    den = {c: math.log(tot[c] + V) for c in (M.S, M.T)}
    z = math.log(ny[M.S]) - math.log(ny[M.T])
    for g in M.grams(text):
        if g in nb.vocab and g not in removed:
            cs = cnt[M.S][g] - (g_self[g] if who == M.S else 0)
            ct = cnt[M.T][g] - (g_self[g] if who == M.T else 0)
            z += (math.log(cs + 1) - den[M.S]) - (math.log(ct + 1) - den[M.T])
    return z

ack = [t for t in tr if t["ack"]]
for t in ack:
    t["loo"] = margin_loo(nb_b, t["text_b"], t["who"], len(tr))
stu = [t for t in ack if t["who"] == M.S]; tut = [t for t in ack if t["who"] == M.T]
p, lo, hi, ne = boot_pair_ci(stu, tut, "len_b", lambda t: t["loo"])
print(f"LOO ack (train period) pairwise: S {len(stu)} T {len(tut)} eligible {ne} acc {p:.3f} CI [{lo:.3f},{hi:.3f}]")
ys = [t["who"] for t in ack]; ps = [M.S if t["loo"] > 0 else M.T for t in ack]
print("LOO ack single-turn:", M.fmt(M.metrics(ys, ps)))
# sanity: same LOO procedure on all train short-generic turns
sg = [t for t in tr if t["short"]]
for t in sg:
    t["loo"] = margin_loo(nb_b, t["text_b"], t["who"], len(tr))
stu = [t for t in sg if t["who"] == M.S]; tut = [t for t in sg if t["who"] == M.T]
p, lo, hi, ne = boot_pair_ci(stu, tut, "len_b", lambda t: t["loo"])
print(f"LOO short-generic (train period) pairwise: S {len(stu)} T {len(tut)} eligible {ne} acc {p:.3f} CI [{lo:.3f},{hi:.3f}]")
