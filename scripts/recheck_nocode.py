"""Re-validate the rulers after removing symbol/code-like turns (pre-registered follow-up of experiment 3).

Variants:  filter in {none, loose (style_nb.CODE_RE), strict (keywords/URL/;{} OR >=20 chars with >=30% letter/operator share)}
E1  single-turn attribution on the filtered test pool (NB, Min-max, Cosine Delta, NCD), thr 0 and OOF-tuned, + AUC
E3  bags k in (5,10,25,50): thr 0, per-k OOF-TRAIN-calibrated threshold, AUC (+200-rep turn bootstrap for count-based/NB)
E4  within-person vs cross-person bag distances, k in (25,50): (a) random periods, (b) period-matched (same date block)
Pure Python + numpy, no GPU.  Reuses formal_lib (measures) and formal2_lib (auc).
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
import formal_lib as L  # noqa: E402
from formal_lib import S, T, M  # noqa: E402
import formal2_lib as L2  # noqa: E402

KS = (5, 10, 25, 50)
NBAG = 150          # test bags per person per k
NOOF = 40           # OOF train bags per person per block per k
NBOOT = 200
COUNT = ["M2_N300", "M3_N300", "M5"]          # count-based keys reported (Cosine Delta, Min-max, NB)
ALLKEYS = COUNT + ["M6"]

KW = re.compile(r"https?://|www\.|[;{}]|\b(cout|cin|int|endl|return|include|for|while|if|else|main|vector|string|char|"
                r"bool|std|void|long|double|printf|scanf|def|import|print|float|size_t)\b", re.I)
SYM = re.compile(r"[A-Za-z+\-*/=^_<>\[\]{}()|\\%$#@`]")


def symbol_share(t):
    s = re.sub(r"\s", "", t)
    return (len(SYM.findall(s)) / len(s)) if s else 0.0, len(s)


def flag_strict(t):
    sh, n = symbol_share(t)
    return bool(KW.search(t)) or (n >= 20 and sh >= 0.30)


def flag_loose(t):
    return bool(M.CODE_RE.search(t))


FILTERS = {"none": lambda t: False, "loose": flag_loose, "strict": flag_strict}


def char_share(pool, flag):
    tot = sum(len(t["text_b"]) for t in pool)
    fl = sum(len(t["text_b"]) for t in pool if flag(t["text_b"]))
    return fl / tot if tot else 0.0


def scores_single(model, store, idx, ref):
    X, tot, mar, joined = store.features([[i] for i in idx])
    sc = model.scores_from_counts(X, tot, mar)
    out = {k: sc[k] for k in COUNT}
    out["M6"] = ref.score(joined, "zlib", "xy")
    return out


def scores_bags(model, store, bags, ref, with_ncd=True):
    X, tot, mar, joined = store.features(bags)
    sc = model.scores_from_counts(X, tot, mar)
    out = {k: sc[k] for k in COUNT}
    if with_ncd:
        out["M6"] = ref.score(joined, "zlib", "xy")
    return out


def draw_bags(iS, iT, k, n, rng):
    bS = [iS[rng.choice(len(iS), k, replace=False)] for _ in range(n)]
    bT = [iT[rng.choice(len(iT), k, replace=False)] for _ in range(n)]
    return bS, bT


def run_variant(name, flag, tr_all, te_all, res):
    t0 = time.perf_counter()
    tr = [t for t in tr_all if not flag(t["text_b"])]
    te = [t for t in te_all if not flag(t["text_b"])]
    cnt = lambda p: (sum(t["who"] == S for t in p), sum(t["who"] == T for t in p))
    r = {"train_S_T": cnt(tr), "test_S_T": cnt(te),
         "removed_train": len(tr_all) - len(tr), "removed_test": len(te_all) - len(te),
         "char_share_flagged": {"train_S": char_share([t for t in tr_all if t["who"] == S], flag),
                                "train_T": char_share([t for t in tr_all if t["who"] == T], flag),
                                "test_S": char_share([t for t in te_all if t["who"] == S], flag),
                                "test_T": char_share([t for t in te_all if t["who"] == T], flag)}}
    print(f"\n=== filter={name}: train S/T {r['train_S_T']}, test S/T {r['test_S_T']}, removed {r['removed_train']}/{r['removed_test']}",
          flush=True)

    model = L.Model(tr)
    ref = L.NCDRef(tr, codecs=("zlib",))
    te_texts = [t["text_b"] for t in te]
    y_te = np.array([1 if t["who"] == S else 0 for t in te])
    store_te = L.Store(model, te_texts)
    sc_te = scores_single(model, store_te, range(len(te)), ref)

    # ---- OOF on TRAIN: single-turn scores + bag scores per block (for thresholds)
    blocks = L.date_blocks(tr)
    oof_single = {k: np.zeros(len(tr)) for k in ALLKEYS}
    oof_bags = {k: {K: ([], []) for K in KS} for k in ALLKEYS}   # scores for S bags, T bags
    rng = np.random.default_rng(0)
    for fi, held in enumerate(blocks):
        hs = set(held)
        train_f = [t for i, t in enumerate(tr) if i not in hs]
        mf = L.Model(train_f)
        rf = L.NCDRef(train_f, codecs=("zlib",))
        sf = L.Store(mf, [tr[i]["text_b"] for i in held])
        sc = scores_single(mf, sf, range(len(held)), rf)
        for k in ALLKEYS:
            oof_single[k][held] = sc[k]
        hS = np.array([j for j, i in enumerate(held) if tr[i]["who"] == S])
        hT = np.array([j for j, i in enumerate(held) if tr[i]["who"] == T])
        for K in KS:
            if len(hS) < K or len(hT) < K:
                continue
            bS, bT = draw_bags(hS, hT, K, NOOF, rng)
            sb = scores_bags(mf, sf, bS + bT, rf)
            for k in ALLKEYS:
                oof_bags[k][K][0].extend(sb[k][:NOOF])
                oof_bags[k][K][1].extend(sb[k][NOOF:])
    y_tr = np.array([1 if t["who"] == S else 0 for t in tr])

    # ---- E1
    e1 = {}
    for k in ALLKEYS:
        thr, _, _ = L.tune_threshold(oof_single[k], y_tr)
        a0, rS0, rT0 = L2.acc_parts(sc_te[k], y_te, 0.0)
        a1, rS1, rT1 = L2.acc_parts(sc_te[k], y_te, thr)
        e1[k] = {"acc_thr0": a0, "recS_thr0": rS0, "recT_thr0": rT0, "acc_tuned": a1, "recS_tuned": rS1,
                 "recT_tuned": rT1, "thr": thr, "auc": L2.auc(sc_te[k], y_te),
                 "ci_thr0": M.clopper_pearson(int(round(a0 * len(y_te))), len(y_te)),
                 "ci_tuned": M.clopper_pearson(int(round(a1 * len(y_te))), len(y_te))}
    e1["n"] = int(len(y_te))
    e1["length_rule"] = None
    r["E1"] = e1
    print("E1 (acc thr0 / tuned / AUC):", {k: (round(v["acc_thr0"], 3), round(v["acc_tuned"], 3), round(v["auc"], 3))
                                           for k, v in e1.items() if isinstance(v, dict)}, flush=True)

    # ---- E3 test bags
    iS = np.array([i for i, t in enumerate(te) if t["who"] == S])
    iT = np.array([i for i, t in enumerate(te) if t["who"] == T])
    rng = np.random.default_rng(0)
    e3 = {}
    thr_k = {k: {} for k in ALLKEYS}
    for K in KS:
        for k in ALLKEYS:
            sS, sT = oof_bags[k][K]
            if sS:
                thr_k[k][K], _, _ = L.tune_threshold(np.array(sS + sT), np.array([1] * len(sS) + [0] * len(sT)))
            else:
                thr_k[k][K] = 0.0
        bS, bT = draw_bags(iS, iT, K, NBAG, rng)
        sb = scores_bags(model, store_te, bS + bT, ref)
        yb = np.array([1] * NBAG + [0] * NBAG)
        e3[K] = {}
        for k in ALLKEYS:
            a0 = L2.acc_parts(sb[k], yb, 0.0)
            a1 = L2.acc_parts(sb[k], yb, thr_k[k][K])
            e3[K][k] = {"acc_thr0": a0[0], "acc_cal": a1[0], "recS_cal": a1[1], "recT_cal": a1[2],
                        "auc": L2.auc(sb[k], yb), "thr": thr_k[k][K]}
        print(f"E3 k={K}:", {k: (round(v['acc_thr0'], 3), round(v['acc_cal'], 3), round(v['auc'], 3)) for k, v in e3[K].items()},
              flush=True)
    # bootstrap (count-based + NB only; thresholds fixed)
    boot = {K: {k: {"acc_cal": [], "auc": []} for k in COUNT} for K in KS}
    rngb = np.random.default_rng(1)
    for _ in range(NBOOT):
        rS = np.unique(rngb.choice(iS, len(iS), replace=True))
        rT = np.unique(rngb.choice(iT, len(iT), replace=True))
        for K in KS:
            if len(rS) < K or len(rT) < K:
                continue
            bS, bT = draw_bags(rS, rT, K, NBAG, rngb)
            sb = scores_bags(model, store_te, bS + bT, ref, with_ncd=False)
            yb = np.array([1] * NBAG + [0] * NBAG)
            for k in COUNT:
                boot[K][k]["acc_cal"].append(L2.acc_parts(sb[k], yb, thr_k[k][K])[0])
                boot[K][k]["auc"].append(L2.auc(sb[k], yb))
    for K in KS:
        for k in COUNT:
            for m in ("acc_cal", "auc"):
                v = boot[K][k][m]
                e3[K][k][m + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    r["E3"] = {str(K): v for K, v in e3.items()}

    # ---- E4 within vs cross (train), random periods vs period-matched
    def bag_dist(meas, Xa, ta, Xb, tb):
        return float(model.dist(meas, 300, Xa[None, :], np.array([ta]), Xb[None, :], np.array([tb]))[0])

    def ncd_pair(a, b):
        x, y = a.encode("utf-8"), b.encode("utf-8")
        return L.ncd(x, y, L.clen(x, "zlib"), L.clen(y, "zlib"), "zlib", "xy")

    store_tr = L.Store(model, [t["text_b"] for t in tr])
    jS = np.array([i for i, t in enumerate(tr) if t["who"] == S])
    jT = np.array([i for i, t in enumerate(tr) if t["who"] == T])
    blk_of = np.zeros(len(tr), int)
    for fi, held in enumerate(blocks):
        blk_of[held] = fi
    e4 = {}
    for mode in ("random", "period_matched"):
        for K in (25, 50):
            rng4 = np.random.default_rng(2)
            res4 = {m: {"within_T": [], "within_S": [], "cross": []} for m in ("M1", "M2", "M3", "M4", "M6")}
            nrep = 1000
            done = 0
            tries = 0
            while done < nrep and tries < nrep * 5:
                tries += 1
                if mode == "random":
                    pS, pT = jS, jT
                else:
                    fi = rng4.integers(len(blocks))
                    pS = jS[blk_of[jS] == fi]
                    pT = jT[blk_of[jT] == fi]
                if len(pS) < 2 * K or len(pT) < 2 * K:
                    continue
                sel_S = rng4.choice(pS, 2 * K, replace=False)
                sel_T = rng4.choice(pT, 2 * K, replace=False)
                bags = [sel_T[:K], sel_T[K:], sel_S[:K], sel_S[K:]]
                X, tot, mar, joined = store_tr.features(bags)
                for m in ("M1", "M2", "M3", "M4"):
                    res4[m]["within_T"].append(bag_dist(m, X[0], tot[0], X[1], tot[1]))
                    res4[m]["within_S"].append(bag_dist(m, X[2], tot[2], X[3], tot[3]))
                    res4[m]["cross"].append(bag_dist(m, X[0], tot[0], X[2], tot[2]))
                res4["M6"]["within_T"].append(ncd_pair(joined[0], joined[1]))
                res4["M6"]["within_S"].append(ncd_pair(joined[2], joined[3]))
                res4["M6"]["cross"].append(ncd_pair(joined[0], joined[2]))
                done += 1
            summ = {}
            for m, d in res4.items():
                wT, wS, cr = map(np.array, (d["within_T"], d["within_S"], d["cross"]))
                if len(cr) == 0:
                    continue
                summ[m] = {"n": int(len(cr)), "within_T": float(wT.mean()), "within_S": float(wS.mean()), "cross": float(cr.mean()),
                           "ratio": float((cr / ((wT + wS) / 2)).mean()),
                           "p_cross_gt_max": float((cr > np.maximum(wT, wS)).mean()),
                           "p_cross_gt_mean": float((cr > (wT + wS) / 2).mean())}
            e4[f"{mode}_k{K}"] = summ
            print(f"E4 {mode} k={K}:", {m: round(v["p_cross_gt_max"], 3) for m, v in summ.items()}, flush=True)
    r["E4"] = e4
    r["seconds"] = time.perf_counter() - t0
    res[name] = r


def main():
    msgs, n_sys, turns = M.build_turns()
    tr_all = [t for t in turns if t["split"] == "train" and t["text_b"]]
    te_all = [t for t in turns if t["split"] == "test" and t["text_b"]]
    res = {"symbol_share_by_block": {}}
    # symbol share by date block (strict rule), both persons, train blocks + test
    blocks = L.date_blocks(tr_all)
    for fi, held in enumerate(blocks):
        pool = [tr_all[i] for i in held]
        res["symbol_share_by_block"][f"train_block{fi}_{pool[0]['date']}..{pool[-1]['date']}"] = {
            "S": char_share([t for t in pool if t["who"] == S], flag_strict),
            "T": char_share([t for t in pool if t["who"] == T], flag_strict)}
    res["symbol_share_by_block"]["test"] = {"S": char_share([t for t in te_all if t["who"] == S], flag_strict),
                                            "T": char_share([t for t in te_all if t["who"] == T], flag_strict)}
    print("strict symbol/code char share by block:", json.dumps(res["symbol_share_by_block"], indent=0, ensure_ascii=False))
    for name, flag in FILTERS.items():
        run_variant(name, flag, tr_all, te_all, res)
    with open(os.path.join(SCR, "recheck_nocode.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\nsaved recheck_nocode.json")


if __name__ == "__main__":
    main()
