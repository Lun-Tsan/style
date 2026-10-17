"""Ruler validation E1 / E3 / E4 / E5 (port of recheck_nocode.py) with pre-registered pass criteria.

run(turns, flag, text_fn) -> dict with the same structure as recheck_nocode.json plus E5;
judge(result) -> pass/fail per criterion (config.E3_NB_K10_MIN, E4_MIN, E4_MAX_DROP, E5_NB_MIN).
Pure Python + numpy; ~80 s per rule with default sizes.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .. import config as C
from ..flag_symbols import CODE_RE, DIGIT_RE
from ..split import date_blocks
from .ncd import NCDRef, ncd_pair
from .profiles import Model, Store
from .stats import acc_parts, auc, clopper_pearson, pct_ci, tune_threshold

S, T = C.PARTNER, C.TARGET
KS = (5, 10, 25, 50)
COUNT = ["M2", "M3", "M5"]
ALLKEYS = COUNT + ["M6"]
E4_MEAS = ("M1", "M2", "M3", "M4", "M6")

# agree/acknowledge subset (addon.py definition, verbatim): short generic turn whose punctuation-stripped core is
# made only of acknowledgement tokens
ACK_RE = re.compile(r"^(好|嗯|恩|對|可以|OK|Ok|ok|okay|喔|哦|噢|沒錯|確實|是|了解|收到|沒問題|行|當然|"
                    r"啊|呀|的|吧|耶|欸|呢|啦|了|哈|唷|喲|囉|哇)+$")
PUNCT = re.compile(r"[\s!！～~。，,.、?？…:：）)(（]+")


def is_short(text: str) -> bool:
    return len(re.sub(r"\s", "", text)) <= 10 and not CODE_RE.search(text) and not DIGIT_RE.search(text)


def is_ack(text: str) -> bool:
    core = PUNCT.sub("", text)
    return is_short(text) and bool(core) and bool(ACK_RE.match(core))


def _char_share(texts, flags):
    tot = sum(len(x) for x in texts)
    return (sum(len(x) for x, f in zip(texts, flags) if f) / tot) if tot else 0.0


def _draw(iS, iT, k, n, rng):
    bS = [iS[rng.choice(len(iS), k, replace=False)] for _ in range(n)]
    bT = [iT[rng.choice(len(iT), k, replace=False)] for _ in range(n)]
    return bS, bT


def _scores(model, store, bags, ref, with_ncd=True):
    X, tot, mar, joined = store.features(bags)
    sc = model.scores_from_counts(X, tot, mar)
    out = {k: sc[k] for k in COUNT}
    if with_ncd:
        out["M6"] = ref.score(joined)
    return out


def run(turns, flag, text_fn=lambda t: t.ruler_text, name="v1", ks=KS, nbag=150, noof=40, nboot=200,
        e4_reps=1000, e4_ks=(25, 50), log=print):
    """turns: chronological Turn list (train + test). flag(turn) -> True if the turn is excluded (symbol-like).
    text_fn(turn) -> the text the rulers see (ruler_text in v1; text_b to reproduce experiment 4)."""
    t0 = time.perf_counter()
    tr_all = [t for t in turns if t.split == "train" and text_fn(t)]
    te_all = [t for t in turns if t.split == "test" and text_fn(t)]
    tr = [t for t in tr_all if not flag(t)]
    te = [t for t in te_all if not flag(t)]
    cnt = lambda p: [sum(t.who == S for t in p), sum(t.who == T for t in p)]
    r = {"name": name, "train_S_T": cnt(tr), "test_S_T": cnt(te),
         "removed_train": len(tr_all) - len(tr), "removed_test": len(te_all) - len(te),
         "removed_train_S_T": [a - b for a, b in zip(cnt(tr_all), cnt(tr))],
         "char_share_flagged": {f"{sp}_{C.CODE[P]}": _char_share([text_fn(t) for t in pool if t.who == P],
                                                         [flag(t) for t in pool if t.who == P])
                                for sp, pool in (("train", tr_all), ("test", te_all)) for P in (S, T)}}
    log(f"[{name}] train S/T {r['train_S_T']} test S/T {r['test_S_T']} removed {r['removed_train']}/{r['removed_test']}")

    texts_tr, whos_tr = [text_fn(t) for t in tr], [t.who for t in tr]
    model = Model(texts_tr, whos_tr)
    ref = NCDRef(texts_tr, whos_tr)
    te_texts = [text_fn(t) for t in te]
    y_te = np.array([1 if t.who == S else 0 for t in te])
    store_te = Store(model, te_texts)
    sc_te = _scores(model, store_te, [[i] for i in range(len(te))], ref)

    # ---- out-of-fold TRAIN scores (single turns and bags) over contiguous date blocks
    blocks = date_blocks(tr)
    oof_single = {k: np.zeros(len(tr)) for k in ALLKEYS}
    oof_bags = {k: {K: ([], []) for K in ks} for k in ALLKEYS}
    rng = np.random.default_rng(0)
    for fi, held in enumerate(blocks):
        hs = set(held)
        tf = [t for i, t in enumerate(tr) if i not in hs]
        mf = Model([text_fn(t) for t in tf], [t.who for t in tf])
        rf = NCDRef([text_fn(t) for t in tf], [t.who for t in tf])
        sf = Store(mf, [text_fn(tr[i]) for i in held])
        sc = _scores(mf, sf, [[j] for j in range(len(held))], rf)
        for k in ALLKEYS:
            oof_single[k][held] = sc[k]
        hS = np.array([j for j, i in enumerate(held) if tr[i].who == S])
        hT = np.array([j for j, i in enumerate(held) if tr[i].who == T])
        for K in ks:
            if len(hS) < K or len(hT) < K:
                continue
            bS, bT = _draw(hS, hT, K, noof, rng)
            sb = _scores(mf, sf, bS + bT, rf)
            for k in ALLKEYS:
                oof_bags[k][K][0].extend(sb[k][:noof])
                oof_bags[k][K][1].extend(sb[k][noof:])
    y_tr = np.array([1 if t.who == S else 0 for t in tr])
    r["oof_single_nb_margin"] = oof_single["M5"].tolist()     # for the Calibrator (partner-likeness; negate for T)
    r["oof_y_is_S"] = y_tr.tolist()

    # ---- E1 single turn
    e1 = {"n": int(len(y_te))}
    for k in ALLKEYS:
        thr, _, _ = tune_threshold(oof_single[k], y_tr)
        a0, rS0, rT0 = acc_parts(sc_te[k], y_te, 0.0)
        a1, rS1, rT1 = acc_parts(sc_te[k], y_te, thr)
        e1[k] = {"acc_thr0": a0, "recS_thr0": rS0, "recT_thr0": rT0, "acc_tuned": a1, "recS_tuned": rS1,
                 "recT_tuned": rT1, "thr": thr, "auc": auc(sc_te[k], y_te),
                 "ci_thr0": clopper_pearson(int(round(a0 * len(y_te))), len(y_te)),
                 "ci_tuned": clopper_pearson(int(round(a1 * len(y_te))), len(y_te))}
    r["E1"] = e1
    log(f"[{name}] E1 acc thr0/tuned/AUC: " + ", ".join(f"{k} {e1[k]['acc_thr0']:.3f}/{e1[k]['acc_tuned']:.3f}/{e1[k]['auc']:.3f}" for k in ALLKEYS))

    # ---- E3 test bags with per-k OOF thresholds + AUC (+ turn bootstrap for count-based measures)
    iS = np.array([i for i, t in enumerate(te) if t.who == S])
    iT = np.array([i for i, t in enumerate(te) if t.who == T])
    rng = np.random.default_rng(0)
    e3, thr_k = {}, {k: {} for k in ALLKEYS}
    for K in ks:
        for k in ALLKEYS:
            sS, sT = oof_bags[k][K]
            thr_k[k][K] = tune_threshold(np.array(sS + sT), np.array([1] * len(sS) + [0] * len(sT)))[0] if sS else 0.0
        bS, bT = _draw(iS, iT, K, nbag, rng)
        sb = _scores(model, store_te, bS + bT, ref)
        yb = np.array([1] * nbag + [0] * nbag)
        e3[str(K)] = {}
        for k in ALLKEYS:
            a0 = acc_parts(sb[k], yb, 0.0)
            a1 = acc_parts(sb[k], yb, thr_k[k][K])
            e3[str(K)][k] = {"acc_thr0": a0[0], "acc_cal": a1[0], "recS_cal": a1[1], "recT_cal": a1[2],
                             "auc": auc(sb[k], yb), "thr": thr_k[k][K]}
        log(f"[{name}] E3 k={K} cal acc / AUC: " + ", ".join(f"{k} {e3[str(K)][k]['acc_cal']:.3f}/{e3[str(K)][k]['auc']:.3f}" for k in ALLKEYS))
    if nboot:
        boot = {K: {k: {"acc_cal": [], "auc": []} for k in COUNT} for K in ks}
        rngb = np.random.default_rng(1)
        for _ in range(nboot):
            rS = np.unique(rngb.choice(iS, len(iS), replace=True))
            rT = np.unique(rngb.choice(iT, len(iT), replace=True))
            for K in ks:
                if len(rS) < K or len(rT) < K:
                    continue
                bS, bT = _draw(rS, rT, K, nbag, rngb)
                sb = _scores(model, store_te, bS + bT, ref, with_ncd=False)
                yb = np.array([1] * nbag + [0] * nbag)
                for k in COUNT:
                    boot[K][k]["acc_cal"].append(acc_parts(sb[k], yb, thr_k[k][K])[0])
                    boot[K][k]["auc"].append(auc(sb[k], yb))
        for K in ks:
            for k in COUNT:
                for m in ("acc_cal", "auc"):
                    e3[str(K)][k][m + "_ci"] = pct_ci(boot[K][k][m])
    r["E3"] = e3

    # ---- E4 within-person vs cross-person bag distances (TRAIN), random vs period-matched
    def bag_dist(meas, Xa, ta, Xb, tb):
        return float(model.dist(meas, C.VOCAB_N, Xa[None, :], np.array([ta]), Xb[None, :], np.array([tb]))[0])

    store_tr = Store(model, texts_tr)
    jS = np.array([i for i, t in enumerate(tr) if t.who == S])
    jT = np.array([i for i, t in enumerate(tr) if t.who == T])
    blk_of = np.zeros(len(tr), int)
    for fi, held in enumerate(blocks):
        blk_of[held] = fi
    e4 = {}
    for mode in ("random", "period_matched"):
        for K in e4_ks:
            rng4 = np.random.default_rng(2)
            res4 = {m: {"within_T": [], "within_S": [], "cross": []} for m in E4_MEAS}
            done = tries = 0
            while done < e4_reps and tries < e4_reps * 5:
                tries += 1
                if mode == "random":
                    pS, pT = jS, jT
                else:
                    fi = rng4.integers(len(blocks))
                    pS, pT = jS[blk_of[jS] == fi], jT[blk_of[jT] == fi]
                if len(pS) < 2 * K or len(pT) < 2 * K:
                    continue
                sel_S = rng4.choice(pS, 2 * K, replace=False)
                sel_T = rng4.choice(pT, 2 * K, replace=False)
                X, tot, mar, joined = store_tr.features([sel_T[:K], sel_T[K:], sel_S[:K], sel_S[K:]])
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
                summ[m] = {"n": int(len(cr)), "within_T": float(wT.mean()), "within_S": float(wS.mean()),
                           "cross": float(cr.mean()), "ratio": float((cr / ((wT + wS) / 2)).mean()),
                           "p_cross_gt_max": float((cr > np.maximum(wT, wS)).mean()),
                           "p_cross_gt_mean": float((cr > (wT + wS) / 2).mean())}
            e4[f"{mode}_k{K}"] = summ
            log(f"[{name}] E4 {mode} k={K} P(cross>max within): " + ", ".join(f"{m} {v['p_cross_gt_max']:.3f}" for m, v in summ.items()))
    r["E4"] = e4

    # ---- E5 content control: OOF single-turn AUC on agree/acknowledge TRAIN turns
    ack = np.array([is_ack(text_fn(t)) for t in tr])
    e5 = {"n": int(ack.sum()), "n_S": int((ack & (y_tr == 1)).sum()), "n_T": int((ack & (y_tr == 0)).sum())}
    for k in ALLKEYS:
        e5[k + "_auc"] = auc(oof_single[k][ack], y_tr[ack]) if ack.sum() else float("nan")
    r["E5"] = e5
    log(f"[{name}] E5 ack OOF AUC (n={e5['n']}): " + ", ".join(f"{k} {e5[k + '_auc']:.3f}" for k in ALLKEYS))
    r["markers"] = model.markers(12)
    r["seconds"] = time.perf_counter() - t0
    return r


def judge(res: dict, ref_k50: dict | None = C.EXP4_LOOSE_K50) -> dict:
    """Pre-registered criteria (PIPELINE_SPEC §3.5 / FORMAL_DISTANCE_EXPERIMENT §E.6)."""
    e4 = res["E4"]["random_k50"]
    checks = {"E3_NB_k10_acc_cal>=%.2f" % C.E3_NB_K10_MIN: res["E3"]["10"]["M5"]["acc_cal"] >= C.E3_NB_K10_MIN}
    for m, thr in C.E4_MIN.items():
        checks[f"E4_{m}_k50>={thr:.2f}"] = e4[m]["p_cross_gt_max"] >= thr
    if ref_k50:
        for m, v in ref_k50.items():
            if m in e4:
                checks[f"E4_{m}_k50_drop<={C.E4_MAX_DROP:.2f}_vs_exp4"] = e4[m]["p_cross_gt_max"] >= v - C.E4_MAX_DROP
    checks[f"E5_NB_ack_auc>={C.E5_NB_MIN:.2f}"] = res["E5"]["M5_auc"] >= C.E5_NB_MIN
    return {"pass": all(checks.values()), "checks": checks}


def summary_md(res: dict, verdict: dict) -> str:
    L = [f"# Ruler validation: {res['name']}", "",
         f"train S/T {res['train_S_T']}, test S/T {res['test_S_T']}, removed {res['removed_train']} train / {res['removed_test']} test "
         f"(train S/T removed {res['removed_train_S_T']}); seconds {res['seconds']:.0f}", "",
         "## E1 single turn (acc thr0 [CP] / tuned / AUC)", "", "| measure | thr0 | tuned | AUC |", "|---|---|---|---|"]
    for k in ALLKEYS:
        v = res["E1"][k]
        L.append(f"| {k} | {v['acc_thr0']:.3f} [{v['ci_thr0'][0]:.3f}, {v['ci_thr0'][1]:.3f}] | {v['acc_tuned']:.3f} | {v['auc']:.3f} |")
    L += ["", "## E3 bags: calibrated accuracy (AUC)", "", "| k | " + " | ".join(ALLKEYS) + " |", "|---|" + "---|" * len(ALLKEYS)]
    for K, row in res["E3"].items():
        L.append(f"| {K} | " + " | ".join(f"{row[k]['acc_cal']:.3f} ({row[k]['auc']:.3f})" for k in ALLKEYS) + " |")
    L += ["", "## E4 P(cross > max within)", "", "| mode | " + " | ".join(E4_MEAS) + " |", "|---|" + "---|" * len(E4_MEAS)]
    for mode, row in res["E4"].items():
        L.append(f"| {mode} | " + " | ".join(f"{row[m]['p_cross_gt_max']:.3f}" for m in E4_MEAS if m in row) + " |")
    e5 = res["E5"]
    L += ["", f"## E5 ack subset OOF AUC (n = {e5['n']}; S {e5['n_S']} / T {e5['n_T']})", "",
          ", ".join(f"{k} {e5[k + '_auc']:.3f}" for k in ALLKEYS), "", "## Verdict", "",
          f"**{'PASS' if verdict['pass'] else 'FAIL'}**", ""]
    L += [f"- {'OK ' if ok else 'NO '} {name}" for name, ok in verdict["checks"].items()]
    return "\n".join(L) + "\n"
