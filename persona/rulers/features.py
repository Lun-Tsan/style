"""Interpretable (R1) features per turn, bag aggregates, tolerance bands, discrimination and control charts.

Features are computed from turn.msgs (raw messages incl. placeholder-only messages); the ruler population is the
non-symbol turns, so punctuation width is judged on , . ! ? : ; (brackets and slashes are removed by the symbol rule).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .. import config as C
from ..normalize import EMO_ASCII_RE, EMO_LINE_RE, PH
from .stats import auc

S, T = C.PARTNER, C.TARGET

PARTICLE_T = set("喔哦噢啊呀啦耶欸齁嘛吧")
PARTICLE_S = set("嗎")
TRAIL = re.compile(r"[\s!！?？。.,，~～…]+$")
HALF_PUNCT = set(",.!?:;")
FULL_PUNCT = set("，。！？：；、")   # ， 。 ！ ？ ： ； 、 (full-width forms)
PUNCT_STRIP = re.compile(r"[\s!！～~。，,.、?？…:：）)(（]+")
CATCH_T = {"好啊", "沒錯", "讚喔", "讚", "好喔", "對啊", "可以啊", "好啊好啊"}
CATCH_S = {"嗯嗯", "好的", "我懂了", "了解", "收到", "好", "嗯"}
LAUGH = re.compile(r"哈{2,}|XD|xd|笑死|w{3,}")
QUESTION = re.compile(r"[嗎?？]")
LETTER = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9０-９]")
WS = re.compile(r"\s")

Z_CAP = 5.0   # |z| cap in the weighted distance, so near-constant features cannot dominate
FEATURES = ["particle_T", "particle_S", "fullwidth_punct", "end_punct", "sticker", "burst", "msg_len", "tilde",
            "catch_T", "catch_S", "question", "letters", "digits", "laugh", "first_person", "emoticon"]


def _core(t: str) -> str:
    return PUNCT_STRIP.sub("", t)


def turn_features(turn) -> dict:
    """One value per feature for one turn (NaN when undefined, e.g. no punctuation for fullwidth_punct)."""
    msgs = [b for b in turn.msgs]
    text_msgs = [b for b in msgs if b.strip() not in PH]
    f = dict.fromkeys(FEATURES, float("nan"))
    f["sticker"] = float(np.mean([b.strip() in PH for b in msgs]))
    f["burst"] = float(len(msgs))
    if not text_msgs:
        return f
    ends = [TRAIL.sub("", b) for b in text_msgs]
    f["particle_T"] = float(np.mean([bool(e) and e[-1] in PARTICLE_T for e in ends]))
    f["particle_S"] = float(np.mean([bool(e) and e[-1] in PARTICLE_S for e in ends]))
    half = sum(ch in HALF_PUNCT for b in text_msgs for ch in b)
    full = sum(ch in FULL_PUNCT for b in text_msgs for ch in b)
    f["fullwidth_punct"] = full / (half + full) if half + full else float("nan")
    f["end_punct"] = float(np.mean([bool(b.rstrip()) and (b.rstrip()[-1] in HALF_PUNCT | FULL_PUNCT | set("～~…")) for b in text_msgs]))
    f["msg_len"] = float(np.mean([len(WS.sub("", b)) for b in text_msgs]))
    f["tilde"] = float(np.mean([("～" in b) or ("~" in b) for b in text_msgs]))
    cores = [_core(b) for b in text_msgs]
    f["catch_T"] = float(np.mean([c in CATCH_T for c in cores]))
    f["catch_S"] = float(np.mean([c in CATCH_S for c in cores]))
    f["question"] = float(np.mean([bool(QUESTION.search(b)) for b in text_msgs]))
    chars = sum(len(WS.sub("", b)) for b in text_msgs)
    f["letters"] = sum(len(LETTER.findall(b)) for b in text_msgs) / chars if chars else float("nan")
    f["digits"] = sum(len(DIGIT.findall(b)) for b in text_msgs) / chars if chars else float("nan")
    f["laugh"] = float(np.mean([bool(LAUGH.search(b)) for b in text_msgs]))
    f["first_person"] = float(np.mean(["我" in b for b in text_msgs]))
    f["emoticon"] = float(np.mean([bool(EMO_LINE_RE.search(b) or EMO_ASCII_RE.search(b)) for b in text_msgs]))
    return f


def bag_features(turns) -> dict:
    """Mean of each feature over the turns of a bag (NaN values ignored)."""
    rows = [turn_features(t) for t in turns]
    out = {}
    for k in FEATURES:
        v = np.array([r[k] for r in rows], float)
        v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    return out


def _bags(turns, k, n, rng):
    return [[turns[i] for i in rng.choice(len(turns), k, replace=False)] for _ in range(n)]


def fit_bands(target_turns, k: int = C.K_VALID, n: int = C.BAND_BAGS, q=C.BAND_Q, seed: int = 0):
    """Tolerance band per feature = quantiles q over n random bags of k target turns; also mean and sd."""
    rng = np.random.default_rng(seed)
    rows = [bag_features(b) for b in _bags(target_turns, k, n, rng)]
    bands = {}
    for f in FEATURES:
        v = np.array([r[f] for r in rows], float)
        v = v[~np.isnan(v)]
        bands[f] = {"lo": float(np.quantile(v, q[0])), "hi": float(np.quantile(v, q[1])),
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1))} if len(v) > 1 else None
    return bands


def band_check(feat: dict, bands: dict):
    """How many features of a bag fall inside the band; also the signed distance in sd units."""
    inside, z = {}, {}
    for f, b in bands.items():
        if b is None or np.isnan(feat.get(f, float("nan"))):
            inside[f] = None
            continue
        inside[f] = bool(b["lo"] <= feat[f] <= b["hi"])
        z[f] = (feat[f] - b["mean"]) / b["sd"] if b["sd"] > 0 else 0.0
    n_ok = sum(1 for v in inside.values() if v)
    n_def = sum(1 for v in inside.values() if v is not None)
    return {"n_inside": n_ok, "n_defined": n_def, "inside": inside, "z": z}


def discrimination(target_turns, partner_turns, k: int = C.K_VALID, n: int = 500, seed: int = 0):
    """Bag-level AUC of each feature for T vs S (max of the two directions); below FEATURE_MIN_AUC -> not weighted."""
    rng = np.random.default_rng(seed)
    bT = [bag_features(b) for b in _bags(target_turns, k, n, rng)]
    bS = [bag_features(b) for b in _bags(partner_turns, k, n, rng)]
    out = {}
    for f in FEATURES:
        vT = np.array([r[f] for r in bT], float)
        vS = np.array([r[f] for r in bS], float)
        ok = ~np.isnan(np.concatenate([vT, vS]))
        s = np.concatenate([vT, vS])[ok]
        y = np.concatenate([np.ones(len(vT)), np.zeros(len(vS))])[ok]
        a = auc(s, y)
        out[f] = {"auc_T_vs_S": a, "disc": max(a, 1 - a) if not np.isnan(a) else float("nan"),
                  "mean_T": float(np.nanmean(vT)), "mean_S": float(np.nanmean(vS)),
                  "weighted": bool(not np.isnan(a) and max(a, 1 - a) >= C.FEATURE_MIN_AUC)}
    return out


def discrimination_weights(bands: dict, disc: dict):
    """Weights for the feature distance: proportional to (discrimination - 0.5) over weighted features, normalised.
    (Inverse-variance weighting was tried first and was dominated by near-constant rare features such as laughter.)"""
    w = {f: (max(disc[f]["disc"] - 0.5, 0.0) if bands[f] and bands[f]["sd"] > 0 and disc[f]["weighted"] else 0.0)
         for f in FEATURES}
    tot = sum(w.values())
    return {f: v / tot for f, v in w.items()} if tot else w


inverse_variance_weights = discrimination_weights   # backward-compatible name


def feature_distance(feat: dict, bands: dict, weights: dict) -> float:
    """sum_f w_f |z_f| over weighted features (NaN features skipped)."""
    d = 0.0
    for f, w in weights.items():
        if w and bands[f] and not np.isnan(feat.get(f, float("nan"))) and bands[f]["sd"] > 0:
            d += w * min(abs(feat[f] - bands[f]["mean"]) / bands[f]["sd"], Z_CAP)
    return float(d)


def control_chart(target_turns, bands: dict):
    """Per date block: block mean of each feature and its z vs the band (bags of K_VALID); |z| > 3 flagged."""
    by_block = {}
    for t in target_turns:
        by_block.setdefault(t.block, []).append(t)
    out = {}
    for b, ts in sorted(by_block.items()):
        feat = bag_features(ts)
        scale = {f: (bands[f]["sd"] * np.sqrt(C.K_VALID / max(len(ts), 1)) if bands[f] else None) for f in FEATURES}
        z = {f: ((feat[f] - bands[f]["mean"]) / scale[f] if bands[f] and scale[f] and not np.isnan(feat[f]) else None)
             for f in FEATURES}
        out[str(b)] = {"n_turns": len(ts), "dates": [ts[0].date, ts[-1].date], "mean": feat, "z": z,
                       "out_of_control": [f for f, v in z.items() if v is not None and abs(v) > 3]}
    return out
