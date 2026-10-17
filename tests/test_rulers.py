"""Ruler tests: pre-registered validation thresholds on the v1 rule (slow, ~2 min), calibration, features."""
from __future__ import annotations

import numpy as np

from persona import config as C
from persona.rulers import features as FE, validate as V
from persona.rulers.calibrate import Calibrator
from persona.rulers.stats import Isotonic, auc, clopper_pearson, cllr, tune_threshold
from tests.test_data import data

_RES = {}


def _validation():
    if "v1" not in _RES:
        _, _, turns, _ = data()
        _RES["v1"] = V.run(turns, flag=lambda t: t.symbol, text_fn=lambda t: t.ruler_text, name="v1_test", nboot=0, log=lambda *a: None)
    return _RES["v1"]


def test_validation_thresholds():
    res = _validation()
    verdict = V.judge(res, ref_k50=None if C.SYNTH else C.EXP4_LOOSE_K50)   # experiment-4 numbers are real-data only
    assert verdict["pass"], verdict["checks"]


test_validation_thresholds.slow = True


def test_validation_reproduces_structure():
    res = _validation()
    assert res["E3"]["10"]["M5"]["acc_cal"] >= 0.95
    assert res["E1"]["M5"]["auc"] > 0.85 and res["E1"]["M5"]["acc_tuned"] < 0.85    # single turn still weak
    assert res["E5"]["n"] > 50


test_validation_reproduces_structure.slow = True


def test_calibrator_improves_brier_and_cllr():
    res = _validation()
    llr_T = -np.array(res["oof_single_nb_margin"])
    y_T = 1 - np.array(res["oof_y_is_S"])
    # fit on the first 70% (time order), evaluate on the rest
    n = len(y_T)
    cut = int(n * 0.7)
    cal = Calibrator().fit(llr_T[:cut], y_T[:cut])
    ev = cal.evaluate(llr_T[cut:], y_T[cut:])
    assert ev["brier_cal"] < ev["brier_raw_sigmoid"], ev
    assert ev["cllr_cal"] < 1.0, ev


test_calibrator_improves_brier_and_cllr.slow = True


def test_stats_helpers():
    assert abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9
    assert abs(auc([1, 2, 3, 4], [1, 1, 0, 0])) < 1e-9
    assert abs(auc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-9
    thr, bal, acc = tune_threshold([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    assert 0.4 < thr < 0.6 and bal == 1.0
    lo, hi = clopper_pearson(90, 100)
    assert 0.82 < lo < 0.86 and 0.94 < hi < 0.96
    iso = Isotonic().fit([0, 1, 2, 3, 4, 5], [0, 0, 1, 0, 1, 1])
    p = iso.prob([0, 2.5, 5])
    assert p[0] <= p[1] <= p[2]
    assert cllr([10, 10, -10, -10], [1, 1, 0, 0]) < 0.01 and abs(cllr([0, 0], [1, 0]) - 1.0) < 1e-9


def test_features_and_bands():
    _, _, turns, _ = data()
    trT = [t for t in turns if t.split == "train" and t.who == C.TARGET and t.ruler_text and not t.symbol]
    trS = [t for t in turns if t.split == "train" and t.who == C.PARTNER and t.ruler_text and not t.symbol]
    f = FE.turn_features(trT[0])
    assert set(f) == set(FE.FEATURES)
    bands = FE.fit_bands(trT, n=100)
    assert all(bands[k] is None or bands[k]["lo"] <= bands[k]["hi"] for k in FE.FEATURES)
    chk = FE.band_check(FE.bag_features(trT[:50]), bands)
    assert chk["n_inside"] >= chk["n_defined"] - 4, chk            # a real T bag sits inside most bands
    disc = FE.discrimination(trT, trS, n=60)
    # known contrasts (experiments 1-4 and stage-1 report): sticker S > T, burst T > S, particles T/S, question S > T.
    # Punctuation width is discriminative but its direction is NOT the one claimed on 10/03 (see reports/data_stats):
    # in the ruler population T uses more full-width marks than S, so only its discrimination is asserted.
    assert disc["fullwidth_punct"]["disc"] >= C.FEATURE_MIN_AUC
    assert disc["sticker"]["mean_S"] > disc["sticker"]["mean_T"]
    assert disc["question"]["mean_S"] > disc["question"]["mean_T"]
    assert disc["burst"]["mean_T"] > disc["burst"]["mean_S"]
    assert disc["particle_T"]["mean_T"] > disc["particle_T"]["mean_S"]
    assert disc["particle_S"]["mean_S"] > disc["particle_S"]["mean_T"]
    w = FE.discrimination_weights(bands, disc)
    assert abs(sum(w.values()) - 1.0) < 1e-9
