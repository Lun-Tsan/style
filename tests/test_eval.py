"""Evaluation tests on the model-free controls (PIPELINE_SPEC §8 test 10): C-real near the floor, C-S far from it,
C-shuf keeps style but loses content (checked here through the feature JSD and bands; on-topic needs bge-m3)."""
from __future__ import annotations

import json

from persona import config as C
from persona import evaluate as EV, samples as SM
from persona.generate import CopyChecker
from tests.test_data import data

_E = {}


def evaluator():
    if "e" not in _E:
        msgs, n_sys, turns, segs = data()
        smp = SM.build_samples(segs)
        test = [s for s in smp if s.split == "test"]
        bands = json.load(open(C.REPORTS / "feature_bands.json", encoding="utf-8"))["bands"]
        tr = [t for t in turns if t.split == "train"]
        teS = [t for t in turns if t.split == "test" and t.who == C.PARTNER]
        _E["e"] = (EV.Evaluator(test, tr, teS, bands), test)
    return _E["e"]


def test_controls_define_axes():
    ev, test = evaluator()
    m_real = ev.evaluate(ev.real)
    assert abs(m_real["floor_auc1"] - 0.5) < 0.12, m_real                     # real vs real ~ chance
    assert m_real["sens_auc1"] > 0.70, m_real                                   # real T vs real S separable (0.76 OOF by string)
    assert m_real["copy_rate"] >= 0.0 and m_real["band_inside_mean"] >= m_real["band_defined_mean"] - 5
    m_shuf = ev.evaluate(EV.shuffled_outputs(test))
    # C-shuf has exactly the real replies, so the style measures must agree with C-real
    assert m_shuf["feature_jsd"] < 0.02, m_shuf["feature_jsd"]
    assert abs(m_shuf["ncd_delta"]) < 0.01, m_shuf["ncd_delta"]
    assert abs(m_shuf["div_hill1"] - m_real["div_hill1"]) < 1e-6
    # C-S: the partner's turns as "replies" -> the discriminator must tell them apart
    S_out = [{"sid": i, "msgs": [t]} for i, t in enumerate(ev.partner_texts[:len(test)])]
    m_S = ev.evaluate(S_out)
    assert m_S["d_auc1"] > 0.70, m_S["d_auc1"]
    assert m_S["ncd_delta"] > 0.0, m_S["ncd_delta"]                              # more partner-like than real T


test_controls_define_axes.slow = True


def test_copy_checker_and_diversity():
    ck = CopyChecker(["今天天氣真的很好我們去海邊走走吧", "好啊"], L=6)
    assert ck.is_copy("真的很好我們") and not ck.is_copy("好啊好啊好啊")
    d = EV.diversity([{"sid": 0, "msgs": ["好啊"]}, {"sid": 1, "msgs": ["好啊"]}, {"sid": 2, "msgs": ["沒錯"]}])
    assert d["n"] == 3 and 1.0 < d["hill1"] < 3.0 and d["unique_share"] < 1.0
    sh = EV.shuffled_outputs(evaluator()[1][:20])
    real = EV.real_outputs(evaluator()[1][:20])
    assert all(a["msgs"] != b["msgs"] or len(set(map(tuple, (r["msgs"] for r in real)))) < 20 for a, b in zip(sh, real))
