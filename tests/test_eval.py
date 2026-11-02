"""Evaluation tests on the model-free controls (PIPELINE_SPEC §8 test 10): C-real near the floor, C-S far from it,
C-shuf keeps style but loses content (checked here through the feature JSD and bands; on-topic needs bge-m3).
Also the review fixes that run without a model (CODE_REVIEW_2026-10-17): symbol-like replies leave the rulers (#3;
Leak and MMI too), the NLL is taken on the real replies (#1), C-S keeps its messages (#11), rarefied diversity
compared at the same n (#13), the on-topic threshold comes from training replies (#12), discriminator_scores, and the
data manifest check."""
from __future__ import annotations

import json
import math
import tempfile
import zlib
from pathlib import Path

import numpy as np

from persona import config as C
from persona import evaluate as EV, samples as SM
from persona.flag_symbols import is_symbol_turn
from persona.rulers.stats import auc
from persona.text import CopyChecker, reply_text
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
    assert m_real["symbol_share"] == 0.0                                        # real targets are not symbol-like
    assert m_real["div_n"] == min(C.EVAL["rarefy_n"], len(test)), m_real["div_n"]   # rarefied by default (#13)
    m_shuf = ev.evaluate(EV.shuffled_outputs(test))
    # C-shuf has exactly the real replies, so the style measures must agree with C-real
    assert m_shuf["feature_jsd"] < 0.02, m_shuf["feature_jsd"]
    assert abs(m_shuf["ncd_delta"]) < 0.01, m_shuf["ncd_delta"]
    assert abs(m_shuf["div_hill1"] - m_real["div_hill1"]) < 1e-6
    # C-S: the partner's turns as "replies", built from the messages like C-real (#11); the rulers must separate them
    assert ev.partner_texts == EV.texts_of(ev.partner_outputs)
    assert any(len(o["msgs"]) > 1 for o in ev.partner_outputs)                  # a burst stays several messages
    assert any(not t for t in ev.partner_texts)                                 # sticker-only turns stay in
    m_S = ev.evaluate(ev.partner_outputs[:len(test)])
    assert m_S["symbol_share"] == 0.0
    assert m_S["d_auc1"] > 0.70, m_S["d_auc1"]
    assert m_S["ncd_delta"] > 0.0, m_S["ncd_delta"]                              # more partner-like than real T
    assert m_S["band_inside_mean"] < m_real["band_inside_mean"], (m_S["band_inside_mean"], m_real["band_inside_mean"])


test_controls_define_axes.slow = True


def test_symbol_like_replies_leave_the_rulers():
    """#3 / contract (d): symbol-like replies are dropped from the ruler populations and counted in symbol_share.
    Leak runs with a stand-in counter (OpenCC may be absent) that counts a character of the symbol-like reply, and MMI
    with a stub that records what it is given."""
    ev, test = evaluator()
    bad = ["好啊 (笑)"]
    assert is_symbol_turn(reply_text(bad))
    idx = set(range(0, len(ev.real), 5))
    mixed = [{"sid": o["sid"], "msgs": bad} if i in idx else o for i, o in enumerate(ev.real)]
    kept = [o for i, o in enumerate(ev.real) if i not in idx]
    seen, orig = [], (EV.leak_chars, EV.mmi)
    EV.leak_chars = lambda t: t.count("好")
    EV.mmi = lambda model, tok, samples, outputs, **kw: seen.append((samples, outputs)) or 0.0
    try:
        m_mixed, m_kept = ev.evaluate(mixed, base_model=object(), tok=object()), ev.evaluate(kept)
    finally:
        EV.leak_chars, EV.mmi = orig
    assert abs(m_mixed["symbol_share"] - len(idx) / len(mixed)) < 1e-12 and m_kept["symbol_share"] == 0.0
    assert m_mixed["leak_per_100"] is not None
    for k in ("d_auc1", f"d_auc{C.K_ATTR}", "d_n", "ncd_delta", "feature_jsd", "copy_rate", "leak_per_100",
              "band_inside_mean", "div_distinct2", "div_hill1", "div_n", "div_distinct2_ratio", "div_hill1_ratio"):
        a, b = m_mixed[k], m_kept[k]
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), (k, a, b)
    # MMI: the kept replies with their own contexts
    samples, outputs = seen[0]
    assert len(seen) == 1 and outputs == kept and [s.sid for s in samples] == [o["sid"] for o in kept]


test_symbol_like_replies_leave_the_rulers.slow = True


def test_diversity_ratio_at_the_same_n():
    """#13: diversity is compared with C-real at the same n. Real replies with the odd-indexed ones replaced by a
    symbol-like reply keep fewer than rarefy_n replies, each of them real; their ratios are taken against C-real
    rarefied to their own div_n (on the synthetic test set the raw Hill-1 against C-real at rarefy_n is 0.69 of it)."""
    ev, test = evaluator()
    m_real = ev.evaluate(ev.real)
    assert m_real["div_hill1_ratio"] == m_real["div_distinct2_ratio"] == 1.0
    half = ev.evaluate([{"sid": o["sid"], "msgs": ["好啊 (笑)"]} if i % 2 else o for i, o in enumerate(ev.real)])
    assert half["div_n"] < m_real["div_n"] == C.EVAL["rarefy_n"], (half["div_n"], m_real["div_n"])
    ref = EV.diversity(ev.real, half["div_n"])
    assert half["div_hill1_ratio"] == half["div_hill1"] / ref["hill1"]
    assert half["div_distinct2_ratio"] == half["div_distinct2"] / ref["distinct2"]
    floor = C.LINES["diversity_ratio"]
    assert half["div_hill1_ratio"] >= floor and half["div_distinct2_ratio"] >= floor


test_diversity_ratio_at_the_same_n.slow = True


def test_nll_is_taken_on_the_real_replies():
    """#1: the model is scored on the real test replies, not on the system's outputs (model_nll replaced by a stub)."""
    ev, test = evaluator()
    seen = []
    orig = EV.model_nll
    EV.model_nll = lambda model, tok, samples, outputs, **kw: seen.append((samples, outputs)) or 0.0
    try:
        m = ev.evaluate(EV.shuffled_outputs(test), model=object(), tok=object())
    finally:
        EV.model_nll = orig
    assert m["nll"] == 0.0 and len(seen) == 1
    assert seen[0][0] is ev.samples and seen[0][1] is ev.real


test_nll_is_taken_on_the_real_replies.slow = True


def test_discriminator_scores():
    """Contract (c): out-of-fold margins over the non-empty texts in input order; discriminator is built on them."""
    real = [f"好啊{i}喔" for i in range(30)] + [""]
    gen = [f"嗯嗯{i}嗎" for i in range(25)]
    s_real, s_gen = EV.discriminator_scores(real, gen)
    assert isinstance(s_real, np.ndarray) and len(s_real) == 30 and len(s_gen) == 25
    d = EV.discriminator(real, gen)
    assert d["auc1"] == auc(np.concatenate([s_gen, s_real]), [1] * 25 + [0] * 30) and d["auc1"] > 0.9, d
    r2, g2 = EV.discriminator_scores(real[::-1], gen)          # input order carries through to the scores
    assert np.array_equal(r2, s_real[::-1]) and np.array_equal(g2, s_gen)


class _Enc:
    """Stand-in for bge-m3 (no model, no torch): a fixed pseudo-random unit vector per text."""

    def encode(self, texts):
        out = np.zeros((len(texts), 16))
        for i, t in enumerate(texts):
            v = np.random.default_rng(zlib.crc32(t.encode("utf-8"))).normal(size=16)
            out[i] = v / np.linalg.norm(v)
        return out


def test_ontopic_threshold_from_training_replies():
    """#12 / contract (b): the threshold is fitted on training-period replies; test samples passed in are ignored."""
    from persona import embed
    _, _, turns, segs = data()
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    thr = embed.fit_ontopic_threshold(_Enc(), smp)
    assert thr == embed.fit_ontopic_threshold(_Enc(), train_all)
    assert thr == embed.null_threshold(_Enc(), [reply_text(s.target.msgs) for s in train_all if s.target.text_b])
    try:
        embed.fit_ontopic_threshold(_Enc(), [s for s in smp if s.split == "test"])
    except ValueError:
        pass
    else:
        raise AssertionError("test samples alone gave an on-topic threshold")


def test_manifest_check_raises_on_tampered_copy():
    """data.load_all() compares the data with C.REPORTS/data_manifest.json and names the earliest differing key."""
    from persona import data as DA
    _, _, turns, segs = data()
    man = json.loads(json.dumps(DA.manifest(turns, segs)))
    assert man["samples"]["test"] == sum(s.split == "test" for s in SM.build_samples(segs))
    old = C.REPORTS
    with tempfile.TemporaryDirectory() as tmp:
        C.REPORTS = Path(tmp)
        try:
            path = C.REPORTS / DA.MANIFEST
            path.write_text(json.dumps(man), encoding="utf-8")
            DA.load_all()                                               # an untouched copy passes
            for key, edit in (("samples.dev", lambda m: m["samples"].update(dev=m["samples"]["dev"] + 1)),
                              ("DEV_FRAC", lambda m: m.update(DEV_FRAC=0.2))):   # a constant is named before counts
                edit(man)
                path.write_text(json.dumps(man), encoding="utf-8")
                try:
                    DA.load_all()
                except ValueError as e:
                    assert f"'{key}'" in str(e), str(e)
                else:
                    raise AssertionError(f"tampered manifest ({key}) passed")
        finally:
            C.REPORTS = old


def test_copy_checker_and_diversity():
    ck = CopyChecker(["今天天氣真的很好我們去海邊走走吧", "好啊"], L=6)
    assert ck.is_copy("真的很好我們") and not ck.is_copy("好啊好啊好啊")
    d = EV.diversity([{"sid": 0, "msgs": ["好啊"]}, {"sid": 1, "msgs": ["好啊"]}, {"sid": 2, "msgs": ["沒錯"]}])
    assert d["n"] == 3 and 1.0 < d["hill1"] < 3.0 and d["unique_share"] < 1.0
    many = [{"sid": i, "msgs": [f"好啊{i % 37}"]} for i in range(150)]
    d1 = EV.diversity(many, C.EVAL["rarefy_n"])
    assert d1["n"] == C.EVAL["rarefy_n"] and d1 == EV.diversity(many[::-1], C.EVAL["rarefy_n"])   # order-free (#13)
    sh = EV.shuffled_outputs(evaluator()[1][:20])
    real = EV.real_outputs(evaluator()[1][:20])
    assert all(a["msgs"] != b["msgs"] or len(set(map(tuple, (r["msgs"] for r in real)))) < 20 for a, b in zip(sh, real))
