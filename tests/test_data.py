"""Data-layer tests: parser counts, turns, segments, placeholders, symbol rule, split, subsets, samples."""
from __future__ import annotations

from collections import Counter

from persona import config as C
from persona import parse_line, samples as SM, split as SP, turns as TU
from persona.flag_symbols import CODE_PH, flag_raw, is_symbol_turn
from persona.normalize import PH, deid, emoticon_placeholders, show

_CACHE = {}


def data():
    if "d" not in _CACHE:
        msgs, n_sys = parse_line.load()
        turns = TU.build_turns(msgs)
        for t in turns:
            t.symbol = is_symbol_turn(t.ruler_text)
        segs = TU.assign_segments(turns)
        _CACHE["d"] = (msgs, n_sys, turns, segs)
    return _CACHE["d"]


def test_parse_counts():
    msgs, n_sys, turns, segs = data()
    if C.SYNTH:                                   # counts known by construction (persona/synth.py)
        from persona.synth import generate
        truth = generate()[1]
        want = (truth["n_msgs"], truth["n_sys"], truth["text_turns_S_T"])
        assert (len(turns), len(segs)) == (truth["n_turns"], truth["n_segments"]), (len(turns), len(segs), truth)
    else:
        want = (4967, 23, {"train": (785, 862), "test": (134, 149)})
    assert (len(msgs), n_sys, TU.counts(turns)) == want, (len(msgs), n_sys, TU.counts(turns), want)
    assert all(m.date and m.time and m.who in C.SPEAKERS for m in msgs)


def test_turn_rules():
    _, _, turns, _ = data()
    for a, b in zip(turns, turns[1:]):
        same = a.who == b.who and a.date == b.date and b.t_start - a.t_end <= C.GAP_MIN
        assert not same, "two consecutive turns should have been merged"
        assert a.abs_end <= b.abs_start
    assert all(t.idx == i for i, t in enumerate(turns))


def test_segments():
    _, _, turns, segs = data()
    assert sum(len(s) for s in segs) == len(turns)
    gap = C.SEG_GAP_H * 60
    for s in segs:
        for a, b in zip(s, s[1:]):
            assert b.abs_start - a.abs_end <= gap
    for s1, s2 in zip(segs, segs[1:]):
        assert s2[0].abs_start - s1[-1].abs_end > gap
    assert all(t.seg >= 0 for t in turns)


def test_placeholders_and_display():
    assert show("\n好啊") == "[貼圖]\n好啊(表情)"
    assert emoticon_placeholders("好啊(cry)QQ") == "好啊"
    assert emoticon_placeholders("x(abc)") == "x(abc)"          # unknown code is not an emoticon
    t = TU.Turn(C.TARGET, "2026-01-01", 0, 0, ["貼圖", "好啊"])
    assert t.has_sticker and t.text_b == "好啊" and t.text_a == "\n好啊" and t.n_msgs == 2


def test_symbol_rule():
    assert is_symbol_turn(emoticon_placeholders("好喔(cry)")) is False        # whitelist keeps emoticons
    assert flag_raw("好喔(cry)") is True                                    # raw loose rule flags them
    assert is_symbol_turn("int main() {") is True
    assert is_symbol_turn("x = 3") is True
    assert is_symbol_turn("7/20 見") is True                                  # slash: known cost of the loose rule
    assert is_symbol_turn("好啊,明天見。") is False                           # commas/periods are never flagged
    assert is_symbol_turn(None) is False
    _, _, turns, _ = data()
    raw = sum(flag_raw(t.text_b) for t in turns if t.text_b)
    v1 = sum(t.symbol for t in turns)
    assert 0 < v1 < raw, (v1, raw)                                            # whitelist rescues some turns


def test_deid():
    s, hits = deid("打 0912345678 或 a.b@c.com 看 https://x.y/z")
    assert "[電話]" in s and "[信箱]" in s and "[網址]" in s and "0912" not in s
    assert hits == Counter({"mobile": 1, "email": 1, "url": 1}), hits
    assert deid("好啊")[0] == "好啊"


def test_samples():
    _, _, turns, segs = data()
    smp = SM.build_samples(segs)
    assert all(s.target.who == C.TARGET and not s.target.symbol for s in smp)
    assert all(len(s.ctx) <= C.CTX_TURNS for s in smp)
    assert all(all(c.seg == s.target.seg for c in s.ctx) for s in smp)
    assert all(all(c.idx < s.target.idx for c in s.ctx) for s in smp)
    n_T_text = sum(1 for t in turns if t.who == C.TARGET and not t.symbol)
    assert len(smp) == n_T_text, (len(smp), n_T_text)
    st = SM.stats(smp, turns, segs)
    assert st["train"]["n_samples"] > 600 and st["test"]["n_samples"] > 100
    assert not any(isinstance(v, str) and len(v) > 40 for v in st.values())   # aggregates only


def test_split_dev_and_subsets():
    _, _, turns, segs = data()
    smp = [s for s in SM.build_samples(segs) if s.split == "train"]
    train, dev = SP.dev_cut(smp)
    assert len(train) + len(dev) == len(smp)
    assert not ({s.target.seg for s in train} & {s.target.seg for s in dev})
    assert max(s.target.date for s in train) <= min(s.target.date for s in dev)
    assert abs(len(dev) / len(smp) - C.DEV_FRAC) < 0.05
    subs = SP.nested_subsets(train)
    prev = set()
    for size in C.SUBSET_SIZES:
        ids = {s.sid for s in subs[size]}
        assert prev <= ids
        assert len(ids) == (size if size else len(train))
        prev = ids
        if size:
            assert SP.jsd_hist(SP.month_hist(subs[size]), SP.month_hist(train)) < 0.05
    blocks = SP.date_blocks([t for t in turns if t.split == "train" and t.text_b])
    assert len(blocks) == C.N_BLOCKS and sum(len(b) for b in blocks) == sum(1 for t in turns if t.split == "train" and t.text_b)


def test_context_placeholder_for_symbol_turns():
    from persona.prompt import msg_lines
    t = TU.Turn(C.PARTNER, "2026-01-01", 0, 0, ["int main() {"])
    t.symbol = True
    assert msg_lines(t) == [f"對方: {CODE_PH}"]
