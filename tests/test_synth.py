"""Synthetic export (persona/synth.py) and parser edge cases; no private data needed, so these run everywhere."""
from __future__ import annotations

import re

from persona import config as C
from persona import parse_line, turns as TU
from persona.synth import generate


def test_synth_roundtrip_counts():
    lines, truth = generate(seed=1)
    msgs, n_sys = parse_line.parse(lines, C.SPEAKERS)
    turns = TU.build_turns(msgs)
    segs = TU.assign_segments(turns)
    assert (len(msgs), n_sys) == (truth["n_msgs"], truth["n_sys"]), (len(msgs), n_sys, truth)
    assert (len(turns), len(segs)) == (truth["n_turns"], truth["n_segments"]), (len(turns), len(segs), truth)
    assert TU.counts(turns) == truth["text_turns_S_T"], (TU.counts(turns), truth)
    assert sum(t.date >= C.SPLIT_DATE for t in turns) > 100                     # both splits populated
    assert any(a.who == b.who and a.date != b.date and b.abs_start - a.abs_end <= C.GAP_MIN
               for a, b in zip(turns, turns[1:])), "no same-speaker turn split at midnight was generated"


def test_synth_deterministic():
    assert generate(seed=3) == generate(seed=3)
    assert generate(seed=3)[0] != generate(seed=4)[0]


def test_parse_prefix_names_system_and_continuation():
    lines = ["[LINE] 與王的聊天記錄", "儲存日期:2026.01.02 10:00", "",
             "2026.01.01 星期四",
             "09:00 王 小明 好啊",           # target "王 小明" (longer name wins over its prefix "王")
             "09:01 王 你好",                 # partner "王"
             "第二行",                        # continuation of the previous message
             "09:02 王已收回訊息",            # system line (no space after the name) -> dropped and counted
             "這行也丟掉",                    # continuation after a system line -> dropped
             "09:03 王 小明 貼圖"]
    msgs, n_sys = parse_line.parse(lines, ["王", "王 小明"])
    assert [(m.who, m.text) for m in msgs] == [("王 小明", "好啊"), ("王", "你好\n第二行"), ("王 小明", "貼圖")]
    assert n_sys == 1 and msgs[2].kind == "sticker" and msgs[0].date == "2026-01-01"


def test_synth_not_in_real_export():
    """On the machine with the private export: no synthetic message of >= 12 characters occurs in it verbatim."""
    if C.SYNTH or not C.CHAT_PATH.exists():
        print("  (no private export here; nothing to compare)")
        return
    real = C.CHAT_PATH.read_text(encoding="utf-8-sig")
    msgs, _ = parse_line.parse(generate()[0], C.SPEAKERS)
    long_ = {m.text for m in msgs if len(re.sub(r"\s", "", m.text)) >= 12}
    hits = [m for m in long_ if m in real]
    assert not hits, f"{len(hits)} synthetic messages occur in the real export"   # count only, never the text
