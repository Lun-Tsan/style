"""Prompt builder tests (no tokenizer needed)."""
from __future__ import annotations

from persona import config as C
from persona import turns as TU
from persona.prompt import ME, OTHER, parse_generation, render_parts


def _turn(who, msgs, symbol=False):
    t = TU.Turn(who, "2026-01-01", 0, 0, list(msgs))
    t.symbol = symbol
    return t


def test_render_format():
    ctx = [_turn(C.PARTNER, ["這題我不會", "貼圖"]), _turn(C.TARGET, ["哪一題", "拍給我看"]), _turn(C.PARTNER, ["x = 3"], symbol=True)]
    tgt = _turn(C.TARGET, ["好啊", "我看看"])
    p, t = render_parts(ctx, tgt)
    assert p == f"{OTHER}: 這題我不會\n{OTHER}: [貼圖]\n{ME}: 哪一題\n{ME}: 拍給我看\n{OTHER}: [程式碼/算式]\n{ME}: ", p
    assert t == f"好啊\n{ME}: 我看看", t
    assert p.endswith(f"{ME}: ")


def test_render_empty_context_and_sticker_target():
    p, t = render_parts([], _turn(C.TARGET, ["貼圖"]))
    assert p == f"{ME}: " and t == "[貼圖]"


def test_multiline_message_indent():
    p, t = render_parts([_turn(C.PARTNER, ["第一行\n第二行"])], _turn(C.TARGET, ["好"]))
    assert p == f"{OTHER}: 第一行\n  第二行\n{ME}: "
    assert not any(line.startswith((f"{ME}:", f"{OTHER}:")) for line in p.split("\n")[1:-1])


def test_deid_in_prompt():
    p, t = render_parts([_turn(C.PARTNER, ["我的信箱 a@b.com"])], _turn(C.TARGET, ["好"]))
    assert "a@b.com" not in p and "[信箱]" in p


def test_parse_generation():
    assert parse_generation("好啊\n我: 我看看\n對方: 謝謝\n我: 不客氣") == ["好啊", "我看看"]
    assert parse_generation("好啊") == ["好啊"]
    assert parse_generation("") == []
    assert parse_generation("第一行\n  第二行\n我: 再一則") == ["第一行\n第二行", "再一則"]
    assert len(parse_generation("\n".join(["a"] + ["我: b"] * 10))) == C.DECODE["max_msgs"]
    # round trip through the renderer
    tgt = _turn(C.TARGET, ["好啊", "我看看"])
    _, t = render_parts([], tgt)
    assert parse_generation(t) == tgt.msgs
