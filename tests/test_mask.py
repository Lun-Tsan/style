"""Loss-mask test with the real tokenizer (offline, from the HF cache): the first target token has mask 1,
the last prompt token has mask 0, eos is appended with mask 1, and the prompt decodes back exactly."""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from persona import config as C
from persona import turns as TU
from persona.prompt import ME, encode, render_parts
from persona.samples import Sample

_TOK = {}


def tok():
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained(C.MODELS["instruct"])
    return _TOK["t"]


def _sample(ctx_msgs, tgt_msgs):
    ctx = [TU.Turn(C.PARTNER if i % 2 == 0 else C.TARGET, "2026-01-01", 0, 0, [m]) for i, m in enumerate(ctx_msgs)]
    return Sample(0, ctx, TU.Turn(C.TARGET, "2026-01-01", 0, 0, list(tgt_msgs)))


def test_mask_boundary():
    t = tok()
    s = _sample(["這題我不會", "哪一題", "第三題"], ["好啊", "我看看"])
    enc = encode(t, s)
    ids, mask = enc["input_ids"], enc["loss_mask"]
    assert len(ids) == len(mask) == enc["n_prompt"] + enc["n_target"]
    assert mask[enc["n_prompt"] - 1] == 0 and mask[enc["n_prompt"]] == 1          # IMPersona off-by-one guard
    assert ids[-1] == t.eos_token_id and mask[-1] == 1
    prompt, target = render_parts(s.ctx, s.target)
    assert t.decode(ids[:enc["n_prompt"]]) == prompt
    assert t.decode(ids[enc["n_prompt"]:-1]) == target
    assert prompt.endswith(f"{ME}: ")


test_mask_boundary.slow = True


def test_truncation_drops_oldest_context():
    t = tok()
    s = _sample(["很長的訊息" * 40] * 12, ["好"])
    enc = encode(t, s, ctx_tokens=200)
    assert enc["n_prompt"] <= 200 or enc["n_ctx_turns"] == 0
    assert enc["n_ctx_turns"] < 12
    enc2 = encode(t, s, ctx_tokens=100000)
    assert enc2["n_ctx_turns"] == 12


test_truncation_drops_oldest_context.slow = True
