"""persona.generate -- batched sampling with the shared prompt builder; copy and leakage checks (PIPELINE_SPEC §5.2)."""
from __future__ import annotations

import torch

from . import config as C
from .normalize import PH, PH_LINE, emoticon_placeholders
from .prompt import parse_generation, render

PH_LINES = set(PH_LINE.values())


@torch.inference_mode()
def sample_continuations(model, tok, prompts, n: int = 1, temperature: float = C.DECODE["temperature"],
                         top_p: float = C.DECODE["top_p"], max_new_tokens: int = C.DECODE["max_new_tokens"],
                         batch_size: int = 16, seed: int = 0):
    """Returns, per prompt, a list of n raw continuations (text after the prompt, special tokens removed)."""
    model.eval()
    side = tok.padding_side
    tok.padding_side = "left"
    torch.manual_seed(seed)
    out_all = [[] for _ in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for a in range(0, len(order), batch_size):
        bi = order[a:a + batch_size]
        enc = tok([prompts[i] for i in bi], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                             num_return_sequences=n, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for j, i in enumerate(bi):
            out_all[i] = texts[j * n:(j + 1) * n]
    tok.padding_side = side
    return out_all


def generate_replies(model, tok, samples, n: int = 1, **kw):
    """[{sid, cands: [[msg, ...], ...]}] with messages parsed from the continuations."""
    prompts = [render(s)[0] for s in samples]
    conts = sample_continuations(model, tok, prompts, n=n, **kw)
    return [{"sid": s.sid, "cands": [parse_generation(c) for c in cs]} for s, cs in zip(samples, conts)]


def reply_text(msgs) -> str:
    """Ruler view of a reply: text messages only (placeholder lines dropped), emoticons as placeholder chars."""
    kept = [m for m in msgs if m.strip() not in PH_LINES and m.strip() not in PH]
    return emoticon_placeholders("\n".join(kept)) if kept else ""


class CopyChecker:
    """Verbatim-copy test: any common substring of >= L characters with a training target (PIPELINE_SPEC §5.3)."""

    def __init__(self, train_texts, L: int = C.DPO["copy_lcs"]):
        self.L = L
        self.sub = set()
        for t in train_texts:
            t = t.replace("\n", " ")
            for i in range(0, max(0, len(t) - L + 1)):
                self.sub.add(t[i:i + L])

    def is_copy(self, text: str) -> bool:
        t = text.replace("\n", " ")
        return any(t[i:i + self.L] in self.sub for i in range(0, max(0, len(t) - self.L + 1)))


_OPENCC = {}


def leak_chars(text: str):
    """Characters changed by OpenCC s2twp (simplified / Mainland terms -> Taiwan); None if OpenCC is not installed."""
    if "cc" not in _OPENCC:
        try:
            import opencc
            _OPENCC["cc"] = opencc.OpenCC("s2twp")
        except Exception:
            _OPENCC["cc"] = None
    cc = _OPENCC["cc"]
    if cc is None:
        return None
    conv = cc.convert(text)
    return sum(1 for a, b in zip(text, conv) if a != b) + abs(len(conv) - len(text))
