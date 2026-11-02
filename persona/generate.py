"""persona.generate -- batched sampling with the shared prompt builder; copy and leakage checks (PIPELINE_SPEC §5.2)."""
from __future__ import annotations

import torch

from . import config as C
from .prompt import parse_generation, render
from .text import CopyChecker, leak_chars, reply_text  # noqa: F401  (torch-free helpers live in persona.text)


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
