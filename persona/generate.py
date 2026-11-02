"""persona.generate -- batched sampling with the shared prompt builder; copy and leakage checks (PIPELINE_SPEC §5.2)."""
from __future__ import annotations

import torch

from . import config as C
from .prompt import encode, parse_generation, render_parts
from .text import CopyChecker, leak_chars, reply_text  # noqa: F401  (torch-free helpers live in persona.text)


def token_batches(lengths, n: int, max_new_tokens: int, token_budget: int, batch_size: int):
    """Prompt index batches in decreasing prompt length, so an out-of-memory error from a long prompt shows up at the
    start of a run rather than late in it. A prompt joins the open batch while the batch holds fewer than batch_size
    prompts and (prompts + 1) * (longest prompt + max_new_tokens) * n <= token_budget; a prompt over the budget runs
    alone (CODE_REVIEW #15: many-shot prompts of 6K-30K tokens)."""
    batches = []
    for i in sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True):
        b = batches[-1] if batches else None
        # b[0] is the longest prompt of b (decreasing order)
        if b and len(b) < batch_size and (len(b) + 1) * (lengths[b[0]] + max_new_tokens) * n <= token_budget:
            b.append(i)
        else:
            batches.append([i])
    return batches


@torch.inference_mode()
def sample_continuations(model, tok, prompts, n: int = 1, temperature: float = C.DECODE["temperature"],
                         top_p: float = C.DECODE["top_p"], max_new_tokens: int = C.DECODE["max_new_tokens"],
                         batch_size: int = C.DECODE["batch_size"], seed: int = 0,
                         token_budget: int = C.DECODE["token_budget"],
                         return_finished: bool = False):
    """Returns, per prompt, a list of n raw continuations (text after the prompt up to its eos, special tokens
    removed). With return_finished=True it returns (continuations, finished), finished[p][k] being True when the
    model emitted eos (or the pad id) in continuation k of prompt p within max_new_tokens (CODE_REVIEW #16).
    Batching: token_batches, with batch_size as an upper bound on prompts per batch."""
    model.eval()
    torch.manual_seed(seed)
    ids = tok(list(prompts), add_special_tokens=False)["input_ids"] if prompts else []   # tokenised once
    stop = {t for t in (tok.eos_token_id, tok.pad_token_id) if t is not None}
    out_all, fin_all = [[] for _ in prompts], [[] for _ in prompts]
    for bi in token_batches([len(x) for x in ids], n, max_new_tokens, token_budget, batch_size):
        width = max(len(ids[i]) for i in bi)
        pad = [width - len(ids[i]) for i in bi]          # left padding: each prompt ends where generation starts
        input_ids = torch.tensor([[tok.pad_token_id] * p + ids[i] for p, i in zip(pad, bi)], device=model.device)
        mask = torch.tensor([[0] * p + [1] * (width - p) for p in pad], device=model.device)
        gen = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=True, temperature=temperature,
                             top_p=top_p, max_new_tokens=max_new_tokens, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        rows = gen[:, width:].tolist()
        # generate writes pad after a row's eos, so either id marks a row that ended. Rows are cut there because a pad
        # sampled mid-reply (pad != eos) does not stop generation, and the text after it could run on to the limit.
        cut = [next((k for k, t in enumerate(r) if t in stop), None) for r in rows]
        texts = tok.batch_decode([r[:k] for r, k in zip(rows, cut)], skip_special_tokens=True)
        for j, i in enumerate(bi):
            out_all[i] = texts[j * n:(j + 1) * n]
            fin_all[i] = [k is not None for k in cut[j * n:(j + 1) * n]]
    return (out_all, fin_all) if return_finished else out_all


def generation_prompt(tok, sample, ctx_tokens: int = C.CTX_TOKENS) -> str:
    """render(sample)[0] cut to the context turns prompt.encode keeps (oldest dropped until the prompt fits ctx_tokens,
    §0: 512), so generation sees the context that training and evaluate.model_nll see. data_sft's further cut for a
    target longer than MAX_SEQ - ctx_tokens tokens depends on the target, so it has no counterpart here."""
    n = encode(tok, sample, ctx_tokens)["n_ctx_turns"]
    return render_parts(sample.ctx[len(sample.ctx) - n:], sample.target)[0]


def generate_replies(model, tok, samples, n: int = 1, **kw):
    """[{sid, cands: [[msg, ...], ...], finished: [bool, ...]}] with messages parsed from the continuations; finished
    is aligned with cands (True = eos within max_new_tokens), so pairs can drop cut-off candidates (CODE_REVIEW #16).
    A continuation that ends its turn with a line of the other speaker (a §5.2 stop) but runs to max_new_tokens
    without eos counts as unfinished, as pairs documents it (no eos before max_new_tokens). Prompts:
    generation_prompt."""
    prompts = [generation_prompt(tok, s) for s in samples]
    conts, fins = sample_continuations(model, tok, prompts, n=n, return_finished=True, **kw)
    return [{"sid": s.sid, "cands": [parse_generation(c) for c in cs], "finished": f}
            for s, cs, f in zip(samples, conts, fins)]
