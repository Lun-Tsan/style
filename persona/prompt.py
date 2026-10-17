"""persona.prompt -- the ONE prompt builder used by training, candidate generation and evaluation.

Format (plain text, identical for base and instruct checkpoints; no chat template):
    對方: 這題我不會
    對方: [貼圖]
    我: 哪一題
    我: 拍給我看
    對方: [程式碼/算式]
    我: <target message 1>
    我: <target message 2><eos>
prompt_text ends with "我: " (role tag + colon + space); target_text = first target message, then "\\n我: " +
each further message of the same turn; encode() appends the tokenizer's eos. A message that spans several lines
keeps its newlines but continuation lines are indented with two spaces, so no line starts with a role tag.
"""
from __future__ import annotations

from . import config as C
from .flag_symbols import CODE_PH
from .normalize import PH, PH_LINE, deid

ROLE = C.ROLE
ME = ROLE[C.TARGET]
OTHER = ROLE[C.PARTNER]


def msg_lines(turn, apply_deid: bool = True):
    role = ROLE[turn.who]
    if turn.symbol:
        return [f"{role}: {CODE_PH}"]
    out = []
    for b in turn.msgs:
        s = b.strip()
        if s in PH:
            out.append(f"{role}: {PH_LINE[s]}")
            continue
        t = deid(b)[0] if apply_deid else b
        out.append(f"{role}: " + t.replace("\n", "\n  "))
    return out


def render_parts(ctx, target, apply_deid: bool = True):
    lines = []
    for tr in ctx:
        lines += msg_lines(tr, apply_deid)
    tlines = msg_lines(target, apply_deid)
    prompt = ("\n".join(lines) + "\n" if lines else "") + f"{ME}: "
    target_text = tlines[0][len(ME) + 2:] + "".join("\n" + l for l in tlines[1:])
    return prompt, target_text


def render(sample, apply_deid: bool = True):
    return render_parts(sample.ctx, sample.target, apply_deid)


def encode(tok, sample, ctx_tokens: int = C.CTX_TOKENS, apply_deid: bool = True):
    """Token ids and loss mask (1 on every target token and on the eos). Context turns are dropped from the oldest
    until the prompt fits ctx_tokens. Prompt and target are tokenised separately, so the boundary is exact."""
    ctx = list(sample.ctx)
    while True:
        prompt, target = render_parts(ctx, sample.target, apply_deid)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(p_ids) <= ctx_tokens or not ctx:
            break
        ctx = ctx[1:]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": p_ids + t_ids, "loss_mask": [0] * len(p_ids) + [1] * len(t_ids),
            "n_prompt": len(p_ids), "n_target": len(t_ids), "n_ctx_turns": len(ctx)}


def parse_generation(text: str, max_msgs: int = C.DECODE["max_msgs"]):
    """Continuation after 'prompt' -> list of the generated messages of the target turn.
    Stops at the first line of the other speaker, at max_msgs, or at the end of text."""
    msgs, cur = [], []
    for line in text.split("\n"):
        if line.startswith(f"{OTHER}: ") or line.startswith(f"{OTHER}:"):
            break
        if line.startswith(f"{ME}: "):
            if cur:
                msgs.append("\n".join(cur))
            cur = [line[len(ME) + 2:]]
        elif line.startswith("  ") and cur:
            cur.append(line[2:])
        elif not msgs and not cur:
            cur = [line]
        else:
            break
        if len(msgs) >= max_msgs:
            break
    if cur and len(msgs) < max_msgs:
        msgs.append("\n".join(cur))
    return [m.strip() for m in msgs if m.strip()]
