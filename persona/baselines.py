"""persona.baselines -- prompting baselines B0 (zero-shot), B1-n (many-shot), B2 (retrieval) for the instruct
checkpoint (PIPELINE_SPEC §4). They use the chat template (thinking off); the context itself is rendered by the
same prompt builder as every other system."""
from __future__ import annotations

import re

import numpy as np

from . import config as C
from .generate import sample_continuations
from .normalize import PH_LINE
from .prompt import ME, OTHER, parse_generation, render

SYSTEM = ("你是對話中的「我」。下面是一段 LINE 對話紀錄,請以「我」的身分接著回覆對方。"
          "用台灣口語的繁體中文;一行一則訊息,可以只回一則,也可以連發多則;要傳貼圖就寫 [貼圖]。"
          "只輸出回覆的訊息內容,每行開頭不要加「我:」,不要解釋。")
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def example_pairs(train_samples, n: int, seed: int = 0):
    """n (partner turn -> T reply) examples, equally spaced in time (seed shifts the phase)."""
    cands = [s for s in train_samples if s.ctx and s.ctx[-1].who == C.PARTNER and not s.ctx[-1].symbol]
    if n is None or n >= len(cands):
        return cands
    rng = np.random.default_rng(seed)
    off = rng.uniform(0, 1)
    return [cands[int(((i + off) * len(cands)) / n) % len(cands)] for i in range(n)]


def example_block(examples):
    lines = []
    for s in examples:
        last = s.ctx[-1]
        for m in last.msgs:
            lines.append(f"{OTHER}: {PH_LINE.get(m.strip(), m)}")
        for m in s.target.msgs:
            lines.append(f"{ME}: {PH_LINE.get(m.strip(), m)}")
        lines.append("")
    return "\n".join(lines)


def build_messages(sample, examples=None):
    prompt, _ = render(sample)
    ctx_text = prompt[:-len(f"{ME}: ")].rstrip() or "(對話開始)"
    user = ""
    if examples:
        user += "以下是「我」過去的回覆範例,照這個風格回:\n\n" + example_block(examples) + "\n"
    user += "現在的對話:\n" + ctx_text + "\n\n請接著以「我」的身分回覆。"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def chat_prompt(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse_chat_output(text: str, max_msgs: int = C.DECODE["max_msgs"]):
    text = THINK_RE.sub("", text).strip()
    msgs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(f"{OTHER}:"):
            break
        line = re.sub(rf"^{ME}[:：]\s*", "", line)
        msgs.append(line)
        if len(msgs) >= max_msgs:
            break
    return msgs


def retrieval_examples(bge, train_samples, sample, k: int = 20):
    """k training examples whose last partner turn is closest (bge-m3 cosine) to the current last partner turn."""
    cands = [s for s in train_samples if s.ctx and s.ctx[-1].who == C.PARTNER and s.ctx[-1].ruler_text and not s.ctx[-1].symbol]
    if not sample.ctx or not sample.ctx[-1].ruler_text:
        return example_pairs(train_samples, k)
    q = sample.ctx[-1].ruler_text
    E = bge.encode([s.ctx[-1].ruler_text for s in cands])
    sims = E @ bge.encode([q])[0]
    top = np.argsort(-sims)[:k]
    return [cands[i] for i in sorted(top, key=lambda i: cands[i].target.date)]


def run_baseline(model, tok, samples, kind: str, train_samples=None, n_shots=None, bge=None, seed: int = 0, **kw):
    """kind in {zero, many, retrieval}. Returns [{sid, msgs}] and the median prompt length in tokens."""
    prompts = []
    for s in samples:
        if kind == "zero":
            ex = None
        elif kind == "many":
            ex = example_pairs(train_samples, n_shots, seed)
        elif kind == "retrieval":
            ex = retrieval_examples(bge, train_samples, s, n_shots or 20)
        else:
            raise ValueError(kind)
        prompts.append(chat_prompt(tok, build_messages(s, ex)))
    n_tok = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    conts = sample_continuations(model, tok, prompts, n=1, seed=seed, **kw)
    outs = [{"sid": s.sid, "msgs": parse_chat_output(c[0])} for s, c in zip(samples, conts)]
    return outs, {"prompt_tokens_median": int(np.median(n_tok)), "prompt_tokens_max": int(max(n_tok))}
