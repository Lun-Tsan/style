"""persona.data_sft -- encoded samples, collation, teacher-forced NLL, canary sentences (memorisation audit)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import config as C
from .prompt import encode
from .samples import Sample
from .turns import Turn

# three fictitious sentences that never occur in the chat; added once each to the training set (PIPELINE_SPEC §5.1)
CANARIES = ["今天的風把海邊的橘色燈塔吹成了三角形", "我把鯨魚的鑰匙放在冰箱第七層了", "星期八的數學課改成在屋頂上跳繩"]


def canary_samples(start_sid: int = 10_000_000):
    out = []
    for i, text in enumerate(CANARIES):
        ctx = [Turn(C.PARTNER, "2026-01-01", 0, 0, ["你今天怎麼樣"])]
        tgt = Turn(C.TARGET, "2026-01-01", 1, 1, [text])
        out.append(Sample(start_sid + i, ctx, tgt))
    return out


def encode_samples(tok, samples, ctx_tokens: int = C.CTX_TOKENS, max_seq: int = C.MAX_SEQ):
    items = []
    for s in samples:
        e = encode(tok, s, ctx_tokens=ctx_tokens)
        if len(e["input_ids"]) > max_seq:                      # very long target: shrink the context budget
            e = encode(tok, s, ctx_tokens=max(16, max_seq - e["n_target"]))
            if len(e["input_ids"]) > max_seq:                  # still too long: truncate the target (rare)
                e["input_ids"] = e["input_ids"][:max_seq]
                e["loss_mask"] = e["loss_mask"][:max_seq]
        e["sid"] = s.sid
        items.append(e)
    return items


def collate(items, pad_id: int, device: str = "cuda"):
    L = max(len(it["input_ids"]) for it in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        ids[i, :n] = torch.tensor(it["input_ids"])
        att[i, :n] = 1
        m = torch.tensor(it["loss_mask"], dtype=torch.bool)
        lab[i, :n] = torch.where(m, ids[i, :n], torch.full((n,), -100, dtype=torch.long))
    return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device)}


def lm_loss(model, batch):
    """Summed token NLL over labelled positions and the number of labelled tokens."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, int((labels != -100).sum())


@torch.no_grad()
def nll_per_item(model, items, pad_id: int, batch_size: int = 4, device: str = "cuda"):
    """Teacher-forced NLL per item: list of (sum_nll, n_tokens); items sorted by length internally."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    res = [None] * len(items)
    for a in range(0, len(order), batch_size):
        bi = order[a:a + batch_size]
        batch = collate([items[i] for i in bi], pad_id, device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        tok_nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                                  reduction="none").view(labels.shape)
        for j, i in enumerate(bi):
            m = labels[j] != -100
            res[i] = (float(tok_nll[j][m].sum()), int(m.sum()))
    return res


def mean_nll(pairs):
    s = sum(p[0] for p in pairs)
    n = sum(p[1] for p in pairs)
    return s / n if n else math.nan
