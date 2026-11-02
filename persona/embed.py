"""persona.embed -- bge-m3 dense embeddings (CLS, L2-normalised), used ONLY for the on-topic gate
(experiment 2: semantic cosine sees topic, not style or polarity). torch is imported inside BGE, so the threshold
helpers import on a machine without it (the tests check them with a stand-in encoder)."""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from . import config as C
from .text import reply_text

NAME = "BAAI/bge-m3"


class BGE:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(NAME)
        self.model = AutoModel.from_pretrained(NAME, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.device = device
        self.cache = {}

    def encode(self, texts, batch: int = 64):
        import torch
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        order = sorted(todo, key=len)
        with torch.inference_mode():
            for a in range(0, len(order), batch):
                bt = order[a:a + batch]
                enc = self.tok(bt, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0].float()
                e = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
                for t, v in zip(bt, e):
                    self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1024), np.float32)

    def cos(self, a_texts, b_texts):
        A, B = self.encode(a_texts), self.encode(b_texts)
        return (A * B).sum(1)


def null_threshold(bge: BGE, real_texts, q: float = C.DPO["ontopic_q"], n_pairs: int = 2000, seed: int = 0) -> float:
    """q-quantile of cosine between a real reply and an unrelated real reply (different context)."""
    rng = np.random.default_rng(seed)
    E = bge.encode(real_texts)
    i = rng.integers(0, len(real_texts), n_pairs)
    j = rng.integers(0, len(real_texts), n_pairs)
    ok = i != j
    c = (E[i[ok]] * E[j[ok]]).sum(1)
    return float(np.quantile(c, q))


def fit_ontopic_threshold(bge: BGE, samples) -> float:
    """On-topic threshold shared by make_pairs and run_eval: the C.DPO["ontopic_q"] null quantile (null_threshold)
    of cosine between unrelated real replies of the TRAINING period (train incl. dev; targets with text). Samples of
    other splits are dropped here, so test replies do not enter it (PIPELINE_SPEC §0: thresholds are calibrated on
    the training period only; CODE_REVIEW #12)."""
    texts = [reply_text(s.target.msgs) for s in samples if s.split == "train" and s.target.text_b]
    if len(texts) < 2:
        raise ValueError(f"on-topic threshold needs training-period replies with text; got {len(texts)}")
    return null_threshold(bge, texts)
