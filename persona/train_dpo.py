"""Stage 2c: DPO on ruler-ranked pairs, own implementation (PIPELINE_SPEC §5.4).

    python -m persona.train_dpo --sft runs/sft_instruct_all_s0 --pairs runs/pairs_instruct_all_s0/pairs.jsonl
    python -m persona.train_dpo --quick     # 0.8B, the quick SFT run and quick pairs, 4 steps

Reference = the SFT model (adapter merged into the weights, frozen); its log-probs are precomputed once so only
one model lives in VRAM. Policy = a new LoRA on the merged weights. Loss = -log sigmoid(beta * (dc - dr)),
dc = logp_pol(chosen) - logp_ref(chosen), dr likewise for rejected; loss only on target tokens.
Early stopping uses the acceptance rulers on dev replies (independent discriminator AUC), not the training score.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import data_sft as D
from . import modeling as MD
from . import samples as SM, split as SP
from .evaluate import pseudo_turn
from .samples import Sample
from .data import load_all


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float):
    """Batched DPO loss and diagnostics; all inputs are summed target-token log-probs (tensors of shape (B,))."""
    dc = pol_c - ref_c
    dr = pol_r - ref_r
    logits = beta * (dc - dr)
    loss = -F.logsigmoid(logits).mean()
    dcd, drd = dc.detach(), dr.detach()
    return loss, {"reward_acc": float((dcd > drd).float().mean()), "margin": float((dcd - drd).mean()),
                  "chosen_delta": float(dcd.mean()), "rejected_delta": float(drd.mean())}


def seq_logp(model, batch):
    """Sum of log p(token) over labelled positions, per sequence -> (B,)."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    lp = torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(1)


def encode_pair_items(tok, samples_by_sid, pairs):
    items = []
    for p in pairs:
        s = samples_by_sid[p["sid"]]
        c = Sample(s.sid, s.ctx, pseudo_turn(p["chosen"], s.target.date))
        r = Sample(s.sid, s.ctx, pseudo_turn(p["rejected"], s.target.date))
        ec, er = D.encode_samples(tok, [c, r])
        items.append((ec, er))
    return items


@torch.no_grad()
def ref_logps(model, items, pad_id):
    model.eval()
    out = []
    for ec, er in items:
        b = D.collate([ec, er], pad_id)
        lp = seq_logp(model, b)
        out.append((float(lp[0]), float(lp[1])))
    return out


def run(sft_dir, pairs_path, out=None, seed=0, beta=C.DPO["beta"], lr=C.DPO["lr"], epochs=C.DPO["epochs"],
        quick=False, model="instruct", eval_every=100, log=print):
    MD.set_seed(seed)
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model)
    out = C.RUNS / (out or ("dpo_quick" if quick else f"dpo_{model}_s{seed}"))
    out.mkdir(parents=True, exist_ok=True)
    msgs, n_sys, turns, segs = load_all()
    smp = SM.build_samples(segs)
    by_sid = {s.sid: s for s in smp}
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    if quick:
        pairs = pairs[:8]
    tok = MD.load_tokenizer(name)
    t0 = time.time()
    base = MD.load_model(name, grad_ckpt=False)
    ref = MD.merge_adapter(MD.load_adapter(base, str(C.RUNS / sft_dir / "adapter_best"))) if sft_dir else base
    items = encode_pair_items(tok, by_sid, pairs)
    refs = ref_logps(ref, items, tok.pad_token_id)
    log(f"{name}: SFT merged, reference log-probs for {len(pairs)} pairs in {time.time() - t0:.0f}s; vram {MD.vram_gb()} GB")
    ref.gradient_checkpointing_enable()
    ref.config.use_cache = False
    pm = MD.add_lora(ref)
    opt = torch.optim.AdamW(MD.trainable_params(pm), lr=lr, weight_decay=0.0)
    accum = 8
    total = math.ceil(len(items) / accum) * epochs
    if quick:
        total = min(total, 4)
    rng = random.Random(seed)
    hist = []
    step = 0
    done = False
    for ep in range(epochs):
        order = list(range(len(items)))
        rng.shuffle(order)
        pm.train()
        acc = {"loss": 0.0, "reward_acc": 0.0, "margin": 0.0, "n": 0}
        for j, i in enumerate(order):
            ec, er = items[i]
            b = D.collate([ec, er], tok.pad_token_id)
            lp = seq_logp(pm, b)
            loss, diag = dpo_loss(lp[0:1], lp[1:2], torch.tensor([refs[i][0]], device=lp.device),
                                  torch.tensor([refs[i][1]], device=lp.device), beta)
            (loss / accum).backward()
            acc["loss"] += float(loss.detach())
            acc["reward_acc"] += diag["reward_acc"]
            acc["margin"] += diag["margin"]
            acc["n"] += 1
            if (j + 1) % accum == 0 or j + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(MD.trainable_params(pm), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "loss": acc["loss"] / acc["n"], "reward_acc": acc["reward_acc"] / acc["n"],
                       "margin": acc["margin"] / acc["n"]}
                hist.append(rec)
                log(f"ep {ep + 1} step {step}/{total} loss {rec['loss']:.4f} reward_acc {rec['reward_acc']:.2f} "
                    f"margin {rec['margin']:.3f} vram {MD.vram_gb()} GB")
                acc = {"loss": 0.0, "reward_acc": 0.0, "margin": 0.0, "n": 0}
                if step >= total:
                    done = True
                    break
        if done:
            break
    pm.save_pretrained(str(out / "adapter_best"))
    metrics = {"model": name, "sft": sft_dir, "pairs": str(pairs_path), "n_pairs": len(pairs), "beta": beta, "lr": lr,
               "steps": step, "history": hist, "vram_gb_max": MD.vram_gb()}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done -> {out}")
    return metrics


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=None, help="run directory name under runs/ (adapter_best inside)")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.sft = a.sft or "sft_quick_all_s0"
        a.pairs = a.pairs or str(C.RUNS / "pairs_quick" / "pairs.jsonl")
    run(a.sft, a.pairs, a.out, a.seed, quick=a.quick, model=a.model)


if __name__ == "__main__":
    main()
