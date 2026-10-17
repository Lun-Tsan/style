"""Stage 2a: LoRA SFT (PIPELINE_SPEC §5.1).

    python -m persona.train_sft --model instruct --size all --seed 0
    python -m persona.train_sft --quick            # 0.8B model, 16 samples, 4 optimiser steps (smoke test)
    python -m persona.train_sft --measure          # 4B model, 3 optimiser steps: seconds/step and VRAM only

Outputs runs/sft_<model>_<size>_s<seed>/: adapter_best/ (LoRA weights, private), metrics.json (loss curve, dev NLL
per epoch, canary audit, timing), config.json.
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

from . import config as C
from . import data_sft as D
from . import modeling as MD
from . import samples as SM, split as SP
from .stage1 import load_all


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def prepare_data(size, seed=0, n_override=None, dev_override=None):
    msgs, n_sys, turns, segs = load_all()
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    train, dev = SP.dev_cut(train_all)
    subsets = SP.nested_subsets(train)
    key = None if size in (None, "all", "None") else int(size)
    tr = subsets[key]
    if n_override:
        tr = tr[:n_override]
    if dev_override:
        dev = dev[:dev_override]
    return tr, dev, key


def run(model="instruct", size=None, seed=0, out=None, epochs=None, targets=MD.LORA_ALL, quick=False,
        measure=False, log=print):
    MD.set_seed(seed)
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model)
    tr, dev, key = prepare_data(size, seed, n_override=16 if quick else (12 if measure else None),
                                dev_override=8 if (quick or measure) else None)
    n_ep = 1 if (quick or measure) else (epochs or C.SFT["epochs"][key])
    tag = "quick" if quick else ("measure" if measure else model)
    out = C.RUNS / (out or f"sft_{tag}_{key or 'all'}_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    tok = MD.load_tokenizer(name)
    t0 = time.time()
    model_ = MD.load_model(name, grad_ckpt=True)
    pm = MD.add_lora(model_, targets=targets)
    tot, trn = MD.n_params(pm)
    log(f"model {name} loaded in {time.time() - t0:.0f}s; params {tot / 1e6:.0f}M, trainable {trn / 1e6:.2f}M; "
        f"train {len(tr)} dev {len(dev)} epochs {n_ep}")
    items_tr = D.encode_samples(tok, tr) + D.encode_samples(tok, D.canary_samples())
    items_dev = D.encode_samples(tok, dev)
    log(f"encoded: train tokens median {int(np.median([len(i['input_ids']) for i in items_tr]))}, "
        f"max {max(len(i['input_ids']) for i in items_tr)}; target tokens median {int(np.median([i['n_target'] for i in items_tr]))}")

    micro, accum = C.SFT["micro_batch"], C.SFT["grad_accum"]
    steps_per_epoch = math.ceil(len(items_tr) / (micro * accum))
    total_steps = steps_per_epoch * n_ep
    if quick:
        total_steps = min(total_steps, 4)
    if measure:
        total_steps = 3
    warm = max(1, round(total_steps * C.SFT["warmup"]))
    opt = torch.optim.AdamW(MD.trainable_params(pm), lr=C.SFT["lr"], weight_decay=C.SFT["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total_steps, warm))
    rng = random.Random(seed)
    hist = {"train_loss": [], "dev_nll": [], "step_seconds": []}
    best, best_ep = math.inf, -1
    step = 0
    dev0 = D.mean_nll(D.nll_per_item(pm, items_dev, tok.pad_token_id)) if items_dev else math.nan
    hist["dev_nll"].append({"epoch": 0, "nll": dev0})
    log(f"dev NLL before training {dev0:.4f}")
    done = False
    for ep in range(1, n_ep + 1):
        pm.train()
        order = list(range(len(items_tr)))
        rng.shuffle(order)
        acc_loss = acc_n = 0.0
        for a in range(0, len(order), micro):
            if done:
                break
            batch = D.collate([items_tr[i] for i in order[a:a + micro]], tok.pad_token_id)
            t1 = time.time()
            loss, n = D.lm_loss(pm, batch)
            (loss / max(n, 1) / accum).backward()
            acc_loss += float(loss.detach())
            acc_n += n
            if ((a // micro) + 1) % accum == 0 or a + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(MD.trainable_params(pm), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                hist["train_loss"].append({"step": step, "loss": acc_loss / max(acc_n, 1)})
                hist["step_seconds"].append(time.time() - t1)
                if step % 10 == 0 or quick or measure:
                    log(f"ep {ep} step {step}/{total_steps} loss/token {acc_loss / max(acc_n, 1):.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} vram {MD.vram_gb()} GB")
                acc_loss = acc_n = 0.0
                if step >= total_steps:
                    done = True
        dev_nll = D.mean_nll(D.nll_per_item(pm, items_dev, tok.pad_token_id)) if items_dev else math.nan
        hist["dev_nll"].append({"epoch": ep, "nll": dev_nll})
        log(f"epoch {ep} dev NLL {dev_nll:.4f}")
        if not measure and (math.isnan(dev_nll) or dev_nll < best):
            best, best_ep = dev_nll, ep
            pm.save_pretrained(str(out / "adapter_best"))
        if done:
            break
    # canary audit: NLL/token of the canaries vs real dev targets of similar length
    can = D.nll_per_item(pm, D.encode_samples(tok, D.canary_samples()), tok.pad_token_id)
    dev_items = D.nll_per_item(pm, items_dev, tok.pad_token_id) if items_dev else []
    audit = {"canary_nll_per_token": [c[0] / c[1] for c in can],
             "dev_nll_per_token": D.mean_nll(dev_items),
             "dev_items_n": len(dev_items)}
    metrics = {"model": name, "size": key, "seed": seed, "epochs": n_ep, "steps": step, "best_epoch": best_ep,
               "best_dev_nll": best, "dev_nll_before": dev0, "history": hist, "canary": audit,
               "seconds_per_step_median": float(np.median(hist["step_seconds"])) if hist["step_seconds"] else None,
               "vram_gb_max": MD.vram_gb(), "n_train": len(tr), "n_dev": len(dev), "lora_targets": list(targets)}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done: best dev NLL {best:.4f} (epoch {best_ep}); canary NLL/token {['%.2f' % x for x in audit['canary_nll_per_token']]} "
        f"vs dev {audit['dev_nll_per_token']:.2f}; median s/step {metrics['seconds_per_step_median']}; -> {out}")
    return metrics


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--size", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn-only", action="store_true", help="LoRA on the token-mixing projections only (ablation)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure", action="store_true")
    a = ap.parse_args(argv)
    run(a.model, a.size, a.seed, a.out, a.epochs, MD.LORA_MIX_ONLY if a.attn_only else MD.LORA_ALL, a.quick, a.measure)


if __name__ == "__main__":
    main()
