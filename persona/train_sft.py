"""Stage 2a: LoRA SFT (PIPELINE_SPEC §5.1).

    python -m persona.train_sft --model instruct --size all --seed 0
    python -m persona.train_sft --quick            # 0.8B model, 16 samples, 4 optimiser steps (smoke test)
    python -m persona.train_sft --measure          # 4B model, 3 optimiser steps: seconds/step and VRAM only
    options: --epochs N, --attn-only (LoRA ablation), --no-retrain, --out NAME, --overwrite

Phase 1 trains on train minus dev and keeps the epoch with the lowest dev NLL. Phase 2 (§2.5, C.SFT["retrain_full"])
trains a new LoRA on the base model with train + dev for that many epochs and the same schedule shape. For a subset
size, phase 2 draws that many samples the same nested, time-uniform way from train + dev, so each point of the
data-size curve keeps its size. --no-retrain, --quick and --measure skip phase 2.

Outputs runs/sft_<model>_<size>_s<seed>[_attn][_ep<N>][_noretrain]/ (suffixes: --attn-only, a non-default --epochs,
no phase 2): adapter_best/ (phase 2, else the dev-selected adapter) and adapter_devsel/ (phase 1, when retrained),
both LoRA weights and private; metrics.json (phase-1 loss curve and dev NLL per epoch, phase-2 loss curve, seconds,
canary audit of adapter_best). An existing non-empty run directory is refused unless --overwrite.
"""
from __future__ import annotations

import argparse
import gc
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
from .data import load_all


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def size_key(size):
    """--size as the key of C.SFT["epochs"] and split.nested_subsets (None = all)."""
    return None if size in (None, "all", "None") else int(size)


def prepare_data(size, seed=0, n_override=None, dev_override=None):
    """(subset of train minus dev, dev, size key, phase-2 set). The phase-2 set is the subset of the same size drawn
    the same way from train + dev; for size all it is train + dev."""
    msgs, n_sys, turns, segs = load_all()
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    train, dev = SP.dev_cut(train_all)
    subsets = SP.nested_subsets(train)
    key = size_key(size)
    tr = subsets[key]
    full = SP.nested_subsets(train_all)[key]
    if n_override:
        tr = tr[:n_override]
    if dev_override:
        dev = dev[:dev_override]
    return tr, dev, key, full


def default_out(tag, key, seed, targets=MD.LORA_ALL, epochs=None, no_retrain=False) -> str:
    """Run name carrying the settings that change adapter_best (CODE_REVIEW #14), so an ablation does not write into
    the main run: _attn (--attn-only), _lora-<modules> (another target set), _ep<N>, _noretrain."""
    t = set(targets)
    lora = ("" if t == set(MD.LORA_ALL) else "_attn" if t == set(MD.LORA_MIX_ONLY)
            else "_lora-" + "-".join(sorted(m.replace("_proj", "") for m in t)))
    return (f"sft_{tag}_{key or 'all'}_s{seed}{lora}" + (f"_ep{epochs}" if epochs else "")
            + ("_noretrain" if no_retrain else ""))


def load_lora(pm, path):
    """Copy the LoRA weights saved at `path` into pm, with the loader that PeftModel.from_pretrained uses."""
    from peft import load_peft_weights, set_peft_model_state_dict
    res = set_peft_model_state_dict(pm, load_peft_weights(str(path)))
    missing = [k for k in res.missing_keys if "lora_" in k and "default" in k]     # peft load_adapter's filter
    if missing:                      # a partial load would audit other weights than the saved ones
        raise RuntimeError(f"{path}: {len(missing)} LoRA tensors were not loaded")


def canary_audit(pm, tok, can_items, items_dev, dev_trained=0):
    """Memorisation audit (§5.1): NLL/token of each canary vs the mean NLL/token of the real dev targets.
    dev_items_trained counts dev targets inside this model's training data (phase 2: the baseline is not held out)."""
    can = D.nll_per_item(pm, can_items, tok.pad_token_id)
    dev_items = D.nll_per_item(pm, items_dev, tok.pad_token_id) if items_dev else []
    return {"canary_nll_per_token": [c[0] / c[1] for c in can], "dev_nll_per_token": D.mean_nll(dev_items),
            "dev_items_n": len(dev_items), "dev_items_trained": dev_trained}


def train_epochs(pm, tok, items_tr, items_dev, n_ep, seed, save_dir, quick=False, measure=False, log=print):
    """n_ep epochs of the §5.1 schedule (AdamW; warmup, then cosine decay over the whole run).
    items_dev given (phase 1): dev NLL before training and after each epoch; the adapter of the lowest-dev-NLL epoch
    is saved to save_dir. items_dev None (phase 2): the adapter after the last epoch is saved to save_dir.
    --measure does not save an adapter."""
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
    t0 = time.time()
    if items_dev is not None:
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
        if items_dev is not None:
            dev_nll = D.mean_nll(D.nll_per_item(pm, items_dev, tok.pad_token_id)) if items_dev else math.nan
            hist["dev_nll"].append({"epoch": ep, "nll": dev_nll})
            log(f"epoch {ep} dev NLL {dev_nll:.4f}")
            if not measure and (math.isnan(dev_nll) or dev_nll < best):
                best, best_ep = dev_nll, ep
                pm.save_pretrained(str(save_dir))
        if done:
            break
    res = {"epochs": n_ep, "steps": step, "history": hist, "seconds": time.time() - t0,
           "seconds_per_step_median": float(np.median(hist["step_seconds"])) if hist["step_seconds"] else None}
    if items_dev is not None:
        res.update(best_epoch=best_ep, best_dev_nll=best)
    else:                            # phase 2: dev is training data, so the last epoch is the model
        del hist["dev_nll"]
        if not measure:
            pm.save_pretrained(str(save_dir))
    return res


def run(model="instruct", size=None, seed=0, out=None, epochs=None, targets=MD.LORA_ALL, quick=False,
        measure=False, log=print, retrain=None, overwrite=False):
    """retrain None follows C.SFT["retrain_full"]; --quick and --measure do not retrain. An existing non-empty run
    directory is refused unless overwrite (checked before the data and the model are loaded)."""
    MD.set_seed(seed)
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model)
    full_run = not (quick or measure)
    retrain = full_run and (C.SFT["retrain_full"] if retrain is None else bool(retrain))
    key = size_key(size)
    n_ep = 1 if not full_run else (epochs or C.SFT["epochs"][key])
    tag = "quick" if quick else ("measure" if measure else model)
    ep_tag = n_ep if full_run and epochs and epochs != C.SFT["epochs"].get(key) else None
    out = C.RUNS / (out or default_out(tag, key, seed, targets, ep_tag, no_retrain=full_run and not retrain))
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists and is not empty; pass --overwrite to write into it")
    out.mkdir(parents=True, exist_ok=True)
    tr, dev, _, full = prepare_data(size, seed, n_override=16 if quick else (12 if measure else None),
                                    dev_override=8 if (quick or measure) else None)
    tok = MD.load_tokenizer(name)
    t0 = time.time()
    model_ = MD.load_model(name, grad_ckpt=True)
    pm = MD.add_lora(model_, targets=targets)
    tot, trn = MD.n_params(pm)
    log(f"model {name} loaded in {time.time() - t0:.0f}s; params {tot / 1e6:.0f}M, trainable {trn / 1e6:.2f}M; "
        f"train {len(tr)} dev {len(dev)} epochs {n_ep}" + (f"; phase 2 on train + dev {len(full)}" if retrain else ""))
    can = D.encode_samples(tok, D.canary_samples())
    items_tr = D.encode_samples(tok, tr) + can
    items_dev = D.encode_samples(tok, dev)
    log(f"encoded: train tokens median {int(np.median([len(i['input_ids']) for i in items_tr]))}, "
        f"max {max(len(i['input_ids']) for i in items_tr)}; target tokens median {int(np.median([i['n_target'] for i in items_tr]))}")
    p1 = train_epochs(pm, tok, items_tr, items_dev, n_ep, seed, out / ("adapter_devsel" if retrain else "adapter_best"),
                      quick=quick, measure=measure, log=log)
    p2 = audit_devsel = None
    n_dev_trained = 0
    if retrain:
        # CODE_REVIEW #9. The dev-selected adapter is audited here, while dev is still held out from it.
        load_lora(pm, out / "adapter_devsel")
        audit_devsel = canary_audit(pm, tok, can, items_dev)
        del pm, model_                # the phase-1 optimizer and scheduler went out of scope in train_epochs
        gc.collect()
        torch.cuda.empty_cache()      # free the phase-1 weights before the base model is loaded again
        MD.set_seed(seed)             # same RNG state as at the start of phase 1 (LoRA init, dropout)
        t0 = time.time()
        model_ = MD.load_model(name, grad_ckpt=True)
        pm = MD.add_lora(model_, targets=targets)
        dev_sids = {s.sid for s in dev}
        n_dev_trained = sum(s.sid in dev_sids for s in full)
        log(f"phase 2: model reloaded in {time.time() - t0:.0f}s; new LoRA, train + dev {len(full)} samples "
            f"({n_dev_trained} from dev), {p1['best_epoch']} epochs")
        p2 = train_epochs(pm, tok, D.encode_samples(tok, full) + can, None, p1["best_epoch"], seed,
                          out / "adapter_best", log=log)
    if not measure:                   # CODE_REVIEW #17: audit the weights saved as adapter_best, not the last epoch's
        load_lora(pm, out / "adapter_best")
    audit = canary_audit(pm, tok, can, items_dev, n_dev_trained)
    metrics = {"model": name, "size": key, "seed": seed, "epochs": n_ep, "steps": p1["steps"],
               "best_epoch": p1["best_epoch"], "best_dev_nll": p1["best_dev_nll"],
               "dev_nll_before": p1["history"]["dev_nll"][0]["nll"], "history": p1["history"],
               "seconds": p1["seconds"], "canary": audit, "seconds_per_step_median": p1["seconds_per_step_median"],
               "vram_gb_max": MD.vram_gb(), "n_train": len(tr), "n_dev": len(dev), "lora_targets": list(targets),
               "retrained": retrain, "canary_devsel": audit_devsel,
               "phase2": None if p2 is None else {"n_train": len(full), "n_dev_in_train": n_dev_trained, **p2}}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done: best dev NLL {p1['best_dev_nll']:.4f} (epoch {p1['best_epoch']})"
        + (f"; retrained on train + dev for {p2['epochs']} epochs in {p2['seconds']:.0f}s" if p2 else "")
        + f"; canary NLL/token {['%.2f' % x for x in audit['canary_nll_per_token']]} vs dev "
        f"{audit['dev_nll_per_token']:.2f}; median s/step {metrics['seconds_per_step_median']}; -> {out}")
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
    ap.add_argument("--no-retrain", action="store_true",
                    help="skip phase 2 (retraining on train + dev); adapter_best is then the dev-selected adapter")
    ap.add_argument("--overwrite", action="store_true", help="write into an existing non-empty run directory")
    a = ap.parse_args(argv)
    run(a.model, a.size, a.seed, a.out, a.epochs, MD.LORA_MIX_ONLY if a.attn_only else MD.LORA_ALL, a.quick, a.measure,
        retrain=False if a.no_retrain else None, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
