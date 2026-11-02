"""Stage 2c: DPO on ruler-ranked pairs, own implementation (PIPELINE_SPEC §5.4).

    python -m persona.train_dpo --sft sft_instruct_all_s0 --pairs runs/pairs_sft_instruct_all_s0_s0/pairs.jsonl
    python -m persona.train_dpo --quick     # 0.8B, quick SFT run and pairs: 1 optimizer step, dev evals at steps 0, 1

--sft is a run directory name under runs/, not a path. Output: runs/dpo_<sft>_<pairs run>_s<seed>/ (pairs run = the
directory holding pairs.jsonl); an existing non-empty one is refused unless --overwrite.
Reference = the SFT model (adapter merged into the weights, frozen); its log-probs are precomputed before training so
only one model lives in VRAM. Policy = a new LoRA on the merged weights. Loss = -log sigmoid(beta * (dc - dr)),
dc = logp_pol(chosen) - logp_ref(chosen), dr likewise for rejected; loss only on target tokens.
Checkpoint selection uses the acceptance rulers on dev generations, not the training score: at step 0 (the SFT
starting point), at each multiple of --eval-every optimizer steps and after the last step, one reply per context of a
fixed seeded dev subset is scored against the real replies of those contexts (D-AUC10, band_inside_mean, feature JSD,
symbol-like share); SELECTION picks the checkpoint. Dev replies go to dev_outputs.jsonl (private), the evaluation
history to metrics.json.
The evaluated model must not have trained on dev. If the SFT run's adapter_best did (train_sft phase 2 on train +
dev), DPO runs twice, as §2.5 retrains the SFT epochs: from its adapter_devsel with the dev evaluations (selected
checkpoint -> adapter_devsel), then from its adapter_best for the same number of steps without evaluations
(checkpoint at the selected step -> adapter_best, end -> adapter_last). Otherwise one run: selected checkpoint ->
adapter_best, end -> adapter_last. run_eval --dpo stacks adapter_best on the SFT run's adapter_best.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import data_sft as D
from . import modeling as MD
from .data import load_all, train_dev_test
from .evaluate import band_metrics, discriminator, feature_jsd, pseudo_turn, real_outputs, texts_of
from .flag_symbols import is_symbol_turn
from .generate import generate_replies
from .samples import Sample
from .text import reply_text

SYMBOL_TOL = 0.05       # SELECTION guard: symbol-like share allowed above the step-0 share (3 of 60 dev replies)
# Selection rule, set in this module. CODE_REVIEW #8 (DPO had no evaluation) states no rule; PIPELINE_SPEC §5.4
# registers "stop when D-AUC no longer decreases", while this rule evaluates to the end and keeps the best record:
# a deviation still to be recorded in §10.
SELECTION = (f"symbol_share above the step-0 share + {SYMBOL_TOL} ranks behind; then lowest auc10; ties: higher band, "
             "then the earlier step")


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


def default_out(sft_dir, pairs_path, seed: int = 0, model: str = "instruct") -> str:
    """dpo_<SFT run>_<pairs run>_s<seed>, so DPO runs on different SFT runs or pair sets do not overwrite each other
    (CODE_REVIEW #14); the pairs run is the name of the directory holding pairs.jsonl."""
    sft = Path(sft_dir).name if sft_dir else f"untrained_{model}"
    return f"dpo_{sft}_{Path(pairs_path).absolute().parent.name}_s{seed}"


def dev_subset(dev, n: int, seed: int):
    """The dev contexts the evaluations of one run share: n drawn with the run seed, kept in time order."""
    idx = np.random.default_rng(seed).choice(len(dev), min(n, len(dev)), replace=False)
    return [dev[i] for i in sorted(idx.tolist())]


def dev_metrics(outputs, real, bands) -> dict:
    """Acceptance rulers (PIPELINE_SPEC §6.1) on the policy's dev replies vs the real replies of the same contexts.
    Symbol-like replies leave the ruler populations as in evaluate.Evaluator (the data layer removed symbol turns
    from the real targets); their share is kept as symbol_share."""
    sym = [is_symbol_turn(reply_text(o["msgs"])) for o in outputs]
    kept = [o for o, s in zip(outputs, sym) if not s]
    d = discriminator(texts_of(real), texts_of(kept))
    return {"auc10": d["auc10"], "auc1": d["auc1"], "d_n": d["n"],
            "band": band_metrics(kept, bands)["band_inside_mean"], "jsd": feature_jsd(kept, real),
            "symbol_share": float(np.mean(sym)) if sym else math.nan}


def selection_key(rec, symbol_base: float = math.nan, symbol_tol: float = SYMBOL_TOL) -> tuple:
    """Sort key of SELECTION. Symbol-like replies leave the ruler populations while real targets hold no symbol turns,
    so a policy could lower D-AUC10 by making dev replies symbol-like: a record whose symbol_share exceeds symbol_base
    (the step-0 share) by more than symbol_tol ranks behind the others (a NaN symbol_base turns this off). Then lower
    D-AUC10, then higher band_inside_mean. NaN (e.g. D-AUC10 with fewer than 10 texts per side, as in --quick) ranks
    behind any number."""
    auc, band, sym = rec["auc10"], rec["band"], rec.get("symbol_share", math.nan)
    over = not math.isnan(symbol_base) and not sym <= symbol_base + symbol_tol + 1e-9    # 1e-9: shares are k / n
    return (over, math.inf if math.isnan(auc) else auc, math.inf if math.isnan(band) else -band)


def generate_dev(pm, tok, samples, seed: int):
    """One reply per dev context (cands[0]). Generation runs in eval mode, where transformers (5.17) does not apply
    gradient checkpointing, so checkpointing stays enabled; use_cache is switched on for the call. Restored afterwards:
    train mode, use_cache, and the torch RNG state (generate re-seeds torch; restoring it keeps the training dropout
    stream independent of --eval-every)."""
    cfg = pm.get_base_model().config
    use_cache = cfg.use_cache
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        pm.eval()
        cfg.use_cache = True
        rows = generate_replies(pm, tok, samples, n=1, seed=seed)
    finally:
        cfg.use_cache = use_cache
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        pm.train()
    return [{"sid": r["sid"], "msgs": r["cands"][0]} for r in rows]


def sft_dev_trained(sft_dir) -> int:
    """Dev samples in the training data of runs/<sft_dir>/adapter_best: phase2.n_dev_in_train of the SFT run's
    metrics.json (train_sft); 0 for a run without phase 2 and without an SFT run."""
    if not sft_dir:
        return 0
    p = C.RUNS / sft_dir / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; it tells whether {sft_dir}/adapter_best was trained on dev")
    return int((json.loads(p.read_text(encoding="utf-8")).get("phase2") or {}).get("n_dev_in_train") or 0)


def run(sft_dir, pairs_path, out=None, seed=0, beta=C.DPO["beta"], lr=C.DPO["lr"], epochs=C.DPO["epochs"],
        quick=False, model="instruct", eval_every=C.DPO["eval_every"], log=print, overwrite=False,
        dev_eval_n=C.DPO["dev_eval_n"]):
    out = C.RUNS / (out or default_out(sft_dir, pairs_path, seed, model))
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists and is not empty; pass --overwrite to write into it")
    if sft_dir and not (C.RUNS / sft_dir / "adapter_best").is_dir():
        raise FileNotFoundError(f"{C.RUNS / sft_dir / 'adapter_best'} not found; --sft takes a run name under runs/")
    # the dev evaluations need a model that did not train on dev (module docstring)
    n_dev_sft = sft_dev_trained(sft_dir)
    sel_adapter = "adapter_devsel" if n_dev_sft else "adapter_best"
    if n_dev_sft and not (C.RUNS / sft_dir / sel_adapter).is_dir():
        raise FileNotFoundError(f"{C.RUNS / sft_dir / sel_adapter} not found; {sft_dir}/adapter_best was trained on "
                                f"{n_dev_sft} dev samples, so the dev selection needs that adapter")
    MD.set_seed(seed)
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model)
    out.mkdir(parents=True, exist_ok=True)
    msgs, n_sys, turns, segs = load_all()
    _, dev, _, train_all = train_dev_test(segs)
    by_sid = {s.sid: s for s in train_all}           # pairs come from training contexts
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    if quick:
        pairs = pairs[:8]
    on_dev = len({p["sid"] for p in pairs} & {s.sid for s in dev})
    if on_dev:
        raise ValueError(f"{pairs_path}: {on_dev} pair contexts are in dev; the dev evaluation needs held-out ones")
    dev_eval = dev_subset(dev, 8 if quick else dev_eval_n, seed)
    real_dev = real_outputs(dev_eval)
    bands = json.load(open(C.REPORTS / "feature_bands.json", encoding="utf-8"))["bands"]
    tok = MD.load_tokenizer(name)
    items = encode_pair_items(tok, by_sid, pairs)
    accum = 8
    total = math.ceil(len(items) / accum) * epochs
    if quick:
        total = min(total, 4)
    log(f"dev evaluation: {len(dev_eval)} of {len(dev)} dev contexts, at step 0, at each multiple of {eval_every} "
        f"steps and after the last step ({total})"
        + (f"; on {sft_dir}/adapter_devsel, then retrained on adapter_best ({n_dev_sft} dev samples in its training "
           "data)" if n_dev_sft else ""))
    evals, best = [], None
    best_dir = "adapter_devsel" if n_dev_sft else "adapter_best"

    def train_phase(adapter, selecting):
        """DPO with the SFT run's `adapter` merged as the reference, `total` optimizer steps -> (steps, loss history).
        selecting: dev evaluations, SELECTION-best checkpoint -> best_dir. Otherwise (retraining) the checkpoint after
        best["step"] steps -> adapter_best. Both phases start from the same RNG state (LoRA init, dropout, pair
        order)."""
        MD.set_seed(seed)
        t0 = time.time()
        base = MD.load_model(name, grad_ckpt=False)
        ref = MD.merge_adapter(MD.load_adapter(base, str(C.RUNS / sft_dir / adapter))) if sft_dir else base
        refs = ref_logps(ref, items, tok.pad_token_id)
        log(f"{name}: reference {sft_dir + '/' + adapter if sft_dir else 'untrained'} merged, log-probs for "
            f"{len(pairs)} pairs in {time.time() - t0:.0f}s; vram {MD.vram_gb()} GB")
        ref.gradient_checkpointing_enable()
        ref.config.use_cache = False
        pm = MD.add_lora(ref)
        opt = torch.optim.AdamW(MD.trainable_params(pm), lr=lr, weight_decay=0.0)
        hist = []

        def evaluate_dev(step):
            nonlocal best
            t1 = time.time()
            outs = generate_dev(pm, tok, dev_eval, seed)
            rec = {"step": step, **dev_metrics(outs, real_dev, bands), "seconds": round(time.time() - t1, 1)}
            # generated text stays in the run directory; opened with "w" at this run's earliest evaluation, so a run
            # that stops before it leaves the directory empty (a plain re-run is then accepted)
            with open(out / "dev_outputs.jsonl", "a" if evals else "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "outputs": outs}, ensure_ascii=False) + "\n")
            evals.append(rec)
            sym0 = evals[0]["symbol_share"]              # step 0: the SFT starting point
            new_best = best is None or selection_key(rec, sym0) < selection_key(best, sym0)
            if new_best:
                best = rec
                pm.save_pretrained(str(out / best_dir))
            log(f"dev eval step {step}: D-AUC10 {rec['auc10']:.3f} band {rec['band']:.2f} feature JSD {rec['jsd']:.3f} "
                f"symbol-like {rec['symbol_share']:.0%} ({rec['seconds']:.0f}s)"
                + (f" -> {best_dir}" if new_best else ""))

        # step 0 (LoRA B = 0: policy = reference) is a candidate, so a DPO run that only moves the dev rulers away
        # from the SFT model keeps the SFT model (a no-op adapter)
        if selecting:
            evaluate_dev(0)
        elif best["step"] == 0:
            pm.save_pretrained(str(out / "adapter_best"))
        rng = random.Random(seed)
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
                    if selecting:
                        if eval_every and step % eval_every == 0:
                            evaluate_dev(step)
                    elif step == best["step"]:
                        pm.save_pretrained(str(out / "adapter_best"))
                    if step >= total:
                        done = True
                        break
            if done:
                break
        if selecting and evals[-1]["step"] != step:
            evaluate_dev(step)                           # once more after the last step
        if not (selecting and n_dev_sft):                # the end of a run on adapter_devsel is not kept
            pm.save_pretrained(str(out / "adapter_last"))
        return step, hist

    steps, hist = train_phase(sel_adapter, True)
    retrain = None
    if n_dev_sft:
        gc.collect()
        torch.cuda.empty_cache()                         # the selection phase's model went out of scope
        steps2, hist2 = train_phase("adapter_best", False)
        retrain = {"sft_adapter": "adapter_best", "steps": steps2, "history": hist2, "adapter_best_step": best["step"]}
    metrics = {"model": name, "sft": sft_dir, "pairs": str(pairs_path), "n_pairs": len(pairs), "beta": beta, "lr": lr,
               "seed": seed, "steps": steps, "history": hist, "eval_every": eval_every, "dev_eval_n": len(dev_eval),
               "dev_eval_sids": [s.sid for s in dev_eval], "selection": SELECTION, "symbol_tol": SYMBOL_TOL,
               "best_step": best["step"], "eval_history": evals,
               # dev held out from the evaluated model: an SFT adapter that did not train on dev (sft_n_dev_in_train
               # counts the dev samples of the SFT run's adapter_best)
               "dev_heldout": True, "selection_sft_adapter": sel_adapter if sft_dir else None,
               "sft_n_dev_in_train": n_dev_sft, "retrain": retrain, "vram_gb_max": MD.vram_gb()}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done: {best_dir} = step {best['step']} (D-AUC10 {best['auc10']:.3f}, band {best['band']:.2f}, symbol-like "
        f"{best['symbol_share']:.0%})"
        + (f"; retrained on {sft_dir}/adapter_best: adapter_best = step {best['step']}" if n_dev_sft else "")
        + f", adapter_last = step {retrain['steps'] if retrain else steps} -> {out}")
    return metrics


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=None, help="run directory name under runs/ (adapter_best inside), not a path")
    ap.add_argument("--pairs", default=None, help="pairs.jsonl written by make_pairs")
    ap.add_argument("--out", default=None, help="run name under runs/ (default dpo_<sft>_<pairs run>_s<seed>)")
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=C.DPO["eval_every"],
                    help="optimizer steps between dev evaluations; step 0 and the last step are evaluated too")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="write into an existing non-empty run directory")
    a = ap.parse_args(argv)
    if a.quick:
        a.sft = a.sft or "sft_quick_all_s0"
        a.pairs = a.pairs or str(C.RUNS / "pairs_quick" / "pairs.jsonl")
    if not a.pairs:
        ap.error("--pairs is required (or --quick)")
    run(a.sft, a.pairs, a.out, a.seed, quick=a.quick, model=a.model, eval_every=a.eval_every, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
