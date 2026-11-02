"""Stage 2d: evaluate a system on the test contexts with the acceptance rulers (PIPELINE_SPEC §6).

    python -m persona.run_eval --system controls                      # C-real, C-shuf, C-S (no model)
    python -m persona.run_eval --system base                          # untrained model, raw continuation
    python -m persona.run_eval --system baseline --kind zero          # B0
    python -m persona.run_eval --system baseline --kind many --shots 50
    python -m persona.run_eval --system baseline --kind retrieval --shots 20
    python -m persona.run_eval --system sft --run sft_instruct_all_s0 [--dpo dpo_instruct_s0]
    options: --seed, --quick (0.8B, 12 contexts), --no-bge (skip on-topic), --mmi (coherence under the untrained model)

Outputs runs/eval_<name>/outputs.jsonl + metrics.json; reports/eval_table.json accumulates one row per name.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from . import config as C
from . import evaluate as EV
from . import modeling as MD
from . import samples as SM, split as SP
from .generate import generate_replies
from .data import load_all


def build_evaluator(turns, test, use_bge=True, log=print):
    bands = json.load(open(C.REPORTS / "feature_bands.json", encoding="utf-8"))["bands"]
    tr = [t for t in turns if t.split == "train"]
    teS = [t for t in turns if t.split == "test" and t.who == C.PARTNER]
    bge = thr = None
    if use_bge:
        from .embed import BGE
        bge = BGE()
    ev = EV.Evaluator(test, tr, teS, bands, bge, thr)
    if use_bge:
        from .embed import null_threshold
        ev.thr = null_threshold(bge, [t for t in ev.real_texts if t])
        log(f"on-topic threshold (null {C.DPO['ontopic_q']:.0%} quantile of cosine between unrelated real replies): {ev.thr:.3f}")
    return ev


def load_system(model_key, run=None, dpo=None, quick=False):
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model_key)
    tok = MD.load_tokenizer(name)
    m = MD.load_model(name)
    if run:
        m = MD.load_adapter(m, str(C.RUNS / run / "adapter_best"))
        if dpo:
            m = MD.merge_adapter(m)
            m = MD.load_adapter(m, str(C.RUNS / dpo / "adapter_best"))
    return m, tok


def save(name, outputs, metrics, extra=None):
    out = C.RUNS / f"eval_{name}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "outputs.jsonl", "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    metrics = dict(metrics, **(extra or {}))
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    tab_path = C.REPORTS / "eval_table.json"
    tab = json.load(open(tab_path, encoding="utf-8")) if tab_path.exists() else {}
    tab[name] = metrics
    tab_path.write_text(json.dumps(tab, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["controls", "base", "baseline", "sft"])
    ap.add_argument("--kind", default="zero", choices=["zero", "many", "retrieval"])
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--dpo", default=None)
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-bge", action="store_true")
    ap.add_argument("--mmi", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    msgs, n_sys, turns, segs = load_all()
    smp = SM.build_samples(segs)
    test = [s for s in smp if s.split == "test"]
    train_all = [s for s in smp if s.split == "train"]
    if a.quick:
        test = test[:12]
    ev = build_evaluator(turns, test, use_bge=not a.no_bge)
    print(f"evaluator ready ({len(test)} test contexts) in {time.time() - t0:.0f}s; floor AUC1 {ev._floor['auc1']:.3f}, "
          f"sensitivity AUC1 {ev._sens['auc1']:.3f}")

    if a.system == "controls":
        for name, outs in (("C-real", ev.real), ("C-shuf", EV.shuffled_outputs(test, a.seed)),
                           ("C-S", [{"sid": i, "msgs": [t]} for i, t in enumerate(ev.partner_texts[:len(test)])])):
            m = save(name + ("_quick" if a.quick else ""), outs, ev.evaluate(outs))
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
        return

    extra = {"system": a.system, "seed": a.seed, "model": a.model, "quick": a.quick}
    if a.system in ("base", "sft"):
        model, tok = load_system(a.model, a.run if a.system == "sft" else None, a.dpo, a.quick)
        outs = generate_replies(model, tok, test, n=1, seed=a.seed)
        outs = [{"sid": o["sid"], "msgs": o["cands"][0]} for o in outs]
        name = a.name or (f"{'base' if a.system == 'base' else a.run}{'+' + a.dpo if a.dpo else ''}_s{a.seed}")
        m = ev.evaluate(outs, model=model, tok=tok)
        extra.update({"run": a.run, "dpo": a.dpo})
    else:
        from . import baselines as B
        model, tok = load_system(a.model, None, None, a.quick)
        bge = ev.bge if a.kind == "retrieval" else None
        if a.kind == "retrieval" and bge is None:
            from .embed import BGE
            bge = BGE()
        outs, info = B.run_baseline(model, tok, test, a.kind, train_samples=train_all, n_shots=a.shots, bge=bge, seed=a.seed)
        name = a.name or f"B-{a.kind}{('-' + str(a.shots)) if a.shots else ''}_s{a.seed}"
        m = ev.evaluate(outs, model=None, tok=None)
        extra.update({"kind": a.kind, "shots": a.shots, **info})
    if a.mmi:
        del model
        torch.cuda.empty_cache()
        base, tok = load_system(a.model, None, None, a.quick)
        m["mmi"] = EV.mmi(base, tok, test, outs)
    name = name + ("_quick" if a.quick else "")
    m = save(name, outs, m, extra)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k not in ("history",)})
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
