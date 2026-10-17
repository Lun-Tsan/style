"""Stage 2b: sample candidates from an SFT model for the training contexts and build DPO pairs (PIPELINE_SPEC §5.2-5.3).

    python -m persona.make_pairs --sft sft_instruct_all_s0 --out pairs_instruct_all_s0
    python -m persona.make_pairs --quick        # 0.8B quick adapter, 12 contexts, no bge-m3

Outputs runs/<out>/candidates.jsonl (sid, cands), pairs.jsonl, stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config as C
from . import modeling as MD
from . import samples as SM, split as SP
from .generate import CopyChecker, generate_replies
from .pairs import build_pairs, fit_scorer, pairs_feature_jsd
from .rulers import validate as V
from .stage1 import load_all


def run(sft_dir, out, n_cand=C.DPO["n_cand"], seed=0, quick=False, model="instruct", use_bge=True, log=print):
    out = C.RUNS / out
    out.mkdir(parents=True, exist_ok=True)
    msgs, n_sys, turns, segs = load_all()
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    train, dev = SP.dev_cut(train_all)
    if quick:
        train = train[:12]
    tr_turns = [t for t in turns if t.split == "train" and t.ruler_text and not t.symbol]
    # ruler: NB on the training population + isotonic calibration from the stage-1 OOF scores
    val = json.load(open(C.REPORTS / "ruler_validation.json", encoding="utf-8"))
    scorer = fit_scorer(tr_turns, val["oof_single_nb_margin"], val["oof_y_is_S"])
    checker = CopyChecker([t.ruler_text for t in tr_turns if t.who == C.TARGET])
    log(f"scorer cap {scorer.cap:.3f} (calibrated log-odds, {C.DPO['cap_q']:.0%} quantile of real training replies)")
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model)
    tok = MD.load_tokenizer(name)
    t0 = time.time()
    base = MD.load_model(name)
    mdl = MD.load_adapter(base, str(C.RUNS / sft_dir / "adapter_best")) if sft_dir else base
    gen = generate_replies(mdl, tok, train, n=n_cand, seed=seed)
    log(f"generated {n_cand} candidates for {len(train)} contexts in {time.time() - t0:.0f}s; vram {MD.vram_gb()} GB")
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for g in gen:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    del mdl, base
    import torch
    torch.cuda.empty_cache()
    bge = thr = None
    if use_bge and not quick:
        from .embed import BGE, null_threshold
        from .generate import reply_text
        bge = BGE()
        thr = null_threshold(bge, [reply_text([m for m in s.target.msgs]) for s in train_all if s.target.text_b])
        log(f"on-topic threshold (bge-m3 cosine, null {C.DPO['ontopic_q']:.0%} quantile): {thr:.3f}")
    pairs, stats = build_pairs(train, [g["cands"] for g in gen], scorer, checker, bge, thr, seed=seed)
    stats["ontopic_threshold"] = thr
    stats["chosen_feature_jsd_vs_real"] = pairs_feature_jsd(pairs, {s.sid: s for s in train}) if pairs else None
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"pairs {stats['n_pairs']} (ditto {stats['n_ditto']}), cell match {stats.get('contexts_with_cell_match', 0)}/{stats['contexts']}, "
        f"mean score chosen {stats['mean_score_chosen']} rejected {stats['mean_score_rejected']}, "
        f"chosen-vs-real feature JSD {stats['chosen_feature_jsd_vs_real']} -> {out}")
    return pairs, stats


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--n", type=int, default=C.DPO["n_cand"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bge", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.sft = a.sft or "sft_quick_all_s0"
        a.out = a.out or "pairs_quick"
    run(a.sft, a.out or f"pairs_{a.model}_s{a.seed}", a.n, a.seed, a.quick, a.model, not a.no_bge)


if __name__ == "__main__":
    main()
