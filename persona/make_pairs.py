"""Stage 2b: sample candidates from an SFT model for the training contexts and build DPO pairs (PIPELINE_SPEC §5.2-5.3).

    python -m persona.make_pairs --sft sft_instruct_all_s0      # -> runs/pairs_sft_instruct_all_s0_s0
    python -m persona.make_pairs --quick        # 0.8B quick adapter, 12 contexts, no bge-m3 -> runs/pairs_quick

Outputs runs/<out>/candidates.jsonl (sid, cands, finished), pairs.jsonl, stats.json. An existing non-empty
runs/<out> is refused unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config as C
from . import samples as SM, split as SP
from .data import load_all
from .pairs import build_pairs, fit_scorer, pairs_feature_jsd
from .text import CopyChecker


def default_out(sft_dir, model="instruct", seed=0) -> str:
    """Run name carrying the source SFT run and the seed, so pairs of different SFT runs do not overwrite each other."""
    return f"pairs_{Path(sft_dir).name}_s{seed}" if sft_dir else f"pairs_untrained_{model}_s{seed}"


def run(sft_dir, out, n_cand=C.DPO["n_cand"], seed=0, quick=False, model="instruct", use_bge=True, log=print,
        overwrite=False):
    out = C.RUNS / out
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists and is not empty; pass --overwrite to write into it")
    # torch-side modules are imported here, so the module (and the check above) loads on a machine without torch
    from . import modeling as MD
    from .generate import generate_replies
    out.mkdir(parents=True, exist_ok=True)
    msgs, n_sys, turns, segs = load_all()
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    train, dev = SP.dev_cut(train_all)
    if quick:
        train = train[:12]
    tr_turns = [t for t in turns if t.split == "train" and t.ruler_text and not t.symbol]
    # ruler: NB on the training population + isotonic calibration and raw-LLR cap from the stage-1 OOF scores
    val = json.load(open(C.REPORTS / "ruler_validation.json", encoding="utf-8"))
    scorer = fit_scorer(tr_turns, val["oof_single_nb_margin"], val["oof_y_is_S"])
    checker = CopyChecker([t.ruler_text for t in tr_turns if t.who == C.TARGET])
    fs = scorer.fit_stats
    log(f"scorer cap: NB target-LLR {fs['cap_raw']:.2f} ({fs['cap_q']:.0%} quantile of real T OOF LLRs, "
        f"{fs['cap_T_oof_above']:.1%} above it) = calibrated log-odds {fs['cap']:.3f}")
    name = "Qwen/Qwen3.5-0.8B" if quick else MD.model_name(model)
    tok = MD.load_tokenizer(name)
    t0 = time.time()
    base = MD.load_model(name)
    mdl = MD.load_adapter(base, str(C.RUNS / sft_dir / "adapter_best")) if sft_dir else base
    gen = generate_replies(mdl, tok, train, n=n_cand, seed=seed)
    log(f"generated {n_cand} candidates for {len(train)} contexts in {time.time() - t0:.0f}s; vram {MD.vram_gb()} GB")
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for g in gen:
            f.write(json.dumps({**g, "finished": g.get("finished")}, ensure_ascii=False) + "\n")
    del mdl, base
    import torch
    torch.cuda.empty_cache()
    bge = thr = None
    if use_bge and not quick:
        from .embed import BGE, fit_ontopic_threshold
        bge = BGE()
        thr = fit_ontopic_threshold(bge, train_all)      # training period incl. dev (contract b), test stays out
        log(f"on-topic threshold (bge-m3 cosine, null {C.DPO['ontopic_q']:.0%} quantile): {thr:.3f}")
    pairs, stats = build_pairs(train, [g["cands"] for g in gen], scorer, checker, bge, thr, seed=seed,
                               finished=[g.get("finished") for g in gen])
    stats["ontopic_threshold"] = thr
    by_sid = {s.sid: s for s in train}
    stats["chosen_feature_jsd_vs_real"] = pairs_feature_jsd(pairs, by_sid)                  # non-DITTO pairs (#10)
    stats["chosen_feature_jsd_vs_real_all"] = pairs_feature_jsd(pairs, by_sid, include_ditto=True)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"pairs {stats['n_pairs']} (ditto {stats['n_ditto']}), cell match {stats.get('contexts_with_cell_match', 0)}/{stats['contexts']}, "
        f"candidates unfinished {stats['cand_unfinished']} / symbol-like {stats['cand_symbol']} / capped {stats['cand_capped']}, "
        f"mean score chosen (non-DITTO) {stats['mean_score_chosen']} rejected {stats['mean_score_rejected']}, "
        f"chosen-vs-real feature JSD (non-DITTO) {stats['chosen_feature_jsd_vs_real']} -> {out}")
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
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.sft = a.sft or "sft_quick_all_s0"
        a.out = a.out or "pairs_quick"                   # the name train_dpo --quick reads
    run(a.sft, a.out or default_out(a.sft, a.model, a.seed), a.n, a.seed, a.quick, a.model, not a.no_bge,
        overwrite=a.overwrite)


if __name__ == "__main__":
    main()
