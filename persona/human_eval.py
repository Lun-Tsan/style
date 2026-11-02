"""persona.human_eval -- blind items for the target person and a sequential stopping rule (PIPELINE_SPEC §6.2).

make_items(): writes items.jsonl (what the judge sees: de-identified context and replies, the two replies in random
order), key.json (which side is real / which system) and stats.json (counts only) to a local folder. Each test
context goes to one system (kinds A and B), so a context and its replies give at most one item (CODE_REVIEW #6).
sprt_stop(): Wald SPRT on the judge's hit rate; score() runs it for each system separately (CODE_REVIEW #5, #6).

    python -m persona.human_eval make --kind A --systems eval_x,eval_y --out runs/human_A --n 100
    python -m persona.human_eval make --kind B --systems eval_x,eval_y --out runs/human_B    # y = baseline
    python -m persona.human_eval score --dir runs/human_A --answers answers.json

answers.json: {"<item>": "left" | "right"}, the side the judge picked. The folder has to lie under runs/ (items hold
message text); the printed scores carry numbers and system names only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from . import config as C
from .flag_symbols import is_symbol_turn
from .normalize import PH_LINE, deid
from .prompt import render_parts
from .rulers.stats import clopper_pearson
from .text import reply_text

QUESTION = {"A": "哪一個是你寫的?", "B": "哪一個比較像你?"}
HIDDEN = "[略]"         # kind A: stands in for a context turn that is the real reply of another item


def _display_context(sample, hide=frozenset()):
    """The context the judge reads, without the trailing role tag. A turn whose id() is in `hide` (a T turn here)
    shows as the one line '我: [略]', so its text and its message count stay hidden."""
    ctx = [replace(t, msgs=[HIDDEN]) if id(t) in hide else t for t in sample.ctx]
    prompt, _ = render_parts(ctx, sample.target)          # render_parts() de-identifies the context
    return prompt[:-len(f"{C.ROLE[C.TARGET]}: ")].rstrip()


def _unusable(msgs):
    """Why a reply cannot be shown (None if it can): missing, empty, or symbol-like (contract d; real targets are not
    symbol-like, so such a reply would give the answer away, as it would to the discriminator)."""
    if msgs is None:
        return "missing"
    if not any(m.strip() for m in msgs):
        return "empty"
    return "symbol" if is_symbol_turn(reply_text(msgs)) else None


def make_items(samples, outputs_by_system: dict, out_dir, kind: str = "A", seed: int = 0, n: int = 100,
               priority_sids=None, baseline: str | None = None):
    """kind A: real vs system ("which one is mine?"); kind B: each system vs `baseline` ("which is more like me?").
    outputs_by_system: {name: [{sid, msgs}]}; n caps the total number of items.

    Each context goes to one system by seeded permuted blocks along the selection order (each block of len(systems)
    contexts holds one context per system), so a context, its real reply and (kind B) the baseline's reply give at most
    one item. Selection: priority_sids (e.g. contexts where the rulers disagree, §6.2) lead the selection order, the
    other contexts follow in seeded random order until n items exist; priority only decides which contexts are chosen.
    Presentation: time order. Kind A: the real reply of a chosen context also sits in the history of later contexts of
    its conversation segment, where it would answer that item to a judge who reads ahead or back in items.jsonl, so
    such a turn is shown as '我: [略]'; stats.json counts the hidden turns and the real replies still visible in an
    earlier / later item's context (0 by construction).
    Each reply shown is de-identified message by message (normalize.deid; the context via prompt.render_parts). A
    missing, empty or symbol-like system reply gives no item (counted per system in stats.json, which also reports
    symbol_share over all replies of each system, as evaluate does). Returns the number of items."""
    if kind not in QUESTION:
        raise ValueError(f"kind must be A or B, got {kind!r}")
    systems = [k for k in outputs_by_system if k != baseline]
    if not systems or (kind == "B" and baseline not in outputs_by_system):
        raise ValueError("kind A needs at least one system; kind B also needs a baseline among outputs_by_system")
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sid = {s.sid: s for s in samples}
    pos = {s.sid: i for i, s in enumerate(samples)}
    real = {s.sid: [PH_LINE.get(m.strip(), m) for m in s.target.msgs] for s in samples}
    outs = {name: {o["sid"]: o["msgs"] for o in rows} for name, rows in outputs_by_system.items()}
    prio = list(dict.fromkeys(sid for sid in (priority_sids or []) if sid in by_sid))
    prio_set = set(prio)
    rest = [s.sid for s in samples if s.sid not in prio_set]
    rng.shuffle(rest)
    walk = prio + rest
    # permuted blocks: each context goes to one system; kind B too, because one context in two items repeats the
    # baseline's reply, and the repeated reply marks the baseline side
    pairs = []
    for b in range(0, len(walk), len(systems)):
        perm = systems[:]
        rng.shuffle(perm)
        pairs += list(zip(walk[b:b + len(systems)], perm))
    chosen, skipped = [], {name: Counter() for name in outputs_by_system}
    for sid, name in pairs:
        if len(chosen) >= n:
            break
        why = _unusable(outs[name].get(sid))
        if why is None and kind == "B":
            why_b = _unusable(outs[baseline].get(sid))
            if why_b:
                skipped[baseline][why_b] += 1
                continue
        if why:
            skipped[name][why] += 1
            continue
        chosen.append((sid, name))
    n_prio = sum(sid in prio_set for sid, _ in chosen)
    chosen.sort(key=lambda p: pos[p[0]])                        # time order (one item per context)
    targets = {id(by_sid[sid].target): j for j, (sid, _) in enumerate(chosen)}
    hide = set(targets) if kind == "A" else set()               # kind A: shown, these turns answer other items

    items, key, hits = [], [], Counter()

    def shown(msgs):
        out = []
        for m in msgs:
            t, h = deid(PH_LINE.get(m.strip(), m))
            hits.update(h)
            out.append(t)
        return out

    for sid, name in chosen:
        if kind == "A":
            a, b, tag_a, tag_b = real[sid], outs[name][sid], "real", name
        else:
            a, b, tag_a, tag_b = outs[name][sid], outs[baseline][sid], name, baseline
        if rng.random() < 0.5:
            a, b, tag_a, tag_b = b, a, tag_b, tag_a
        iid = len(items)
        items.append({"item": iid, "context": _display_context(by_sid[sid], hide), "left": shown(a), "right": shown(b),
                      "question": QUESTION[kind]})
        key.append({"item": iid, "sid": sid, "left": tag_a, "right": tag_b, "system": name})
    stats = {"kind": kind, "seed": seed, "n_items": len(items), "n_contexts": len(samples), "n_priority": n_prio,
             "items_per_system": dict(Counter(k["system"] for k in key)),
             "skipped": {name: dict(c) for name, c in skipped.items()},          # among the contexts visited
             "symbol_share": {name: sum(is_symbol_turn(reply_text(o["msgs"])) for o in rows) / max(1, len(rows))
                              for name, rows in outputs_by_system.items()},
             "deid_hits": dict(hits)}
    if kind == "A":         # (j, i): the real reply of item j sits in the context of item i (same segment)
        inside = [(targets[id(t)], i, id(t) in hide) for i, (sid, _) in enumerate(chosen) for t in by_sid[sid].ctx
                  if id(t) in targets]
        stats["real_hidden_in_context"] = sum(h for _, _, h in inside)
        stats["real_in_earlier_context"] = len({j for j, i, h in inside if i < j and not h})
        stats["real_in_later_context"] = len({j for j, i, h in inside if i > j and not h})
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(items)


def _llr(n_correct: int, n: int, h: dict) -> float:
    return (n_correct * math.log(h["p_high"] / h["p_low"])
            + (n - n_correct) * math.log((1 - h["p_high"]) / (1 - h["p_low"])))


def sprt_stop(n_correct: int, n: int, kind: str = "A"):
    """Wald SPRT (Wald 1945) after n items with n_correct hits; parameters C.HUMAN[kind]: simple hypotheses
    p = p_low vs p = p_high (the indifference zone around p_line), log LR = n_correct log(p_high / p_low)
    + (n - n_correct) log((1 - p_high) / (1 - p_low)). Returns (stop, verdict): log LR >= log((1 - beta) / alpha)
    -> (True, "above"); log LR <= log(beta / (1 - alpha)) -> (True, "below"); otherwise (False, "undecided"), i.e.
    continue, until n reaches C.HUMAN["max_n"], where it stops as (True, "undecided") and score() reports the point
    estimate with its Clopper-Pearson interval. Wald's bounds put the error rates near alpha ("above" at p_low) and
    beta ("below" at p_high); the truncation at max_n moves some sequences to "undecided" (simulated in
    tests/test_human.py). For kind A, "below" favours p_low (the judge spots the real reply less often), the side
    line 3 accepts."""
    h = C.HUMAN[kind]
    llr = _llr(n_correct, n, h)
    if llr >= math.log((1 - h["beta"]) / h["alpha"]):
        return True, "above"
    if llr <= math.log(h["beta"] / (1 - h["alpha"])):
        return True, "below"
    return n >= C.HUMAN["max_n"], "undecided"


def ci_stop(n_correct: int, n: int, p0: float = 0.65, max_n: int = 100):
    """The stopping rule of the 2026-10-13 version (kept for the record; not used): stop when the exact 95%
    Clopper-Pearson interval lies above or below p0, or at max_n. Not used because checking a 95% interval after each
    item raises the error rate above nominal: at a true accuracy of 0.65 it stops at "below" in 12.3% of simulated
    sequences against 2.5% nominal (CODE_REVIEW_2026-10-17 #5; 11.9% in a re-simulation of 20,000 sequences on
    2026-11-02)."""
    if n == 0:
        return False, "undecided"
    lo, hi = clopper_pearson(n_correct, n)
    if hi < p0:
        return True, "below"
    if lo > p0:
        return True, "above"
    return n >= max_n, "undecided"


def _sequence(hits, kind: str, n_items: int):
    """One system's SPRT over its answered items in item order; answers after the stopping point stay in n only."""
    k = 0
    stop, verdict, reason, n_stop, k_stop = False, "undecided", None, None, None
    for i, hit in enumerate(hits, 1):
        k += hit
        stop, verdict = sprt_stop(k, i, kind)
        if stop:
            reason, n_stop, k_stop = ("max_n" if verdict == "undecided" else "bound"), i, k
            break
    if not stop and hits and len(hits) >= n_items:          # this system has no items left
        stop, reason, n_stop, k_stop = True, "exhausted", len(hits), k
    nc = sum(hits)
    return {"n_items": n_items, "n": len(hits), "n_correct": nc, "acc": nc / len(hits) if hits else None,
            "ci": list(clopper_pearson(nc, len(hits))) if hits else None,
            "stop": stop, "verdict": verdict, "reason": reason, "n_stop": n_stop,
            "acc_stop": k_stop / n_stop if stop else None,
            "ci_stop": list(clopper_pearson(k_stop, n_stop)) if stop else None,
            "llr": _llr(k_stop, n_stop, C.HUMAN[kind]) if stop else _llr(k, len(hits), C.HUMAN[kind])}


def score(answers, key, kind: str = "A"):
    """answers: {item: 'left' | 'right'}; key: the key.json list of make_items. A hit is picking the real reply
    (kind A) or the system over the baseline (kind B). per_system: each system's own SPRT (sprt_stop) over its items
    in item order, stopping at a bound, at C.HUMAN["max_n"] items, or when its items run out ("exhausted");
    acc_stop / ci_stop are the point estimate and Clopper-Pearson interval at the stop. The pooled numbers over all
    systems carry no verdict: pooling tests no single system (CODE_REVIEW #6)."""
    by = {int(k["item"]): k for k in key}
    n_items = Counter(k["system"] for k in key)
    seq = {name: [] for name in n_items}
    for iid, side in sorted((int(i), s) for i, s in answers.items()):
        if side not in ("left", "right"):
            raise ValueError(f"item {iid}: the answer must be 'left' or 'right'")
        if iid not in by:
            raise ValueError(f"item {iid} is not in the key")
        k = by[iid]
        seq[k["system"]].append((k[side] == "real") if kind == "A" else (k[side] == k["system"]))
    n = sum(len(v) for v in seq.values())
    nc = sum(sum(v) for v in seq.values())
    return {"kind": kind, "n": n, "n_correct": nc, "acc": nc / n if n else None,
            "ci": list(clopper_pearson(nc, n)) if n else None,
            "per_system": {name: _sequence(v, kind, n_items[name]) for name, v in seq.items()}}


def private_dir(path) -> Path:
    """A blind-test folder: a relative path that starts with runs/ is taken from the project root, another relative
    path from C.RUNS; the result has to lie inside C.RUNS (git-ignored), because items hold message text."""
    p = Path(path)
    if not p.is_absolute():
        p = C.PROJECT / p if p.parts[:1] == ("runs",) else C.RUNS / p
    p, root = p.resolve(), C.RUNS.resolve()
    if root not in p.parents:
        raise ValueError(f"{p} is not inside {root}; blind-test files stay in the private runs folder")
    return p


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m persona.human_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write items.jsonl, key.json and stats.json into a folder under runs/")
    mk.add_argument("--kind", choices=["A", "B"], default="A")
    mk.add_argument("--systems", required=True, help="comma-separated eval runs (runs/eval_<name>/outputs.jsonl)")
    mk.add_argument("--baseline", default=None,
                    help="kind B: the system the others face (default: the last of --systems)")
    mk.add_argument("--out", required=True)
    mk.add_argument("--n", type=int, default=100)
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--priority", default=None, help="JSON file with a list of test sids to show ahead of the others")
    mk.add_argument("--overwrite", action="store_true")
    sc = sub.add_parser("score", help="per-system SPRT verdicts from the judge's answers (numbers only)")
    sc.add_argument("--dir", required=True)
    sc.add_argument("--answers", required=True)
    sc.add_argument("--kind", choices=["A", "B"], default=None, help="default: A if the key has real replies")
    a = ap.parse_args(argv)
    from .report import lines_from_human, load_outputs
    if a.cmd == "make":
        out = private_dir(a.out)
        if not a.overwrite and out.is_dir() and any(out.iterdir()):
            raise FileExistsError(f"{out} exists and is not empty; pass --overwrite to replace it")
        loaded = [load_outputs(nm.strip()) for nm in a.systems.split(",") if nm.strip()]
        if a.kind == "B" and a.baseline and a.baseline.removeprefix("eval_") not in dict(loaded):
            loaded.append(load_outputs(a.baseline))
        baseline = (a.baseline.removeprefix("eval_") if a.baseline else loaded[-1][0]) if a.kind == "B" else None
        from .data import load_all, train_dev_test
        test = train_dev_test(load_all()[3])[2]
        prio = json.loads(Path(a.priority).read_text(encoding="utf-8")) if a.priority else None
        make_items(test, dict(loaded), out, a.kind, a.seed, a.n, prio, baseline)
        print(f"{out}:", (out / "stats.json").read_text(encoding="utf-8"))
        return
    d = private_dir(a.dir)
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    answers = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    kind = a.kind or ("A" if any("real" in (k["left"], k["right"]) for k in key) else "B")
    res = score(answers, key, kind)
    for name, r in res["per_system"].items():
        print(f"{name}: answered {r['n']}/{r['n_items']}, hits {r['n_correct']}, stop {r['stop']} ({r['reason']}), "
              f"verdict {r['verdict']}, acc at stop {r['acc_stop']}")
        if r["reason"] == "exhausted":
            print(f"{name}: items ran out after {r['n_stop']} answers, before an SPRT bound or max_n "
                  f"{C.HUMAN['max_n']}; line 3 stays undecided until more items of this system are answered")
    print(json.dumps({"score": res, "line3": lines_from_human(res)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
