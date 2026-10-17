"""persona.human_eval -- blind items for the target person and a sequential stopping rule (PIPELINE_SPEC §6.2).

make_items(): writes items.jsonl (what the judge sees: de-identified context, two replies in random order) and
key.json (which side is real / which system) to a local folder. sprt_stop(): stop when the exact 95% interval of
the judge's accuracy lies entirely above or below p0 (0.65), or at max_n items.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config as C
from .normalize import PH_LINE
from .prompt import render
from .rulers.stats import clopper_pearson


def _display_context(sample):
    prompt, _ = render(sample)
    return prompt[:-len(f"{C.ROLE[C.TARGET]}: ")].rstrip()


def make_items(samples, outputs_by_system: dict, out_dir, kind: str = "A", seed: int = 0, n: int = 100,
               priority_sids=None, baseline: str | None = None):
    """kind A: real vs each system (which one is mine?); kind B: best system vs baseline (which is more like me?).
    outputs_by_system: {name: [{sid, msgs}]}, aligned with samples."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sid = {s.sid: s for s in samples}
    real = {s.sid: [PH_LINE.get(m.strip(), m) for m in s.target.msgs] for s in samples}
    order = list(priority_sids or []) + [s.sid for s in samples if s.sid not in set(priority_sids or [])]
    items, key = [], []
    systems = [k for k in outputs_by_system if k != baseline]
    for sid in order:
        for name in systems:
            o = next((x for x in outputs_by_system[name] if x["sid"] == sid), None)
            if o is None or not o["msgs"]:
                continue
            if kind == "A":
                a, b, tag_a, tag_b = real[sid], o["msgs"], "real", name
            else:
                ob = next((x for x in outputs_by_system[baseline] if x["sid"] == sid), None)
                if ob is None or not ob["msgs"]:
                    continue
                a, b, tag_a, tag_b = o["msgs"], ob["msgs"], name, baseline
            if rng.random() < 0.5:
                a, b, tag_a, tag_b = b, a, tag_b, tag_a
            iid = len(items)
            items.append({"item": iid, "context": _display_context(by_sid[sid]), "left": a, "right": b,
                          "question": "哪一個是你寫的?" if kind == "A" else "哪一個比較像你?"})
            key.append({"item": iid, "sid": sid, "left": tag_a, "right": tag_b, "system": name})
            if len(items) >= n:
                break
        if len(items) >= n:
            break
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(items)


def sprt_stop(n_correct: int, n: int, p0: float = 0.65, max_n: int = 100):
    """(stop, verdict) with verdict in {'below', 'above', 'undecided'}; CI = exact Clopper-Pearson 95%."""
    if n == 0:
        return False, "undecided"
    lo, hi = clopper_pearson(n_correct, n)
    if hi < p0:
        return True, "below"
    if lo > p0:
        return True, "above"
    return n >= max_n, "undecided"


def score(answers, key, kind: str = "A"):
    """answers: {item: 'left'|'right'}; kind A accuracy of spotting the real reply; kind B preference for the system."""
    by = {k["item"]: k for k in key}
    n = correct = 0
    per_system = {}
    for iid, side in answers.items():
        k = by[int(iid)]
        n += 1
        tag = k[side]
        hit = (tag == "real") if kind == "A" else (tag == k["system"])
        correct += hit
        d = per_system.setdefault(k["system"], [0, 0])
        d[0] += hit
        d[1] += 1
    stop, verdict = sprt_stop(correct, n)
    return {"n": n, "acc": correct / n if n else None, "ci": clopper_pearson(correct, n) if n else None,
            "stop": stop, "verdict": verdict, "per_system": {k: v[0] / v[1] for k, v in per_system.items()}}
