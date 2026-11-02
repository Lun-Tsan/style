"""persona.report -- paper table, paired bootstrap over test contexts, acceptance lines (PIPELINE_SPEC §6.1, §6.3).

    python -m persona.report --table [reports/eval_table.json]      # markdown, one row per system, seeds pooled
    python -m persona.report --compare eval_a eval_b [--n-boot 2000] [--seed 0]    # paired bootstrap a - b

paired_bootstrap(): any metric of aligned outputs, resampling test contexts (2,000 resamples, §6.1).
d_auc_bootstrap(): D-AUC_1 / D-AUC_k from out-of-fold discriminator scores computed once per system (fast path).
acceptance(): lines 1-2 from the metrics (+ the D-AUC bootstrap against the best prompting system);
lines_from_human(): line 3 from human_eval.score(). Output: numbers and system names, no message text.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from . import config as C
from . import evaluate as EV
from .flag_symbols import is_symbol_turn
from .rulers.stats import auc, pct_ci
from .text import reply_text

MAIN = (("NLL", "nll"), ("D-AUC1", "d_auc1"), (f"D-AUC{C.K_ATTR}", f"d_auc{C.K_ATTR}"), ("NCD-Δ", "ncd_delta"),
        ("Band", "band_inside_mean"), ("JSD", "feature_jsd"), ("OnTopic", "ontopic"), ("MMI", "mmi"),
        ("Copy", "copy_rate"), ("Leak", "leak_per_100"), ("distinct-2", "div_distinct2"), ("Hill-1", "div_hill1"),
        ("symbol", "symbol_share"))
CONTROLS = ("C-real", "C-shuf", "C-S")      # then C-base: the untrained model (untrained_<model>; earlier rows: base)
_SEED = re.compile(r"_s\d+(?=$|[+_])")      # training and generation seed tokens in run_eval row names


def _num(v):
    """float, or None for a missing, NaN or non-numeric value."""
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ----------------------------------------------------------------------------- table
def system_name(row: str) -> str:
    """eval_table.json row name without its seed tokens, so the seeds of one system share a row."""
    return _SEED.sub("", row)


def _rank(name: str) -> int:
    base = name.removesuffix("_quick")
    if base in CONTROLS:
        return CONTROLS.index(base)
    return len(CONTROLS) + (0 if base == "base" or base.startswith("untrained") else 1)


def table(eval_table_path=None) -> str:
    """Markdown table of reports/eval_table.json: one row per system name with the seed suffix stripped, mean ± sd
    (sample sd, ddof 1) over its seeds for the main metrics of §6.1. Controls lead (C-real, C-shuf, C-S, then C-base);
    the other systems keep the order in which they entered the file. "—": the metric is missing or NaN."""
    path = Path(eval_table_path or C.REPORTS / "eval_table.json")
    groups = {}
    for row, m in json.loads(path.read_text(encoding="utf-8")).items():
        groups.setdefault(system_name(row), []).append(m)
    out = ["| system | seeds | " + " | ".join(h for h, _ in MAIN) + " |", "|" + "---|" * (len(MAIN) + 2)]
    for g in sorted(groups, key=_rank):                      # sorted() is stable: file order within a rank
        cells = []
        for _, k in MAIN:
            v = [x for x in (_num(m.get(k)) for m in groups[g]) if x is not None]
            cells.append("—" if not v else f"{v[0]:.3f}" if len(v) == 1 else
                         f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
        out.append(f"| {g} | {len(groups[g])} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- paired bootstrap
def paired_bootstrap(metric_fn, outputs_a, outputs_b, n_boot: int = C.EVAL["boot_n"], seed: int = 0):
    """Paired bootstrap over test contexts (§6.1): outputs_a[i] and outputs_b[i] answer the same test context i.
    Each resample draws len(outputs_a) context indices with replacement and recomputes metric_fn(outputs) for both
    systems on the same indices. Returns diff (a - b on the full lists), diff_mean, ci (2.5 / 97.5 percentiles) and
    p_gt0 (share of resamples with a - b > 0) over the n_valid resamples where both values are numbers, n_boot, n."""
    a, b = list(outputs_a), list(outputs_b)
    if len(a) != len(b) or not a:
        raise ValueError(f"paired bootstrap needs two aligned, non-empty lists (got {len(a)} and {len(b)})")
    if all(isinstance(o, dict) and "sid" in o for o in a + b) and [o["sid"] for o in a] != [o["sid"] for o in b]:
        raise ValueError("outputs_a and outputs_b are not aligned by sid")

    def val(outs):
        v = metric_fn(outs)
        return math.nan if v is None else float(v)

    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        d[r] = val([a[i] for i in idx]) - val([b[i] for i in idx])
    ok = d[~np.isnan(d)]
    return {"diff": val(a) - val(b), "diff_mean": float(ok.mean()) if len(ok) else math.nan, "ci": pct_ci(ok),
            "p_gt0": float((ok > 0).mean()) if len(ok) else math.nan, "n_boot": n_boot, "n_valid": int(len(ok)),
            "n": len(a)}


def _bag_auc(s_gen, s_real, k, nbags, rng):
    """D-AUC_k exactly as evaluate.discriminator draws it (nbags k-bags without replacement per class)."""
    kk = min(k, len(s_real), len(s_gen))
    bg = [s_gen[rng.choice(len(s_gen), kk, replace=False)].sum() for _ in range(nbags)]
    br = [s_real[rng.choice(len(s_real), kk, replace=False)].sum() for _ in range(nbags)]
    return auc(np.array(bg + br), np.array([1] * nbags + [0] * nbags))


def _bag_sums(pool, kk, nbags, rng):
    """nbags sums of kk distinct entries of pool (random keys + argpartition = draws without replacement)."""
    sel = np.argpartition(rng.random((nbags, len(pool))), kk - 1, axis=1)[:, :kk]
    return pool[sel].sum(1)


def _auc2(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def d_auc_bootstrap(real_texts, gen_a_texts, gen_b_texts, k: int = C.K_ATTR, n_boot: int = C.EVAL["boot_n"],
                    seed: int = 0, folds: int = 5, nbags: int = 150):
    """Fast paired bootstrap of D-AUC_1 and D-AUC_k, system a - system b (lower = harder to tell from T).
    The three lists are aligned with the test contexts, in the ruler view (evaluate.texts_of). Symbol-like generated
    replies (contract d) and empty texts leave the populations, as in evaluate.Evaluator.evaluate.
    evaluate.discriminator_scores runs once per system; each resample of test contexts recomputes the AUCs from
    these fixed out-of-fold scores. This ignores refit variability: the discriminators are not retrained per resample,
    so the interval covers the sampling of test contexts given the fitted discriminators, not the variance of the
    fitting. With the same texts and seed the point estimates equal evaluate.discriminator's (seed 0 = eval_table)."""
    n = len(real_texts)
    if not len(gen_a_texts) == len(gen_b_texts) == n or not n:
        raise ValueError("real_texts, gen_a_texts and gen_b_texts must be aligned with the same, non-empty contexts")
    real = [t or "" for t in real_texts]
    kb = f"auc{k}"
    per = []
    for gen_texts in (gen_a_texts, gen_b_texts):
        gen = ["" if not t or is_symbol_turn(t) else t for t in gen_texts]
        s = {"n_gen": sum(map(bool, gen)), "n_symbol": sum(bool(t) and is_symbol_turn(t) for t in gen_texts),
             "auc1": math.nan, kb: math.nan}
        s["ok"] = min(sum(map(bool, real)), s["n_gen"]) >= 2 * folds          # evaluate.discriminator's minimum
        if s["ok"]:
            s_real, s_gen = EV.discriminator_scores(real, gen, folds, seed)
            s["auc1"] = auc(np.concatenate([s_gen, s_real]), np.array([1] * len(s_gen) + [0] * len(s_real)))
            s[kb] = _bag_auc(s_gen, s_real, k, nbags, np.random.default_rng(seed))
            r, g = np.full(n, np.nan), np.full(n, np.nan)       # out-of-fold score of each context, NaN if left out
            r[[i for i, t in enumerate(real) if t]] = s_real
            g[[i for i, t in enumerate(gen) if t]] = s_gen
            s["r"], s["g"] = r, g
            gv, rv = g[~np.isnan(g)], r[~np.isnan(r)]
            s["M"] = (gv[:, None] > rv[None, :]) + 0.5 * (gv[:, None] == rv[None, :])    # pairwise wins, ties 1/2
        per.append(s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))                     # the same resampled contexts for both systems
    W = np.zeros((n_boot, n))
    np.add.at(W, (np.arange(n_boot)[:, None], idx), 1.0)      # multiplicity of each context in each resample
    boot = {key: np.full((2, n_boot), np.nan) for key in ("auc1", kb)}
    for j, s in enumerate(per):
        if not s["ok"]:
            continue
        wg, wr = W[:, ~np.isnan(s["g"])], W[:, ~np.isnan(s["r"])]
        den = wg.sum(1) * wr.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot["auc1"][j] = np.where(den > 0, ((wg @ s["M"]) * wr).sum(1) / den, np.nan)
        for b in range(n_boot):
            pg, pr = s["g"][idx[b]], s["r"][idx[b]]
            pg, pr = pg[~np.isnan(pg)], pr[~np.isnan(pr)]
            kk = min(k, len(pg), len(pr))
            if kk:
                boot[kb][j, b] = _auc2(_bag_sums(pg, kk, nbags, rng), _bag_sums(pr, kk, nbags, rng))

    def summary(key):
        d = boot[key][0] - boot[key][1]
        d = d[~np.isnan(d)]
        va, vb = per[0][key], per[1][key]
        return {"a": va, "b": vb, "diff": va - vb, "diff_mean": float(d.mean()) if len(d) else math.nan,
                "ci": pct_ci(d), "p_gt0": float((d > 0).mean()) if len(d) else math.nan, "n_valid": int(len(d))}

    return {"n": n, "n_real": sum(map(bool, real)), "n_gen_a": per[0]["n_gen"], "n_gen_b": per[1]["n_gen"],
            "symbol_a": per[0]["n_symbol"], "symbol_b": per[1]["n_symbol"], "k": k, "folds": folds, "nbags": nbags,
            "seed": seed, "n_boot": n_boot, "auc1": summary("auc1"), kb: summary(kb),
            "refit": "not resampled: out-of-fold scores fixed per system"}


def load_outputs(name):
    """(system name, rows) of runs/eval_<name>/outputs.jsonl; name may carry the eval_ prefix. The rows hold message
    text: callers keep them in memory or under runs/."""
    label = name.removeprefix("eval_")
    with open(C.RUNS / f"eval_{label}" / "outputs.jsonl", encoding="utf-8") as f:
        return label, [json.loads(line) for line in f if line.strip()]


def compare(name_a, name_b, n_boot: int = C.EVAL["boot_n"], seed: int = 0):
    """Paired bootstrap a - b over the test contexts both runs answered (C-S answers partner turns, not test contexts):
    D-AUC_1 / D-AUC_k by the fast path, and symbol_share, sticker_only_share and copy_rate as evaluate defines them.
    seed sets folds, bags and resamples; seed 0 reproduces the D-AUC point estimates of eval_table.json."""
    from .data import load_all, train_dev_test
    from .text import CopyChecker
    _, _, turns, segs = load_all()
    test = train_dev_test(segs)[2]
    (la, ra), (lb, rb) = load_outputs(name_a), load_outputs(name_b)
    A, B = {o["sid"]: o for o in ra}, {o["sid"]: o for o in rb}
    keep = [s for s in test if s.sid in A and s.sid in B]
    oa, ob = [A[s.sid] for s in keep], [B[s.sid] for s in keep]
    tr = [t for t in turns if t.split == "train" and t.ruler_text and not t.symbol]
    checker = CopyChecker([t.ruler_text for t in tr if t.who == C.TARGET])

    def flags(o):
        t = reply_text(o["msgs"])
        sym = is_symbol_turn(t)
        return {"sid": o["sid"], "sym": sym, "sticker": not t, "copy": checker.is_copy(t) if t and not sym else None}

    def mean(v):
        return float(np.mean(v)) if len(v) else math.nan

    shares = {"symbol_share": lambda fs: mean([f["sym"] for f in fs]),
              "sticker_only_share": lambda fs: mean([f["sticker"] for f in fs]),
              "copy_rate": lambda fs: mean([f["copy"] for f in fs if f["copy"] is not None])}
    fa, fb = [flags(o) for o in oa], [flags(o) for o in ob]
    return {"a": la, "b": lb, "n_contexts": len(keep), "n_a": len(ra), "n_b": len(rb), "seed": seed,
            "d_auc": d_auc_bootstrap(EV.texts_of(EV.real_outputs(keep)), EV.texts_of(oa), EV.texts_of(ob),
                                     n_boot=n_boot, seed=seed),
            **{name: paired_bootstrap(fn, fa, fb, n_boot, seed) for name, fn in shares.items()}}


# ----------------------------------------------------------------------------- acceptance lines (§6.3)
def _all(parts):
    """Three-valued AND: False if a part is False, None if a part is None (not decidable), else True."""
    parts = list(parts)
    if any(p is False for p in parts):
        return False
    return None if any(p is None for p in parts) else True


def _part(value, threshold, op):
    v, t = _num(value), _num(threshold)
    ok = None if v is None or t is None else bool(v <= t if op == "<=" else v >= t)
    return {"value": v, "threshold": t, "op": op, "pass": ok}


def _div_ratio(m, r, k):
    """System / C-real for diversity metric k at the same n (§6.1): evaluate's <k>_ratio (C-real rarefied to the
    system's div_n); for a row without it, the two values when both rows were rarefied to the same div_n; else None."""
    v = _num(m.get(f"{k}_ratio"))
    if v is None and _num(m.get("div_n")) is not None and _num(m.get("div_n")) == _num(r.get("div_n")):
        a, b = _num(m.get(k)), _num(r.get(k))
        v = a / b if a is not None and b else None
    return v


def acceptance(metrics, c_real_metrics, boot_vs_best_prompting=None):
    """Lines 1-2 of §6.3 (thresholds C.LINES) for one system's metrics (an eval_table.json row, or means over its
    seeds) against C-real on the same test contexts. A part is True / False / None (not decidable: metric missing or
    NaN, Leak without OpenCC, no bootstrap, diversity of two rows at different n); a line is False if a part is False,
    None if a part is None, else True.
    line 1: D-AUC_1 <= 0.75; D-AUC_10 <= 0.90; D-AUC_1 significantly below the best prompting system's, i.e. the
            95% CI of D-AUC_1(system) - D-AUC_1(prompting) lies below 0. boot_vs_best_prompting: the result of
            d_auc_bootstrap(real, system, best prompting) or of compare(system, best prompting).
    line 2: Band >= C-real Band - 1; distinct-2 and Hill-1 >= 0.80 x C-real's at the same n (_div_ratio; a system
            that keeps fewer replies than C.EVAL["rarefy_n"] is compared at its own div_n); Copy <= 0.02;
            Leak <= C-real's.
    symbol_share goes into details without a threshold (§6.3 registers no threshold for it)."""
    L, m, r = C.LINES, metrics, c_real_metrics
    boot = None
    if boot_vs_best_prompting is not None:
        boot = boot_vs_best_prompting.get("d_auc", boot_vs_best_prompting)
        boot = boot.get("auc1", boot)
    ci = (boot or {}).get("ci")
    sig = {"value": ci, "threshold": 0.0, "op": "ci <", "pass": None}
    if ci and all(_num(v) is not None for v in ci):
        sig["pass"] = bool(ci[1] < 0)
    real_band = _num(r.get("band_inside_mean"))
    band_min = None if real_band is None else real_band - L["band_margin"]
    line1 = {"d_auc1": _part(m.get("d_auc1"), L["d_auc1"], "<="),
             "d_auc10": _part(m.get("d_auc10"), L["d_auc10"], "<="),
             "below_best_prompting": sig}
    line2 = {"band": _part(m.get("band_inside_mean"), band_min, ">="),
             **{f"{k}_ratio": _part(_div_ratio(m, r, k), L["diversity_ratio"], ">=")
                for k in ("div_distinct2", "div_hill1")},
             "copy": _part(m.get("copy_rate"), L["copy_max"], "<="),
             "leak": _part(m.get("leak_per_100"), r.get("leak_per_100"), "<=")}
    return {"line1": _all(p["pass"] for p in line1.values()), "line2": _all(p["pass"] for p in line2.values()),
            "details": {"line1": line1, "line2": line2, "symbol_share": _num(m.get("symbol_share"))}}


def lines_from_human(score_dict):
    """Line 3 of §6.3 from human_eval.score(): kind A passes for a system when the judge's accuracy at its stopping
    point is <= C.LINES["human_A"]; kind B when the preference for it over the baseline (the best prompting system) at
    the stopping point is >= C.LINES["human_B"]; None while its sequence continues. At an SPRT bound the verdict and
    this point-estimate rule agree for the registered C.HUMAN parameters (checked in tests/test_human.py); at max_n the
    point estimate decides (§6.3). §6.2 stops only at a bound or at max_n, so a sequence whose items ran out before
    either ("exhausted") stays None with more_items_needed True. score_dict: one score() result or {"A": ..., "B": ...};
    line3 per system = kind A and kind B (False if one fails, None while one is missing, running or short of items)."""
    scores = {score_dict["kind"]: score_dict} if "per_system" in score_dict else dict(score_dict)
    parts = {}
    for kind, sc in scores.items():
        thr = C.LINES[f"human_{kind}"]
        for name, r in sc["per_system"].items():
            short = r["reason"] == "exhausted" and r["n_stop"] < C.HUMAN["max_n"]
            ok = None if not r["stop"] or short else bool(r["acc_stop"] <= thr if kind == "A" else r["acc_stop"] >= thr)
            parts.setdefault(name, {})[kind] = {"pass": ok, "acc_stop": r["acc_stop"], "ci_stop": r["ci_stop"],
                                                "n_stop": r["n_stop"], "verdict": r["verdict"], "reason": r["reason"],
                                                "more_items_needed": short}
    return {"line3": {name: _all(p.get(kd, {}).get("pass") for kd in ("A", "B")) for name, p in parts.items()},
            "parts": parts}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m persona.report")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--table", nargs="?", const=str(C.REPORTS / "eval_table.json"), metavar="EVAL_TABLE",
                   help="markdown table of eval_table.json (default: reports/eval_table.json)")
    g.add_argument("--compare", nargs=2, metavar=("EVAL_A", "EVAL_B"), help="paired bootstrap a - b (numbers only)")
    ap.add_argument("--n-boot", type=int, default=C.EVAL["boot_n"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.table:
        print(table(a.table))
    else:
        print(json.dumps(compare(*a.compare, n_boot=a.n_boot, seed=a.seed), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
