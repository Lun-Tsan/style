"""Stage 1: data layer + ruler validation + feature bands -> reports/ (PIPELINE_SPEC.md §7 step 1-2).

    python -m persona.stage1              # v1 rule (emoticon whitelist), full reports
    python -m persona.stage1 --reproduce  # experiment-4 loose rule on raw text_b; compares with the known numbers
    python -m persona.stage1 --quick      # fewer reps (smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config as C
from . import samples as SM, split as SP, turns as TU
from .data import load_all as load_all_data
from .flag_symbols import flag_raw
from .rulers import features as FE, validate as V
from .rulers.calibrate import Calibrator


load_all = load_all_data      # kept for the drivers that import it from here


def _dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    msgs, n_sys, turns, segs = load_all()
    print(f"messages {len(msgs)} (system lines {n_sys}); turns {len(turns)}; segments {len(segs)}; "
          f"text-turn counts {TU.counts(turns)}")
    sizes = dict(nbag=60, noof=20, nboot=0, e4_reps=200) if a.quick else {}

    if a.reproduce:
        res = V.run(turns, flag=lambda t: flag_raw(t.text_b), text_fn=lambda t: t.text_b, name="reproduce_exp4_loose", **sizes)
        e4 = res["E4"]["random_k50"]
        print("reproduce vs experiment 4 (random k=50 P(cross>max within)):",
              {m: (round(e4[m]["p_cross_gt_max"], 3), C.EXP4_LOOSE_K50[m]) for m in C.EXP4_LOOSE_K50})
        print("E1 NB thr0 (exp4: 0.772):", round(res["E1"]["M5"]["acc_thr0"], 3),
              "| E3 NB k=10 cal (exp4: 1.000):", round(res["E3"]["10"]["M5"]["acc_cal"], 3))
        _dump(res, C.REPORTS / "ruler_validation_reproduce.json")
        return

    # ---- data layer
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    test = [s for s in smp if s.split == "test"]
    train, dev = SP.dev_cut(train_all)
    subsets = SP.nested_subsets(train)
    st = SM.stats(smp, turns, segs)
    st["dev"] = {"n": len(dev), "dates": [dev[0].date, dev[-1].date] if dev else None}
    st["train_minus_dev"] = {"n": len(train)}
    full = SP.month_hist(train)
    st["subsets"] = {str(k): {"n": len(v), "month_jsd_vs_all": SP.jsd_hist(SP.month_hist(v), full)} for k, v in subsets.items()}
    _dump(st, C.REPORTS / "data_stats.json")
    print(f"samples: train {len(train_all)} (train-dev {len(train)} / dev {len(dev)}), test {len(test)}; "
          f"subset month-JSD {[round(v['month_jsd_vs_all'], 3) for v in st['subsets'].values()]}")

    # ---- ruler validation on the v1 rule
    res = V.run(turns, flag=lambda t: t.symbol, text_fn=lambda t: t.ruler_text, name="v1_whitelist", **sizes)
    verdict = V.judge(res)
    _dump(res, C.REPORTS / "ruler_validation.json")
    (C.REPORTS / "ruler_validation.md").write_text(V.summary_md(res, verdict), encoding="utf-8")
    print("verdict:", "PASS" if verdict["pass"] else "FAIL", verdict["checks"])

    # ---- calibration of the single-turn NB LLR (target-likeness = -margin) on OOF training scores
    cal = Calibrator().fit(-np.array(res["oof_single_nb_margin"]), 1 - np.array(res["oof_y_is_S"]))
    _dump({"fit_on_oof_train": cal.fit_stats, "iso_x": cal.iso.x.tolist(), "iso_p": cal.iso.p.tolist()},
          C.REPORTS / "nb_calibration.json")
    print("NB calibration (OOF train):", {k: round(v, 3) for k, v in cal.fit_stats.items()})

    # ---- interpretable features: bands, discrimination, control chart (ruler population = non-symbol train turns)
    trT = [t for t in turns if t.split == "train" and t.who == C.TARGET and t.ruler_text and not t.symbol]
    trS = [t for t in turns if t.split == "train" and t.who == C.PARTNER and t.ruler_text and not t.symbol]
    nb_bags = 200 if a.quick else C.BAND_BAGS
    bands = FE.fit_bands(trT, n=nb_bags)
    disc = FE.discrimination(trT, trS, n=100 if a.quick else 500)
    weights = FE.discrimination_weights(bands, disc)
    chart = FE.control_chart(trT, bands)
    _dump({"bands": bands, "discrimination": disc, "weights": weights, "control_chart": chart},
          C.REPORTS / "feature_bands.json")
    print("features weighted:", [f for f in FE.FEATURES if disc[f]["weighted"]])
    print("features not weighted (disc < %.2f):" % C.FEATURE_MIN_AUC, [f for f in FE.FEATURES if not disc[f]["weighted"]])
    print("out-of-control blocks:", {b: v["out_of_control"] for b, v in chart.items() if v["out_of_control"]})
    print(f"done in {time.perf_counter() - t0:.0f} s -> {C.REPORTS}")


if __name__ == "__main__":
    main()
