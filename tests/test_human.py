"""Blind-test and report tests that need no model, torch or chat data (CODE_REVIEW_2026-10-17 #5, #6, #7): Wald SPRT
bounds and error rates (simulation), items (one system per context for kinds A and B, time order, kind-A contexts
that hide the real replies of other items, de-identified replies, symbol-like replies left out), per-system scoring
and line 3 (a sequence whose items ran out stays undecided), the paired bootstrap, the D-AUC fast path, the table,
lines 1-2 (diversity at the same n) and both CLIs. The inputs are made up in this file (no message of a chat
export)."""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
from pathlib import Path

import numpy as np

import persona.data as D
from persona import config as C
from persona import evaluate as EV, human_eval as HE, report as R, samples as SM, turns as TU
from persona.flag_symbols import is_symbol_turn
from persona.text import reply_text

PHONE_REAL, PHONE_GEN = "0912345678", "0911222333"


def _llr(k, n, h):
    return k * math.log(h["p_high"] / h["p_low"]) + (n - k) * math.log((1 - h["p_high"]) / (1 - h["p_low"]))


def _segment(n_pairs=30):
    """One conversation segment: partner turn, target turn, ... (target replies hold a phone number now and then)."""
    seg = []
    for i in range(n_pairs):
        seg.append(TU.Turn(C.PARTNER, "2026-08-01", 2 * i, 2 * i, [f"第{i}題怎麼做"]))
        seg.append(TU.Turn(C.TARGET, "2026-08-01", 2 * i + 1, 2 * i + 1,
                           [f"回覆{i}"] + ([f"打{PHONE_REAL}給我"] if i % 3 == 0 else [])))
    return seg


def _outputs(samples):
    """Three systems: z is symbol-like on sids divisible by 5, y lacks sid 7, x is empty on sid 8."""
    outs = {}
    for name in ("x", "y", "z"):
        rows = []
        for s in samples:
            msgs = [f"{name}說{s.sid}"] + ([PHONE_GEN] if s.sid % 4 == 0 else [])
            if name == "z" and s.sid % 5 == 0:
                msgs = ["好啊 (笑)"]
            if name == "x" and s.sid == 8:
                msgs = []
            if not (name == "y" and s.sid == 7):
                rows.append({"sid": s.sid, "msgs": msgs})
        outs[name] = rows
    return outs


def _run(main, argv):
    """What a CLI main() prints; stdout is a UTF-8 text wrapper because main() reconfigures it."""
    buf = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    with contextlib.redirect_stdout(buf):
        main(argv)
        buf.flush()
    return buf.buffer.getvalue().decode("utf-8")


def _read(out):
    items = [json.loads(line) for line in open(out / "items.jsonl", encoding="utf-8")]
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    return items, key, stats


# ----------------------------------------------------------------------------- #5 SPRT
def test_sprt_bounds():
    for kind in ("A", "B"):
        h = C.HUMAN[kind]
        up, lo = math.log((1 - h["beta"]) / h["alpha"]), math.log(h["beta"] / (1 - h["alpha"]))
        for n in range(C.HUMAN["max_n"] + 3):
            for k in range(n + 1):
                l = _llr(k, n, h)
                want = (True, "above") if l >= up else (True, "below") if l <= lo else (n >= C.HUMAN["max_n"],
                                                                                         "undecided")
                got = HE.sprt_stop(k, n, kind)
                assert got == want, (kind, k, n, got, want)
                # line 3 reads the point estimate at the stop; at a bound it agrees with the verdict (n <= max_n)
                if got[1] != "undecided" and n <= C.HUMAN["max_n"]:
                    above_line = k / n > h["p_line"] if kind == "A" else k / n >= h["p_line"]
                    assert above_line == (got[1] == "above"), (kind, k, n)
    # hand-computed for kind A (0.55 vs 0.75, alpha = beta = 0.05, bounds +-2.944): after 10 items
    assert [HE.sprt_stop(k, 10) for k in (10, 9, 4, 3)] == [(True, "above"), (False, "undecided"),
                                                            (False, "undecided"), (True, "below")]
    assert HE.sprt_stop(0, 0) == (False, "undecided") and HE.sprt_stop(65, 100) == (True, "undecided")
    # the old interval rule is kept as ci_stop with its old answers
    assert [HE.ci_stop(30, 30), HE.ci_stop(10, 30), HE.ci_stop(20, 30)] == [(True, "above"), (True, "below"),
                                                                            (False, "undecided")]


def test_sprt_error_rates():
    """2,000 simulated judges per case: at p_high "below" <= beta + 0.03, at p_low "above" <= alpha + 0.03."""
    rng = np.random.default_rng(0)
    for kind in ("A", "B"):
        h = C.HUMAN[kind]
        for p, wrong, rate in ((h["p_high"], "below", h["beta"]), (h["p_low"], "above", h["alpha"])):
            verdicts = []
            for seq in rng.random((2000, C.HUMAN["max_n"])) < p:
                k = 0
                for i, hit in enumerate(seq, 1):
                    k += int(hit)
                    stop, v = HE.sprt_stop(k, i, kind)
                    if stop:
                        break
                verdicts.append(v)
            share = verdicts.count(wrong) / len(verdicts)
            assert share <= rate + 0.03, (kind, p, wrong, share)


# ----------------------------------------------------------------------------- #6 items
def _no_answer_in_context(items, key, stats):
    """Kind A: no item's context shows the real reply of a chosen context (in _segment each reply is unique); the
    replies that sit in a later context of the segment show as '我: [略]'."""
    answers = {f"我: 回覆{k['sid']}" for k in key}
    assert not any(answers & set(it["context"].split("\n")) for it in items)
    assert stats["real_in_earlier_context"] == stats["real_in_later_context"] == 0
    assert 0 < stats["real_hidden_in_context"] == sum(it["context"].count(f"我: {HE.HIDDEN}") for it in items)


def test_kind_a_items_one_system_per_context_and_deid():
    samples = SM.build_samples([_segment(30)])
    outs = _outputs(samples)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "a"
        n_items = HE.make_items(samples, outs, out, kind="A", seed=3, n=100)
        items, key, stats = _read(out)
        raw = (out / "items.jsonl").read_text(encoding="utf-8")
        assert n_items == len(items) == len(key) == stats["n_items"]
        sids = [k["sid"] for k in key]
        assert len(sids) == len(set(sids))                              # a context (real reply) gives one item
        assert sids == sorted(sids)                                     # time order
        _no_answer_in_context(items, key, stats)
        # permuted blocks: 30 contexts / 3 systems -> 10 contexts each, shown or skipped
        for name in outs:
            assert stats["items_per_system"].get(name, 0) + sum(stats["skipped"][name].values()) == 10, stats
        assert stats["symbol_share"]["z"] == 6 / 30 and stats["symbol_share"]["x"] == 0
        by_sid = {s.sid: s for s in samples}
        orig = {(name, o["sid"]): o["msgs"] for name, rows in outs.items() for o in rows}
        for it, k in zip(items, key):
            assert {k["left"], k["right"]} == {"real", k["system"]}
            sys_msgs = it["left"] if k["left"] == k["system"] else it["right"]
            real_msgs = it["left"] if k["left"] == "real" else it["right"]
            assert real_msgs == [m.replace(PHONE_REAL, "[電話]") for m in by_sid[k["sid"]].target.msgs]
            assert sys_msgs == [m.replace(PHONE_GEN, "[電話]") for m in orig[(k["system"], k["sid"])]]
            assert not is_symbol_turn(reply_text(orig[(k["system"], k["sid"])]))      # z's symbol-like ones left out
            assert (k["system"], k["sid"]) not in (("y", 7), ("x", 8))
        assert PHONE_REAL not in raw and PHONE_GEN not in raw and "[電話]" in raw
        assert stats["deid_hits"].get("mobile", 0) > 0
        # priority decides which contexts are chosen, not where they stand: time order; n caps the item count
        HE.make_items(samples, outs, out, kind="A", seed=3, n=12, priority_sids=[22, 13])
        items2, key2, stats2 = _read(out)
        sids2 = [k["sid"] for k in key2]
        assert len(key2) == 12 and {22, 13} <= set(sids2) and sids2 == sorted(sids2) and stats2["n_priority"] == 2
        _no_answer_in_context(items2, key2, stats2)
        # the same seed gives the same files
        HE.make_items(samples, outs, Path(tmp) / "b", kind="A", seed=3, n=100)
        assert (Path(tmp) / "b" / "items.jsonl").read_text(encoding="utf-8") == raw


def test_kind_b_items():
    samples = SM.build_samples([_segment(12)])
    outs = _outputs(samples)
    with tempfile.TemporaryDirectory() as tmp:
        HE.make_items(samples, {"x": outs["x"], "z": outs["z"]}, tmp, kind="B", seed=0, baseline="z")
        items, key, stats = _read(Path(tmp))
        assert all({k["left"], k["right"]} == {"x", "z"} and k["system"] == "x" for k in key)
        assert {k["sid"] for k in key} == {s.sid for s in samples} - {0, 5, 10, 8}     # z symbol-like, x empty
        assert PHONE_GEN not in (Path(tmp) / "items.jsonl").read_text(encoding="utf-8")
        assert stats["skipped"]["z"]["symbol"] == 3 and stats["skipped"]["x"]["empty"] == 1
        # two systems against the baseline: one item per context, so no context or baseline reply shows twice
        HE.make_items(samples, outs, Path(tmp) / "xy", kind="B", seed=0, baseline="z")
        _, key2, stats2 = _read(Path(tmp) / "xy")
        sids = [k["sid"] for k in key2]
        assert len(sids) == len(set(sids)) and sids == sorted(sids) and {k["system"] for k in key2} == {"x", "y"}
        assert len(key2) + sum(sum(c.values()) for c in stats2["skipped"].values()) == len(samples)


# ----------------------------------------------------------------------------- #6 scoring, line 3
def _key(systems, n_each, seed=0):
    rng = np.random.default_rng(seed)
    key = []
    for i in range(n_each * len(systems)):
        name = systems[i % len(systems)]
        real_left = bool(rng.random() < 0.5)
        key.append({"item": i, "sid": i, "left": "real" if real_left else name,
                    "right": name if real_left else "real", "system": name})
    return key


def test_score_per_system_and_line3():
    key = _key(["x", "y"], 30)
    # the judge spots x on each item and y on no item: per-system SPRTs stop at 10 and 6 items; pooled acc is 0.5
    ans = {str(k["item"]): (("left" if k["left"] == "real" else "right") if k["system"] == "x" else
                            ("left" if k["left"] != "real" else "right")) for k in key}
    sc = HE.score(ans, key, "A")
    x, y = sc["per_system"]["x"], sc["per_system"]["y"]
    assert (x["verdict"], x["reason"], x["n_stop"], x["acc_stop"]) == ("above", "bound", 10, 1.0)
    assert (y["verdict"], y["reason"], y["n_stop"], y["acc_stop"]) == ("below", "bound", 6, 0.0)
    assert x["n"] == y["n"] == 30 and sc["n"] == 60 and sc["acc"] == 0.5 and "verdict" not in sc
    l3 = R.lines_from_human(sc)
    assert l3["line3"] == {"x": False, "y": None}                     # y passes kind A; kind B still missing
    # items run out without a bound -> "exhausted", point estimate at the stop; unanswered -> continue
    key_w = _key(["w"], 5)
    hits = [True, False, True, False, True]
    ans_w = {str(k["item"]): (("left" if k["left"] == "real" else "right") if h else
                              ("left" if k["left"] != "real" else "right")) for k, h in zip(key_w, hits)}
    w = HE.score(ans_w, key_w)["per_system"]["w"]
    assert (w["stop"], w["reason"], w["verdict"], w["acc_stop"]) == (True, "exhausted", "undecided", 0.6)
    w3 = HE.score(dict(list(ans_w.items())[:3]), key_w)["per_system"]["w"]
    assert (w3["stop"], w3["reason"], w3["acc_stop"], w3["n"]) == (False, None, None, 3)
    # §6.2 stops at a bound or at max_n only: line 3 waits for more items instead of reading acc 0.6 at 5 items
    lw = R.lines_from_human(HE.score(ans_w, key_w))
    assert lw["parts"]["w"]["A"]["pass"] is None and lw["parts"]["w"]["A"]["more_items_needed"] is True
    assert l3["parts"]["x"]["A"]["more_items_needed"] is False
    # kind B: a hit is preferring the system over the baseline; y preferred on each item -> above -> line 3 for y
    key_b = [{"item": i, "sid": i, "left": "y", "right": "B0", "system": "y"} for i in range(20)]
    sb = HE.score({str(i): "left" for i in range(20)}, key_b, "B")
    assert sb["per_system"]["y"]["verdict"] == "above"
    assert R.lines_from_human({"A": sc, "B": sb})["line3"] == {"x": False, "y": True}
    try:
        HE.score({"0": "middle"}, key)
        raise AssertionError("an answer other than left/right was accepted")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- #7 bootstrap
def test_paired_bootstrap_copy_and_shift():
    rng = np.random.default_rng(0)
    a = [{"sid": i, "x": float(v)} for i, v in enumerate(rng.normal(0.0, 1.0, 200))]
    mean_x = lambda outs: float(np.mean([o["x"] for o in outs]))
    same = R.paired_bootstrap(mean_x, a, [dict(o) for o in a], n_boot=300, seed=1)
    assert same["diff"] == 0.0 and same["ci"][0] <= 0.0 <= same["ci"][1] and same["p_gt0"] == 0.0
    shift = 0.3
    b = [{"sid": o["sid"], "x": o["x"] + shift + e} for o, e in zip(a, rng.normal(0.0, 0.1, 200))]
    res = R.paired_bootstrap(mean_x, a, b, seed=1)
    assert res["n_boot"] == C.EVAL["boot_n"] == res["n_valid"] and res["n"] == 200
    assert abs(res["diff"] + shift) < 0.03 and res["ci"][0] <= res["diff"] <= res["ci"][1] < 0.0
    # pairing: the CI reflects the per-context noise (sd 0.1), not the spread between contexts (sd 1)
    assert res["ci"][1] - res["ci"][0] < 0.1 and res["p_gt0"] == 0.0
    try:
        R.paired_bootstrap(mean_x, a, b[1:] + b[:1])
        raise AssertionError("misaligned outputs were accepted")
    except ValueError:
        pass


def _texts(rng, alphabet, n):
    return ["".join(rng.choice(list(alphabet), int(rng.integers(3, 9)))) for _ in range(n)]


def test_d_auc_bootstrap_fast_path():
    rng = np.random.default_rng(0)
    real = _texts(rng, "好啊喔嗯對我看等一下可以", 80)
    far = _texts(rng, "您請問謝是的這樣沒問題嗎", 80)
    res = R.d_auc_bootstrap(real, list(real), far, n_boot=300, seed=0)
    for side, gen in (("a", real), ("b", far)):                     # point estimates = evaluate.discriminator
        d = EV.discriminator(real, gen)
        assert res["auc1"][side] == d["auc1"] and res[f"auc{C.K_ATTR}"][side] == d[f"auc{C.K_ATTR}"], (side, res, d)
    assert res["auc1"]["a"] == 0.5 and res["auc1"]["b"] > 0.9       # a copy of the real replies vs another style
    assert res["auc1"]["ci"][1] < 0 and res[f"auc{C.K_ATTR}"]["ci"][1] <= 0 and res["auc1"]["p_gt0"] == 0.0
    same = R.d_auc_bootstrap(real, far, list(far), n_boot=100, seed=0)
    assert same["auc1"]["diff"] == 0.0 and same["auc1"]["ci"] == [0.0, 0.0]
    # contract (d): symbol-like generated replies leave the population as in Evaluator.evaluate
    mixed = ["好啊 (笑)" if i % 8 == 0 else t for i, t in enumerate(far)]
    res2 = R.d_auc_bootstrap(real, mixed, far, n_boot=50, seed=0)
    assert res2["symbol_a"] == 10 and res2["n_gen_a"] == 70
    assert res2["auc1"]["a"] == EV.discriminator(real, [t for t in mixed if not is_symbol_turn(t)])["auc1"]


# ----------------------------------------------------------------------------- #7 table and lines 1-2
def test_table_and_acceptance():
    tab = {"B-zero_instruct_s0": {"d_auc1": 0.8}, "C-shuf": {"d_auc1": 0.5}, "C-real": {"d_auc1": 0.5, "nll": None},
           "B-zero_instruct_s1": {"d_auc1": 0.9}, "untrained_instruct_s0": {"d_auc1": 0.85},
           "sft_instruct_all_s0_s0": {"d_auc1": 0.6, "nll": 1.0}, "sft_instruct_all_s1_s0": {"d_auc1": 0.7, "nll": 1.2},
           "C-S": {"d_auc1": 0.76, "band_inside_mean": float("nan")}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "eval_table.json"
        p.write_text(json.dumps(tab), encoding="utf-8")
        md = R.table(p).splitlines()
    rows = [line.split(" | ")[0].lstrip("| ") for line in md[2:]]
    assert rows == ["C-real", "C-shuf", "C-S", "untrained_instruct", "B-zero_instruct", "sft_instruct_all"], rows
    cells = {r: line.split(" | ") for r, line in zip(rows, md[2:])}
    assert cells["B-zero_instruct"][1:4] == ["2", "—", "0.850 ± 0.071"]          # seeds, NLL, D-AUC1
    assert cells["sft_instruct_all"][2] == "1.100 ± 0.141" and cells["C-S"][6] == "—"       # NLL; Band NaN

    c_real = {"band_inside_mean": 12.0, "div_distinct2": 0.6, "div_hill1": 100.0, "div_n": 100, "leak_per_100": 0.5}
    # 70 replies kept: Hill-1 60 is 0.6 of C-real's at 100 replies, 0.85 of C-real's rarefied to 70 (the ratio)
    good = {"d_auc1": 0.7, "d_auc10": 0.85, "band_inside_mean": 11.2, "div_distinct2": 0.5, "div_hill1": 60.0,
            "div_n": 70, "div_distinct2_ratio": 0.9, "div_hill1_ratio": 0.85, "copy_rate": 0.01, "leak_per_100": 0.3,
            "symbol_share": 0.03}
    boot = {"auc1": {"ci": [-0.10, -0.02]}}
    a = R.acceptance(good, c_real, boot)
    assert (a["line1"], a["line2"]) == (True, True) and a["details"]["symbol_share"] == 0.03
    assert R.acceptance(good, c_real)["line1"] is None                       # no bootstrap: not decidable yet
    assert R.acceptance(good, dict(c_real, leak_per_100=None), boot)["line2"] is None      # OpenCC absent
    assert R.acceptance(dict(good, d_auc1=0.8), c_real)["line1"] is False    # a failed part decides alone
    assert R.acceptance(good, c_real, {"auc1": {"ci": [-0.05, 0.01]}})["line1"] is False   # CI holds 0
    assert R.acceptance(good, c_real, {"d_auc": boot})["line1"] is True      # compare() result
    for bad in ({"band_inside_mean": 10.9}, {"div_hill1_ratio": 0.79}, {"copy_rate": 0.03}, {"leak_per_100": 0.6}):
        assert R.acceptance(dict(good, **bad), c_real, boot)["line2"] is False, bad
    # a row without the ratios: the raw values decide only at the same div_n as C-real
    raw = {k: v for k, v in good.items() if not k.endswith("_ratio")}
    assert R.acceptance(raw, c_real, boot)["line2"] is None
    assert R.acceptance(dict(raw, div_n=100, div_hill1=85.0), c_real, boot)["line2"] is True
    assert R.acceptance(dict(raw, div_n=100, div_hill1=79.0), c_real, boot)["line2"] is False


# ----------------------------------------------------------------------------- CLIs (contract e)
def test_clis_stay_under_runs():
    samples = SM.build_samples([_segment(30)])
    old = (C.RUNS, D.load_all, D.train_dev_test)
    with tempfile.TemporaryDirectory() as tmp:
        C.RUNS = Path(tmp) / "runs"
        D.load_all = lambda *a, **k: (None, None, [], None)          # no training turns: copy checker empty
        D.train_dev_test = lambda segs: ([], [], samples, [])
        try:
            for name, fn in (("x", lambda s: s.target.msgs), ("y", lambda s: [f"您好{s.sid}請問"])):
                (C.RUNS / f"eval_{name}").mkdir(parents=True)
                with open(C.RUNS / f"eval_{name}" / "outputs.jsonl", "w", encoding="utf-8") as f:
                    for s in samples:
                        f.write(json.dumps({"sid": s.sid, "msgs": fn(s)}, ensure_ascii=False) + "\n")
            argv = ["make", "--kind", "A", "--systems", "eval_x,y", "--out", "human_A", "--n", "12"]
            _run(HE.main, argv)
            key = json.loads((C.RUNS / "human_A" / "key.json").read_text(encoding="utf-8"))
            assert len(key) == 12 and {k["system"] for k in key} == {"x", "y"}
            for bad_argv in (argv, argv[:-2] + ["--out", "runs/human_A"], argv[:-2] + ["--out", "/tmp/elsewhere"]):
                try:
                    HE.main(bad_argv)                                # existing folder / outside C.RUNS: refused
                    raise AssertionError(bad_argv)
                except (FileExistsError, ValueError) as e:
                    assert "\n" not in str(e)
            _run(HE.main, argv + ["--overwrite"])
            _run(HE.main, ["make", "--kind", "B", "--systems", "x,y", "--out", "human_B"])
            kb = json.loads((C.RUNS / "human_B" / "key.json").read_text(encoding="utf-8"))
            assert all(k["system"] == "x" and {k["left"], k["right"]} == {"x", "y"} for k in kb)
            ans = Path(tmp) / "answers.json"
            ans.write_text(json.dumps({str(k["item"]): "left" for k in key}), encoding="utf-8")
            printed = _run(HE.main, ["score", "--dir", "human_A", "--answers", str(ans)])
            assert '"line3"' in printed and "您好" not in printed and "回覆" not in printed     # numbers only
            cmp = json.loads(_run(R.main, ["--compare", "eval_x", "y", "--n-boot", "50"]))
            assert cmp["n_contexts"] == 30 and cmp["d_auc"]["auc1"]["a"] == 0.5 and cmp["symbol_share"]["diff"] == 0
        finally:
            C.RUNS, D.load_all, D.train_dev_test = old
