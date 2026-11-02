"""Training-side unit tests that need no GPU: pair construction and the scorer cap on synthetic inputs, DPO loss,
baseline parsing, human-eval stopping rule."""
from __future__ import annotations

import math
import pathlib
import tempfile

import numpy as np

from persona import config as C
from persona import turns as TU
from persona.human_eval import sprt_stop
from persona.pairs import build_pairs, cell, cell_level, fit_scorer
from persona.samples import Sample
from persona.text import CopyChecker


class FakeScorer:
    """Identity calibration over a table of raw target-LLRs (0 for other replies): score = min(raw, cap_raw)."""
    cap_raw = 2.0
    fit_stats = {"cap_raw": 2.0}
    table = {"好啊": 1.5, "好喔": 1.8, "沒錯": 1.3, "我不知道耶": -1.6, "嗯嗯": -2.0, "好的": -1.4}

    def raw(self, msgs):
        return self.table.get("\n".join(msgs), 0.0)

    def score(self, msgs):
        return min(self.raw(msgs), self.cap_raw)


def _sample(sid, target_msgs):
    ctx = [TU.Turn(C.PARTNER, "2026-01-01", 0, 0, ["明天要上課嗎"])]
    return Sample(sid, ctx, TU.Turn(C.TARGET, "2026-01-01", 1, 1, target_msgs))


def test_cell():
    assert cell(["好啊"]) == (0, False, 1, True)
    assert cell(["[貼圖]"]) == (0, True, 1, False)
    assert cell(["這題要先把式子展開", "再代入"]) == (1, False, 2, False)
    assert cell(["a", "b", "c", "d"])[2] == 3


def test_build_pairs_selection_and_ditto_cap():
    checker = CopyChecker(["這是一句很長的訓練回覆用來測試抄襲檢查器"], L=12)
    samples = [_sample(0, ["好啊"]), _sample(1, ["好啊"]), _sample(2, ["沒錯"]), _sample(3, ["好啊"])]
    cands = [
        [["好喔"], ["好啊"], ["嗯嗯"], ["我不知道耶"]],        # chosen 好喔 (same cell, highest), rejected 嗯嗯
        [["這是一句很長的訓練回覆用來測試抄襲檢查器"], ["好的"]],   # copy -> no chosen -> ditto (real 好啊), rejected 好的
        [["嗯嗯"]],                                              # no chosen -> ditto, rejected 嗯嗯
        [["好喔"], ["好喔"], ["好喔"]],                           # no rejected -> no pair
    ]
    pairs, st = build_pairs(samples, cands, FakeScorer(), checker, delta=math.log(3), ditto_max=1.0, seed=0)
    assert st["contexts"] == 4 and st["contexts_no_rejected"] == 1 and st["rejected_copy"] == 1
    assert st["candidates"] == 10 and st["candidates_unique"] == st["cand_eligible"] == 8
    assert st["cand_symbol"] == st["cand_unfinished"] == st["cand_capped"] == 0 and st["cap_raw"] == 2.0
    assert [p["sid"] for p in pairs] == [0, 1, 2]
    assert pairs[0]["chosen"] == ["好喔"] and pairs[0]["rejected"] == ["嗯嗯"] and not pairs[0]["ditto"]
    assert pairs[1]["ditto"] and pairs[1]["chosen"] == ["好啊"] and pairs[1]["rejected"] == ["好的"]
    assert pairs[2]["ditto"] and pairs[2]["rejected"] == ["嗯嗯"]
    # the chosen mean leaves out the DITTO pairs (their chosen is the real reply); the overall mean is kept apart
    assert st["mean_score_chosen"] == 1.8 and abs(st["mean_score_chosen_all"] - (1.8 + 1.5 + 1.3) / 3) < 1e-12
    # with the default cap (50%) the two ditto pairs are thinned so that ditto / total <= 0.5
    pairs2, st2 = build_pairs(samples, cands, FakeScorer(), checker, delta=math.log(3), ditto_max=0.5, seed=0)
    assert st2["n_pairs"] == 2 and st2["n_ditto"] == 1 and st2["ditto_dropped"] == 1
    assert pairs2[0]["sid"] == 0


def test_build_pairs_cell_ladder():
    checker = CopyChecker(["x"], L=12)
    samples = [_sample(0, ["好啊"])]                                 # real cell (0, False, 1, True)
    cands = [[["這題要先把式子展開啦"], ["嗯嗯"]]]                       # candidate cell (1, False, 1, True): level 1
    class Sc(FakeScorer):
        table = {**FakeScorer.table, "這題要先把式子展開啦": 1.9}
    pairs, st = build_pairs(samples, cands, Sc(), checker, delta=math.log(3), ditto_max=1.0, seed=0)
    assert len(pairs) == 1 and not pairs[0]["ditto"] and pairs[0]["level"] == 1 and st["chosen_level_1"] == 1
    assert cell_level((0, True, 1, False), (0, False, 1, False)) == 2
    assert cell_level((3, False, 1, False), (0, False, 1, False)) == 2
    assert cell_level((1, False, 3, False), (0, False, 1, True)) == 1


def test_build_pairs_raw_cap_ties_and_exclusions():
    checker = CopyChecker(["x"], L=12)
    class Sc(FakeScorer):
        table = {**FakeScorer.table, "好呀": 2.6, "對啊": 3.4, "是啊": 1.9, "好啊(笑)": 5.0, "可以啊": 4.0,
                 "嗯": -3.0, "喔": -3.0}
    class NoCap(Sc):
        cap_raw = math.inf
        fit_stats = {"cap_raw": math.inf}
    samples = [_sample(0, ["好啊"])]
    cands = [[["好呀"], ["對啊"], ["是啊"], ["好啊(笑)"], ["可以啊"], ["嗯"], ["喔"]]]
    fin = [[True, True, True, True, False, True, True]]
    def run(sc, seed, finished=fin):
        return build_pairs(samples, cands, sc, checker, delta=math.log(3), ditto_max=1.0, seed=seed, finished=finished)
    _, st = run(Sc(), 0)
    # 好啊(笑) is symbol-like and 可以啊 unfinished: they take neither role although their raw LLRs are the highest
    assert st["cand_symbol"] == 1 and st["cand_unfinished"] == 1 and st["cand_eligible"] == 5
    assert abs(st["symbol_share"] - 1 / 7) < 1e-12
    assert st["cand_capped"] == 2 and st["cand_capped_share"] == 0.4
    # 好呀 (2.6) and 對啊 (3.4) are capped at 2.0 and tie, 是啊 (1.9) stays below; 嗯 and 喔 tie for the lowest
    picks = [run(Sc(), seed)[0][0] for seed in range(20)]
    assert {tuple(p["chosen"]) for p in picks} == {("好呀",), ("對啊",)}
    assert {tuple(p["rejected"]) for p in picks} == {("嗯",), ("喔",)}
    assert all(p["score_chosen"] == 2.0 and p["score_rejected"] == -3.0 and not p["ditto"] for p in picks)
    assert run(Sc(), 3) == run(Sc(), 3)                                  # fixed seed -> same pairs and stats
    # without the cap the highest raw LLR (對啊) is chosen for each of the 20 seeds
    assert {tuple(run(NoCap(), seed)[0][0]["chosen"]) for seed in range(20)} == {("對啊",)}
    # a calibrator that is flat below cap_raw (its 0.999 ceiling, as in reports/) does not move the cap: the
    # capped raw LLR ranks, so 是啊 (1.9) stays out of the tie although all three score 1.5
    class Flat(Sc):
        def score(self, msgs):
            return min(super().score(msgs), 1.5)
    assert {tuple(run(Flat(), seed)[0][0]["chosen"]) for seed in range(20)} == {("好呀",), ("對啊",)}
    # a missing finished list counts as all finished; a duplicate that finished in one of its draws stays eligible
    assert run(Sc(), 0, finished=[None])[1]["cand_unfinished"] == 0
    assert build_pairs(samples, cands, Sc(), checker, delta=math.log(3))[1]["cand_eligible"] == 6
    _, st2 = run(Sc(), 0, finished=None)
    assert st2["cand_unfinished"] == 0
    _, st3 = build_pairs(samples, [[["可以啊"], ["可以啊"], ["嗯"]]], Sc(), checker, delta=math.log(3),
                         finished=[[False, True, True]])
    assert st3["candidates_unique"] == st3["cand_eligible"] == 2 and st3["cand_unfinished"] == 0


def test_fit_scorer_raw_cap():
    rng = np.random.default_rng(0)
    alphabet = {C.TARGET: "好啊喔耶欸齁嘛吧對呀", C.PARTNER: "請問老師這題怎麼算嗎謝謝"}
    turns = []
    for i in range(80):
        who = C.TARGET if i % 2 else C.PARTNER
        turns.append(TU.Turn(who, "2026-01-01", i, i, ["".join(rng.choice(list(alphabet[who]), int(rng.integers(2, 9))))]))
    y_S = np.array([t.who == C.PARTNER for t in turns], int)
    margin = np.where(y_S == 1, 2.0, -2.0) + rng.normal(0, 2, len(turns))     # stand-in for the OOF NB margins
    sc = fit_scorer(turns, margin, y_S, cap_q=0.95)
    llr_T = -margin[y_S == 0]
    # the cap is the 95% quantile of the T OOF target-LLRs (raw scale); 2 of the 40 T turns lie above it
    assert sc.cap_raw == float(np.quantile(llr_T, 0.95)) == sc.fit_stats["cap_raw"]
    assert sc.fit_stats["cap_T_oof_above"] == 0.05 and sc.fit_stats["cap"] == float(sc.cal.logodds([sc.cap_raw])[0])
    for t in turns:
        assert sc.score(t.msgs) == float(sc.cal.logodds([min(sc.raw(t.msgs), sc.cap_raw)])[0])
    capped = [t for t in turns if sc.raw(t.msgs) > sc.cap_raw]
    assert capped and len({sc.score(t.msgs) for t in capped}) == 1
    assert sc.raw(["[貼圖]"]) is None and sc.score(["[貼圖]"]) == 0.0
    # OOF arrays that do not line up with the turns (length or speaker) are refused
    for args in ((turns[:-1], margin, y_S), (turns, margin, 1 - y_S)):
        try:
            fit_scorer(*args)
        except ValueError:
            continue
        raise AssertionError("misaligned OOF arrays were accepted")


def test_make_pairs_out_dir():
    from persona import make_pairs as MP                 # torch is imported inside run(), after the directory check
    assert MP.default_out("sft_instruct_all_s0", "instruct", 1) == "pairs_sft_instruct_all_s0_s1"
    assert MP.default_out("sft_instruct_all_s0/", "instruct", 0) == "pairs_sft_instruct_all_s0_s0"
    assert MP.default_out(None, "base", 2) == "pairs_untrained_base_s2"
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats.json").write_text("{}", encoding="utf-8")
        try:
            MP.run("sft_x", d)                           # absolute path: C.RUNS / d is d
        except FileExistsError as e:
            assert d in str(e) and "\n" not in str(e)
        else:
            raise AssertionError("a non-empty run directory was accepted without --overwrite")


def test_dpo_loss_matches_formula():
    import torch                                   # skipped by the runner when torch is absent
    from persona.train_dpo import dpo_loss
    pc, pr, rc, rr = torch.tensor([-10.0, -5.0]), torch.tensor([-12.0, -4.0]), torch.tensor([-11.0, -6.0]), torch.tensor([-11.0, -6.0])
    loss, diag = dpo_loss(pc, pr, rc, rr, beta=0.1)
    manual = -torch.nn.functional.logsigmoid(0.1 * ((pc - rc) - (pr - rr))).mean()
    assert abs(float(loss) - float(manual)) < 1e-6
    assert diag["reward_acc"] == 0.5 and abs(diag["margin"] - ((1 - (-1)) + ((1) - (2))) / 2) < 1e-6


def test_parse_chat_output():
    from persona.baselines import parse_chat_output   # imports torch through persona.generate
    assert parse_chat_output("<think>想一下</think>\n我: 好啊\n我：我看看\n\n對方: 謝謝") == ["好啊", "我看看"]
    assert parse_chat_output("好啊") == ["好啊"]
    assert parse_chat_output("") == []


def test_sprt_stop():
    assert sprt_stop(0, 0) == (False, "undecided")
    assert sprt_stop(30, 30) == (True, "above")
    assert sprt_stop(10, 30) == (True, "below")
    assert sprt_stop(20, 30) == (False, "undecided")
    assert sprt_stop(65, 100)[0] is True


def test_token_batches():
    import torch  # noqa: F401  (generate imports torch at module level; skipped by the runner without it)
    from persona.generate import token_batches
    import random
    rng = random.Random(0)
    for _ in range(200):
        lengths = [rng.randint(1, 900) for _ in range(rng.randint(0, 40))]
        n, mx, budget, bs = rng.choice([1, 8]), 64, rng.choice([1000, 12000]), rng.choice([2, 16])
        batches = token_batches(lengths, n, mx, budget, bs)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(lengths)))                                   # each prompt once
        assert [lengths[i] for i in flat] == sorted(lengths, reverse=True)                 # longest first
        for b in batches:
            assert len(b) <= bs
            if len(b) > 1:
                assert len(b) * (lengths[b[0]] + mx) * n <= budget                         # within budget
    assert token_batches([5000, 10, 10], 8, 64, 12000, 16) == [[0], [1, 2]]                 # over budget runs alone


def test_dpo_selection_key_and_dev_subset():
    import torch  # noqa: F401
    import math
    from persona.train_dpo import default_out, dev_subset, selection_key
    recs = [{"step": 0, "auc10": 0.55, "band": 13.0, "symbol_share": 0.00},
            {"step": 10, "auc10": 0.50, "band": 13.5, "symbol_share": 0.02},
            {"step": 20, "auc10": 0.45, "band": 14.0, "symbol_share": 0.20},     # symbol-like replies: ranks behind
            {"step": 30, "auc10": math.nan, "band": 14.0, "symbol_share": 0.00},  # NaN ranks behind any number
            {"step": 40, "auc10": 0.50, "band": 13.5, "symbol_share": 0.00}]      # tie with step 10: earlier step wins
    order = sorted(recs, key=lambda r: (selection_key(r, recs[0]["symbol_share"]), r["step"]))
    assert [r["step"] for r in order] == [10, 40, 0, 30, 20], [r["step"] for r in order]
    assert selection_key(recs[2], math.nan) < selection_key(recs[1], math.nan)       # no step-0 share: guard off
    dev = list(range(118))
    a, b = dev_subset(dev, 60, 0), dev_subset(dev, 60, 0)
    assert a == b and len(a) == 60 and a == sorted(a) and len(set(a)) == 60
    assert dev_subset(dev, 60, 1) != a and dev_subset(dev, 500, 0) == dev
    assert default_out("sft_instruct_all_s0", "runs/pairs_sft_instruct_all_s0_s0/pairs.jsonl", 2) == \
        "dpo_sft_instruct_all_s0_pairs_sft_instruct_all_s0_s0_s2"


def test_sft_run_names():
    import torch  # noqa: F401
    from persona import modeling as MD
    from persona.train_sft import default_out, size_key
    assert size_key("all") is None and size_key(None) is None and size_key("250") == 250
    assert default_out("instruct", None, 0) == "sft_instruct_all_s0"
    assert default_out("instruct", 100, 1, targets=MD.LORA_MIX_ONLY) == "sft_instruct_100_s1_attn"
    assert default_out("instruct", None, 0, epochs=5, no_retrain=True) == "sft_instruct_all_s0_ep5_noretrain"
