"""Training-side unit tests that need no GPU: pair construction on synthetic candidates, DPO loss, baseline parsing,
human-eval stopping rule."""
from __future__ import annotations

import math

import torch

from persona import config as C
from persona import turns as TU
from persona.baselines import parse_chat_output
from persona.generate import CopyChecker
from persona.human_eval import sprt_stop
from persona.pairs import build_pairs, cell, cell_level
from persona.samples import Sample
from persona.train_dpo import dpo_loss


class FakeScorer:
    cap = 2.0

    def score(self, msgs):
        t = "\n".join(msgs)
        return min(self.cap, {"好啊": 1.5, "好喔": 1.8, "沒錯": 1.3, "我不知道耶": -1.6, "嗯嗯": -2.0, "好的": -1.4}.get(t, 0.0))


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
    assert [p["sid"] for p in pairs] == [0, 1, 2]
    assert pairs[0]["chosen"] == ["好喔"] and pairs[0]["rejected"] == ["嗯嗯"] and not pairs[0]["ditto"]
    assert pairs[1]["ditto"] and pairs[1]["chosen"] == ["好啊"] and pairs[1]["rejected"] == ["好的"]
    assert pairs[2]["ditto"] and pairs[2]["rejected"] == ["嗯嗯"]
    # with the default cap (50%) the two ditto pairs are thinned so that ditto / total <= 0.5
    pairs2, st2 = build_pairs(samples, cands, FakeScorer(), checker, delta=math.log(3), ditto_max=0.5, seed=0)
    assert st2["n_pairs"] == 2 and st2["n_ditto"] == 1 and st2["ditto_dropped"] == 1
    assert pairs2[0]["sid"] == 0


def test_build_pairs_cell_ladder():
    checker = CopyChecker(["x"], L=12)
    samples = [_sample(0, ["好啊"])]                                 # real cell (0, False, 1, True)
    cands = [[["這題要先把式子展開啦"], ["嗯嗯"]]]                       # candidate cell (1, False, 1, True): level 1
    class Sc(FakeScorer):
        def score(self, msgs):
            return {"這題要先把式子展開啦": 1.9}.get("\n".join(msgs), super().score(msgs))
    pairs, st = build_pairs(samples, cands, Sc(), checker, delta=math.log(3), ditto_max=1.0, seed=0)
    assert len(pairs) == 1 and not pairs[0]["ditto"] and pairs[0]["level"] == 1 and st["chosen_level_1"] == 1
    assert cell_level((0, True, 1, False), (0, False, 1, False)) == 2
    assert cell_level((3, False, 1, False), (0, False, 1, False)) == 2
    assert cell_level((1, False, 3, False), (0, False, 1, True)) == 1


def test_dpo_loss_matches_formula():
    pc, pr, rc, rr = torch.tensor([-10.0, -5.0]), torch.tensor([-12.0, -4.0]), torch.tensor([-11.0, -6.0]), torch.tensor([-11.0, -6.0])
    loss, diag = dpo_loss(pc, pr, rc, rr, beta=0.1)
    manual = -torch.nn.functional.logsigmoid(0.1 * ((pc - rc) - (pr - rr))).mean()
    assert abs(float(loss) - float(manual)) < 1e-6
    assert diag["reward_acc"] == 0.5 and abs(diag["margin"] - ((1 - (-1)) + ((1) - (2))) / 2) < 1e-6


def test_parse_chat_output():
    assert parse_chat_output("<think>想一下</think>\n我: 好啊\n我：我看看\n\n對方: 謝謝") == ["好啊", "我看看"]
    assert parse_chat_output("好啊") == ["好啊"]
    assert parse_chat_output("") == []


def test_sprt_stop():
    assert sprt_stop(0, 0) == (False, "undecided")
    assert sprt_stop(30, 30) == (True, "above")
    assert sprt_stop(10, 30) == (True, "below")
    assert sprt_stop(20, 30) == (False, "undecided")
    assert sprt_stop(65, 100)[0] is True
