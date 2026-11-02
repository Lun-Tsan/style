"""persona.data -- the one place that turns the LINE export into turns, segments and samples.

Every driver (stage1, train_sft, make_pairs, train_dpo, run_eval) calls load_all(); the split, dev cut, date blocks
and subsets are recomputed deterministically from config, so a change to those constants changes every later run.
"""
from __future__ import annotations

from . import config as C
from . import parse_line, samples as SM, split as SP, turns as TU
from .flag_symbols import is_symbol_turn


def load_all():
    """messages -> turns (with symbol flags, segments, blocks) -> segments."""
    msgs, n_sys = parse_line.load()
    turns = TU.build_turns(msgs)
    for t in turns:
        t.symbol = is_symbol_turn(t.ruler_text)
    segs = TU.assign_segments(turns)
    SP.assign_blocks([t for t in turns if t.split == "train" and t.ruler_text and not t.symbol])
    return msgs, n_sys, turns, segs


def train_dev_test(segs):
    """Samples split the way every stage uses them: (train minus dev, dev, test, train incl. dev)."""
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    test = [s for s in smp if s.split == "test"]
    train, dev = SP.dev_cut(train_all)
    return train, dev, test, train_all
