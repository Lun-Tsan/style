"""persona.data -- the one place that turns the LINE export into turns, segments and samples.

The drivers (stage1, train_sft, make_pairs, train_dpo, run_eval) call load_all(); the split, dev cut, date blocks
and subsets are recomputed deterministically from config, so a change to those constants changes later runs.
stage1 writes manifest() to reports/data_manifest.json; load_all() raises when the data differ from that file (if it
exists), so a changed constant stops the later stages until stage 1 is re-run on the new data.
"""
from __future__ import annotations

import json

from . import config as C
from . import parse_line, samples as SM, split as SP, turns as TU
from .flag_symbols import is_symbol_turn

MANIFEST = "data_manifest.json"      # under C.REPORTS, written by stage1


def load_all(check_manifest: bool = True):
    """messages -> turns (with symbol flags, segments, blocks) -> segments. With check_manifest, raises ValueError
    when the data differ from C.REPORTS/data_manifest.json (if that file exists); stage1 passes False."""
    msgs, n_sys = parse_line.load()
    turns = TU.build_turns(msgs)
    for t in turns:
        t.symbol = is_symbol_turn(t.ruler_text)
    segs = TU.assign_segments(turns)
    tr = [t for t in turns if t.split == "train" and not t.symbol]
    SP.assign_blocks([t for t in tr if t.ruler_text])      # blocks of the text population (= rulers.validate OOF folds)
    # sticker-only turns join the block of their date (else of the latest earlier text turn), so the feature control
    # chart can include them (CODE_REVIEW #4) without moving the block boundaries of the text population
    by_date, last = {t.date: t.block for t in tr if t.ruler_text}, 0
    for t in tr:
        if t.ruler_text:
            last = t.block
        else:
            t.block = by_date.get(t.date, last)
    if check_manifest:
        verify_manifest(turns, segs)
    return msgs, n_sys, turns, segs


def train_dev_test(segs):
    """Samples split the way every stage uses them: (train minus dev, dev, test, train incl. dev)."""
    smp = SM.build_samples(segs)
    train_all = [s for s in smp if s.split == "train"]
    test = [s for s in smp if s.split == "test"]
    train, dev = SP.dev_cut(train_all)
    return train, dev, test, train_all


def manifest(turns, segs) -> dict:
    """Split constants and the counts they produce (numbers and role codes only, no text). The constants come
    before the counts, so a changed constant is the key verify_manifest() names."""
    train, dev, test, _ = train_dev_test(segs)

    def by(pool):
        return {sp: {C.CODE[P]: sum(t.split == sp and t.who == P for t in pool) for P in C.SPEAKERS}
                for sp in ("train", "test")}

    return {"SPLIT_DATE": C.SPLIT_DATE, "GAP_MIN": C.GAP_MIN, "SEG_GAP_H": C.SEG_GAP_H, "DEV_FRAC": C.DEV_FRAC,
            "SUBSET_SIZES": list(C.SUBSET_SIZES), "CTX_TURNS": C.CTX_TURNS,
            "segments": len(segs),
            "turns": by(turns),
            "text_turns": by([t for t in turns if t.text_b]),
            "symbol_turns": by([t for t in turns if t.symbol]),
            "samples": {"train": len(train), "dev": len(dev), "test": len(test)},      # train = train minus dev
            "subsets": {("all" if k is None else str(k)): len(v) for k, v in SP.nested_subsets(train).items()}}


def _first_diff(old, new, path=()):
    """(dotted key, old value, new value) of the earliest difference, keys in file order; None if equal."""
    if isinstance(old, dict) and isinstance(new, dict):
        for k in list(old) + [k for k in new if k not in old]:
            if k not in old or k not in new:
                return ".".join(path + (k,)), old.get(k), new.get(k)
            d = _first_diff(old[k], new[k], path + (k,))
            if d:
                return d
        return None
    return None if old == new else (".".join(path), old, new)


def verify_manifest(turns, segs):
    """Raise ValueError naming the earliest key where these data differ from the stage-1 manifest, if it exists."""
    path = C.REPORTS / MANIFEST
    if not path.exists():
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    new = json.loads(json.dumps(manifest(turns, segs)))     # same types as the file (tuples -> lists, keys -> str)
    d = _first_diff(old, new)
    if d:
        raise ValueError(f"data differ from {path} at '{d[0]}' ({d[1]} -> {d[2]}); re-run python -m persona.stage1 "
                         f"if the change is intended")
