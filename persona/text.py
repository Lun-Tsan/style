"""persona.text -- torch-free text helpers shared by generation, pair building and evaluation.

reply_text(): the ruler view of a reply; CopyChecker: verbatim-copy test against training targets; leak_chars():
OpenCC-based simplified / Mainland-term leakage count. Kept out of generate.py so that evaluate, pairs and
human_eval import without torch (controls, pair logic and the blind test run on a CPU-only machine).
"""
from __future__ import annotations

from . import config as C
from .normalize import PH, PH_LINE, emoticon_placeholders

PH_LINES = set(PH_LINE.values())


def reply_text(msgs) -> str:
    """Ruler view of a reply: text messages only (placeholder lines dropped), emoticons as placeholder chars."""
    kept = [m for m in msgs if m.strip() not in PH_LINES and m.strip() not in PH]
    return emoticon_placeholders("\n".join(kept)) if kept else ""


class CopyChecker:
    """Verbatim-copy test: any common substring of >= L characters with a training target (PIPELINE_SPEC §5.3)."""

    def __init__(self, train_texts, L: int = C.DPO["copy_lcs"]):
        self.L = L
        self.sub = set()
        for t in train_texts:
            t = t.replace("\n", " ")
            for i in range(0, max(0, len(t) - L + 1)):
                self.sub.add(t[i:i + L])

    def is_copy(self, text: str) -> bool:
        t = text.replace("\n", " ")
        return any(t[i:i + self.L] in self.sub for i in range(0, max(0, len(t) - self.L + 1)))


_OPENCC = {}


def leak_chars(text: str):
    """Number of characters OpenCC s2twp (simplified / Mainland terms -> Taiwan) would change; None if OpenCC is
    not installed. Counted with a difflib alignment, so a conversion that changes the length (e.g. 软件 -> 軟體 is
    same length, but 鼠标 -> 滑鼠 vs. terms of different length) does not shift every later character."""
    if "cc" not in _OPENCC:
        try:
            import opencc
            _OPENCC["cc"] = opencc.OpenCC("s2twp")
        except Exception:
            _OPENCC["cc"] = None
    cc = _OPENCC["cc"]
    if cc is None:
        return None
    conv = cc.convert(text)
    if conv == text:
        return 0
    import difflib
    changed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, text, conv, autojunk=False).get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed
