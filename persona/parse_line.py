"""persona.parse_line -- LINE text export -> messages.

Format (validated on chathistory.txt; reproduces the turn counts of experiments 1-4 exactly):
  'YYYY.MM.DD 星期X'                 date line
  'HH:MM<space>speaker<space>text'   new message; speaker names may contain spaces, so they are matched by prefix
  'HH:MM<space>other'                system line (e.g. unsent-message notice) -> dropped and counted
  any other line                     continuation of the previous message (joined with '\\n'; blank lines kept)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config as C
from .normalize import PH

DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2}) 星期[一二三四五六日]$")
TIME_RE = re.compile(r"^(\d{1,2}:\d{2}) (.*)$")
KIND = {"貼圖": "sticker", "圖片": "image", "影片": "video"}


@dataclass
class Msg:
    date: str      # YYYY-MM-DD
    time: str      # HH:MM
    who: str
    text: str

    @property
    def kind(self) -> str:
        return KIND.get(self.text.strip(), "text")

    @property
    def is_placeholder(self) -> bool:
        return self.text.strip() in PH


def read_lines(path: Path = C.CHAT_PATH):
    return Path(path).read_text(encoding="utf-8-sig").splitlines()


def detect_speakers(lines, speakers=C.SPEAKERS, min_count=100):
    """Which configured speakers occur at least min_count times as a message prefix."""
    c = Counter()
    for line in lines:
        m = TIME_RE.match(line)
        if m:
            for sp in speakers:
                if m.group(2).startswith(sp + " ") or m.group(2) == sp:
                    c[sp] += 1
    return [sp for sp, n in c.most_common() if n >= min_count]


def parse(lines, speakers):
    """Return (messages, number of system lines dropped)."""
    msgs, n_sys, date, cur = [], 0, None, None
    sps = sorted(speakers, key=len, reverse=True)
    for line in lines:
        d = DATE_RE.match(line)
        if d:
            date = f"{d.group(1)}-{d.group(2)}-{d.group(3)}"
            cur = None
            continue
        m = TIME_RE.match(line)
        if m:
            rest = m.group(2)
            for sp in sps:
                if rest.startswith(sp + " ") or rest == sp:
                    cur = Msg(date, m.group(1), sp, rest[len(sp) + 1:])
                    msgs.append(cur)
                    break
            else:
                n_sys += 1
                cur = None          # continuation lines after a system line are dropped
            continue
        if cur is not None:
            cur.text += "\n" + line
    return msgs, n_sys


def load(path: Path = C.CHAT_PATH):
    if C.SYNTH:
        from .synth import write
        write(path)                 # deterministic, so regenerating each time never leaves a stale export
    elif not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; without the private export use PERSONA_SYNTH=1 "
                                f"(tests: python tests/run_tests.py --synth)")
    lines = read_lines(path)
    found = detect_speakers(lines)
    if set(found) != set(C.SPEAKERS):
        raise ValueError(f"speakers found {found} differ from config {C.SPEAKERS}")
    return parse(lines, found)
