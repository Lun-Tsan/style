"""Shim for the missing read-only module analyze_chat (Desktop/tmp/qwen_chat_analysis/analyze_chat.py is no longer on disk).

Re-implements detect_speakers(lines) and parse(lines, speakers) for this LINE export format:
  'YYYY.MM.DD 星期X'      -> date line
  'HH:MM <speaker> <text>' -> new message
  'HH:MM <other>'          -> system line (counted in n_sys, e.g. unsent-message notices)
  any other line           -> continuation of the previous message (joined with '\n'; blank lines kept)
Install with formal_chatshim.install() BEFORE `import style_nb`. Validated by reproducing style_nb's known turn
counts and NB baselines exactly (see formal_shimcheck.py).
"""
import re
import sys
import types
from collections import Counter
from dataclasses import dataclass

from _local import S, T  # noqa: E402  (local, git-ignored)
DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2}) 星期[一二三四五六日]$")
TIME_RE = re.compile(r"^(\d{1,2}:\d{2}) (.*)$")
SYS_CONT = "drop"   # continuation lines after a system line: 'drop' or 'prev'


@dataclass
class Msg:
    date: str
    time: str
    who: str
    text: str


def detect_speakers(lines, min_count=100):
    c = Counter()
    for l in lines:
        m = TIME_RE.match(l)
        if m:
            for sp in (S, T):
                if m.group(2).startswith(sp + " ") or m.group(2) == sp:
                    c[sp] += 1
    return [sp for sp, n in c.most_common() if n >= min_count]


def parse(lines, speakers):
    msgs, n_sys, date, cur = [], 0, None, None
    sps = sorted(speakers, key=len, reverse=True)
    for l in lines:
        d = DATE_RE.match(l)
        if d:
            date = f"{d.group(1)}-{d.group(2)}-{d.group(3)}"
            cur = None
            continue
        m = TIME_RE.match(l)
        if m:
            rest = m.group(2)
            for sp in sps:
                if rest.startswith(sp + " ") or rest == sp:
                    cur = Msg(date, m.group(1), sp, rest[len(sp) + 1:])
                    msgs.append(cur)
                    break
            else:
                n_sys += 1
                cur = None if SYS_CONT == "drop" else cur
            continue
        if cur is not None:
            cur.text += "\n" + l
    return msgs, n_sys


def install():
    mod = types.ModuleType("analyze_chat")
    mod.detect_speakers = detect_speakers
    mod.parse = parse
    mod.Msg = Msg
    sys.modules["analyze_chat"] = mod
    return mod
