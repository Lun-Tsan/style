"""persona.flag_symbols -- symbol/code-like turn rule (experiment 4: the loose rule validated best).

A turn is symbol-like if its ruler_text (text_b with emoticons already replaced by placeholder characters)
contains any of  ; { } ( ) = < > # [ ] _ \\ / * + & | % ^ $ @ `  , a URL scheme, or a programming keyword.
Flagged turns: never targets, never in ruler fitting/scoring; in contexts they become CODE_PH.
"""
from __future__ import annotations

import re

CODE_RE = re.compile(r"[;{}()=<>#\[\]_\\/*+&|%^$@`]|https?:|\b(cout|cin|int|endl|return|include|for|while|if|else|"
                     r"main|vector|string|char|bool|std|void|long|double|printf|scanf)\b", re.I)
DIGIT_RE = re.compile(r"[0-9０-９]")
CODE_PH = "[程式碼/算式]"


def is_symbol_turn(ruler_text: str | None) -> bool:
    return bool(ruler_text) and bool(CODE_RE.search(ruler_text))


def flag_raw(text_b: str | None) -> bool:
    """Experiment-4 loose rule on raw text_b (no emoticon whitelist); kept for reproduction only."""
    return bool(text_b) and bool(CODE_RE.search(text_b))
