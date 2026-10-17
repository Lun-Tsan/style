"""persona.normalize -- placeholders (stickers, emoticons) and de-identification.

Three text views of a turn exist downstream:
  * raw messages        -> prompts (after deid); what the generator sees and produces
  * text_a / text_b     -> as in the validated experiments (placeholder char / placeholder-only messages removed)
  * ruler_text          -> text_b with emoticons replaced by one private-use char each; the symbol rule and every
                           ruler operate on this string, so an emoticon is never mistaken for code
"""
from __future__ import annotations

import re
from collections import Counter

# whole-message placeholders of the LINE export (a message whose body is exactly this word)
PH = {"貼圖": "", "圖片": "", "影片": ""}
PH_SHOW = {v: f"[{k}]" for k, v in PH.items()}          # private-use char -> display
PH_LINE = {k: f"[{k}]" for k in PH}                      # placeholder message -> prompt line

# emoticons: LINE emoticon codes as exported in text, e.g. (cry) (wave); and ASCII emoticons.
# Observed in the training data (counts over all turns, 2026-10-13): (cry) 8, (emoji) 8, (wave) 3, (wow) 3,
# (please) 2, (sad) 1, (tear) 1, (upset) 1; QQ 6.  The list below adds common codes that may appear later.
EMO_LINE_RE = re.compile(r"\((?:cry|emoji|wave|wow|please|sad|tear|upset|smile|laugh|angry|love|heart|shy|sweat|"
                         r"sleepy|cool|blush|kiss|hug|wink|grin|sob|yawn)\)")
EMO_ASCII_RE = re.compile(r"\^\^|\^_\^|\^o\^|QAQ|QQ|Q_Q|T_T|XDD*|xdd*|:\)|:\(|:D|:P|;\)|= =|orz|Orz|<3")
EMO_PH_LINE, EMO_PH_ASCII = "", ""
EMO_SHOW = {EMO_PH_LINE: "(表情)", EMO_PH_ASCII: "(顏文字)"}


def emoticon_placeholders(text: str) -> str:
    """Replace every emoticon by a single private-use character (LINE code -> U+E010, ASCII -> U+E011)."""
    return EMO_ASCII_RE.sub(EMO_PH_ASCII, EMO_LINE_RE.sub(EMO_PH_LINE, text))


def show(text: str) -> str:
    """Display form: private-use characters back to readable tags (for reports; never prints raw messages)."""
    for k, v in {**PH_SHOW, **EMO_SHOW}.items():
        text = text.replace(k, v)
    return text


# de-identification (Taiwan-specific patterns + e-mail + URL). Counts on the training data, 2026-10-13:
# mobile 0, landline 1, national id 0, e-mail 4, URL 83 (URL turns are symbol-flagged anyway), address 0.
DEID_RULES = [
    ("url", re.compile(r"https?://\S+|www\.\S+"), "[網址]"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[信箱]"),
    ("mobile", re.compile(r"09\d{2}-?\d{3}-?\d{3}"), "[電話]"),
    ("landline", re.compile(r"(?<!\d)0\d{1,2}-?\d{6,8}(?!\d)"), "[電話]"),
    ("twid", re.compile(r"(?<![A-Z0-9])[A-Z][12]\d{8}(?![0-9])"), "[證號]"),
    ("address", re.compile(r"[一-鿿]{1,8}(?:路|街|大道)[一-鿿\d段巷弄之]*\d+號"), "[地址]"),
]


def deid(text: str):
    """Return (text with identifiers replaced, Counter of rule hits)."""
    hits = Counter()
    for name, rx, rep in DEID_RULES:
        text, n = rx.subn(rep, text)
        if n:
            hits[name] += n
    return text, hits
