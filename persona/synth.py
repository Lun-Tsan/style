"""persona.synth -- synthetic two-person LINE export, for tests and smoke runs without the private chat.

Every message is assembled from the phrase lists below, written for this file; nothing is copied from a real chat.
The two fake people differ in the directions the interpretable features take on the real pair (stage-1 reports):
T bursts more, ends messages with particles and uses full-width marks; S sends more stickers and asks more
questions ending in 嗎 with half-width marks. Both draw from one topic vocabulary and one set of short acks, so a
single turn is hard to attribute while a bag of turns is easy -- the shape the rulers were validated on.

    python -m persona.synth [--out PATH] [--seed 0]   # export + <out>.truth.json
    PERSONA_SYNTH=1 python -m persona.stage1 --quick   # stage 1 on the synthetic export (outputs under runs/synth/)
    python tests/run_tests.py --synth                  # data-layer and ruler tests on the synthetic export

generate() also returns the counts the parser and the turn/segment rules must reproduce (messages, system lines,
turns, segments, text turns per split and speaker). They are known by construction -- turns are built first and
then timed so that the rules of persona.turns hold -- so the parser is checked against an independent count.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

from . import config as C

WEEKDAY = "一二三四五六日"
START = dt.datetime(2025, 6, 20, 15, 0)
END = dt.date(2026, 9, 19)
SEG_GAP_MIN = 7 * 60          # > SEG_GAP_H, so consecutive segments never merge
SEG_GAP_MEAN = 36 * 60        # mean extra gap between segments (exponential) -> ~230 segments over the period

NOUN = ["這題", "第三題", "第二小題", "作業", "講義", "考卷", "這個式子", "迴圈", "陣列", "函數", "遞迴", "指標",
        "二次函數", "機率", "向量", "斜率", "這行", "答案", "筆記", "段考範圍", "上次那題", "錯的地方"]
VERB = ["看一下", "寫完", "算一次", "跑一次", "改一下", "想一下", "再試試", "整理一下", "對答案", "畫圖", "代進去", "展開"]
WHEN = ["明天", "後天", "禮拜六", "禮拜天", "今天晚上", "下禮拜"]
CLOCK = ["7點", "8點", "8點半", "9點", "下午3點", "晚上10點"]
ACK_BOTH = ["對", "好", "OK", "可以", "沒問題"]

ACK_T = ["好啊", "好喔", "對啊", "可以啊", "沒錯", "讚喔", "讚", "好啊好啊"]
COMMENT_T = ["不用急", "慢慢來", "差一點點", "這邊很常錯", "方向是對的", "這樣就對了", "快好了", "寫得不錯",
             "這個要記起來", "我看一下", "我等等看", "我晚點回你", "我剛剛看到", "我也是這樣想", "先休息一下"]
LAUGH_T = ["哈哈哈", "哈哈哈哈", "XD", "笑死", "哈哈"]
PARTICLE_T = "喔啊啦耶欸齁吧嘛"
EMO_T = ["XD", "(wave)", "(wow)", "^^"]

ACK_S = ["嗯嗯", "好的", "我懂了", "了解", "收到", "好", "嗯"]
REPORT_S = ["我寫完了", "我不會", "我卡住了", "我懂了", "我看不懂", "我再試試看", "我算出來了", "我傳給你", "我晚點寫",
            "我忘記了", "{n}我寫完了", "{n}我不太懂", "我覺得{n}怪怪的"]
COMMENT_S = ["好難", "原來如此", "真的假的", "等我一下", "我先去吃飯", "今天好累", "快好了", "我等等傳"]
LAUGH_S = ["哈哈", "XDD", "哈哈哈"]
EMO_S = ["(cry)", "(please)", "QQ", "(sad)", "(tear)"]

CODE = ["for (int i = 0; i < n; i++)", "cout << ans << endl;", "f(x) = 2x + 1", "a[i] = a[i-1] + 2",
        "x^2 - 5x + 6 = 0", "if (n % 2 == 0) return 1;", "int main() {\n    int n;\n    cin >> n;\n}",
        "vector<int> v(n);", "sum += a[i];", "y = 3x - 4", "(a+b)^2 = a^2 + 2ab + b^2", "while (l < r)"]
CODE_TALK = ["第{d}行改成 i <= n", "{d}/{m} 晚上見", "答案是 x = {d}", "https://example.com/hw{d}",
             "講義在 https://example.com/notes", "這行 = 改成 ==", "第{d}題是 (b)"]
FAKE_ID = ["我的信箱 tutor.lin@example.com", "有事打 0900-000-000"]   # de-identification hits, obviously fake


def _p(r, prob):
    return r.random() < prob


INSTR = ["{n}先{v}", "你先{v}", "{n}要先{v}", "再{v}就好", "{n}{v}看看", "記得{v}", "先把{n}{v}"]
ASK = ["{n}寫到哪了", "有看到{n}", "{n}會了", "要不要{v}", "還有問題", "{n}懂了", "{n}要怎麼{v}", "{n}可以{v}嗎",
       "{w}有上課嗎", "{n}是什麼意思", "這樣對嗎", "可以再講一次嗎", "{n}要交嗎", "{c}可以嗎", "{n}有範圍嗎"]
ASK_T, ASK_S = ASK[:6], ASK[6:]

# per role: share of each message kind, how often the role picks from its own phrase list (the rest from the
# other's), and marker rates. Content lists are shared, so most of the difference is in rates, not in words.
STYLE_TEXT = {
    "T": dict(kinds=dict(ack=0.24, instr=0.24, comment=0.18, ask=0.14, report=0.06, sched=0.09, laugh=0.05),
              own=dict(ack=0.95, comment=0.7, ask=0.7, laugh=0.8), ma=0.08, particle=0.45, tilde=0.12, punct=0.28,
              full=0.75, joint=0.12, emo=0.03),
    "S": dict(kinds=dict(ack=0.24, instr=0.08, comment=0.15, ask=0.27, report=0.17, sched=0.05, laugh=0.04),
              own=dict(ack=0.95, comment=0.7, ask=0.7, laugh=0.8), ma=0.35, particle=0.12, tilde=0.04, punct=0.28,
              full=0.3, joint=0.12, emo=0.06),
}
OWN = {"T": dict(ack=ACK_T, comment=COMMENT_T, ask=ASK_T, laugh=LAUGH_T, emo=EMO_T),
       "S": dict(ack=ACK_S, comment=COMMENT_S, ask=ASK_S, laugh=LAUGH_S, emo=EMO_S)}


def _pick(r, role, kind):
    """A phrase from the role's own list with probability own[kind], otherwise from the other role's list."""
    other = "S" if role == "T" else "T"
    return r.choice(OWN[role if _p(r, STYLE_TEXT[role]["own"][kind]) else other][kind])


def _punct(r, st, full, half):
    return full if _p(r, st["full"]) else half


def _text(r, role):
    st = STYLE_TEXT[role]
    kind = r.choices(list(st["kinds"]), weights=list(st["kinds"].values()))[0]
    if kind == "ack":
        s = _pick(r, role, "ack") if _p(r, 0.95) else r.choice(ACK_BOTH)
        return s + ("～" if _p(r, st["tilde"]) else _punct(r, st, "！", "!") if _p(r, 0.08) else "")
    if kind == "laugh":
        return _pick(r, role, "laugh")
    if kind == "ask":
        s = _fill(r, _pick(r, role, "ask"))
        if not s.endswith("嗎") and _p(r, st["ma"]):
            s += "嗎"
        return s + (_punct(r, st, "？", "?") if _p(r, 0.65) else "") + (r.choice(OWN[role]["emo"]) if _p(r, st["emo"]) else "")
    if kind == "instr":
        s = _fill(r, r.choice(INSTR))
    elif kind == "comment":
        s = _pick(r, role, "comment")
    elif kind == "report":
        s = _fill(r, r.choice(REPORT_S))
    else:
        s = f"{r.choice(WHEN)}{r.choice(CLOCK)}" + r.choice(["上課", "可以", "見", ""])
    if _p(r, st["joint"]):
        s += _punct(r, st, "，", ",") + _pick(r, role, "comment")
    if _p(r, st["particle"]):
        s += r.choice(PARTICLE_T)
    if _p(r, st["tilde"]):
        s += "～"
    elif _p(r, st["punct"]):
        s += _punct(r, st, r.choice("。！…"), r.choice(".!"))
    if _p(r, st["emo"]):
        s += r.choice(OWN[role]["emo"])
    return s


def _fill(r, s):
    return s.format(n=r.choice(NOUN), v=r.choice(VERB), w=r.choice(WHEN), c=r.choice(CLOCK),
                    d=r.randint(1, 12), m=r.randint(1, 12))


STYLE = {  # per role: burst continuation prob, sticker/image per message, sticker-only turn, symbol turn
    "T": dict(burst=0.58, sticker=0.04, image=0.01, sticker_turn=0.07, symbol_turn=0.06),
    "S": dict(burst=0.38, sticker=0.2, image=0.03, sticker_turn=0.15, symbol_turn=0.10),
}


def _turn_msgs(r, role):
    st = STYLE[role]
    if _p(r, st["sticker_turn"]):
        return ["貼圖"] * (2 if _p(r, 0.2) else 1)
    n = 1
    while n < 7 and _p(r, st["burst"]):
        n += 1
    if _p(r, st["symbol_turn"]):
        out = [r.choice(CODE) if _p(r, 0.6) else _fill(r, r.choice(CODE_TALK))]
        return out + [_text(r, role) for _ in range(n - 1)]
    out = []
    for _ in range(n):
        if _p(r, st["sticker"]):
            out.append("貼圖")
        elif _p(r, st["image"]):
            out.append("影片" if _p(r, 0.1) else "圖片")
        elif _p(r, 0.004):
            out.append(r.choice(FAKE_ID))
        elif _p(r, 0.02):
            out.append(_text(r, role) + "\n" + _text(r, role))          # one message over two lines
        else:
            out.append(_text(r, role))
    return out


def _abs(t: dt.datetime) -> int:
    return t.toordinal() * 1440 + t.hour * 60 + t.minute


def _segment(r, t0):
    """One segment: list of (role, [(abs_min, text), ...]) with in-segment gaps well below SEG_GAP_H."""
    n = max(1, round(r.lognormvariate(2.15, 0.7)))
    role = "S" if _p(r, 0.6) else "T"
    out, t = [], t0
    for i in range(n):
        if i:
            same = _p(r, 0.07)
            role = role if same else ("T" if role == "S" else "S")
            u = r.random()
            gap = r.randint(0, 5) if u < 0.7 else r.randint(6, 40) if u < 0.9 else r.randint(41, 150)
            if same:
                gap = max(gap, C.GAP_MIN + 1)                          # same speaker again -> a new turn
            t = out[-1][1][-1][0] + gap
        msgs = _turn_msgs(r, role)
        times = [t]
        for _ in msgs[1:]:
            times.append(times[-1] + (0 if _p(r, 0.6) else r.randint(1, 3)))
        day = times[0] // 1440
        k = sum(x // 1440 == day for x in times)
        out.append((role, list(zip(times[:k], msgs[:k]))))
        if k < len(msgs):                                              # crosses midnight -> two turns (new date)
            out.append((role, list(zip(times[k:], msgs[k:]))))
    return out


def generate(seed: int = 0, target: str | None = None, partner: str | None = None):
    """Return (export lines, truth). Speaker names default to the configured ones."""
    r = random.Random(seed)
    name = {"T": target or C.TARGET, "S": partner or C.PARTNER}
    end_abs = _abs(dt.datetime.combine(END, dt.time(23, 59)))
    segs, t0 = [], _abs(START)
    while True:
        sg = _segment(r, t0)
        if sg[-1][1][-1][0] > end_abs:
            break
        segs.append(sg)
        t0 = sg[-1][1][-1][0] + SEG_GAP_MIN + int(r.expovariate(1 / SEG_GAP_MEAN))
        if 60 <= t0 % 1440 < 9 * 60:                                   # nobody chats 01:00-09:00
            t0 = t0 - t0 % 1440 + 9 * 60 + r.randint(0, 600)

    lines = [f"[LINE] 與{name['S']}的聊天記錄", f"儲存日期:{END:%Y.%m.%d} 21:03", ""]
    cur = None
    n_msgs = n_sys = 0
    text_turns = {"train": [0, 0], "test": [0, 0]}
    for sg in segs:
        for role, msgs in sg:
            d = dt.date.fromordinal(msgs[0][0] // 1440)
            if any(m.strip() not in ("貼圖", "圖片", "影片") for _, m in msgs):
                text_turns["test" if d.isoformat() >= C.SPLIT_DATE else "train"][0 if role == "S" else 1] += 1
            for a, text in msgs:
                d = dt.date.fromordinal(a // 1440)
                if d != cur:
                    if cur is not None:
                        lines.append("")
                    lines.append(f"{d:%Y.%m.%d} 星期{WEEKDAY[d.weekday()]}")
                    cur = d
                hhmm = f"{a % 1440 // 60:02d}:{a % 60:02d}"
                if _p(r, 0.005):
                    lines.append(f"{hhmm} {name[r.choice('ST')]}已收回訊息")   # system line: no space after the name
                    n_sys += 1
                lines.extend(f"{hhmm} {name[role]} {text}".split("\n"))
                n_msgs += 1
    truth = {"seed": seed, "n_msgs": n_msgs, "n_sys": n_sys, "n_turns": sum(len(sg) for sg in segs),
             "n_segments": len(segs), "text_turns_S_T": {k: tuple(v) for k, v in text_turns.items()}}
    return lines, truth


def write(path: Path = None, seed: int = 0):
    """Write the export (UTF-8 with BOM, like a LINE export) and <path>.truth.json; return truth."""
    path = Path(path or C.CHAT_PATH)
    lines, truth = generate(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    path.with_suffix(".truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.PROJECT / "runs" / "synth" / "chathistory_synth.txt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(write(Path(a.out), a.seed))


if __name__ == "__main__":
    main()
