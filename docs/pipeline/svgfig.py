"""Tiny declarative SVG drawing helper for the pipeline figures (no dependencies).

Boxes sit on a 20 px grid; arrows attach to box edges and get a labelled elbow; text widths are estimated
(CJK 1.0 em, ASCII 0.55 em) so label backgrounds and box heights fit. Standalone files: explicit light colours,
so they read the same in a browser, on GitHub and in a paper draft.
"""
from __future__ import annotations

import html
import math
import re

FONT = "Noto Sans TC, PingFang TC, Microsoft JhengHei, WenQuanYi Zen Hei, Helvetica, Arial, sans-serif"
MONO = "JetBrains Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"

INK, MUTED, BORDER, BG = "#1f2937", "#6b7280", "#9ca3af", "#ffffff"
FILL = {
    "step": "#f8fafc", "artifact": "#f3f4f6", "model": "#ede9fe", "note": "#ffffff", "group": "none",
}
STATUS = {                                     # fill, stroke
    "done": ("#e8f5e9", "#2e7d32"),            # ran on the real data, numbers in reports/
    "todo": ("#fff8e1", "#b45309"),            # code exists, formal run not done
    "plan": ("#ffffff", "#9ca3af"),            # planned only
}
RED = "#c62828"
BLUE = "#1d4ed8"


def est_width(s: str, size: float) -> float:
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o > 0x2E7F:                       # CJK, full-width punctuation
            w += 1.0
        elif ch in "mwMW@%":
            w += 0.8
        elif ch in "il.,:;'| ()[]":
            w += 0.32
        else:
            w += 0.56
    return w * size


class Fig:
    def __init__(self, w: int, h: int, title: str, caption: str = ""):
        self.w, self.h, self.title, self.caption = w, h, title, caption
        self.parts: list[str] = []
        self.boxes: dict[str, tuple[float, float, float, float]] = {}
        self._legend = None

    # ---------------------------------------------------------------- primitives
    def text(self, x, y, s, size=12, anchor="start", weight="normal", color=INK, mono=False, italic=False):
        fam = MONO if mono else FONT
        st = f'font-family="{fam}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}"'
        if italic:
            st += ' font-style="italic"'
        self.parts.append(f'<text x="{x:.0f}" y="{y:.0f}" {st}>{html.escape(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke=BORDER, rx=6, dash=None, sw=1.2):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, pts, stroke=INK, sw=1.4, dash=None, arrow=True, color_id=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#arrow{color_id or ""})"' if arrow else ""
        p = " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{sw}"{d}{mk}/>')

    # ---------------------------------------------------------------- boxes
    def box(self, key, x, y, w, h, title, lines=(), kind="step", status=None, mono_lines=True, title_size=13,
            line_size=11, dash=None, private=False, flags=(), tag=None):
        fill = FILL[kind]
        stroke = BORDER
        if status:
            fill, stroke = STATUS[status]
        if kind == "artifact":
            # document shape: a folded corner drawn as a small triangle
            self.rect(x, y, w, h, fill, stroke, rx=3, dash=dash or ("4,3" if private else None))
            self.parts.append(f'<polygon points="{x + w - 14:.0f},{y:.0f} {x + w:.0f},{y + 14:.0f} {x + w - 14:.0f},{y + 14:.0f}" '
                              f'fill="{BG}" stroke="{stroke}" stroke-width="1"/>')
        else:
            self.rect(x, y, w, h, fill, stroke, rx=8 if kind == "step" else 4, dash=dash, sw=1.6 if kind == "model" else 1.2)
        self.boxes[key] = (x, y, w, h)
        ty = y + 19
        self.text(x + 10, ty, title, size=title_size, weight="bold")
        if tag:
            self.text(x + w - 10, ty, tag, size=10, anchor="end", color=MUTED)
        if private:
            self.text(x + w - 20, y + h - 7, "私有", size=9, anchor="end", color=RED)
        for s in lines:
            ty += line_size + 4
            mono = mono_lines and not s.startswith("·") and not s.startswith("→")
            self.text(x + 10, ty, s, size=line_size, mono=mono, color=INK if mono else MUTED)
        for i, f in enumerate(flags):
            cx, cy = x + w - 12 - 26 * i, y - 1
            self.parts.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="10" fill="{RED}"/>')
            self.text(cx, cy + 3.5, f, size=9, anchor="middle", color="#ffffff", weight="bold")
        return key

    def group(self, x, y, w, h, title, color=MUTED, dash="6,4"):
        self.rect(x, y, w, h, "none", color, rx=10, dash=dash, sw=1.2)
        self.text(x + 12, y + 16, title, size=11, color=color, weight="bold")

    # ---------------------------------------------------------------- arrows
    def _anchor(self, key, side):
        x, y, w, h = self.boxes[key]
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[side]

    def arrow(self, a, b, label=None, sides=None, dash=None, color=INK, label_dy=-6, via=None, label_at=0.5,
              shift=0.0, label_side="auto"):
        """Elbow arrow from box a to box b. sides = (from_side, to_side) or auto from relative position."""
        ax, ay, aw, ah = self.boxes[a]
        bx, by, bw, bh = self.boxes[b]
        if sides is None:
            if bx >= ax + aw:
                sides = ("r", "l")
            elif bx + bw <= ax:
                sides = ("l", "r")
            elif by >= ay + ah:
                sides = ("b", "t")
            else:
                sides = ("t", "b")
        p0 = list(self._anchor(a, sides[0]))
        p1 = list(self._anchor(b, sides[1]))
        if sides[0] in "tb":
            p0[0] += shift
        else:
            p0[1] += shift
        if sides[1] in "tb":
            p1[0] += shift
        else:
            p1[1] += shift
        if via is not None:
            pts = [tuple(p0)] + list(via) + [tuple(p1)]
        elif sides[0] in "lr" and sides[1] in "lr":
            mx = (p0[0] + p1[0]) / 2
            pts = [tuple(p0), (mx, p0[1]), (mx, p1[1]), tuple(p1)] if abs(p0[1] - p1[1]) > 1 else [tuple(p0), tuple(p1)]
        elif sides[0] in "tb" and sides[1] in "tb":
            my = (p0[1] + p1[1]) / 2
            pts = [tuple(p0), (p0[0], my), (p1[0], my), tuple(p1)] if abs(p0[0] - p1[0]) > 1 else [tuple(p0), tuple(p1)]
        elif sides[0] in "lr":
            pts = [tuple(p0), (p1[0], p0[1]), tuple(p1)]
        else:
            pts = [tuple(p0), (p0[0], p1[1]), tuple(p1)]
        cid = "red" if color == RED else "blue" if color == BLUE else "muted" if color == MUTED else None
        self.line(pts, stroke=color, dash=dash, color_id=cid)
        if label:
            # label at a fraction of the polyline length
            seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            tot = sum(seg) or 1
            want = tot * label_at
            acc = 0.0
            for i, s in enumerate(seg):
                if acc + s >= want:
                    t = (want - acc) / s if s else 0
                    lx = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t
                    ly = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t
                    vertical = abs(pts[i + 1][0] - pts[i][0]) < 1
                    break
                acc += s
            size = 10.5
            tw = est_width(label, size) + 8
            if vertical or label_side in ("r", "l"):
                if label_side == "l":
                    lx -= 6 + tw / 2
                else:
                    lx += 6 + tw / 2
                ly += 4
            else:
                ly += label_dy
            self.rect(lx - tw / 2, ly - size + 1, tw, size + 5, BG, "none", rx=2)
            self.text(lx, ly, label, size=size, anchor="middle", color=color if color != INK else MUTED)

    # ---------------------------------------------------------------- legend / output
    def legend(self, x, y, items, cols=None):
        """Legend items: (kind, label), kind in STATUS keys, 'artifact', 'model', 'private', 'flag', 'dash', 'blue'.
        Drawn at render time, in one row laid out by label width, just above the caption (y is ignored; kept for
        call-site compatibility)."""
        self._legend = (x, items)

    def _draw_legend(self, x, y, items):
        for kind, label in items:
            cx, cy = x, y
            if kind in STATUS:
                f, s = STATUS[kind]
                self.rect(cx, cy, 26, 14, f, s, rx=3)
            elif kind == "artifact":
                self.rect(cx, cy, 26, 14, FILL["artifact"], BORDER, rx=2)
            elif kind == "model":
                self.rect(cx, cy, 26, 14, FILL["model"], BORDER, rx=2, sw=1.6)
            elif kind == "private":
                self.rect(cx, cy, 26, 14, FILL["artifact"], BORDER, rx=2, dash="4,3")
            elif kind == "flag":
                self.parts.append(f'<circle cx="{cx + 13}" cy="{cy + 7}" r="8" fill="{RED}"/>')
                self.text(cx + 13, cy + 10.5, "#", size=9, anchor="middle", color="#ffffff", weight="bold")
            elif kind == "dash":
                self.line([(cx, cy + 7), (cx + 26, cy + 7)], stroke=MUTED, dash="5,4", arrow=False)
            elif kind == "blue":
                self.line([(cx, cy + 7), (cx + 26, cy + 7)], stroke=BLUE, arrow=False)
            self.text(cx + 34, cy + 11, label, size=10.5, color=MUTED)
            x += 34 + est_width(label, 10.5) + 40

    def render(self) -> str:
        defs = "".join(
            f'<marker id="arrow{sfx}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>'
            for sfx, col in (("", INK), ("red", RED), ("blue", BLUE), ("muted", MUTED)))
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-label="{html.escape(self.title)}">'
                f'<title>{html.escape(self.title)}</title><defs>{defs}</defs>'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{BG}"/>')
        title = (f'<text x="20" y="30" font-family="{FONT}" font-size="17" font-weight="bold" fill="{INK}">{html.escape(self.title)}</text>')
        cap, lines = "", []
        if self.caption:                      # one sentence per line, so a long caption never runs off the canvas
            sents = [t for t in re.split(r"(?<=。)", self.caption) if t]
            cur = ""
            for t in sents:
                if cur and est_width(cur + t, 10.5) > self.w - 40:
                    lines.append(cur)
                    cur = t
                else:
                    cur += t
            lines.append(cur)
            y = self.h - 14 - 15 * (len(lines) - 1)
            cap = "".join(f'<text x="20" y="{y + 15 * i}" font-family="{FONT}" font-size="10.5" fill="{MUTED}">{html.escape(l)}</text>'
                          for i, l in enumerate(lines))
        if self._legend:                      # one row, just above the caption
            self._draw_legend(self._legend[0], self.h - 14 - 15 * len(lines) - 30, self._legend[1])
        return head + title + "".join(self.parts) + cap + "</svg>\n"
