"""Import graph of persona/ (computed from the source, so the figure follows the code).

python docs/pipeline/depgraph.py            # prints layers and edges
Used by make_figures.py for 07_dependencies.svg. Only `from . import x` / `from .x import y` / `from ..x import y`
forms are recognised (the package uses no absolute self-imports); lazy imports inside functions count as edges
too, drawn dashed, because they still tie the modules together at run time.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "persona"
EXTERNAL = ("torch", "transformers", "peft", "opencc")   # GPU-side / optional dependencies worth showing

IMPORT_RE = re.compile(r"^(?P<indent>[ \t]*)from (?P<dots>\.+)(?P<mod>[\w.]*) import (?P<names>[^\n]+)", re.M)
EXT_RE = re.compile(r"^[ \t]*(?:import|from) (" + "|".join(EXTERNAL) + r")\b", re.M)


def modules():
    out = {}
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        name = ".".join(p.relative_to(PKG).with_suffix("").parts)
        src = p.read_text(encoding="utf-8")
        edges = {}                                   # target -> "top" | "lazy"
        for m in IMPORT_RE.finditer(src):
            pkg = p.parent
            for _ in range(len(m["dots"]) - 1):
                pkg = pkg.parent
            lazy = bool(m["indent"])
            if m["mod"]:
                tgt = pkg / m["mod"].replace(".", "/")
                targets = [tgt]
            else:
                targets = [pkg / n.strip().split(" as ")[0].strip() for n in m["names"].split(",")]
            for t in targets:
                if (t.with_suffix(".py")).exists():
                    key = ".".join(t.relative_to(PKG).parts)
                elif (t / "__init__.py").exists():
                    key = ".".join(t.relative_to(PKG).parts)
                else:
                    continue
                if key == name:
                    continue
                edges[key] = "lazy" if (lazy and edges.get(key) != "top") else "top"
        ext = sorted({m.group(1) for m in EXT_RE.finditer(src)})
        ext_top = sorted({m.group(1) for m in EXT_RE.finditer(src) if not m.group(0)[0].isspace()})
        out[name] = {"edges": edges, "ext": ext, "ext_top": ext_top, "lines": src.count("\n")}
    return out


def layers(mods, include_lazy=True):
    """Longest path from a leaf (module importing nothing internal) -> layer index. Lazy imports count too by
    default, so a module drawn above another is one that needs it at some point; cycles are cut at the first repeat."""
    top = {m: [t for t, k in d["edges"].items() if (include_lazy or k == "top") and t in mods] for m, d in mods.items()}
    memo = {}

    def depth(m, seen=()):
        if m in memo:
            return memo[m]
        if m in seen:
            return 0
        memo[m] = 1 + max([depth(t, seen + (m,)) for t in top[m]], default=-1)
        return memo[m]

    return {m: depth(m) for m in mods}


def torch_closure(mods):
    """Modules that import torch (or transformers/peft) at import time, directly or through a top-level import."""
    direct = {m for m, d in mods.items() if any(e in ("torch", "transformers", "peft") for e in d["ext_top"])}
    changed = True
    while changed:
        changed = False
        for m, d in mods.items():
            if m not in direct and any(t in direct for t, k in d["edges"].items() if k == "top"):
                direct.add(m)
                changed = True
    return direct


if __name__ == "__main__":
    mods = modules()
    lay = layers(mods)
    tc = torch_closure(mods)
    for L in sorted(set(lay.values())):
        print(f"layer {L}:", ", ".join(sorted(m for m in mods if lay[m] == L)))
    print("torch at import time:", sorted(tc))
    for m, d in sorted(mods.items()):
        for t, k in sorted(d["edges"].items()):
            print(f"  {m} -> {t} ({k})")
