"""Load addon.py's definitions (ACK_RE, PUNCT, core, boot_pair_ci) verbatim WITHOUT running its top-level experiment code."""
import ast
import os
import re
import random
import numpy as np
from collections import Counter

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "addon.py")


def load():
    src = open(_SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("core", "boot_pair_ci"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("ACK_RE", "PUNCT") for t in node.targets):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"re": re, "np": np, "random": random, "Counter": Counter}
    exec(compile(mod, _SRC, "exec"), ns)
    return ns["ACK_RE"], ns["PUNCT"], ns["core"], ns["boot_pair_ci"]
