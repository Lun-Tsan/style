"""Minimal test runner (no pytest in the venv): python tests/run_tests.py [--fast] [--synth] [name_substring]

--synth   run on the synthetic export of persona/synth.py (PERSONA_SYNTH=1) instead of the private chat
A module whose import fails only because torch / transformers / peft is missing is skipped, not failed.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

MODULES = ["tests.test_synth", "tests.test_data", "tests.test_prompt", "tests.test_rulers", "tests.test_mask",
           "tests.test_eval", "tests.test_train", "tests.test_human"]
OPTIONAL = {"torch", "transformers", "peft"}     # GPU-side dependencies, absent e.g. in a cloud container


def main(argv):
    fast = "--fast" in argv
    if "--synth" in argv:
        os.environ["PERSONA_SYNTH"] = "1"         # before persona.config is imported
    pat = [a for a in argv if not a.startswith("--")]
    n_ok = n_fail = n_skip = 0
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name not in OPTIONAL:
                raise
            print(f"SKIP {modname} (no {e.name})")
            n_skip += 1
            continue
        for name in dir(mod):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if pat and not any(p in name for p in pat):
                continue
            if fast and getattr(fn, "slow", False):
                n_skip += 1
                continue
            t0 = time.perf_counter()
            try:
                fn()
                print(f"PASS {modname}.{name} ({time.perf_counter() - t0:.1f}s)")
                n_ok += 1
            except ModuleNotFoundError as e:
                if e.name not in OPTIONAL:
                    raise
                print(f"SKIP {modname}.{name} (no {e.name})")
                n_skip += 1
            except Exception:
                print(f"FAIL {modname}.{name}")
                traceback.print_exc()
                n_fail += 1
    print(f"\n{n_ok} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
