"""persona.config -- every fixed constant of pipeline v1 (PIPELINE_SPEC.md §0). Change numbers here, nowhere else."""
from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT / "reports"
RUNS = PROJECT / "runs"

# people and data location: real LINE display names live in persona/local_speakers.json (git-ignored):
#   {"target": "<display name of the person to imitate>", "partner": "<the other person>", "chat_path": "<export>"}
# Without that file the package loads with placeholder names and cannot parse a real export.
_LOCAL = Path(__file__).with_name("local_speakers.json")
_loc = json.load(open(_LOCAL, encoding="utf-8")) if _LOCAL.exists() else {}
CHAT_PATH = Path(_loc.get("chat_path", str(PROJECT / "chathistory.txt")))
TARGET = _loc.get("target", "T")      # T: the person to imitate
PARTNER = _loc.get("partner", "S")    # S: contrast author for the rulers (consent needed before publication)
SPEAKERS = (PARTNER, TARGET)
CODE = {TARGET: "T", PARTNER: "S"}    # role codes used in every report (never the display names)
ROLE = {TARGET: "我", PARTNER: "對方"}

# units, split, context ------------------------------------------------------------------------------------------
SPLIT_DATE = "2026-07-20"   # target turns dated >= this -> test
GAP_MIN = 30                # turn = same speaker, same day, gaps <= 30 min
SEG_GAP_H = 6.0             # segment = turns with gaps <= 6 h (ablation: 3 h)
DEV_FRAC = 0.15             # last 15% of the training period (by date, whole segments) -> dev
CTX_TURNS = 12              # context = up to 12 preceding turns of the same segment
CTX_TOKENS = 512            # prompt token budget (oldest context turns dropped first)
MAX_SEQ = 768
SUBSET_SIZES = (100, 250, 500, None)   # nested, time-uniform
N_BLOCKS = 5                # contiguous date blocks for out-of-fold calibration

# rulers ---------------------------------------------------------------------------------------------------------
VOCAB_N = 300               # char 1-3-gram vocabulary used by the distances
VOCAB_MAX = 1000            # count vectors are kept at this width (top-300 is a prefix)
Z_REF_BAGS = 200            # per person; z-reference for Burrows/Cosine Delta (two-author deviation)
Z_REF_K = 25
NCD_BAGS = 10
NCD_CHARS = 3000
K_ATTR = 10                 # planning figure: attribution bag size
K_VALID = 50                # planning figure: ruler-validation bag size
BAND_Q = (0.05, 0.95)       # feature tolerance band = this quantile range over T training bags of K_VALID turns
BAND_BAGS = 1000
FEATURE_MIN_AUC = 0.60      # features below this within-T vs cross AUC are reported but not weighted

# pre-registered validation thresholds (PIPELINE_SPEC.md §3.5, FORMAL_DISTANCE_EXPERIMENT.md §E.6) ------------------
E3_NB_K10_MIN = 0.95
E4_MIN = {"M2": 0.95, "M6": 0.95, "M3": 0.90}   # P(cross > max within) at k = 50, random bags
E4_MAX_DROP = 0.02                               # allowed drop vs the loose-rule row of experiment 4
E5_NB_MIN = 0.90
EXP4_LOOSE_K50 = {"M1": 0.909, "M2": 0.964, "M3": 0.942, "M4": 0.894, "M6": 0.964}   # experiment 4, random k=50

# models and training (stage 2) ----------------------------------------------------------------------------------
MODELS = {"instruct": "Qwen/Qwen3.5-4B", "base": "Qwen/Qwen3.5-4B-Base"}   # base is NOT in the HF cache yet
LORA = dict(r=8, alpha=16, dropout=0.05,
            targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
LORA_ATTN_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")
SFT = dict(lr=1e-4, warmup=0.03, micro_batch=2, grad_accum=4, weight_decay=0.0,
           epochs={100: 5, 250: 4, 500: 3, None: 3})
DECODE = dict(temperature=0.8, top_p=0.95, max_new_tokens=64, max_msgs=6)
DPO = dict(beta=0.1, lr=5e-6, epochs=1, n_cand=8, delta=math.log(3), cap_q=0.95, ditto_max=0.5,
           copy_lcs=12, ontopic_q=0.90)
SEEDS = (0, 1, 2)
