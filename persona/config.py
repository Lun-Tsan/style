"""persona.config -- every fixed constant of pipeline v1 (PIPELINE_SPEC.md §0). Change numbers here, nowhere else."""
from __future__ import annotations

import json
import math
import os
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

# synthetic mode (PERSONA_SYNTH=1): the fake export of persona/synth.py with fake names; every output goes under
# runs/synth/ so the real reports/ and runs/ are never overwritten
SYNTH = os.environ.get("PERSONA_SYNTH") == "1"
if SYNTH:
    RUNS = PROJECT / "runs" / "synth"
    REPORTS = RUNS / "reports"
    CHAT_PATH = RUNS / "chathistory_synth.txt"
    TARGET, PARTNER = "林老師 Lin", "小陳"
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
DECODE = dict(temperature=0.8, top_p=0.95, max_new_tokens=64, max_msgs=6,
              token_budget=12000,   # generation: (prompts + 1) * (longest prompt + max_new_tokens) * n per batch; raise if VRAM allows
              batch_size=16)        # generation: upper bound on prompts per batch
DPO = dict(beta=0.1, lr=5e-6, epochs=1, n_cand=8, delta=math.log(3), cap_q=0.95, ditto_max=0.5,
           copy_lcs=12, ontopic_q=0.90,
           eval_every=10,        # DPO: evaluate dev generations with the acceptance rulers every N optimizer steps
           dev_eval_n=60)        # DPO: number of dev contexts generated at each evaluation
SFT["retrain_full"] = True      # after dev selects the epoch count, retrain on train + dev for that many epochs (§2.5)

# evaluation (§6.1) and human blind test (§6.2)
EVAL = dict(rarefy_n=100,       # diversity (distinct-2, Hill-1) is computed on this many replies per system
            boot_n=2000)        # paired bootstrap over test contexts (report.py)
HUMAN = dict(max_n=100,
             # kind A ("which one is mine?"): line 3 passes when accuracy <= p_line; Wald SPRT between p_low and
             # p_high (indifference zone around the line), error rates alpha (wrongly "above") and beta (wrongly "below")
             A=dict(p_line=0.65, p_low=0.55, p_high=0.75, alpha=0.05, beta=0.05),
             # kind B ("which is more like me?", system vs baseline): passes when preference >= p_line
             B=dict(p_line=0.60, p_low=0.50, p_high=0.70, alpha=0.05, beta=0.05))
SEEDS = (0, 1, 2)

# acceptance lines (PIPELINE_SPEC §6.3), registered in advance; after B0 / C-base are seen they may only be tightened.
# Applied by report.acceptance() (lines 1-2) and report.lines_from_human() (line 3).
LINES = dict(d_auc1=0.75, d_auc10=0.90,          # line 1: D-AUC_1 and D-AUC_10 at most these (+ paired bootstrap)
             band_margin=1,                       # line 2: Band >= C-real Band - band_margin
             diversity_ratio=0.80,                # line 2: distinct-2 and Hill-1 >= this share of C-real's
             copy_max=0.02,                       # line 2: Copy at most this share (Leak: at most C-real's)
             human_A=HUMAN["A"]["p_line"],        # line 3: kind-A accuracy at the stopping point at most this (0.65)
             human_B=HUMAN["B"]["p_line"])        # line 3: kind-B preference over the best prompting system >= this (0.60)
