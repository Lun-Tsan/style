# 附錄:正式風格距離實驗 — 完整報告與審查紀錄

**日期**:2026-10-10 **資料**:`chathistory.txt`(私人 LINE 紀錄;本文件只含統計數字與 ≤ 3 字的 n-gram,無訊息原文)
**流程**:Opus 5.5 實作與評估 → 獨立審查(重寫程式重算、找資料外洩與統計錯誤)→ 修正重跑 → 第二輪審查
**程式**:session scratchpad 內 `formal_*.py`(第一版)、`formal2_*.py`(修正版)、`formalv_*.py`(審查者)。純 Python + numpy,無 GPU、無新安裝。
**主文件**:`PIPELINE_TECH_SURVEY.md` 的「實驗紀錄」節有中文摘要與分析。
**2026-10-13 補充**:§E「拿掉符號類訊息後重驗」(10/10 預先登記的下一步;單次執行、尚未獨立審查)。

---

# A. 最終報告(修正版,英文原文)

# Formal stylometric distances vs the NB author-LM on single LINE turns (post-review run)

## Method (summary)
- **Data.** A unit is a turn (a run of same-speaker messages, as in style_nb). Text is text_b with no normalisation. Train is everything before 2026-07-20: S 785 / T 862 turns. Test te_b is S 134 / T 149 (median 16 chars). te_c is S 51 / T 35 (median 7). style_nb's missing `analyze_chat` was replaced by a shim (formal_chatshim.py); the reviewer's own parser confirmed that it reproduces every baseline.
- **Features.** Character 1–3-grams including newline. The vocabulary is the top-N n-grams in pooled TRAIN (N = 300 main, 100/1000 in the ablation). f = count / all 1–3-grams. One profile per person, built from all of that person's TRAIN turns.
- **Measures.** M1 Burrows Delta, M2 Cosine Delta (+ M2b, cosine on raw f), M3 Min-max, M4 JSD (add-0.5, log2), M5 `style_nb.NB.margin` as-is, M6 NCD (zlib-9, test text first, 10 × ~3,000-char TRAIN reference bags per person), M7 Craig's Zeta (10-turn TRAIN segments, summed Z, as specified). Added Zeta variants: **M7m** = −mean Z over the text's distinct V1000 n-grams; **M7p** = (n_S − n_T)/(n_S + n_T) over the top-100 marker n-grams per person.
- **Deviation from canonical Delta:** M1/M2 z-scores use 400 random TRAIN bags of 25 turns (200 per person, seed 0), not a multi-author corpus, because only two authors exist.
- **Decision rules.** Threshold 0 (closer profile / sign) and an OOF-tuned threshold. The tuned threshold maximises balanced accuracy on out-of-fold TRAIN scores from 5 contiguous date blocks, with everything refit per fold. **New for bags:** a separate threshold per bag size k, tuned on OOF TRAIN bags (60 per person per block per k, k distinct held-out turns each) and applied unchanged to the test bags. AUC is reported for every measure.
- **Statistics.** Clopper-Pearson for single turns. E1 paired tests use exact McNemar plus a paired 2-way bootstrap. Pairs and bags use a 2-way bootstrap (S and T test turns resampled separately, 2000 reps, seed 0); the E3 bootstrap redraws the bags without ever repeating an original turn within a bag. **Every E3 result rests on 134 S / 149 T test turns.** The 300 bags per person overlap heavily (a k=50 S bag holds 37% of the S pool), so n = 600 bags would overstate the sample.

## Changes after review
1. **(High)** E3 now has a per-k OOF-TRAIN-bag threshold and AUC for every measure.
   - The threshold-0 sign rule is miscalibrated for summed scores. The NB margin and summed Zeta grow with k: NB's test S bags average +312 at k=50 while T bags average −1,538, so the error is all on S.
   - The rule is also miscalibrated for distance differences under the train→test shift.
   - Interpretation (ii) and (iv) are rewritten.
2. **(Medium)** Keep/drop decisions now use AUC and TRAIN-calibrated accuracy. Min-max is kept as a candidate.
3. **(Medium)** Cosine Delta is no longer recommended as the bag-level ruler; it is downgraded to provisional.
4. **(Low)** E3 n is stated in turns.
   - The bootstrap no longer allows duplicate turns within a bag. At k=50 under threshold 0, the bootstrap means move toward the point estimates:

     | measure | point | bootstrap mean, old | bootstrap mean, new |
     |---|---|---|---|
     | Cosine Delta | .993 | .967 | .987 |
     | NCD | .993 | .962 | .978 |
     | Burrows Delta | .847 | .796 | .834 |

   - Bias-corrected intervals were checked as well. They are unstable where accuracy piles up at 1.0, so the percentile intervals are reported.
5. **(Low)** E1 now has McNemar tests and a paired bootstrap against NB.
6. **(Low)** The z-reference was redrawn with 10 seeds for M1 and M2.
7. **(Low)** The post-hoc subset is relabelled "symbol/code-like (CODE_RE)".
8. **(Low)** The length-normalised and marker-share Zeta variants were tested before deciding whether to drop Zeta.
9. **Added after the review, not requested:**
   - TRAIN-period AUC within each date block.
   - E5c: out-of-fold single-turn AUC on agree/ack turns.
   - Both change the bag-level conclusion; see interpretation (ii).
10. **Reproduction checks:** all 126 threshold-0 E3 cells, every E1 threshold and every E1 accuracy were reproduced exactly from formal_run.json.

## E1 single-turn attribution, te_b (n = 283)
The difference column compares each measure with NB under the same rule (threshold 0 vs NB sign; tuned vs NB tuned). McNemar counts are (measure right & NB wrong / measure wrong & NB right).

| measure | rule | acc [95% CP] | bal acc | rec S | rec T | diff vs NB [paired boot] | McNemar p (b/c) |
|---|---|---|---|---|---|---|---|
| majority (T) | - | 0.527 [0.467, 0.586] | .500 | .000 | 1.000 | | |
| length rule (len_b<=16 -> S) | - | 0.615 [0.555, 0.672] | .616 | .634 | .597 | | |
| M1 Burrows Delta | closer | 0.604 [0.545, 0.662] | .622 | .948 | .295 | -.127 [-.180, -.071] | .006 (62/98) |
| M1 Burrows Delta | tuned | 0.548 [0.488, 0.607] | .570 | .985 | .154 | -.240 [-.290, -.187] | <.001 |
| M2 Cosine Delta | closer | 0.664 [0.606, 0.719] | .663 | .642 | .685 | -.067 [-.124, -.011] | .042 (30/49) |
| M2 Cosine Delta | tuned | 0.647 [0.588, 0.702] | .648 | .664 | .631 | -.141 [-.201, -.081] | <.001 |
| M2b cosine raw f | closer | 0.541 [0.481, 0.600] | .516 | .052 | .980 | -.191 [-.233, -.148] | <.001 |
| M2b cosine raw f | tuned | 0.671 [0.613, 0.726] | .670 | .649 | .691 | -.117 [-.170, -.067] | <.001 |
| M3 Min-max | closer | 0.604 [0.545, 0.662] | .588 | .276 | .899 | -.127 [-.177, -.078] | <.001 |
| M3 Min-max | tuned | 0.696 [0.639, 0.749] | .699 | .754 | .644 | -.092 [-.145, -.042] | .002 (19/45) |
| M4 JSD | closer | 0.530 [0.470, 0.589] | .504 | .015 | .993 | -.201 [-.244, -.159] | <.001 |
| M4 JSD | tuned | 0.548 [0.488, 0.607] | .523 | .052 | .993 | -.240 [-.286, -.191] | <.001 |
| **M5 NB LLR** | sign | **0.731 [0.676, 0.782]** | .719 | .485 | .953 | - | - |
| **M5 NB LLR** | tuned | **0.788 [0.736, 0.834]** | .784 | .701 | .866 | - | - |
| M6 NCD zlib | closer | 0.629 [0.570, 0.685] | .642 | .881 | .403 | -.102 [-.163, -.046] | .018 (56/85) |
| M6 NCD zlib | tuned | 0.590 [0.530, 0.648] | .608 | .955 | .262 | -.198 [-.254, -.141] | <.001 |
| M7 Zeta summed | sign | 0.565 [0.505, 0.624] | .542 | .097 | .987 | -.166 [-.208, -.124] | <.001 |
| M7 Zeta summed | tuned | 0.693 [0.635, 0.746] | .689 | .627 | .752 | -.095 [-.145, -.046] | <.001 |
| M7m Zeta mean-Z | sign | 0.565 [0.505, 0.624] | .542 | .097 | .987 | -.166 [-.208, -.124] | <.001 |
| M7m Zeta mean-Z | tuned | 0.707 [0.650, 0.759] | .702 | .604 | .799 | -.081 [-.134, -.028] | .004 (19/42) |
| M7p Zeta marker share | sign | 0.668 [0.610, 0.722] | .658 | .478 | .839 | -.064 [-.113, -.014] | .025 (20/38) |
| M7p Zeta marker share | tuned | 0.689 [0.632, 0.743] | .689 | .679 | .698 | -.099 [-.152, -.042] | .001 (21/49) |

**Tuned formal measures vs NB with the untuned sign rule (0.731):**

| measure (tuned) | diff vs NB sign [paired boot] | McNemar p |
|---|---|---|
| M7m Zeta mean-Z | −.025 [−.074, .025] | .43 |
| M3 Min-max | −.035 [−.095, .021] | .36 |
| M7 Zeta summed | −.039 [−.095, .021] | .27 |
| M7p Zeta marker share | −.042 [−.095, .011] | .21 |
| M2b cosine raw f | −.060 [−.120, .000] | .09 |
| M2 Cosine Delta | −.085 [−.141, −.028] | .014 |
| M6 NCD zlib | −.141 [−.198, −.088] | .003 |
| M1 Burrows Delta, M4 JSD | −.184 | < .001 |

## E1 te_c, short generic turns (n = 86)

| measure | closer/sign acc [CP] | tuned acc [CP] | diff vs NB (closer/sign) | diff vs NB (tuned) |
|---|---|---|---|---|
| majority (T) | 0.407 [0.302, 0.518] | - | | |
| length rule | 0.593 [0.482, 0.698] | - | | |
| M1 Burrows | 0.593 [0.482, 0.698] | 0.593 | -.070 [-.163, .012] | -.128 [-.221, -.035] |
| M2 Cosine Delta | 0.674 [0.565, 0.772] | 0.663 [0.553, 0.761] | +.012 [-.093, .105] | -.058 [-.163, .047] |
| M2b raw-f cosine | 0.453 [0.346, 0.565] | 0.593 [0.482, 0.698] | -.209 [-.291, -.128] | -.128 [-.233, -.023] |
| M3 Min-max | 0.535 [0.424, 0.643] | 0.628 [0.517, 0.730] | -.128 [-.221, -.023] | -.093 [-.186, .012] |
| M4 JSD | 0.407 [0.302, 0.518] | 0.407 | -.256 [-.349, -.174] | -.314 [-.407, -.221] |
| **M5 NB** | 0.663 [0.553, 0.761] | 0.721 [0.614, 0.812] | - | - |
| M6 NCD zlib | 0.674 [0.565, 0.772] | 0.663 [0.553, 0.761] | +.012 [-.093, .116] | -.058 [-.151, .035] |
| M7 Zeta summed | 0.488 [0.379, 0.599] | 0.674 [0.565, 0.772] | -.174 [-.267, -.093] | -.047 [-.140, .035] |
| M7m mean-Z | 0.488 [0.379, 0.599] | 0.616 [0.505, 0.719] | -.174 | -.105 [-.209, .000] |
| M7p marker share | 0.581 [0.470, 0.687] | 0.674 [0.565, 0.772] | -.081 [-.174, .000] | -.047 [-.151, .058] |

At n = 86, Cosine Delta, NCD and the Zeta variants cannot be told apart from NB, and neither can NB be called reliable here.

## E2 pairwise: pick the more S-like turn in each pair
These are the same 200 sampled pairs as style_nb (length within ±30%). te_b has 3,314 eligible pairs and te_c has 700.

| measure | te_b acc [CP] | te_b 2-way boot | te_b diff vs NB [boot] | te_b exhaustive [boot] | te_c acc [CP] | te_c diff vs NB [boot] |
|---|---|---|---|---|---|---|
| M1 Burrows | 0.665 [0.595, 0.730] | [0.536, 0.787] | -0.225 [-0.356, -0.106] | 0.712 [0.637, 0.781] | 0.720 [0.652, 0.781] | -0.147 [-0.291, -0.019] |
| M2 Cosine Delta | 0.650 [0.580, 0.716] | [0.516, 0.777] | -0.240 [-0.380, -0.116] | 0.667 [0.585, 0.742] | 0.700 [0.631, 0.763] | -0.167 [-0.310, -0.046] |
| M2b raw-f cosine | 0.630 [0.559, 0.697] | [0.500, 0.752] | -0.260 [-0.402, -0.133] | 0.626 [0.544, 0.699] | 0.670 [0.600, 0.735] | -0.198 [-0.379, -0.025] |
| M3 Min-max | 0.685 [0.616, 0.749] | [0.550, 0.815] | -0.205 [-0.353, -0.068] | 0.716 [0.639, 0.782] | 0.725 [0.658, 0.786] | -0.142 [-0.318, 0.021] |
| M4 JSD | 0.710 [0.642, 0.772] | [0.585, 0.822] | -0.180 [-0.310, -0.069] | 0.718 [0.641, 0.787] | 0.760 [0.695, 0.817] | -0.108 [-0.252, 0.017] |
| **M5 NB LLR** | **0.890 [0.838, 0.930]** | [0.804, 0.959] | - | 0.881 [0.829, 0.925] | **0.870 [0.815, 0.913]** | - |
| M6 NCD zlib | 0.775 [0.711, 0.831] | [0.658, 0.873] | -0.115 [-0.221, -0.030] | 0.766 [0.694, 0.829] | 0.755 [0.689, 0.813] | -0.115 [-0.255, 0.005] |
| M7 Zeta summed | 0.710 [0.642, 0.772] | [0.580, 0.824] | -0.180 [-0.295, -0.085] | 0.729 [0.658, 0.794] | 0.765 [0.700, 0.822] | -0.105 [-0.215, -0.010] |
| M7m mean-Z | 0.745 [0.679, 0.804] | [0.622, 0.856] | -0.145 [-0.259, -0.047] | - | 0.725 [0.658, 0.786] | -0.145 [-0.271, -0.027] |
| M7p marker share | 0.645 [0.574, 0.711] | [0.548, 0.780] | -0.225 [-0.336, -0.119] | - | 0.650 [0.580, 0.716] | -0.200 [-0.323, -0.079] |

## E3 bag-size curve
Each k uses 300 test bags per person, drawn from 134 S / 149 T test turns (the bags overlap). Cells show the point estimate with a [2.5, 97.5] interval from a 2-way turn bootstrap, 2000 reps, with thresholds held fixed.

**(a) Threshold 0 / closer profile (the spec rule)**

| measure | k=1 | k=3 | k=5 | k=10 | k=25 | k=50 | first k >= .95 |
|---|---|---|---|---|---|---|---|
| M1 Burrows | .610 [.57,.67] | .798 [.74,.87] | .798 [.71,.88] | .657 [.54,.77] | .710 [.51,.87] | .847 [.50,.99] | not by 50 |
| M2 Cosine Delta | .623 [.59,.73] | .765 [.67,.82] | .808 [.71,.87] | .870 [.76,.94] | .922 [.79,1.00] | .993 [.92,1.00] | 50 |
| M2b raw-f cos | .523 [.49,.54] | .523 [.47,.56] | .532 [.46,.58] | .558 [.46,.66] | .683 [.50,.85] | .802 [.50,.98] | not by 50 |
| M3 Min-max | .583 [.53,.65] | .577 [.53,.63] | .555 [.48,.60] | .557 [.45,.66] | .683 [.50,.85] | .800 [.50,.97] | not by 50 |
| M4 JSD | .508 [.49,.52] | .523 [.47,.56] | .530 [.46,.58] | .553 [.46,.66] | .683 [.50,.85] | .818 [.50,.98] | not by 50 |
| M5 NB LLR | .738 [.66,.77] | .755 [.69,.82] | .772 [.67,.83] | .797 [.66,.88] | .818 [.64,.95] | .877 [.57,1.00] | not by 50 |
| M6 NCD zlib | .615 [.58,.70] | .722 [.68,.81] | .812 [.74,.88] | .920 [.82,.97] | .933 [.80,1.00] | .993 [.88,1.00] | 50 |
| M7 Zeta summed | .528 | .510 | .502 | .500 | .500 | .500 | not by 50 |
| M7m mean-Z | .528 | .510 | .502 | .500 | .500 | .500 | not by 50 |
| M7p marker share | .628 [.59,.72] | .752 [.67,.81] | .763 [.66,.82] | .707 [.61,.83] | .735 [.54,.89] | .833 [.50,.98] | not by 50 |

**(b) Per-k threshold tuned on OOF TRAIN bags (new; no test data used)**

| measure | k=1 | k=3 | k=5 | k=10 | k=25 | k=50 | first k >= .95 |
|---|---|---|---|---|---|---|---|
| M1 Burrows | .537 [.53,.60] | .672 [.60,.73] | .757 [.68,.83] | .740 [.64,.84] | .855 [.67,.96] | .810 [.56,.99] | not by 50 |
| M2 Cosine Delta | .623 [.59,.73] | .768 [.67,.82] | .803 [.70,.87] | .868 [.75,.94] | .913 [.75,.99] | .987 [.85,1.00] | 50 |
| M2b raw-f cos | .652 [.58,.71] | .688 [.63,.79] | .680 [.59,.78] | .770 [.62,.85] | .803 [.59,.91] | .823 [.55,1.00] | not by 50 |
| M3 Min-max | .713 [.63,.76] | .767 [.71,.85] | .830 [.72,.89] | .865 [.73,.94] | .925 [.79,.98] | .970 [.85,1.00] | 50 |
| M4 JSD | .540 [.50,.55] | .690 [.63,.76] | .787 [.69,.86] | .875 [.75,.93] | .840 [.65,.95] | .853 [.58,1.00] | not by 50 |
| **M5 NB LLR** | **.800 [.72,.83]** | **.887 [.85,.95]** | **.942 [.85,.97]** | **.955 [.88,1.00]** | **.992 [.97,1.00]** | **1.000 [1.00,1.00]** | **10** |
| M6 NCD zlib | .587 [.57,.67] | .693 [.63,.76] | .732 [.66,.81] | .835 [.75,.93] | .918 [.76,1.00] | .993 [.88,1.00] | 50 |
| M7 Zeta summed | .635 [.57,.70] | .748 [.70,.85] | .787 [.69,.86] | .770 [.66,.88] | .883 [.71,.99] | .968 [.84,1.00] | 50 |
| M7m mean-Z | .682 [.64,.76] | .813 [.76,.89] | .883 [.79,.93] | .903 [.78,.96] | .890 [.73,.99] | .848 [.58,1.00] | not by 50 |
| M7p marker share | .712 [.64,.76] | .772 [.70,.84] | .822 [.72,.88] | .828 [.70,.92] | .832 [.61,.96] | .830 [.46,.98] | not by 50 |

**(c) Bag-level AUC (threshold-free)**

| measure | k=1 | k=3 | k=5 | k=10 | k=25 | k=50 | first k with AUC >= .95 |
|---|---|---|---|---|---|---|---|
| M1 Burrows | .749 [.68,.82] | .868 [.79,.93] | .924 [.81,.97] | .931 [.79,.99] | .942 [.78,1.00] | .999 [.98,1.00] | 50 |
| M2 Cosine Delta | .689 [.64,.78] | .834 [.73,.90] | .887 [.77,.94] | .924 [.81,.99] | .985 [.91,1.00] | 1.000 | 25 |
| M2b raw-f cos | .708 [.62,.78] | .777 [.69,.87] | .831 [.68,.90] | .846 [.66,.95] | .896 [.63,1.00] | .933 [.55,1.00] | not by 50 |
| M3 Min-max | .737 [.67,.81] | .835 [.76,.92] | .904 [.78,.95] | .916 [.77,.99] | .988 [.92,1.00] | 1.000 | 25 |
| M4 JSD | .769 [.70,.84] | .868 [.79,.94] | .923 [.81,.97] | .934 [.77,1.00] | .914 [.66,1.00] | .946 [.60,1.00] | not by 50 |
| **M5 NB LLR** | **.892 [.82,.92]** | **.942 [.88,.99]** | **.970 [.88,1.00]** | **.981 [.94,1.00]** | **1.000** | **1.000** | **5** |
| M6 NCD zlib | .682 [.64,.78] | .857 [.77,.92] | .927 [.83,.97] | .951 [.84,1.00] | .990 [.94,1.00] | 1.000 | 10 |
| M7 Zeta summed | .720 [.66,.80] | .792 [.73,.90] | .866 [.74,.94] | .905 [.77,.98] | .988 [.90,1.00] | .998 [.99,1.00] | 25 |
| M7m mean-Z | .746 [.70,.83] | .900 [.83,.95] | .941 [.86,.98] | .945 [.84,.99] | .942 [.77,1.00] | .960 [.68,1.00] | 50 |
| M7p marker share | .722 [.65,.80] | .841 [.77,.92] | .904 [.79,.95] | .911 [.79,.98] | .931 [.73,1.00] | .931 [.53,1.00] | not by 50 |

**Calibrated NB in detail:**

| k | recall S | recall T | bootstrap reps with accuracy ≥ .95 |
|---|---|---|---|
| 10 | .96 | .95 | 57% |
| 25 | 1.00 | .98 | 99.7% |

**Paired bootstrap, calibrated accuracy minus NB:**

| k | Cosine Delta | Min-max | NCD |
|---|---|---|---|
| 10 | −.099 [−.167, −.047] | −.103 [−.165, −.052] | −.107 [−.188, −.030] |
| 25 | −.101 [−.232, −.013] | −.080 [−.192, −.017] | −.087 [−.228, −.002] |
| 50 | −.028 [−.145, .000] | −.030 [−.145, .000] | −.024 [−.123, .000] |

AUC difference, NCD − NB, at k=10: −.036 [−.105, −.001].

**Threshold transfer is not uniformly helpful.** For NCD the calibrated threshold underperforms threshold 0 at k=10 (.835 vs .920). The TRAIN-tuned threshold assumes the train-period offset, and T's test-period text shifted: T's symbol/code-like character share is .38 in test vs .60 in train.

## E3 TRAIN-period check: AUC within each held-out date block (new)
These are out-of-fold scores, refit per fold, with 60 S + 60 T bags per block and k. Cells show min / mean across the 5 blocks.

| measure | k=1 | k=10 | k=25 | k=50 | k=50 per block (b0..b4) |
|---|---|---|---|---|---|
| M1 Burrows | .625/.729 | .812/.869 | .640/.872 | .603/.897 | .993 .969 .921 1.000 .603 |
| M2 Cosine Delta | .664/.713 | .797/.871 | .792/.925 | .839/.965 | .992 1.000 .839 1.000 .997 |
| M2b raw-f cos | .605/.688 | .730/.805 | .632/.817 | .527/.793 | .921 .831 .690 .997 .527 |
| M3 Min-max | .581/.689 | .813/.846 | .767/.875 | .764/.908 | .953 1.000 .821 1.000 .764 |
| M4 JSD | .650/.729 | .800/.849 | .675/.864 | .611/.841 | .963 .832 .797 1.000 .611 |
| **M5 NB LLR** | .814/.878 | .810/.919 | .837/.950 | .852/.966 | 1.000 1.000 .852 1.000 .979 |
| **M6 NCD zlib** | .717/.815 | **.926/.980** | **.985/.997** | **.992/.998** | 1.000 1.000 .992 1.000 .998 |
| M7 Zeta summed | .731/.780 | .856/.931 | .922/.975 | .917/.984 | 1.000 1.000 .917 1.000 1.000 |
| M7m mean-Z | .743/.803 | .889/.934 | .856/.952 | .779/.935 | 1.000 .981 .915 1.000 .779 |
| M7p marker share | .724/.764 | .923/.959 | .921/.979 | .873/.970 | 1.000 .998 .981 1.000 .873 |

The weak blocks are the ones where both people's text is mostly symbol/code-like:

| period | S symbol/code-like character share | T symbol/code-like character share |
|---|---|---|
| block 2 (2025-09-25..11-29) | .79 | .64 |
| block 4 (2026-03..07) | .75 | .72 |
| test period | .70 | .38 |

The test period's contrast makes it easy for NB. **NB's k=10–25 result is therefore specific to the test period.** NCD is the only measure whose within-block AUC stays ≥ .93 at k=10 and ≥ .985 at k=25 in every training block.

## z-reference seed spread (M1/M2, 10 redraws of the 400 reference bags)
Cells show seed 0 (mean; min–max).

| measure | E1 acc thr0 | E1 tuned | E2 pairs | E3 k=10 cal | E3 k=25 cal | E3 k=50 cal | AUC k=10 | AUC k=25 |
|---|---|---|---|---|---|---|---|---|
| M1 N300 | .604 (.602; .569–.633) | .548 (.556; .509–.594) | .665 (.670; .655–.680) | .740 (.788; .732–.832) | .855 (.822; .763–.863) | .810 (.822; .773–.857) | .931 (.930; .928–.933) | .942 (.940; .933–.946) |
| M2 N300 | .664 (.658; .625–.678) | .647 (.659; .643–.678) | .650 (.659; .650–.675) | .868 (.850; .792–.873) | .913 (.910; .897–.922) | .987 (.979; .957–.992) | .924 (.916; .901–.924) | .985 (.980; .968–.985) |
| M2 N1000 | .664 (.661; .640–.675) | .671 (.664; .647–.689) | .685 (.671; .655–.685) | .855 (.847; .812–.873) | .910 (.895; .857–.910) | .985 (.976; .953–.985) | .917 (.907; .893–.917) | .975 (.969; .958–.975) |

- **Seed 0 is a favourable draw for Cosine Delta.** It sits at or near the top of the range on most bag-level numbers.
- **Burrows Delta's calibrated bag accuracy moves by up to .10 between seeds.** Its AUC barely moves.

## E4 ruler validation: TRAIN bags of k = 50 (2000 reps; unchanged)
Pass criterion: cross-person distance exceeds both within-person distances in at least 95% of reps.

| measure | within T | within S | cross | cross / mean within | P(cross > max within) | P(cross > mean within) |
|---|---|---|---|---|---|---|
| M1 | 0.732 [0.427, 1.096] | 0.862 [0.502, 1.287] | 0.880 [0.579, 1.237] | 1.139 [0.652, 1.866] | 0.436 | 0.657 |
| M2 | 0.927 [0.424, 1.342] | 0.952 [0.249, 1.365] | 1.091 [0.489, 1.424] | 1.231 [0.476, 2.358] | 0.532 | 0.705 |
| M2b | 0.272 [0.037, 0.708] | 0.367 [0.011, 0.823] | 0.369 [0.026, 0.754] | 2.178 [0.073, 12.465] | 0.371 | 0.585 |
| M3 | 0.538 [0.316, 0.765] | 0.617 [0.311, 0.859] | 0.633 [0.400, 0.819] | 1.136 [0.614, 1.884] | 0.403 | 0.653 |
| M4 | 0.166 [0.044, 0.367] | 0.221 [0.046, 0.466] | 0.229 [0.075, 0.431] | 1.484 [0.301, 4.769] | 0.418 | 0.620 |
| M6 | 0.867 [0.812, 0.919] | 0.870 [0.782, 0.958] | 0.905 [0.860, 0.952] | 1.043 [0.969, 1.108] | 0.699 | 0.890 |

**All six fail.** Random bags mix all training periods, and variation in symbol/code-like content between periods is larger than the difference between the two people. NB is not a distance, so E4 does not apply to it.

## E5 content control: agree/acknowledge TRAIN turns (addon.py definition)
- **Ack pool:** S has 60 turns but only 13 distinct strings; T has 50 turns and 20 distinct strings; 3 strings are used by both.
- **E5a (spec intent, k reduced):** identical strings are excluded and the two compared bags never share a string. Two disjoint S bags of 10 distinct strings cannot be formed from 13, so k = 6 instead of 10 (1000 reps).
- **E5b (spec k, no exclusion):** k = 10, turn-disjoint halves, identical strings allowed (1000 reps).

| measure | E5a cross/within | E5a P(cross > max within) | E5b cross/within | E5b P(cross > max within) |
|---|---|---|---|---|
| M1 | 1.577 [0.812, 2.486] | 0.823 | 2.671 [1.382, 4.779] | 0.983 |
| M2 | 1.968 [0.797, 3.742] | 0.798 | 6.708 [1.921, 19.304] | 0.995 |
| M2b | 1.732 [0.772, 3.305] | 0.735 | 5.181 [1.583, 13.528] | 0.986 |
| M3 | 1.317 [0.904, 1.824] | 0.786 | 1.949 [1.290, 3.084] | 0.994 |
| M4 | 1.837 [0.697, 3.512] | 0.777 | 4.366 [1.599, 9.728] | 0.989 |
| M6 | 1.227 [0.985, 1.484] | 0.853 | 1.461 [1.134, 1.889] | 0.980 |

**E5c (new): out-of-fold single-turn AUC on the same ack turns**, with 2-way bootstrap intervals. "Unseen" means the exact string never occurs in that fold's training turns.

| measure | all ack (S 60 / T 50 turns) | unseen strings (S 9 / T 16) |
|---|---|---|
| **M5 NB LLR** | **.944 [.896, .983]** | **.965 [.875, 1.000]** |
| M7m Zeta mean-Z | .910 [.843, .968] | .785 [.549, .972] |
| M7 Zeta summed | .901 [.830, .957] | .792 [.569, .958] |
| M7p marker share | .847 [.775, .905] | .653 [.455, .851] |
| M2 Cosine Delta | .815 [.717, .903] | .660 [.438, .861] |
| M4 JSD | .724 [.625, .816] | .646 [.417, .854] |
| M6 NCD zlib | .701 [.597, .790] | .806 [.611, .944] |
| M2b raw-f cosine | .685 [.580, .783] | .538 [.306, .764] |
| M1 Burrows | .676 [.568, .775] | .736 [.521, .910] |
| M3 Min-max | .651 [.543, .754] | .302 [.111, .528] |

With content held fixed, NB still separates the two people. Among the formal measures, the Zeta variants come closest. Min-max does not separate them on unseen strings.

## M7 Zeta marker n-grams (TRAIN, V1000; Z = p_T − p_S; ⏎ = newline)
- **T markers:** `喔` +0.65, `喔⏎` +0.55, `啊` +0.46, `⏎沒` +0.45, `就` +0.43, `就是` +0.43, `⏎這` +0.42, `是要` +0.41, `⏎所` +0.41, `⏎所以` +0.41, `問題` +0.41, `啊⏎` +0.39, `個` +0.37, `這個` +0.37, `來` +0.37
- **S markers:** `嗎⏎我` −0.39, `我不` −0.32, `怎麼` −0.31, `⏎我` −0.31, `不知` −0.29, `不知道` −0.29, `嗎⏎` −0.29, `可是` −0.26, `物` −0.26, `我想` −0.25, `怎` −0.24, `不太` −0.24, `不好` −0.22, `嗎` −0.22, `嗯嗯` −0.22
- **Length drift:** the mean Z over V1000 is +0.062, so the summed score drifts toward T as texts get longer. Both the mean-Z and marker-share variants stop working under threshold 0: at k ≥ 10 every bag is called T for mean-Z, and accuracy never passes .84 for marker share. Calibrated, they peak at k=10 (mean-Z .903) and then degrade.

## Ablation: vocabulary size N and NCD variant
Columns: E1 te_b accuracy (threshold 0 / tuned) and E2 pairwise accuracy on te_b and te_c. The E3 thr0 column is k=10 / k=50. The E3 calibrated and AUC columns are k=10 / 25 / 50.

| variant | E1 te_b thr0/tuned | E2 te_b | E2 te_c | E3 thr0 k10/k50 | E3 cal k10/25/50 | AUC k10/25/50 |
|---|---|---|---|---|---|---|
| M1 N100 | .640/.654 | .665 | .685 | .622/.823 | .812/.863/.952 | .882/.937/.990 |
| M1 N300 | .604/.548 | .665 | .720 | .657/.847 | .740/.855/.810 | .931/.942/.999 |
| M1 N1000 | .523/.523 | .690 | .765 | .820/.857 | .860/.888/.925 | .926/.966/.999 |
| M2 N100 | .615/.640 | .630 | .660 | .807/.963 | .745/.895/.932 | .876/.959/.996 |
| M2 N300 | .664/.647 | .650 | .700 | .870/.993 | .868/.913/.987 | .924/.985/1.000 |
| M2 N1000 | .664/.671 | .685 | .735 | .852/.985 | .855/.910/.985 | .917/.975/.999 |
| M2b N100 | .541/.654 | .610 | .645 | .558/.793 | .748/.790/.810 | .825/.886/.930 |
| M2b N300 | .541/.671 | .630 | .670 | .558/.802 | .770/.803/.823 | .846/.896/.933 |
| M2b N1000 | .551/.678 | .620 | .715 | .558/.805 | .777/.833/.820 | .856/.900/.935 |
| M3 N100 | .597/.696 | .675 | .670 | .555/.747 | .827/.878/.908 | .880/.980/.987 |
| M3 N300 | .604/.696 | .685 | .725 | .557/.800 | .865/.925/.970 | .916/.988/1.000 |
| M3 N1000 | .633/.693 | .675 | .725 | .572/.807 | .865/.935/**.997** | .926/.991/1.000 |
| M4 N100 | .530/.555 | .645 | .690 | .555/.797 | .818/.828/.747 | .903/.908/.938 |
| M4 N300 | .530/.548 | .710 | .760 | .553/.818 | .875/.840/.853 | .934/.914/.946 |
| M4 N1000 | .530/.714 | .740 | .805 | .568/.795 | .880/.795/.862 | .938/.928/.981 |
| M7 N100 | .569/.636 | .675 | .645 | .500/.500 | .667/.672/.558 | .726/.797/.809 |
| M7 N300 | .562/.678 | .710 | .705 | .500/.500 | .758/.763/.777 | .851/.950/.959 |
| M7 N1000 | .565/.693 | .710 | .765 | .500/.500 | .770/.883/.968 | .905/.988/.998 |
| M6 zlib xy (main) | .629/.590 | .775 | .755 | .920/.993 | .835/.918/.993 | .951/.990/1.000 |
| M6 zlib yx | .576/.534 | .795 | .770 | .885/.982 | .807/.918/.973 | .953/.989/1.000 |
| M6 lzma xy | .675/.647 | .775 | .745 | .907/.965 | - | - |
| M6 lzma yx | .636/.594 | .800 | .810 | .873/.968 | - | - |

- **lzma:** about 70× slower than zlib for little gain, so it was not recalibrated.
- **N = 1000** helps Min-max at bag level.

## Post-hoc diagnostic (not pre-specified; a lead only): symbol/code-like turns (CODE_RE)
- **What CODE_RE catches:** it fires on code, and also on `( ) + / = [ ] > * _`, so many matched turns are maths/physics expressions or emoticons rather than code.
- **Character share in matched turns:** train S .68, train T .60, test S .70, test T .38.
- **Which S bags fail at k=50 (threshold 0):** the misattributed ones have less symbol/code-like text, e.g. Min-max .51 vs .73, NB .49 vs .69.
- **Refit without these turns.** Everything was refit on non-symbol/code TRAIN turns (S 679 / T 706) and tested on non-symbol/code test turns (S 115 / T 126, n = 241). Point estimates, threshold 0:

| measure | E1 acc [CP] | E3 k=10 |
|---|---|---|
| Min-max | .772 [.714, .823] | .985 |
| NCD | .718 [.656, .774] | .978 |
| NB | .772 [.714, .823] | .837 |

  - NB's .837 is under threshold 0, which the main result shows is miscalibrated for NB.
  - E4 pass shares rise to .86–.97.
- **Same mechanism in the date blocks:** the per-block AUC table shows NB and the count-based distances are weakest in the training blocks where both people send mostly symbol/code-like text.

## E6 runtime (seconds, this machine; no GPU)
| measure | fit | E1 score (283 turns) | E3 score (3,600 bags) |
|---|---|---|---|
| M1 / M2 | 0.192 (z-reference) | 0.009 / 0.010 | 0.156 / 0.155 |
| M2b / M3 / M4 | shared | 0.005 / 0.005 / 0.024 | 0.084 / 0.083 / 0.464 |
| M5 NB | 0.048 | 0.012 | 3.705 |
| M6 zlib-xy / yx | 0.083 | 3.556 / 1.695 | 16.238 / 17.131 |
| M6 lzma-xy / yx | shared | 76.996 / 54.003 | 1100.674 / 781.462 |
| M7 (+ M7m/M7p) | 0.031 | 0.001 | 0.017 |

- **Shared work:** vocabulary and profiles take 0.071 s.
- **Wall times:**

  | script | wall time | processes |
  |---|---|---|
  | formal_run.py | 257 s | – |
  | formal2_cal.py (all OOF fits, calibration, paired tests) | 16 s | 16 |
  | formal2_boot.py (2000 reps) | 1,638 s | 24 |
  | formal2_zseed.py | 67 s | – |
  | formal2_foldauc.py | 23 s | – |

- **Leakage:** none found. Vocabulary, profiles, z-reference, NB, Zeta, NCD references and every threshold (single-turn and per-k bag) are fit on TRAIN only, or out-of-fold within TRAIN. Test turns are used only for scoring.

## Interpretation
1. **(i) A single short message cannot be attributed reliably by any formal distance.**
   - On te_b (n = 283) the best formal results are tuned mean-Z Zeta .707 [.650, .759] and Min-max .696 [.639, .749].
   - These are statistically indistinguishable from NB's untuned sign rule, .731: paired difference −.025 [−.074, .025] and −.035 [−.095, .021]; McNemar p .43 and .36.
   - Every tuned formal measure is clearly below NB tuned at .788 [.736, .834]: differences −.08 to −.24, all CIs exclude 0.
   - On matched pairs, NB gets .890 against .775 for the best formal measure (NCD), a difference of −.115 [−.221, −.030].
   - On 7-character te_c turns (n = 86) nothing is reliable, NB included: .66–.72.
2. **(ii) Bag size needed for ≥ .95, once each measure's threshold is set on TRAIN only.**
   - NB reaches .955 [.88, 1.00] at k = 10 (57% of reps ≥ .95) and .992 [.97, 1.00] at k = 25 (99.7% of reps). Its AUC is ≥ .95 from k = 5.
   - The earlier "NB stops at .877 at k = 50" was a threshold-0 artifact: the summed log-likelihood ratio grows with k.
   - No formal distance reaches .95 before k = 50, even calibrated: Cosine Delta .987 [.85, 1.00], NCD .993 [.88, 1.00], Min-max .970 [.85, 1.00], summed Zeta .968.
   - **Caveat:** NB's k = 10–25 figure is specific to the test period. On held-out training blocks where both people send mostly symbol/code-like text, NB's within-block AUC falls to .85 at k = 50. NCD stays ≥ .99 in every block. A conservative planning figure is therefore k ≥ 25, with NB and NCD both checked.
3. **(iii) No distance passes the within < cross order check.**
   - E4 pass shares are .37–.70.
   - Under content control, E5a gives .74–.85; E5b passes only when literal string repeats are allowed.
   - In the content-controlled single-turn check (E5c), NB separates the two people with AUC .944 [.896, .983], and .965 [.875, 1.000] on strings it never saw (n = 25). So the author-LM does capture style when content is held fixed; the distances mostly do not, with Min-max at .30 on unseen strings.
4. **(iv) Recommendation.**
   - **Primary ruler:** adopt the NB author-LM LLR at both levels. For bags, use a per-k threshold tuned on out-of-fold TRAIN bags, never the sign rule.
   - **Model-free cross-check:** NCD-zlib, reported as AUC or pairwise accuracy at k ≥ 25. It is the most stable across periods, but its threshold does not transfer.
   - **Provisional:** Cosine Delta. It fails E4 and E5a, needs k = 50, and depends on the z-reference draw (seed 0 is near the best of 10).
   - **Candidate:** Min-max. Its AUC is .988 at k = 25 and calibrated accuracy .997 at k = 50 with N = 1000, and it is best in the symbol/code-free refit. It fails the content checks.
   - **Drop, based on AUC and calibrated accuracy:**
     - Raw-f cosine: AUC ≤ .935.
     - JSD: AUC ≤ .946, and its accuracy is not monotone in k.
     - Burrows Delta: seed-sensitive, its calibrated accuracy is not monotone in k, and its worst block AUC is .60.
   - **Zeta:** the signal is real (summed AUC .988 at k = 25; E5c .90), but none of its three scorings matches NB at any k. Keep the marker lists as descriptions only.

## Headline numbers

- E1 te_b NB LLR single-turn acc, sign rule: 0.731 [0.676, 0.782] n=283
- E1 te_b NB LLR single-turn acc, OOF-tuned threshold: 0.788 [0.736, 0.834] n=283
- E1 te_b best formal single-turn acc (M7m mean-Z Zeta, tuned): 0.707 [0.65, 0.759] n=283
- E1 te_b Min-max (tuned) minus NB (sign), paired 2-way bootstrap: -0.035 [-0.095, 0.021] n=283
- E1 te_b Min-max (tuned) minus NB (tuned), paired 2-way bootstrap: -0.092 [-0.145, -0.042] n=283
- E2 te_b NB pairwise acc (200 sampled pairs, CP): 0.89 [0.838, 0.93] n=200
- E2 te_b NCD-zlib minus NB pairwise (2-way boot): -0.115 [-0.221, -0.03] n=200
- E3 NB bag acc k=10, per-k OOF-TRAIN threshold (2-way turn boot; n = 134 S + 149 T test turns): 0.955 [0.883, 0.998] n=283
- E3 NB bag acc k=25, per-k OOF-TRAIN threshold (turn boot): 0.992 [0.972, 1] n=283
- E3 NB bag AUC k=10 (turn boot): 0.981 [0.939, 1] n=283
- E3 NB bag acc k=50, threshold 0 (spec sign rule; miscalibrated): 0.877 [0.572, 0.997] n=283
- E3 Cosine Delta bag acc k=50, OOF-TRAIN threshold: 0.987 [0.855, 1] n=283
- E3 NCD-zlib bag acc k=50, OOF-TRAIN threshold: 0.993 [0.877, 1] n=283
- E3 Min-max bag acc k=50, OOF-TRAIN threshold: 0.97 [0.855, 1] n=283
- E3 NCD-zlib bag AUC k=10: 0.951 [0.842, 0.999] n=283
- TRAIN per-block OOF AUC k=50, worst block: NB: 0.852 [, ] n=120
- TRAIN per-block OOF AUC k=50, worst block: NCD-zlib: 0.992 [, ] n=120
- E4 Cosine Delta share of reps cross > max within (k=50, 2000 reps): 0.532 [, ] n=2000
- E5a NCD share cross > max within (ack, k=6, 1000 reps; best distance): 0.853 [, ] n=1000
- E5c NB OOF single-turn AUC on ack turns (2-way boot): 0.944 [0.896, 0.983] n=110
- E5c NB OOF AUC on ack strings unseen in fold train: 0.965 [0.875, 1] n=25

---

# B. 第二輪審查(對修正版)

verdict=verdict=minor-issues
[medium] The calibrated E3 results (table b) and the per-block AUC claims depend on which TRAIN calibration bags are drawn. The bootstrap holds the thresholds fixed, so its CIs leave out this calibration uncertainty. I redrew the out-of-fold (OOF) calibration bags twice, with my own code and the same test bags and blocks. NB calibrated accuracy at k=10 came out .937 and .947 against the reported .955, so the bolded claim that NB first reaches ≥.95 at k=10 is borderline and depends on the draw; at k=25 NB is robust (.995 in both redraws vs .992 reported). The redraws also moved NCD k=10 calibrated accuracy from .835 to .903, Cosine Delta k=25 from .913 to .898 and Min-max k=50 from .970 to .957. NCD's worst within-block AUC at k=25 fell from .985 to .948, and NB's block-4 AUC at k=25 from .914 to .815. The qualitative conclusions survive: NB is best, every formal distance needs k=50 to reach .95 calibrated, and NCD is the most stable measure across blocks. The precise figures 'k=10', '≥.985' and 'NCD calibrated .835 vs threshold-0 .920' do not.
   fix: Repeat the OOF bag calibration over several draws (e.g. 10 seeds), or use more OOF bags. Report the mean and range of calibrated accuracy and of within-block AUC. State the NB bag size needed for ≥.95 as 'k=10–25 (borderline at 10, robust at 25)' and lead with k=25. Soften the NCD 'calibration underperforms' magnitude.
[low] The text says NCD is 'the only measure whose within-block AUC stays ≥ .93 at k=10'. In formal2_foldauc.json its actual minimum is .9264, so the claim holds only after rounding up. Zeta marker share (M7p) has a k=10 minimum of .9232, essentially the same, so 'only' is not supported at k=10.
   fix: Say '≥ .92 at k=10 (NCD and Zeta marker share)', and base the uniqueness claim on k=25/50, where NCD's lead is clearer, giving the draw spread from the medium issue.
[low] The unit wording in interpretation (i) overclaims. It says 'a single short message', but every evaluated unit is a turn: a run of one speaker's messages joined within 30 minutes, with test median 16 characters, and te_c median 7. The method section discloses this, but the plain-language answer to the user's question about a single LINE message does not.
   fix: Phrase (i) as 'a single turn (often one message; median 16 characters)', and give the share of test turns that are a single message.
[low] The comparison is selected in the formal measures' favour. The 'best formal' headline (Zeta mean-Z, tuned, .707) is a post-hoc variant picked as the maximum of about 20 measure×rule rows. It is then compared with NB's untuned sign rule to conclude 'indistinguishable'. The like-for-like comparison, tuned against tuned, puts every formal measure below NB (Min-max −.092 [−.145, −.039], McNemar p=.002, which I reproduced). The report does show both, so the conclusion is not wrong, but the headline ordering flatters the formal measures.
   fix: Lead the E1 conclusion with tuned vs tuned. Present tuned-formal vs NB-sign as secondary, and note that M7m and M7p were added after the review.
[low] E3 has no bag-level length-only baseline (the spec required one only for E1). I computed it on the same test bags: length-only AUC is .660, .533 and .430 at k=10, 25 and 50, and accuracy with a threshold calibrated on OOF TRAIN bags is ≤.71. On length-matched S-vs-T bag pairs (within ±10%), NB still scores AUC .998 at k=10 and NCD .979. So bag-level separability is not a length artifact. This supports the report, but it is missing from it.
   fix: Add a length-only row to E3 tables (b) and (c), plus the length-matched bag-pair AUC.
[low] Both parser shims (formal_chatshim and my own independent parser) reproduce every pool size and every sampled or single-turn baseline exactly. The exhaustive te_b NB pairwise figure, however, is .8810 against .880 in the original Oct-3 output (rv_style_eval_rerun.out), produced with the now-missing analyze_chat. A few turn texts probably differ slightly, e.g. in blank-line handling. This is immaterial, but 'reproduces every baseline' is not literally true.
   fix: Note that the exhaustive figure is .881 under the shim vs .880 originally, and treat the shim as a near-exact reconstruction.

##### RECOMPUTED
- E1 te_b NB sign-rule accuracy (n=283): reported 0.731 vs 0.7314 | Exact, including recall S .485 / T .953 and CP [.676, .782]. My parser (formalv_r3_data.py) is independent of formal_chatshim and reproduces all pools (785/862, 134/149, 51/35). Verifier scripts are in the scratchpad: formalv_r3_data.py, formalv_r3_main.py, formalv_r3_extra.py, formalv_r3_small.py.
- E1 te_b NB OOF-tuned accuracy: reported 0.788 vs 0.788 | Own 5 contiguous date blocks: [437, 230, 331, 325, 324] turns, identical to the experiment's. Threshold −3.499.
- E1 te_b Cosine Delta N300, closer-profile accuracy: reported 0.664 vs 0.664 | Own implementation (vocabulary, z-reference of 400 bags with rng(0), profiles). Recall S .642 / T .685.
- E1 te_b Cosine Delta N300, tuned accuracy: reported 0.647 vs 0.647 | 
- E1 te_b Min-max N300, tuned accuracy: reported 0.696 vs 0.696 | 
- E1 te_b Zeta mean-Z, tuned accuracy: reported 0.707 vs 0.707 | 
- E1 Min-max (tuned) minus NB (sign), paired difference: reported -0.035 vs -0.035 | 2-way bootstrap [−.095, +.021]; McNemar 44/54, p=.363. Matches.
- E1 Min-max (tuned) minus NB (tuned): reported -0.092 vs -0.092 | Bootstrap [−.145, −.039]; McNemar 19/45, p=.002.
- E1 te_b Burrows Delta, closer: reported 0.604 vs 0.604 | 
- E1 te_b JSD, closer: reported 0.53 vs 0.53 | 
- E1 te_b NCD zlib, closer: reported 0.629 vs 0.629 | 
- E1 te_b length rule (len_b<=16 -> S): reported 0.615 vs 0.615 | 
- E2 te_b Min-max pairwise (same 200 pairs): reported 0.685 vs 0.685 | CP [.616, .749]. Cosine Delta .650, Burrows .665, JSD .710, raw-f cosine .630, NCD .775 and NB .890 also match.
- E2 te_c Min-max pairwise: reported 0.725 vs 0.725 | 
- E2 te_b NB exhaustive pairwise: reported 0.881 vs 0.881 | The original Oct-3 baseline printed .880; see the low-severity parser issue.
- E3 Cosine Delta k=25, threshold 0 (test bags): reported 0.922 vs 0.922 | Same draw recipe as the experiment. AUC .9849 vs .985 reported. k=10 .870 and k=50 .993 also match.
- E3 Cosine Delta k=25, calibrated on OOF TRAIN bags: reported 0.913 vs 0.898 | My own OOF calibration-bag draw. The difference reflects calibration-draw variance.
- E3 NB k=10, calibrated: reported 0.955 vs 0.937 | A second independent OOF draw gives .947. Two of three draws fall below .95.
- E3 NB k=25, calibrated: reported 0.992 vs 0.995 | Both redraws give .995, so k=25 is robust.
- E3 NB k=50, threshold 0: reported 0.877 vs 0.877 | Recall S .753 / T 1.000, confirming the sign rule is miscalibrated.
- E3 Min-max k=50, calibrated: reported 0.97 vs 0.957 | 
- E3 NCD zlib k=10, calibrated: reported 0.835 vs 0.903 | Test AUC .951 matches exactly.
- E3 NCD zlib k=50, calibrated: reported 0.993 vs 0.992 | 
- TRAIN within-block OOF AUC, NCD k=25, worst block: reported 0.985 vs 0.948 | Own OOF draw. At k=50 the worst block is .992, equal to the report.
- TRAIN within-block OOF AUC, NB k=50, worst block: reported 0.852 vs 0.814 | 
- E4 Cosine Delta k=50: share of reps with cross > max within: reported 0.532 vs 0.54 | 200 reps with my own draws. Within T .948 (reported .927), within S .965 (.952), cross 1.095 (1.091), ratio 1.191 (1.231), P(cross > mean within) .710 (.705). It fails the order check, as reported.
- Zeta mean Z over V1000 (length drift): reported 0.062 vs 0.0619 | The top markers match: T 喔 +.65, 喔⏎ +.55, 啊 +.46; S 嗎⏎我 −.39, 我不 −.32, 怎麼 −.31.

---

# C. 第一輪審查(對第一版;促成修正的意見)

verdict=verdict=minor-issues
[high] The E3 bag-size conclusions are an artifact of using threshold 0. Interpretation (ii) says NB 'stops at 0.877 at k=50' and never reaches .95, and that only Cosine Delta and NCD reach .95, at k=50. Recommendation (iv) then adopts Cosine Delta as the bag-level ruler. But the NB margin is a sum over the bag, so its bias toward T grows with k. All NB errors are S bags called T (recall T 1.00, recall S .75), while NB still ranks the bags almost perfectly: AUC .985 at k=10 and 1.000 at k=25 and k=50; Cosine Delta's AUC is .901 at k=10 and .983 at k=25. I tuned a threshold on TRAIN only (out-of-fold bags from the same 5 contiguous date blocks, everything refit per fold, one threshold per k) and applied it to test bags. NB then scores .955 at k=10, .998 at k=25 and 1.000 at k=50. With a 2-way turn bootstrap (100 reps, thresholds held fixed): k=10 .959 [.876, 1.000], k=25 .994 [.962, 1.000]. Cosine Delta with the same procedure scores .833 at k=10 and .917 at k=25 (bootstrap .855 / .905). So with a threshold set on TRAIN only, NB is also the best bag-level attributor, and about 10–25 messages are enough for >= .95, not 50. The computed numbers are correct and follow the spec's sign rule. The conclusions drawn from them are not.
   fix: Add E3 rows that use a threshold tuned per k on out-of-fold TRAIN bags (date blocks), plus bag-level AUC, for every measure. Rewrite interpretation (ii): NB reaches >= .95 at about k=10–25 once the threshold is set on TRAIN. Rewrite (iv): recommend the NB LLR as both the per-message and the bag-level ruler, with Cosine Delta / NCD as model-free cross-checks. State that the threshold-0 sign rule is miscalibrated for summed scores.
[medium] The same calibration artifact affects the recommendation to drop measures. Min-max, raw-f cosine and JSD are called 'degenerate one-class' at bag level, but under the closer-profile rule they separate the bags well. Min-max AUC is .985 at k=25 and 1.000 at k=50, and with a TRAIN-OOF bag threshold it reaches .960 at k=50 (.800 under threshold 0). Raw-f cosine improves less (AUC .918 at k=50). Dropping measures on threshold-0 accuracy confuses miscalibration with lack of signal. The likely cause is a shift between periods: T's code-like share of characters falls from .60 in train to .38 in test while S stays at about .70.
   fix: Base keep/drop decisions on AUC or on accuracy with a TRAIN-tuned threshold. Keep Min-max as a candidate. It is also the best formal measure in the no-code refit.
[medium] The report contradicts itself. It recommends Cosine Delta as the formal bag-level ruler and says to 'report results at k >= 10'. Yet Cosine Delta fails the report's own ruler validation: E4 pass share .532 and E5a .798. Its k=10 accuracy is .870 [.75, .94], below .95, and it reaches .95 only at k=50, where the bootstrap lower bound is .83.
   fix: Either recommend the TRAIN-calibrated NB (see the high-severity issue), or mark Cosine Delta as provisional: it failed E4/E5a and needs k >= 50.
[low] The E3 headline numbers give n=600 bags, which overstates the sample. All bags are drawn from only 134 S and 149 T test turns (about 5.8k and 5.3k characters), and at k=50 each S bag contains 37% of the S pool. The turn-level bootstrap CI is the right method, but n should be stated in turns.
   fix: Report n as 134 S / 149 T test turns in the E3 headline numbers. Note that the 300 bags per person overlap heavily.
[low] The E3 bootstrap distributions sit below the point estimates: Cosine Delta at k=50 has a bootstrap mean of .967 vs .993; Burrows Delta at k=50 .796 vs .847; NCD .962 vs .993. Resampling turns with replacement puts duplicate turns into the bags, which shrinks their effective size, so the percentile CIs are biased somewhat low and conservative.
   fix: Mention the bias. Optionally use a bias-corrected percentile interval, or draw bags without duplicate original turns.
[low] E1 has no paired test between NB and the best formal measure. Min-max with tuned threshold scores .696 [.639, .749] and NB with sign rule .731 [.676, .782]; the CIs overlap. That NB is better on single messages is supported by E2's paired bootstrap (every te_b difference excludes 0) and by NB tuned at .788, not by E1 alone.
   fix: Add McNemar or a paired bootstrap for E1, NB vs each measure, on the same 283 turns.
[low] Burrows Delta (and Cosine Delta to a lesser degree) depends on the random draw of the z-reference bags. With an independent draw (same design, different RNG stream) I got Burrows Delta .580 vs the reported .604 single-turn, and Cosine Delta .657 vs .664. The z-reference draw is not disclosed as a source of variance.
   fix: Repeat the z-reference with several seeds and report the spread, or average over the seeds.
[low] The post-hoc 'code-like' label is loose. style_nb.CODE_RE also fires on ( ) + / = [ ] > * _. Of the 304 turns it matches, 160 contain no code keyword, URL, ; or {}, with median length 26 characters. These are likely math or physics expressions or emoticons, not code.
   fix: Call the subset 'symbol/code-like (CODE_RE)' and keep the no-code result as a lead only, as the report already does.
[low] Zeta's collapse at bag level comes from the unnormalised sum the spec asks for (mean Z +0.062, so the score drifts toward T as texts get longer). It is not inherent to Zeta. With an OOF-tuned threshold Zeta scores .693 on single turns, close to the best formal measure. The report's wording 'summed Zeta' is accurate, but the drop decision should say that a length-normalised score was not tested.
   fix: Optionally add mean-Z (length-normalised) or the canonical marker-proportion Zeta before dropping it.

##### RECOMPUTED
- E1 te_b NB single-turn acc (sign), n=283: reported 0.731 vs recomputed 0.731 | Exact match, CI [.676, .782], recall S .485 / T .953. Uses my own LINE parser (formalv_data.py, written without reference to formal_chatshim), which reproduces every style_nb baseline: .731, pairwise .890 / .870, exhaustive .881 / .852, length-only .460. The shim is therefore validated. Verifier scripts are in the scratchpad: formalv_data.py, formalv_base.py, formalv_main.py, formalv_oofbag.py, formalv_boot.py, formalv_peekjson.py.
- E2 te_b NB pairwise acc, same 200 pairs: reported 0.89 vs recomputed 0.89 | Pairs come from style_nb.sample_pairs with Random(0), in the same pool order. 3,314 eligible pairs.
- E1 te_b Cosine Delta N300 single-turn acc (closer profile): reported 0.664 vs recomputed 0.657 | Independent z-reference (400 TRAIN bags of 25 turns, different RNG stream), CI [.599, .712]. te_c: .674 vs .674.
- E1 te_b Burrows Delta N300 single-turn acc: reported 0.604 vs recomputed 0.58 | Sensitive to which z-reference bags are drawn.
- E1 te_b Min-max N300 single-turn acc (thr 0): reported 0.604 vs recomputed 0.604 | Exact. Min-max needs no random reference.
- E2 te_b Min-max pairwise acc, 200 sampled pairs: reported 0.685 vs recomputed 0.685 | Exact, CP [.616, .749]. Cosine Delta pairwise: .655 vs reported .650.
- E3 Cosine Delta bag acc k=25 (threshold 0): reported 0.922 vs recomputed 0.93 | 300 bags per person from test turns with my own draws and direct counting on the joined text. k=50: .983 vs .993. Turn-bootstrap mean with a TRAIN-tuned threshold: .905 [.755, .990].
- E3 NB bag acc k=50 (sign rule): reported 0.877 vs recomputed 0.89 | With my own bag draws. The NB bag AUC at k=50 is 1.000, so every error comes from the threshold, not from a lack of signal.
- E3 NB bag acc k=10 with TRAIN-OOF bag threshold (new): reported 0.797 vs recomputed 0.955 | The reported value uses the sign rule; mine uses a threshold tuned on out-of-fold TRAIN bags (5 contiguous date blocks, refit per fold), which never sees test. Turn-bootstrap: .959 [.876, 1.000]. Cosine Delta under the same procedure: .833, bootstrap .855 [.723, .955].
- E3 NB bag acc k=25 with TRAIN-OOF bag threshold (new): reported 0.818 vs recomputed 0.998 | Turn-bootstrap: .994 [.962, 1.000], with 98% of reps >= .95. Cosine Delta: .917. Min-max at k=50 under the same rule: .960, vs .800 with threshold 0.
- E4 Cosine Delta k=50 share of reps cross > max within: reported 0.532 vs recomputed 0.545 | 200 reps. Within T .922, within S .939, cross 1.094 (reported .927 / .952 / 1.091). Cross / mean-within ratio 1.242 vs 1.231. Share cross > mean within .725 vs .705. Cosine Delta fails the order check, which confirms the report.

---

# D. 第一版的結論(已被第一輪審查推翻的部分,保留供對照)

## Interpretation
1. **A single short turn cannot be attributed reliably by any formal distance (test median 16 characters, 7 in te_c).**
   - On te_b (n = 283) the best result in the main setting is Min-max with a tuned threshold, 0.696 [0.639, 0.749]. The NB author-LM gets 0.731 [0.676, 0.782], or 0.788 tuned.
   - With the closer-profile rule, several measures collapse to near one-class output: JSD, raw-f cosine and Zeta call over 97% of turns T; Burrows Delta calls 95% S.
   - On the same 200 pairs, NB gets 0.890 [0.838, 0.930]. The best formal measure is NCD at 0.775 [0.711, 0.831], a difference of −0.115 [−0.221, −0.030]. Every count-based distance gets 0.63–0.71, and all their differences from NB exclude 0 on te_b.
2. **With all turns included, only Cosine Delta and NCD (zlib) reach ≥ 0.95, and only at k = 50 (both 0.993).**
   - Their bootstrap lower bounds are 0.83 and 0.84, so k = 50 is not a safe minimum.
   - NB stops at 0.877 at k = 50. Raw-f cosine, Min-max and JSD reach about 0.80, and Zeta falls to 0.50 because of the length drift above.
3. **None of the rulers passes the within < cross order check at k = 50 on all training turns** (pass shares 0.37–0.53; NCD 0.70).
   - Under content control they pass only when identical strings may repeat (E5b, ≥ 0.98).
   - With repeats excluded (E5a, k = 6) the pass shares are 0.74–0.85.
4. **Code-like turns appear to be the cause (post hoc).** They carry 60–70% of the training characters, so the bag profiles largely encode code, not style.
   - Removing them lifts the E4 pass shares to 0.86–0.97.
   - Min-max then reaches 0.948 at k = 5 and 0.985 at k = 10, and NCD 0.978 at k = 10.
   - Min-max single-turn accuracy then equals NB: 0.772 [0.714, 0.823] vs 0.772.
5. **Recommendation:**
   - Keep the NB author-LM log-likelihood ratio as the per-message ruler.
   - Adopt Cosine Delta (M2) as the formal bag-level ruler, with NCD-zlib as a model-free cross-check, and report results at k ≥ 10.
   - Re-evaluate Min-max on code-free text as a pre-registered next test.
   - Drop the summed Zeta score (length-biased), JSD and raw-f cosine (degenerate one-class decisions, failing rulers), and Burrows Delta (its accuracy curve rises and falls as k grows: 0.80 at k = 3, 0.66 at k = 10).
   - Do not use lzma.

---

# E. 拿掉符號類訊息後重驗(2026-10-13;10/10 預先登記的下一步)

**中文要點**
- 10/10 報告「分析與決定」第 6 點預先登記:拿掉符號類訊息後重跑。今天完成:純 Python、無 GPU,每種規則 83 秒;程式 `recheck_nocode.py`、結果 `recheck_nocode.json`(session scratchpad)。**這次沒有獨立審查**(使用者指示不開 subagent),發表前要重算一次。
- 兩種規則。**寬鬆** = `style_nb.CODE_RE`:任一半形符號 `; { } ( ) = < > # [ ] _ \ / * + & | % ^ $ @` 與反引號、網址、程式關鍵字。**嚴格** = 關鍵字 / 網址 / `; { }`,或「≥ 20 個非空白字元且符號占比 ≥ 30%」。
- **寬鬆規則最好**。k = 50 的尺驗證(跨人距離 > 兩個同人距離)通過率 0.91–0.96(隨機袋)、0.92–0.97(同時期袋);Cosine Delta 與 NCD 0.96–0.97,Min-max 0.94。未過濾只有 0.40–0.69;**只做同時期配對、不過濾,只升到 0.46–0.74** → 干擾來自「這一則是不是符號類」,不是時間漂移。
- 分袋判定:過濾後 NB k = 10 達 1.000 [0.99, 1.00](之前 0.94–0.96 臨界);Min-max 0.97、NCD 0.98、Cosine Delta 0.96;k = 25 全部 1.000。
- 單回合仍然分不出來:NB 0.75–0.77(調門檻 0.80–0.81);正式距離 0.69–0.77。
- k = 25 的尺驗證仍只有 0.73–0.89 → **驗證尺(within < cross)要 k = 50;判定作者 k = 10–25 就夠**,兩件事分開。
- **Pipeline ① 定案**:v1 用寬鬆規則(尺的擬合與候選評分都用)。待驗證的兩個修正:顏文字 / LINE 表情代碼先換成佔位符再套規則;前文裡的符號類回合換成 `[程式碼/算式]` 佔位符而不是整段刪掉。半形標點這個風格線索改由 R1 特徵在未過濾文本上另外計算。

## E.0 Setup
- Pre-registered in §A as "re-evaluate on code-free text". Single run on 2026-10-13, pure Python, no GPU, 83 s per rule. Not independently re-implemented this time; reproduce before publication.
- Same parser shim, turn definition, split (train < 2026-07-20), vocabulary (top-300 char 1–3-grams of the filtered pooled TRAIN), z-reference (400 TRAIN bags of 25, seed 0) and 5 contiguous date blocks as §A. Everything is refit on the filtered TRAIN.
- Two rules, applied to the text of a turn:
  - **loose** = `style_nb.CODE_RE`: any of `; { } ( ) = < > # [ ] _ \ / * + & | % ^ $ @` or a backtick, `http(s):`, or a programming keyword (cout, cin, int, endl, return, include, for, while, if, else, main, vector, string, char, bool, std, void, long, double, printf, scanf).
  - **strict** = a keyword, a URL, or `; { }`; or at least 20 non-space characters with a symbol share of at least 0.30 (symbols = ASCII letters and `+ - * / = ^ _ < > [ ] { } ( ) | \ % $ # @` or a backtick).
  - **none** = the §A data, rerun with today's settings as a control.
- E1 single turn: threshold 0, OOF-tuned threshold (5 date blocks), AUC. E3 bags: k in {5, 10, 25, 50}, 150 bags per person, per-k threshold from OOF TRAIN bags (40 per person per block), AUC, 200-rep turn bootstrap for the count-based measures. E4: 1,000 reps at k = 25 and 50, in two modes: **random** (bags drawn from the whole TRAIN period, as in §A) and **period-matched** (all bags of a rep drawn from one date block, so within-person and cross-person pairs face the same period).

## E.1 Data after filtering

| rule | TRAIN S / T (removed) | test S / T (removed) | share of TRAIN chars removed S / T | test S / T |
|---|---|---|---|---|
| none | 785 / 862 | 134 / 149 (n = 283) | 0 / 0 | 0 / 0 |
| loose | 679 / 706 (106 / 156) | 115 / 126 (n = 241; 19 / 23) | .679 / .600 | .701 / .376 |
| strict | 710 / 741 (75 / 121) | 118 / 128 (n = 246; 16 / 21) | .543 / .557 | .636 / .313 |

The loose rule removes 18% of the turns of T and 13.5% of the turns of S, so it also removes turns in which T uses half-width brackets or slashes inside ordinary chat. Whether that is a style cue was re-measured on 2026-10-13 after this run: on all training messages (code included) the full-width share of , . ! ? : ; is T 0.075 vs S 0.246, but on the non-symbol turns the rulers use it is T 0.32 vs S 0.17 on very few marks (T 81, S 159), with a bag-level discrimination of only 0.68. The 10/03 figure of 18% vs 61% could not be reproduced (its script is lost). Most of the half-width punctuation is therefore a code artefact, and removing it with the loose rule costs little; see E.6.

Symbol-like character share by date block (strict definition), S / T: 2025-06-20..08-24 .70 / .45; 08-25..09-20 .43 / .53; 09-25..11-29 .79 / .67; 12-05..2026-03-07 .18 / .32; 03-14..07-19 .31 / .64; test .64 / .31. Both people swing between .2 and .8 across periods, which is the within-person variance that broke E4.

## E.2 Single turn (different test sets per row; not a paired comparison)

| measure | rule | acc, threshold 0 [CP] | acc, OOF-tuned | AUC |
|---|---|---|---|---|
| NB LLR | none | .731 [.676, .782] | .788 [.736, .834] | .875 |
| NB LLR | loose | .772 [.714, .823] | .797 [.740, .846] | .885 |
| NB LLR | strict | .752 [.693, .805] | .813 [.759, .860] | .906 |
| Min-max | none | .604 | .696 | .744 |
| Min-max | loose | .772 [.714, .823] | .763 | .854 |
| Min-max | strict | .728 | .744 | .844 |
| Cosine Delta | none | .664 | .647 | .713 |
| Cosine Delta | loose | .710 | .714 | .805 |
| Cosine Delta | strict | .687 | .728 | .787 |
| NCD-zlib | none | .629 | .590 | .713 |
| NCD-zlib | loose | .718 | .722 | .814 |
| NCD-zlib | strict | .683 | .630 | .797 |

Filtering helps the distances a lot (Min-max AUC .744 to .854, NCD .713 to .814) and NB a little (.875 to .885 / .906). The sign rule of Min-max now equals the sign rule of NB (.772), but the tuned accuracy and the AUC of NB stay ahead. Conclusion (i) of §A stands: a single turn is attributed at .81 or below.

## E.3 Bags: calibrated accuracy (AUC in parentheses); 150 bags per person

| rule | measure | k = 5 | k = 10 | k = 25 | k = 50 |
|---|---|---|---|---|---|
| none | NB | .947 (.978) | .937 (.985) | .993 (1.000) | 1.000 (1.000) |
| none | Min-max | .820 (.906) | .857 (.885) | .917 (.983) | .983 (1.000) |
| none | Cosine Delta | .817 (.889) | .843 (.901) | .930 (.982) | .993 (1.000) |
| none | NCD | .823 (.933) | .833 (.933) | .930 (.997) | .967 (1.000) |
| loose | NB | .983 [.94, .99] (1.000) | **1.000 [.99, 1.00]** (1.000) | 1.000 | 1.000 |
| loose | Min-max | .967 [.91, .97] (.996) | .970 [.95, 1.00] (.998) | 1.000 | 1.000 |
| loose | Cosine Delta | .923 [.85, .95] (.978) | .963 [.94, .99] (.993) | 1.000 | 1.000 |
| loose | NCD | .940 (.990) | .983 (.998) | 1.000 | 1.000 |
| strict | NB | .967 (.999) | 1.000 (1.000) | 1.000 | 1.000 |
| strict | Min-max | .957 (.988) | .980 (1.000) | 1.000 | 1.000 |
| strict | Cosine Delta | .903 (.966) | .967 (.995) | 1.000 | 1.000 |
| strict | NCD | .797 (.992) | .907 (.999) | .980 (1.000) | 1.000 |

The none rows reproduce §A within Monte-Carlo noise (NB k = 10 .937 vs .955 with half the bags). After loose filtering NB is at 1.000 from k = 10 and every measure is at 1.000 from k = 25. NCD under the strict rule shows again that its threshold does not transfer (k = 5 accuracy .797 with AUC .992).

## E.4 Ruler validation: P(cross > max within), 1,000 reps

| rule | mode | k | Burrows Delta | Cosine Delta | Min-max | JSD | NCD |
|---|---|---|---|---|---|---|---|
| none | random | 25 | .423 | .492 | .407 | .417 | .621 |
| none | random | 50 | .439 | .537 | .397 | .426 | .685 |
| none | period-matched | 25 | .457 | .520 | .466 | .482 | .701 |
| none | period-matched | 50 | .520 | .588 | .464 | .458 | .742 |
| loose | random | 25 | .732 | .795 | .829 | .844 | .854 |
| loose | random | 50 | .909 | **.964** | .942 | .894 | **.964** |
| loose | period-matched | 25 | .806 | .823 | .844 | .854 | .886 |
| loose | period-matched | 50 | .951 | **.966** | .944 | .919 | **.973** |
| strict | random | 25 | .747 | .850 | .825 | .891 | .820 |
| strict | random | 50 | .834 | .956 | .848 | .844 | .843 |
| strict | period-matched | 25 | .771 | .843 | .824 | .872 | .841 |
| strict | period-matched | 50 | .886 | .953 | .859 | .877 | .832 |

Cross / mean-within ratio at k = 50, random bags: none 1.13 / 1.23 / 1.13 / 1.46 / 1.04; loose 1.20 / 1.30 / 1.22 / 1.52 / 1.05; period-matched loose 1.27 / 1.43 / 1.27 / 1.66 / 1.06 (same column order). The Monte-Carlo standard error of a pass share is at most .016.

- The none rows reproduce §A (.436 / .532 / .403 / .418 / .699).
- **Period matching without filtering adds only .03–.08.** The nuisance is at the turn level (is this turn a symbol-like turn), not slow drift between periods.
- **Loose filtering is what makes the rulers pass.** At k = 50, Cosine Delta and NCD clear the 95% criterion in both modes; Min-max is at .94; Burrows Delta passes only period-matched; JSD stays below.
- The strict rule leaves maths expressions, emoticons and bracketed fragments in, and three of the five distances stay at .83–.86.
- At k = 25 nothing passes (.73–.89). Ruler validation needs k = 50 bags; attribution needs k = 10–25. These are different requirements and the pipeline treats them separately.

## E.5 Interpretation
1. The §A failure of E4 was caused by symbol/code-like turns, as the post-hoc diagnostic suggested. Removing them with the loose rule moves the within < cross check from .40–.69 to .91–.97 at k = 50.
2. Period drift is real (symbol share swings .2–.8 between blocks) but is not what breaks the rulers; matching periods without filtering barely helps.
3. For attribution the picture is now comfortable: the NB author-LM separates 10-turn bags of the two people without error on the filtered test period (1.000 [.99, 1.00]); the formal distances need 10–25 turns.
4. A single short turn still cannot be attributed reliably (.81 or below), so any per-candidate score used in training is a noisy score and must be used through ranking, margins and bag-level statistics, never as a point judgement.
5. The best-validated ruler set after filtering: NB LLR (attribution, calibrated), NCD-zlib (model-free cross-check, passes E4 in both modes), Cosine Delta (passes E4 but depends on the two-author z-reference; provisional), Min-max (borderline .94; candidate).

## E.6 Decisions for the pipeline
- **Preprocessing (stage 1):** adopt the loose rule as the v1 filter. Turns it flags are excluded as targets and from all ruler fitting and scoring; in the context they are replaced by the placeholder `[程式碼/算式]` so the dialogue flow stays intact.
- Two refinements, to be validated by rerunning this section as a unit test before use: (a) replace LINE emoticon codes such as `(wave)` and ASCII emoticons such as `^^` with placeholders before applying the rule, so emoticons stay in the data as style; (b) keep half-width punctuation as a style cue by computing the R1 punctuation features on the unfiltered text. Acceptance: Cosine Delta and NCD at or above .95 and Min-max at or above .90 at k = 50 (random mode), and not more than .02 below the loose row.
- **Rulers (stage 3):** NB LLR with per-k calibrated thresholds for attribution; NCD and Cosine Delta for the within < cross validation at k = 50; feature-level checks on the unfiltered text.
- **Planning figures:** attribution bag k = 10 (NB) or 25 (any measure); ruler validation bag k = 50.

## E.7 Caveats
- Single run, no independent reimplementation.
- The three rules evaluate on different test subsets (283 / 241 / 246 turns); E.2 is not a paired comparison.
- The loose rule removes a larger share of the turns of T than of S (18% vs 13.5%); the turns flagged only by brackets or slashes are about 31 per person (symmetric). The half-width punctuation cue turned out to be mostly a code artefact (E.1), so the R1 punctuation feature is computed on the ruler population and reported with its weak discrimination (0.68).
- E4 pass shares at k = 25 are .73–.89 for every rule. If a future design needs validated distances at k = 25, the rulers have not earned that.
- Numbers come from 150 bags per person (E3) and 1,000 reps (E4); seeds as in the script.
