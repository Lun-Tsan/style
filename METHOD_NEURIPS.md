# Method, Experiments and Baselines (draft for a NeurIPS-style paper)

**Working title**: Fine-tuning a local language model to imitate one person's Traditional-Chinese chat style from a few hundred replies, with validated stylometric rulers as the training and acceptance signal

**Status (2026-10-13)**: Method section drafted from the validated parts of the pipeline (`PIPELINE_SPEC.md`); the Experiments section has two tiers: Tier 1 is runnable now on the existing data, Tier 2 is planned. Numbers quoted in this draft are from Experiments 1–4 (`PIPELINE_TECH_SURVEY.md`, `FORMAL_DISTANCE_EXPERIMENT.md`); every number that depends on training is marked **[TBD]**. Chinese key points (中文要點) follow each subsection. New references introduced during offline reasoning are marked *(bib to verify)* and must be checked before submission.

**Main claim**: an LLM can be fine-tuned, from a few hundred reply turns, to imitate a specific person's Traditional-Chinese (Taiwan colloquial) chat style, and this can be shown with rulers that were themselves validated on the same kind of text; the rulers are an intermediate step, not the contribution.

---

## 3 Method

### 3.1 Problem setting

We are given the message log of one two-party chat between a *target* person T and a *partner* S, exported from LINE. A **turn** is a maximal run of messages by one speaker on one day with gaps of at most 30 minutes between consecutive messages; a **segment** is a maximal run of turns with gaps of at most 6 hours. The data form a sequence of segments, each a sequence of turns $(u_1, \dots, u_m)$ with speaker labels in $\{T, S\}$.

A **sample** is a pair $(c, y)$ where $y$ is a turn by T and $c$ is the context: the up-to-12 preceding turns of the same segment (both speakers), rendered as text. The goal is a conditional generator $p_\theta(y \mid c)$ whose replies, under the conditions of the held-out period, cannot be told from T's own replies by rulers that *can* tell T from S, while staying on topic. The training set is small: a few hundred samples (here about 700 after filtering), which rules out the data regimes of prior persona work (thousands to tens of thousands of messages).

Three properties of the data drive the design. (i) Turns are short: the median turn has 16 characters, so a single turn carries little stylistic evidence. (ii) The log is a tutoring chat, so a large and time-varying share of the characters (0.2–0.8 depending on the period) are code, formulas or symbols rather than conversation. (iii) Only two authors exist, so classical multi-author stylometry has no reference population.

**中文要點**:單位是回合(同人、同日、間隔 ≤ 30 分);樣本 = 前 12 個回合 → T 的回合;目標是生成的回覆被「分得出 T 與 S 的尺」分不出來,同時不離題。資料三個特性:短、符號類內容多且隨時期變、只有兩位作者。

### 3.2 Overview

The method has three stages (Figure 1 [TBD]).

1. **Rulers** (§3.3): a small set of stylometric measures on bags of turns, each validated on T and S before use, with calibrated thresholds; plus interpretable feature tolerance bands.
2. **Supervised adaptation** (§3.4): LoRA fine-tuning of a 4B base or instruct model on the samples, with loss only on T's turns.
3. **Ruler-guided preference optimisation** (§3.5): sample candidates for each training context, rank them with the calibrated ruler inside interpretable behaviour cells, and run DPO on the resulting pairs against a fixed reference, with a capped score and content constraints.

Evaluation (§3.6) uses *different* rulers from training: an independent discriminator trained on the test period, a model-free compression distance, feature-distribution checks and a blind judgement by T.

**中文要點**:三階段:尺(先驗證、再校準)→ SFT LoRA → 尺排序的 DPO。驗收用另一組尺。

### 3.3 Rulers: validated stylometric distances on bags of turns

**Preprocessing for the rulers.** Turns are kept as text with newlines between messages and single placeholder characters for stickers, images and emoticons. A turn is flagged *symbol-like* if, after emoticon placeholders are inserted, it contains any of `; { } ( ) = < > # [ ] _ \ / * + & | % ^ $ @` or a backtick, a URL, or a programming keyword. Flagged turns are excluded from ruler fitting and scoring and never serve as targets; in contexts they are replaced by a placeholder. This rule was chosen empirically (Exp. 4): without it, within-person variation in symbol content exceeds the between-person difference and every distance fails the validity check below; a density-based rule that keeps formulas and bracketed fragments is weaker.

**Author language model (primary ruler).** Let $q_T$ and $q_S$ be character 1–3-gram multinomial models with add-one smoothing fitted on T's and S's training turns. For a text $x$ the score is the log-likelihood ratio $\ell(x) = \log q_T(x) - \log q_S(x)$ (plus the log prior ratio); for a bag $B$ of turns, $\ell(B) = \sum_{x \in B} \ell(x)$. This is the matched-filter statistic for "written by T", and in our experiments it beats every formal distance at every bag size (Exp. 3). Because $\ell(B)$ is a sum, its decision threshold depends on $|B|$; using 0 is miscalibrated (0.877 accuracy at $k = 50$ against 1.000 calibrated). We therefore tune a threshold per bag size on out-of-fold training bags from five contiguous date blocks, and map single-turn scores to calibrated log-odds by isotonic regression on the same out-of-fold scores (reporting $C_{llr}$ *(bib to verify: Brümmer & du Preez 2006)*).

**Model-free cross-check.** Normalised compression distance with zlib against ten reference bags of about 3,000 characters per author; the bag score is $\mathrm{NCD}(B, S) - \mathrm{NCD}(B, T)$. NCD needs no vocabulary and was the most stable measure across periods (worst-block AUC 0.992 at $k = 50$), but its threshold does not transfer, so it is reported as AUC or pairwise accuracy only.

**Provisional formal distance.** Cosine Delta on the 300 most frequent 1–3-grams, with z-scores taken from 400 random training bags because there is no multi-author reference. It passes the validity check after filtering but depends on the reference draw; it is kept for reporting and will be replaced by a proper multi-author reference when such data exist (Tier 2).

**Interpretable features and tolerance bands.** Sixteen features computed on the ruler population (non-symbol turns) (sentence-final particles in two groups, full-width punctuation share, sticker rate, burst size, message length, the `～` ending, catchphrases, question rate, letter and digit shares, laughter, first person, emoticons). For each feature the 5th–95th percentile over 1,000 training bags of T ($k = 50$) is the **tolerance band**. A bag of replies is penalised for leaving the band in either direction, which is what stops a generator from over-producing T's catchphrases; features whose bag-level T-vs-S AUC is below 0.6 would be reported but not weighted (on our data all sixteen are above 0.6); the weighted feature distance uses weights proportional to each feature's discrimination (AUC minus 0.5) with |z| capped at 5, so near-constant rare features cannot dominate.

**Validity checks (pre-registered, rerun whenever preprocessing changes).** A measure may be used only if (a) on bags of $k = 10$ turns from the held-out period, calibrated attribution accuracy is at least 0.95; (b) on 1,000 random draws of training bags with $k = 50$, the cross-person distance exceeds both within-person distances in at least 95% of draws, both when bags are drawn from the whole period and when all bags of a draw come from the same date block; (c) on the subset of short acknowledgement turns (content held fixed), out-of-fold single-turn AUC is at least 0.90. After filtering, the author LM reaches 1.000 [0.99, 1.00] at $k = 10$ and 0.944 on the content-controlled subset; Cosine Delta and NCD pass (b) at 0.96–0.97; Min-max is at 0.94 (Exp. 4). At $k = 25$ no distance passes (b) (0.73–0.89), so validity is claimed at $k = 50$ and attribution at $k = 10$–25, and the two are kept separate.

**What the rulers cannot do.** A single turn is attributed at 0.80 or below by every measure, and at 0.66–0.72 on seven-character turns. Every use of a per-candidate score in training is therefore treated as a noisy ranking signal (§3.5), never as a point judgement.

**中文要點**:主尺 = 字元 1–3-gram 作者 LM 的 LLR(袋內加總;門檻按袋大小校準、單回合分數用等張回歸校準);交叉檢查 = NCD;Cosine Delta 暫列。14 個可解釋特徵各有 T 的容忍帶(5–95 百分位),兩側都扣。三個有效性檢查:k=10 判定 ≥ 0.95、k=50 的 within<cross ≥ 95%、內容固定子集 AUC ≥ 0.90。單回合 ≤ 0.80,所以逐候選的分數只能當排序訊號。

### 3.4 Supervised adaptation

**Format.** Every sample is rendered by one function used identically in training and inference: one message per line, prefixed with `我:` (T) or `對方:` (S); the target is all lines of T's turn followed by an end-of-sequence token. The model therefore also learns *whether to send another message* (burst size, median 2 for T vs 1 for S), which the rulers treat as a feature. No chat template is used, for either the base or the instruct checkpoint, so the base-vs-instruct comparison is not confounded by templates and the prompt at inference cannot drift from the prompt in training. Contexts are truncated from the oldest turn to 512 tokens; the loss is computed only on target tokens, and a unit test asserts that the first target token is included.

**Fine-tuning.** LoRA with rank 8, $\alpha = 16$, dropout 0.05 on all linear projections of Qwen3.5-4B (base and instruct), bf16, learning rate $10^{-4}$ with cosine decay, effective batch 8, sequence length 768, 3–5 epochs depending on the subset size, with the epoch chosen on a time-ordered development split (last 15% of the training period) and the final model retrained on the full training period with that epoch count. Three fictitious canary sentences are added to the training set to audit memorisation.

**中文要點**:一個 prompt 函式,訓練推論共用;一則一行、每行有標記;目標含整個回合 + EOS(學連發);不用 chat template。LoRA r8 α16 全 linear;dev 選 epoch 後用全訓練期重訓;放金絲雀句稽核記憶。

### 3.5 Ruler-guided preference optimisation

**Candidates.** For each training context $c$ the SFT model draws $n = 8$ candidates (temperature 0.8, top-p 0.95). Each candidate $y$ receives a calibrated log-odds score $s(y)$ from the author LM, capped at the 95th percentile of the scores of T's real replies. The cap implements a *typicality band*: a reply that looks more like T than T's own replies do is a caricature, not an imitation *(bib to verify: Meister et al. 2023, locally typical sampling)*.

**Constraints.** A candidate can be *chosen* only if it is on topic (its bge-m3 cosine to the real reply exceeds the 90th percentile of cosines between unrelated real replies), is not a verbatim copy (no common substring of 12 or more characters with any training target), and contains no simplified-Chinese or Mainland-term leakage (OpenCC s2twp leaves it unchanged). Semantic embeddings are used *only* for this on-topic gate: on our data they separate on-topic from off-topic replies (AUC 0.94) but do not see style (within-person vs cross-person difference 0.018) and do not see polarity (cos(可以啊, 不行耶) = 0.77), so they must not score style.

**Behaviour cells.** Candidates are binned by four interpretable descriptors: length bucket, sticker or not, number of messages in the burst, sentence-final particle or not. The *target cell* of a context is the cell of T's real reply. The chosen reply is the highest-scoring qualifying candidate with $s > +\delta$ in the target cell; if the target cell holds no qualifying candidate, the neighbouring cells (same sticker flag, length bucket $\pm 1$) are searched, then any cell, and the level used is recorded and reported (with eight candidates an exact cell hit is rare); the rejected reply is the lowest-scoring candidate in any cell with $s < -\delta$ ($\delta = \log 3$ in calibrated log-odds). This quality-diversity construction *(bib to verify: Mouret & Clune 2015, MAP-Elites)* makes the chosen set follow T's empirical distribution over behaviours rather than collapsing onto the single safest reply, which is the known failure mode of preference optimisation on dialogue. If no candidate qualifies as chosen, the real reply is used (as in DITTO), limited to half of the pairs; if no candidate qualifies as rejected, the context yields no pair.

**Objective.** Standard DPO with $\beta = 0.1$ against the frozen SFT model as reference, one epoch, with LoRA on top of the merged SFT weights. No KL penalty beyond the implicit one; instead, training stops when the *evaluation* rulers (§3.6), computed on a development sample every 100 steps, stop improving. This follows the observation that KL penalties did not prevent over-optimisation in reward-model fine-tuning (Gao et al. 2023), and the Goodhart principle that an optimised measure stops measuring: the training ruler and the acceptance rulers are different objects.

**Best-of-$n$ as a cheaper first step.** Before DPO we report best-of-8 selection with the same score and constraints (RAFT-style). It costs no training and bounds how far the ruler alone can move the outputs.

**中文要點**:每前文 8 候選;分數 = 校準 log-odds,封頂在真實回覆的 95 百分位(典型帶);約束 = 離題門檻(bge-m3 只做這件事)、抄襲、用語外漏;行為格 = 長度 × 貼圖 × 連發 × 語氣詞,chosen 要在真實回覆的格內且 > +δ,rejected < −δ;沒有 chosen 就用真實回覆(DITTO,≤ 50%)。DPO β 0.1、參考固定、以驗收尺提早停止,不用 KL 懲罰。先報 best-of-8。

### 3.6 Evaluation protocol

**Independent discriminator.** The acceptance ruler is not the training ruler. On the held-out period we train a fresh character-n-gram discriminator between T's real replies and a system's replies to the same contexts (5-fold out-of-fold) and report its AUC on single turns (D-AUC$_1$) and on bags of 10 (D-AUC$_{10}$). Two controls fix the scale: the *noise floor* (real vs real, expected 0.5) and the *sensitivity* (real T vs real S, 0.885 single-turn after filtering). A system is read as "closer to T than S is" when its D-AUC is below the sensitivity value, and as indistinguishable when its interval covers the floor. Because the summed discriminator score drifts with bag size, bag-level results are always AUC or calibrated, never a fixed threshold. Folds are assigned by reply string, so an identical reply can never be in a training fold and a test fold at once; the shuffled-real control, which must sit at 0.5 by construction, verifies this (a per-context split let the discriminator memorise strings and gave 0.08).

**Other automatic measures.** Held-out negative log-likelihood of T's real replies under the model (the cheapest and most direct fit measure); the NCD score; the number of the 16 features inside T's tolerance band and the Jensen–Shannon divergence between generated and real feature distributions; on-topic rate; a coherence score $\log p(y \mid c) - \log p(y)$ under the base model; copy rate; OpenCC leakage per 100 characters; diversity (distinct-2 and the Hill number of order 1 after rarefaction to equal sample sizes *(bib to verify: Hill 1973)*); canary recall.

**Controls that define the axes.** Every measure is also computed for the real replies (ceiling), for *shuffled* real replies (T's replies reassigned to other contexts: perfect style, wrong content — this separates style measures from content measures), for S's real replies (negative control) and for the untrained base model.

**Human judgement.** T performs a blind two-alternative test (real vs system reply for the same context) with a sequential stopping rule *(bib to verify: Wald 1945, SPRT)*: items are added until the 95% interval of T's accuracy lies on one side of 0.65, with at most 100 items; and a preference test (best fine-tuned vs best prompting system). Items where the automatic rulers disagree are sampled first (query-by-committee).

**Acceptance lines (pre-registered; may only be tightened after the baselines are seen).** (1) D-AUC$_1 \le 0.75$ and D-AUC$_{10} \le 0.90$, and a paired-bootstrap improvement over the best prompting baseline whose interval excludes 0; (2) a band count not below that of the real held-out replies minus one (the real replies sit inside 12 of the 15–16 defined bands because of period drift, so an absolute threshold would be stricter than the person), diversity at least 80% of real, copy rate at most 2%, leakage not above the real replies; (3) T's blind accuracy at most 0.65 and preference for the fine-tuned system at least 0.60. Passing (1) and (2) but not (3) is reported as a failure of the rulers, not a success of the model.

**Statistics.** Three training seeds per system; paired bootstrap over test contexts (2,000 resamples) for every system comparison; Clopper–Pearson intervals for single-turn accuracies; turn-level bootstrap for bag-level accuracies, because overlapping bags are not independent.

**中文要點**:驗收尺 = 測試期重訓的獨立分辨器(報 AUC,附底線與靈敏度)、NLL、NCD、特徵帶、JSD、離題率、銜接、抄襲、外漏、多樣性、金絲雀。四個對照組(真實、打亂的真實、S、底模)把指標的座標定出來。本人盲判用序貫停止。三條驗收線預先登記。

### 3.7 Privacy and consent

All processing is local. Speaker names are replaced by role labels; phone numbers, identity numbers, addresses and e-mail addresses are replaced by placeholders with hand-written Taiwan-specific patterns. S's messages are used only as context and as the contrast author for ruler validation; S's consent is required before any publication that reports statistics derived from S's messages. Adapter weights are treated as derived personal data and are not released. The paper reports only aggregate statistics and n-grams of at most eight characters.

**中文要點**:全程本機;去識別化;S 的訊息只當前文與對照作者,發表前要同意;權重不公開;只報統計與 ≤ 8 字的 n-gram。

---

## 4 Experiments

### 4.1 Tier 1 — runnable now (one target person)

**Data.** One LINE log (4,967 messages, 2025-06-20 to 2026-09-19). Training period before 2026-07-20: 862 T turns (706 after symbol filtering) and 785 S turns (679); held-out period: 149 T turns (126) and 134 S turns (115). Median turn length 16 characters. Exact sample counts, context-length and target-length distributions, sticker share and per-month counts are reported in Table [TBD] (`samples.stats()`).

**Systems.**

| id | system | purpose |
|---|---|---|
| B0 | zero-shot, Qwen3.5-4B-Instruct, Chinese system prompt | prompting floor |
| B1-$n$ | many-shot ICL, $n \in \{10, 50, 200, \text{all}\}$ example pairs from training | the strongest prompting alternative with the same data; non-monotone in $n$ is expected |
| B2 | retrieval-augmented ICL, 20 nearest training contexts by bge-m3 | retrieval alternative |
| B3 | best-of-8 on SFT with the training ruler | ruler-only improvement, no preference training |
| SFT-$m$ | LoRA SFT on nested subsets $m \in \{100, 250, 500, \text{all}\}$, base and instruct | data-size curve; base vs instruct |
| SFT+DPO | §3.5 on the better checkpoint, all data | main system |
| C-real / C-shuf / C-S / C-base | controls (§3.6) | define the axes |

**Ablations** (best checkpoint, all data, 3 seeds): LoRA on attention projections only; 3-hour segment gap; recency weighting (2026 samples ×2); pairs without behaviour cells; no score cap; no real-reply fallback; DPO with the training ruler used also for early stopping (to show over-optimisation).

**Pre-registered hypotheses.** H1: SFT-all beats every prompting system on D-AUC$_1$, NLL and the band count. H2: the data-size curve is monotone from 100 to all on NLL but flattens on D-AUC (the discriminator saturates). H3: SFT+DPO improves D-AUC$_{10}$ and band count over SFT-all without lowering the on-topic rate or diversity by more than 10%. H4: the base checkpoint yields higher diversity and lower D-AUC than instruct at equal data; the instruct checkpoint yields a higher on-topic rate. H5: removing the score cap or the behaviour cells raises the training score but worsens the band count and diversity (Goodhart). H6: the blind accuracy of T on SFT+DPO is below that on the best prompting system.

**Compute.** One RTX 4070 Ti SUPER (16 GB). 24 SFT runs at an estimated 10–20 minutes each **[TBD]**; candidate generation about 5,600 samples; DPO runs under an hour each **[TBD]**. The whole tier fits in a few GPU-days.

**Expected outcome against prior work.** IMPersona reports 25% pass rate against acquaintances with 500 private messages and 44% with 13K; our 700 turns sit at the low end, so Tier 1 is expected to show a clear gap from "indistinguishable" on the human line while beating prompting on every automatic line. The contribution of Tier 1 is the validated-ruler methodology and the data-size and base-vs-instruct curves for Traditional-Chinese short messages, which do not exist in the literature.

**中文要點**:一人資料、約 700 訓練樣本;基準線 = zero-shot、many-shot(4 種 n)、檢索式、best-of-8;主系統 = SFT(4 個資料量 × 2 底模)與 SFT+DPO;七項消融;六個預先登記假設;一張 16 GB 卡幾天內跑完。預期:自動線贏 prompting、人判線離「分不出來」還遠。

### 4.2 Tier 2 — planned (generalisation)

- **Multiple consenting participants** (target 5–10, each with at least 500 turns, with partner identifiers in the context), to test whether the rulers' validity checks and the acceptance lines transfer across people, and whether group pre-adaptation followed by per-person LoRA (as in NTT's personalised setting) helps at this data size.
- **Multi-author reference** for Cosine Delta and a generalisability-theory decomposition of feature variance into person × period × topic *(bib to verify: Cronbach et al. 1972)*, which turns the validity check from pass/fail into an estimate of how many turns are needed.
- **Public Taiwanese short-text corpora** (PTT comments, Dcard replies) for cross-author ruler validation only; they are not private messages and cannot replace Tier 1 data.
- **Acquaintance blind tests** in place of self-judgement, following the IMPersona protocol, with the sequential stopping rule.
- **Sticker identity** once an export format with sticker ids is available.

**中文要點**:多人同意資料、多作者參照、變異數分解、公開語料只做尺的跨作者驗證、熟人盲測、貼圖編號。

### 4.3 Baselines from the literature and why they are not directly comparable

IMPersona (Llama-3.1-8B, English iMessage, acquaintance Turing test), DITTO and DIAL (English, population-level or demonstration-level) and Flipping the Dialogue (user-simulator LMs) are the closest methods. None reports Traditional-Chinese short-message results, a single person at the few-hundred-turn scale, or rulers validated on the target pair. We therefore reimplement their *training recipes* as baselines (SFT with the IMPersona hyperparameters; DITTO pairs with the real reply as chosen; DIAL-style discriminator-ranked pairs without cells or cap) rather than comparing against their published numbers.

**中文要點**:最近的四個方法都沒有繁中短訊、單人幾百則、驗證過的尺;所以重現它們的訓練配方當基準線,不比論文數字。

---

## 5 Limitations (to be stated)

- One target person and one partner in Tier 1; partner accommodation effects (emoji and style converging to the interlocutor) cannot be separated from T's style.
- The held-out period is 126 turns; intervals are wide and all comparisons are paired.
- The symbol filter removes 18% of T's turns and 13.5% of S's; the turns flagged only by brackets or slashes are symmetric (about 31 per person), and the half-width punctuation cue reported on 10/03 turned out to be mostly a code artefact (on the filtered chat it is weak: bag-level AUC 0.68).
- The author LM is fitted with S as the only contrast author; a one-class detector trained on T alone *(bib to verify: Forrest et al. 1994, negative selection)* is planned but not validated.
- Self-judgement by T is weaker evidence than judgement by acquaintances.
- Exp. 4 (the filter decision) is a single run without independent reimplementation.

**中文要點**:單人單對象、測試集小、過濾規則拿掉部分風格線索、對照作者只有一位、本人自判、實驗 4 未獨立重算。
