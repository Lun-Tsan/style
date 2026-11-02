# Handover:用 LINE 聊天紀錄微調本機 LLM 模仿一個人的聊天風格

更新:2026-10-14。給接手的人或新的 session 看;讀完這份應該知道我們在做什麼、做到哪、哪些數字可信、哪些還沒做、什麼不能碰。

> **TL;DR (English).** We fine-tune a local Qwen3.5-4B with LoRA on ~700 reply turns of one person's private LINE chat (Traditional Chinese, Taiwan colloquial, median 16 characters per turn) so that the model replies in that person's style. Before training anything we built and validated stylometric "rulers" (a char-n-gram author LM, compression distance, frequency-profile distances, 16 interpretable features with tolerance bands) on the target pair, with calibrated thresholds; the rulers are used as the training signal (ruler-ranked DPO pairs inside behaviour cells) and, with a *different* set of rulers, as the acceptance test. Data layer, rulers and the training/evaluation code exist and are tested (30 tests); the formal experiments (baselines, SFT grid, DPO, human blind test) have not been run yet. The chat data is private and never leaves the local machine.

---

## 1. 一句話

拿一個人(使用者本人,代號 **T**)在 LINE 上的幾百則回覆,LoRA 微調本機的 Qwen3.5-4B,讓模型回訊息的風格像他。方法上的重點是**先把「尺」做好、驗證過、校準過,再拿尺當訓練訊號和驗收標準**;尺是手段,論文的主張是「少量資料可以把風格學起來,而且能用可驗證的方式證明」。

對話的另一方(學生,代號 **S**)只當尺的對照作者;他的訊息只用在前文和對照,發表前要取得同意。

## 2. 跟既有研究的關係(不說「第一個」,只說文獻裡沒找到什麼)

| 最接近的工作 | 他們做的 | 我們不一樣的地方 |
|---|---|---|
| IMPersona(Princeton 2025;Llama-3.1-8B LoRA,英文 iMessage,熟人盲測) | 500 則 → 25% 騙過熟人;13K 則 + 記憶 → 44% | 繁中短訊、單人幾百則;他們的品質過濾規則(以空白切詞)在中文會刪光回覆,不能照搬;他們的 loss 遮罩少算第一個 token,我們有測試擋 |
| DITTO(真實回覆 = chosen) | 示範層級的 DPO | 我們混用:真實回覆只在沒有合格候選時當 chosen,且 ≤ 50% |
| DIAL(分辨器排序候選 → DPO) | 群體層級、學出來的分辨器 | 我們用可解釋的文體尺 + 行為格(長度 × 貼圖 × 連發 × 語氣詞),單一真人 |
| Flipping the Dialogue(ICLR 2026) | 群體層級:原始預訓練版比 instruct 更像使用者 | 我們把 Base vs Instruct 列為單人層級的待驗證假設 |
| Eder 2015、Baumler 2026、Gao 2023、Marko 2022 | 作者鑑定需 2,500–5,000 字才穩;作者向量與本人自評 r=0.24;KL 懲罰擋不住過度最佳化;emoji 用法會配合對象 | 分袋 + 校準門檻解短樣本;人判是必要的;訓練尺 ≠ 驗收尺;單一對象寫進限制 |

完整的文獻對照與 10 個「文獻找不到」的空白在 `PIPELINE_TECH_SURVEY.md`。

## 3. 資料(私人,不能離開這台機器)

- `Downloads/chathistory.txt`(路徑與兩人的 LINE 顯示名稱放在 git-ignored 的 `persona/local_speakers.json`):LINE 匯出,兩人,4,967 則訊息,2025-06-20 ~ 2026-09-19,家教對話(所以有很多程式碼和算式)。
- **回合** = 同一人、同一天、間隔 ≤ 30 分鐘的連發;**段落** = 間隔 ≤ 6 小時;共 2,291 回合、205 段落。
- **時間切分**:目標回合日期 ≥ 2026-07-20 → test。有文字的回合:train S 785 / T 862,test S 134 / T 149。
- **符號類回合**(程式碼、算式、網址;規則見 §4)不當目標、不進尺:過濾後 train S 698 / T 707,test S 118 / T 126。
- **SFT 樣本**(前 12 個回合 → T 的回合):train 797(扣 dev 118 → 679)、test 146;目標中位數 13 字;11–14% 的目標是純貼圖;一半的訓練前文裡有 `[程式碼/算式]` 佔位符。

**隱私規則**(使用者與指導教授的要求):
- 原文不進任何外部服務、不進雲端、不進 git。文件裡只放統計數字與 ≤ 8 字的 n-gram。
- 含原文的東西:`chathistory.txt`、`runs/*/outputs.jsonl`、`runs/*/candidates.jsonl`、`runs/*/pairs.jsonl`、`human_eval` 的題組、所有 adapter 權重(`runs/*/adapter_best`,微調會記住資料)。`.gitignore` 已排除這些。
- `reports/`、`scripts/*.json` 只有數字與短 n-gram,可以放。

## 4. 已經確定的事(每一項都有數字,來源標在後面)

**尺(實驗 1–4,10/03–10/13)**
1. 兩個人分得開,但**單一回合分不出來**:字元 1–3-gram NB 作者 LM 單回合 0.77(調門檻 0.81、AUC 0.89);所有正式距離更低;7 字短句全部 0.66–0.72。配對判斷 0.89。
2. **要幾則才穩**:NB 分袋 k=10 校準後 0.99–1.00(AUC 1.0 從 k=5 起),k=25 全部方法 1.00。**袋分數是加總,門檻一定要按 k 在訓練期 OOF 校準**;用門檻 0 會得到「NB 在 k=50 只有 0.877」的假象(第一版就是這樣錯的)。
3. **尺的有效性(同人兩半 < 跨人)**:不過濾符號類回合時五種距離全部不過(0.40–0.69);只配對同時期也不過(0.46–0.74);**拿掉符號類回合才過**:k=50 時 Cosine Delta 0.97、NCD 0.98、Min-max 0.95(隨機袋);k=25 仍只有 0.75–0.85。所以「驗證尺用 k=50、判定作者用 k=10–25」兩件事分開。
4. **符號類回合規則(定案)**:寬鬆規則 —— 任一半形符號 `; { } ( ) = < > # [ ] _ \ / * + & | % ^ $ @` 或反引號、網址、程式關鍵字;套規則前先把 LINE 表情代碼 `(cry)` 和 ASCII 顏文字換成佔位符(白名單版本的驗證也 PASS)。嚴格的符號密度規則比較差。
5. **語意向量(bge-m3)看不到風格、抓不到否定**:同人 vs 跨人 cosine 只差 0.018;cos(可以啊, 不行耶) = 0.77;只能用來篩離題(本題 vs 別題 AUC 0.94)。
6. **單回合校準**:等張回歸把 NB 的 LLR 轉成校準 log-odds,Cllr 1.77 → 0.60、Brier 0.176 → 0.130(OOF 訓練期)。
7. **16 個可解釋特徵**(語氣詞兩組、全/半形標點、句尾標點、貼圖率、連發、長度、～、口頭禪兩組、問句、字母、數字、笑聲、第一人稱、顏文字)在 k=50 袋上 T vs S 的鑑別度全部 ≥ 0.60;最強的是第一人稱 1.00、語氣詞 0.99、貼圖 0.99、問句 0.99。容忍帶 = T 訓練袋的 5–95 百分位。

**對照組(146 個測試前文,10/13)**
8. 獨立分辨器(測試期重訓的 NB,fold 按字串分):真人 vs 真人底線 0.46(k=10 袋 0.39),真人 T vs 真人 S 0.76(k=10 袋 1.00)。打亂的真實回覆(風格對、內容錯):風格指標與真實完全相同、離題率只剩 0.10 —— 這組對照證明風格指標和內容指標是分開的。
9. 真人 T 在測試期自己也只有 12.0 / 15.3 個特徵落在訓練期的容忍帶內(時期漂移),所以驗收線 2 定義成「不低於 C-real − 1」,不是絕對數字。
   **待確認 2026-10-17**:特徵帶只用文字回合配、驗收的袋卻含純貼圖回覆,C-real 的貼圖特徵因此永遠在帶外;12.0 / 15.3 至少有一部分是這個不一致,不全是時期漂移(`CODE_REVIEW_2026-10-17.md` #4)。C-S 的 Band 6.6 與 `base_s0_quick` 的 NLL 1.851 也受 #11、#1 影響。
   **2026-11-02 程式已改,真資料的數字還沒重算**:特徵帶、鑑別度、管制圖改用含純貼圖回合的母體(`persona/stage1.py:85-89`、`persona/data.py:27-36`);同時 C-S 改從訊息建(#11)、NLL 改算真實回覆(#1)、離題門檻改在訓練期配(#12)、多樣性稀疏化到 100 則(#13)。合成資料上 C-real 的 Band 從 13.7 升到 14.4(16 個特徵),方向和 #4 的推測一致。真資料的 12.0 / 15.3、C-S 的 6.6、`base_s0_quick` 的 1.851 要在本機重跑 `stage1` 與 `run_eval --system controls --overwrite` 後才有新值;在那之前,上面「時期漂移」的解釋與線 2「≥ C-real − 1」的理由仍待確認。

**程式與速度(10/13)**
10. `persona/` 套件全部寫完,30 個測試全過;新套件**逐格重現**實驗 4 的數字。
11. Qwen3.5-4B(Instruct)LoRA SFT:10.1 GB VRAM、約 1 s/step(短前文);全量 3 epoch 估 5–15 分鐘。Qwen3.5 是混合線性注意力模型,沒裝 `flash-linear-attention` / `causal_conv1d` 時走 PyTorch 參考實作。
    **待確認 2026-11-02**:「約 1 s/step」來自 `step_seconds`,它只量每步 4 個 micro-batch 中的最後一個加上參數更新(`persona/train_sft.py:130, :142`),所以一步的實際時間更長,5–15 分鐘的估計也可能偏低;`metrics.json` 新增的 `seconds`(整段牆鐘時間)下次跑完可以拿來重估。全部資料時,train+dev 重訓最多再加約 1.17 倍的訓練時間(797 / 679 個樣本)。

**被推翻或更正過的說法(寫論文時不要再用)**
- 10/03:「bge-m3 抓得到 可以啊 vs 不行耶」→ 錯,已更正。
- 10/10 第一版:「NB 在 k=50 停在 0.877」→ 門檻 0 的假象。
- 10/03:「全形標點 T 18% vs S 61%」→ 無法重現(腳本遺失);重算:含程式碼時 T 0.075 / S 0.246,非符號回合 T 0.32 / S 0.17 且標點極少 —— 半形標點線索大半是程式碼。
- 10/10 survey 第一版的新意描述太大,DIAL 已經做過「分辨器排序配對」;新意縮小為上面 §2 的差異。

## 5. Pipeline 定案(細節在 `PIPELINE_SPEC.md` §0)

| 項目 | 決定 |
|---|---|
| 格式 | 純文字,一則一行,`我:` / `對方:`,不用 chat template(Base 與 Instruct 相同);目標 = 整個回合 + EOS(學連發);前文最多 12 回合 / 512 token;符號類回合在前文換成 `[程式碼/算式]` |
| 切分 | 時間切分;train 最後 15% 當 dev;子集 100 / 250 / 500 / 全部,巢狀、時間等距 |
| SFT | LoRA r8 α16 dropout 0.05、全部 linear(含線性注意力的 in_proj_qkv / in_proj_z / out_proj);lr 1e-4 cosine、有效 batch 8、epoch 5/4/3/3;dev 選 epoch 後用全訓練期重訓;3 條金絲雀句稽核記憶 |
| 訓練用的尺 | NB 校準 log-odds(封頂在真實回覆的 95 百分位)+ 行為格(長度級距 × 貼圖 × 連發 × 語氣詞)+ 約束(bge-m3 離題門檻 0.59、12 字抄襲、OpenCC 外漏) |
| 配對 | 每前文 8 候選;chosen = 真實回覆所在格(沒有 → 鄰格 → 任一格,層級記錄)中 > +log 3 的最高分;rejected = < −log 3 的最低分;沒有 chosen → 真實回覆(DITTO),≤ 50% |
| DPO | 自寫;β 0.1、lr 5e-6、1 epoch;參考 = SFT 合併後凍結,logp 預先算好(單模型進 VRAM) |
| 驗收尺(≠ 訓練尺) | 測試期重訓的獨立 NB(報 AUC,附底線與靈敏度)、NLL、NCD、16 特徵容忍帶、特徵 JSD、離題率、MMI 銜接、抄襲率、外漏、多樣性(distinct-2、Hill q=1)、金絲雀;四個對照組 C-real / C-shuf / C-S / C-base |
| 三條驗收線(預先登記;看到 B0 / C-base 後只能調嚴) | (1) D-AUC₁ ≤ 0.75 且 D-AUC₁₀ ≤ 0.90,且配對 bootstrap 顯著優於最佳 prompting;(2) Band ≥ C-real − 1、多樣性 ≥ 真實的 80%、抄襲 ≤ 2%、外漏不高於 C-real;(3) 本人盲判「哪個是我寫的」正確率 ≤ 0.65(SPRT 停止,最多 100 題),偏好 ≥ 0.60 |
| 基準線 | B0 zero-shot、B1 many-shot(10/50/200/all)、B2 檢索式、B3 best-of-8;SFT 資料量曲線 × Base/Instruct;SFT+DPO;7 項消融 |

## 6. 還沒做、要決定的

1. **正式實驗一個都還沒跑**(`PIPELINE_SPEC.md` §7 步 3–7):先跑 B0 / C-base 鎖驗收線門檻 → 24 個 SFT 跑(2 底模 × 4 子集 × 3 種子)→ 配對 → DPO → 本人盲判。一張 16 GB 卡幾個 GPU 天。
2. `Qwen/Qwen3.5-4B-Base` **不在 HF 快取**(只有 Instruct 的 `Qwen/Qwen3.5-4B`、0.8B、2B、9B);Base vs Instruct 要先下載約 8 GB。
3. OpenCC 未裝(`opencc-python-reimplemented`),外漏指標目前回 None。
4. 線性注意力核心未裝;1 s/step 若可接受就不用裝(Windows 上不一定裝得起來)。
5. DPO 的提早停止目前只在 epoch 結尾評估,還沒接上「每 N 步用驗收尺看 dev 生成」。
   **更正 2026-10-17**:DPO 目前完全沒有評估(`eval_every` 沒被使用),存的是最後一步(`CODE_REVIEW_2026-10-17.md` #8)。
   **2026-11-02 已接上,GPU 上還沒跑過**:第 0 步、步數為 10 的倍數時(`C.DPO["eval_every"]`)與最後一步之後,在 60 個 dev 前文(`C.DPO["dev_eval_n"]`)上各生成 1 則,依 `symbol_share`、D-AUC₁₀、Band 選檢查點;SFT 的 `adapter_best` 用 train+dev 重訓過時,DPO 跑兩次以保留 dev,時間約加倍(`PIPELINE_SPEC.md` §10)。這個選法與 §5.4 的「D-AUC 不再下降就停」不同,正式 DPO 前要確認。
6. 實驗 4 與 stage-1 的所有數字都是**單次執行、未獨立重算**;實驗 1–3 有另一個 agent 重寫重算過。發表前要重算。
7. 離線推理引入的文獻(Spearman–Brown、Wald SPRT、Brümmer & du Preez Cllr、PAV、概化理論、合成控制、MAP-Elites、典型取樣、Hill 數、Bray–Curtis、負向選擇、Shewhart)全部標了「書目待查證」,寫論文前要回原文查。**2026-10-17**:12 項書目已依索引紀錄查證(年份作者無誤,Spearman 與 Brown 是兩篇;完整書目在 `PROBLEMS_AND_SOLUTIONS.md` 文末),內容不符的 9 處已加「更正」(Spearman–Brown 預測信度不是準確率、Cllr 不等於校準損失、Min–max ≠ Bray–Curtis 只是單調等價、合成控制 / MAP-Elites / 典型取樣 / 負向選擇只能當類比、DiD 不是概化理論特例、CUSUM 是 Page 1954),另 `METHOD_NEURIPS.md` §3.6 的「區間落在 0.65 一側就停」不是 SPRT;還沒做:7 處 *(bib to verify)* 未進 survey 參考文獻、全文未讀、口語尺度 / 稀疏化 / 單類別分類的出處待查。
8. S 的同意(發表前)。
9. Tier 2(多人、多作者參照、熟人盲測、貼圖編號)只有需求,沒有資料。
10. **程式審查(2026-10-17)**:`CODE_REVIEW_2026-10-17.md`,18 項;正式實驗前要修的有 #1 #2 #3 #4 #11 #12 #14,還沒動手。
    **2026-11-02**:18 項都處理了(17 項改程式,#18a 改 `PIPELINE_SPEC.md` §0);#1 #8 #9 #12 #15 #16 #17 需要模型的部分沒有在雲端執行。修完後的複查另找到 14 項,11 項已修。逐項狀態在 `CODE_REVIEW_2026-10-17.md` 文末。**待使用者決定**:(a) 盲判 SPRT 的無差異區要不要調嚴(現在真實正確率 0.70 的系統通過線 3 的機率 0.20,`PIPELINE_SPEC.md` §6.2);(b) DPO 選檢查點的規則(與 §5.4 不同);(c) 配對改用封頂後的原始 LLR 排序(超出審查的修法);(d) 子集 100 / 250 / 500 的重訓資料取法;(e) 題型 A 一組題目放幾個系統(146 個測試前文);(f) `symbol_share` 要不要門檻;(g) 驗收線的三個實作解讀(`PIPELINE_SPEC.md` §6.3)。
11. **2026-11-02 要在本機重跑的**(依序;已存在的輸出目錄要加 `--overwrite`):`python tests/run_tests.py` → `python -m persona.stage1`(重產 `reports/feature_bands.json`,新寫 `reports/data_manifest.json`)→ `python -m persona.run_eval --system controls --overwrite` → 冒煙 `train_sft --quick`、`make_pairs --quick`、`train_dpo --quick`、`run_eval --system sft --run sft_quick_all_s0 --dpo dpo_sft_quick_all_s0_pairs_quick_s0 --quick`(`train_sft --quick` 不做重訓,要測重訓用 `--size 100`)→ `run_eval --system base` 與基準線(B1-200、B1-all 先跑,看記下的 prompt token 數)。2026-11-02 之前在本機產生的 SFT、配對、DPO、評估輸出與 `reports/eval_table.json` 的舊列是舊程式算的(原因列在 `CODE_REVIEW_2026-10-17.md` 文末)。

## 7. 檔案地圖(本機 `Desktop/tmp/qwen_chat_analysis/`;GitHub `Lun-Tsan/style`)

| 檔案 | 內容 | 讀的順序 |
|---|---|---|
| `handover.md` | 本文件 | 1 |
| `PIPELINE_SPEC.md` | 每個元件的輸入/輸出/參數/驗收測試;§0 定案總表;§10 實作狀態與實作時改掉的設計 | 2 |
| `METHOD_NEURIPS.md` | 論文 Method + Experiments 草稿(英文正文 + 每小節中文要點;Tier 1 可跑 / Tier 2 規劃;H1–H6 預登記假設;Limitations) | 3 |
| `PIPELINE_TECH_SURVEY.md` | 文獻對照(每階段 ✅/🟡/⬜)、留空彙整、**實驗 1–4 的中文紀錄與表格**、參考文獻 | 4 |
| `FORMAL_DISTANCE_EXPERIMENT.md` | 實驗 3 的完整英文報告、兩輪審查、被推翻的第一版結論;§E 是實驗 4 | 查數字時 |
| `PROBLEMS_AND_SOLUTIONS.md` | 19 個問題、43 個解法(跨領域:心理計量、序貫分析、雷達 CFAR、法庭科學 LR、免疫負向選擇、MAP-Elites、製程管制……),哪些進 v1 | 想改方法時 |
| `persona/` + `persona/README.md` | 程式碼與執行順序 | 跑東西時 |
| `tests/` | 30 個測試(2026-11-02 為 54 個,見 §9),`python tests/run_tests.py`(`--fast` 跳過需要尺驗證與 tokenizer 的) | — |
| `reports/` | stage-1 產出:`data_stats.json`、`ruler_validation.json/.md`、`nb_calibration.json`、`feature_bands.json`、`eval_table.json`(對照組) | — |
| `runs/` | 冒煙測試的產物(0.8B)與對照組輸出;**含原文,不外流** | — |
| `scripts/` | 實驗 1–4 的原始腳本(`style_nb.py`、`formal_*.py`、`recheck_nocode.py`)與數字結果 | 重算時 |

環境:venv `~/venvs/qwen35`(Python 3.12、torch 2.12.1+cu126、transformers 5.17、peft 0.21;**沒有** pytest、sklearn、opencc、trl);GPU RTX 4070 Ti SUPER 16 GB;模型在 HF 快取(離線載入,`HF_HUB_OFFLINE=1`)。Windows 上要 patch `sdpa_attention.use_gqa_in_sdpa`(`persona/modeling.py` 已做)。

## 8. 工作規則(來自使用者與指導教授)

- **誠實條款**:不寫無法驗證的形容詞(well-trained → pre-trained),不寫 None / every / never 這種絕對句,不說「第一個」;被推翻的結論留在文件裡標明,不刪。
- **三條驗收線 + 可重複性**:每個主張要有數字、信賴區間、種子;門檻預先登記,只能調嚴。
- **概念先於動手**:「請繼續」不等於授權裝套件、寫程式、用 GPU;動手前先問。
- **survey 可以上網;找解法時斷網用深度推理**,離線引入的文獻標「書目待查證」。
- **一次只開 1–2 個 agent**(曾經 203 個並行把額度燒光);subagent 一律用 Opus 5.5;10/13 那天要求完全不開 subagent。
- 論文章節格式:英文正文 + 每小節中文要點;.tex 一句一行。
- Commit:標題短、白話,不加 Co-Authored-By;公開 README 不要寫得像 AI。

## 9. 給雲端 session 的話

雲端 session 拿不到資料、模型、GPU,所以:
- **能做**:讀/改文件與程式、查證「書目待查證」清單、補單元測試、審 `persona/` 的程式、寫論文其他章節、把 `METHOD_NEURIPS.md` 改成 LaTeX。
- **不能做**:跑 `stage1`、訓練、評估、任何需要 `chathistory.txt` 的測試(`tests/test_data.py` 等會因找不到檔案而失敗;只有 `test_prompt`、`test_train`、`test_rulers.test_stats_helpers` 不需要資料)。
- **值得交給雲端的一件事**:寫一個**合成 LINE 匯出產生器**(假的兩人、假的訊息,格式同 `persona/parse_line.py` 的說明),讓資料層與尺的測試在沒有真資料時也能跑;產生器不可含任何真實訊息。
  **2026-10-17 已完成**:`persona/synth.py`;`python tests/run_tests.py --synth` 在雲端跑 23 個測試全過(缺 torch 的 `test_eval`、`test_train`、`test_mask` 記為 SKIP),`PERSONA_SYNTH=1 python -m persona.stage1` 整個跑完、輸出在 `runs/synth/`。合成資料上單回合 NB 0.83 / 調門檻 0.82 / AUC 0.89、k=10 0.99,形狀接近真資料,但只用來驗證程式,不支持任何關於真人的主張。
  **2026-11-02**:不 import torch 的 `persona/text.py`、`persona/data.py` 拆出來之後,evaluate、pairs、human_eval、report 的測試(`test_eval`、`test_train`、`test_human`)也在雲端用合成資料跑:`python tests/run_tests.py --synth` 共 54 個測試,47 過、0 失敗、7 SKIP(`test_mask` 2 個缺 transformers、`test_train` 5 個缺 torch:DPO loss、基準線解析,以及 `token_batches` / DPO 選檢查點 / 執行目錄名的純函式測試,這三個用替身 torch 跑過一次)。生成、NLL、MMI、bge-m3、訓練這些要模型的路徑仍然只能在本機跑。
- 要開雲端 session 前,這個資料夾得先變成 git repo 並推到 GitHub;`.gitignore` 已經寫好(排除 `chathistory.txt`、`runs/`、`*.jsonl`、adapter 權重)。**推之前請再檢查一次 `git status` 裡沒有原文。**

## 10. 建議的下一步順序

1. 裝 OpenCC;決定要不要下載 Base。(2026-11-02:先做 §6 第 11 點的重跑,並決定 §6 第 10 點列的待決定事項。)
2. 跑 `run_eval --system base` 與 `--system baseline --kind zero`(各幾分鐘),把驗收線門檻鎖下來(只能調嚴)。
3. SFT 資料量曲線(先 Instruct、3 種子),每跑完一個用 `run_eval --system sft` 看 NLL 與 D-AUC。
4. 最佳 SFT → `make_pairs` → `train_dpo` → 評估。
5. 本人盲判(`human_eval.make_items`,SPRT 停止)。
6. 找人獨立重算實驗 4 與 stage-1;查證書目;把結果填進 `METHOD_NEURIPS.md` 的 [TBD]。
