# Pipeline 規格 v1:每個元件要做什麼、參數定多少、為什麼

**日期**:2026-10-13
**範圍**:Tier 1 —— 用現有資料(`chathistory.txt`,目標人物 T = Hung LT)現在就能跑的版本。Tier 2(多人、公開語料)只在 §9 列需求。
**依據**:`PIPELINE_TECH_SURVEY.md`(文獻與實驗 1–4)、`FORMAL_DISTANCE_EXPERIMENT.md` §E、`PROBLEMS_AND_SOLUTIONS.md`(P1–P19 / S1–S43)。
**原則**:(1) 每個元件有輸入、輸出、固定參數、驗收測試;(2) 訓練用的尺和驗收用的尺分開;(3) 所有門檻在訓練期校準,測試期只用一次;(4) 全程本機,聊天原文不離開這台機器。
**狀態標記**:✅ 已有程式並驗證過 🔧 要寫(規格已定)⏳ 已定案但待第一次跑完才知道數值

---

## 0. 定案總表(所有先前的 ⏳)

| 項目 | 決定 | 理由 / 依據 |
|---|---|---|
| 目標人物 | T(使用者本人);S 只當尺的對照作者與驗收的靈敏度對照 | 同意問題最小;S 的訊息發表前要取得同意 |
| 單位 | **回合** = 同一人、同一天、間隔 ≤ 30 分鐘的連發 | 與實驗 1–4 一致;單則訊息另外報比例(S38) |
| 對話段落 | 間隔 > 6 小時切段;消融 3 小時 | IMPersona 預設;家教對話段落天然清楚 |
| 前文 | 同段落內最近 12 個回合,token 上限 512,不放時間戳記 | 回合中位數 16 字 → 12 回合約 300 token;回覆延遲不具鑑別力(實驗 1) |
| 說話者標記 | `我:` / `對方:` | 去識別化(S43);訓練與推論同一格式 |
| 連發 | 一則訊息一行、每行都帶標記;目標 = 整個回合的所有行 + `<eos>` | 讓模型學「要不要再發一則」(S2);尺把 ⏎ 當特徵 |
| 貼圖 / 圖片 / 影片 | `[貼圖]` `[圖片]` `[影片]` 當文字,可當目標 | LINE 匯出沒有貼圖編號,只能學何時發(S36) |
| 符號類回合 | **寬鬆規則**(實驗 4)。不當目標、不進尺;前文中換成 `[程式碼/算式]`;套規則前先把顏文字換佔位符 | 實驗 4;S3、S42 |
| 切分 | 目標回合日期 ≥ 2026-07-20 → test;train 內最後 15%(依時間)→ dev | 整段對話為單位;dev 只用來選 epoch 與校準(**2026-11-02**:也用來選 DPO 的檢查點;選完 epoch 後 dev 併入 SFT 重訓,見 §10) |
| 子集 | 100 / 250 / 500 / 全部,巢狀,時間上等距抽 | 各子集的時期組成相同 |
| 底模 | Qwen3.5-4B **Base 與 Instruct 都跑**;兩者都用同一純文字格式,不套 chat template | 消除模板混淆;Flipping the Dialogue(S35) |
| LoRA | r 8、α 16、dropout 0.05、全部 linear 層;lr 1e-4、cosine、warmup 3%、有效 batch 8、epoch 5 / 4 / 3 / 3(100 / 250 / 500 / 全部)、bf16、梯度檢查點、seq 768 | IMPersona + 文獻共識;attention-only 當消融 |
| 解碼 | temperature 0.8、top-p 0.95、max 64 new tokens、停在 `\n對方:` 或 `<eos>`;評估 1 個樣本 / 前文 × 3 個種子 | — |
| 訓練用的尺 | NB 作者 LM 的 LLR(等張校準)+ R1 特徵容忍帶。**更正 2026-11-02**(`CODE_REVIEW_2026-10-17.md` #18a):配對程式沒有用 R1 特徵帶(`rulers.features.feature_distance` 沒有被呼叫)。程式實際用的訓練尺 = NB 校準 log-odds(在原始 LLR 尺度封頂)+ 行為格(長度級距 × 貼圖 × 連發 × 語氣詞)+ 約束(bge-m3 離題、12 字抄襲、OpenCC 外漏);符號類與沒結束的候選不參加(§5.3、§10)。R1 特徵帶用在驗收尺(Band)與 DPO 在 dev 上選檢查點(§5.4、§10) | 實驗 3、4;S10、S28 |
| 驗收用的尺 | 測試期重訓的獨立 NB(real vs generated)+ NCD + R1 分布 + 本人盲判 | 訓練尺 ≠ 驗收尺(S29) |
| 偏好訓練 | 先 best-of-N(RAFT),再 DPO;**自寫 DPO**(不裝 trl);β 0.1、lr 5e-6、1 epoch;參考模型 = SFT 模型,固定 | Gao 2023;遮罩要自己控制 |
| 配對 | 每個前文 8 個候選;chosen = 與真實回覆同一「行為格」且 LLR > +δ 的最高分者;rejected = LLR < −δ 的最低分者;沒有合格 chosen 時用真實回覆(DITTO),上限 50% | DIAL + DITTO + MAP-Elites(S25) |
| 新套件 | 不裝 trl、bitsandbytes;**OpenCC 要裝(裝前先問)**;其餘用現有 venv | 回饋規則 |

---

## 1. 模組地圖

```
persona/
  config.py            所有常數(本文件 §0 的數值)
  parse_line.py        LINE 匯出檔 → 訊息列表                      ✅(以 formal_chatshim 驗證過)
  turns.py             訊息 → 回合 → 段落                          ✅ build_turns;🔧 segments
  normalize.py         佔位符(貼圖/表情)、去識別化                  🔧
  flag_symbols.py      符號類回合判定(寬鬆規則 + 白名單)            ✅ CODE_RE;🔧 白名單
  split.py             時間切分、日期區塊、巢狀子集                  🔧
  samples.py           前文 → 目標回合 的樣本                        🔧
  prompt.py            唯一的 prompt 組裝器 + loss 遮罩              🔧
  rulers/
    features.py        R1 可解釋特徵 + 容忍帶                        🔧
    author_lm.py       NB 字元 1–3-gram 作者 LM                      ✅ style_nb.NB
    ncd.py             NCD-zlib                                      ✅ formal_lib
    delta.py           Cosine Delta(暫列)                           ✅ formal_lib
    calibrate.py       OOF 分數、每 k 門檻、等張校準、Cllr            ✅ tune_threshold;🔧 等張、Cllr
    validate.py        E1 / E3 / E4 / E5 當單元測試                   ✅ recheck_nocode;🔧 包成測試
  baselines.py         zero-shot、many-shot、檢索式 many-shot、對照組  🔧
  train_sft.py         LoRA SFT                                       🔧
  generate.py          批次取樣候選                                   🔧
  pairs.py             DPO 配對(行為格、約束、邊際)                  🔧
  train_dpo.py         自寫 DPO                                       🔧
  evaluate.py          全部自動指標                                   🔧
  human_eval.py        本人盲判題組 + SPRT 停止                        🔧
  report.py            表格、CI、配對 bootstrap                        🔧
  tests/               每個元件的驗收測試                              🔧
```

---

## 2. 資料層(① ②)

### 2.1 `parse_line.parse(path) -> list[Msg]` ✅
- 輸入:UTF-8-sig 文字檔。日期行 `2025.07.18 星期五`;訊息行 `HH:MM<TAB>name<TAB>text`;其餘行併入前一則(多行訊息)。
- 輸出:`Msg(date, time, speaker, text, kind)`,`kind ∈ {text, sticker, image, video, system}`。
- 驗收:訊息總數 4,967;兩位說話者;與 `formal_chatshim` 的回合數一致(train S 785 / T 862,test S 134 / T 149)。

### 2.2 `turns.build_turns(msgs) -> list[Turn]` ✅ / `turns.segments(turns, gap_h=6) -> list[Segment]` 🔧
- 回合:同人、同日、相鄰訊息間隔 ≤ 30 分。`Turn(speaker, date, t_start, msgs: list[str], text_b)`,`text_b` = 訊息以 `\n` 接起來(尺用這個)。
- 段落:相鄰回合間隔 > 6 小時就切;段落是切分與前文的邊界。
- 驗收:回合數同 2.1;段落數列入資料統計表;3 小時版本的段落數也列。

### 2.3 `normalize` 🔧
- `placeholders(text)`:`[貼圖]→`、`[圖片]→`、`[影片]→`(已在 style_nb);**新增** LINE 表情代碼 `\((wave|cry|smile|...)\)` → ``,ASCII 顏文字(白名單:`^^` `^_^` `QQ` `XD` `:)` `:(` `= =` `orz`,可擴充)→ ``。白名單從訓練期統計半形括號片段的前 50 名人工挑。
- `deid(text)`:台灣手機 `09\d{8}`、市話、身分證 `[A-Z][12]\d{8}`、地址(路/街/巷/號 模式)、email → `[電話]` `[證號]` `[地址]` `[信箱]`;說話者改 `我` / `對方`。
- 驗收:佔位符往返可逆(顯示時換回 `[貼圖]`);去識別化在訓練期的命中數列表,人工看一遍(只看命中片段的類別統計,不印原文)。

### 2.4 `flag_symbols.is_symbol_turn(text_after_placeholders) -> bool` ✅ 規則 / 🔧 白名單
- 規則 = 實驗 4 的寬鬆規則 `CODE_RE`:任一 `; { } ( ) = < > # [ ] _ \ / * + & | % ^ $ @` 或反引號、`http(s):`、程式關鍵字。
- 套用順序:先 `normalize.placeholders`(顏文字已變成私用字元,不會被括號規則抓到),再套規則。
- 用途:(a) 被抓到的回合**不當目標**;(b) **不進尺**(擬合、校準、評分都排除);(c) 在前文裡換成 `[程式碼/算式]`。
- 驗收(預先登記,第一個要跑的測試):重跑 `rulers.validate.e4` on 白名單版規則,k=50 隨機袋:Cosine Delta、NCD ≥ 0.95,Min-max ≥ 0.90,且不低於實驗 4 寬鬆列 0.02 以上。不過就退回純寬鬆規則(白名單只在顯示層用)。

### 2.5 `split` 🔧
- `time_split(samples)`:目標回合日期 ≥ 2026-07-20 → test;其餘 train。前文可以含 test 日期之前的回合(那是推論時本來就看得到的歷史)。
- `dev_split(train)`:train 依目標日期排序,最後 15% → dev。dev 用途:SFT 選 epoch、尺的門檻二次確認。**最終模型用 train+dev 以選定的 epoch 數重訓**,報告時註明。(**2026-11-02 已實作**:`train_sft` 階段 2,見 §10;子集 100 / 250 / 500 的重訓資料怎麼取待確認。)
- `date_blocks(train, n=5)`:連續日期區塊,尺的 OOF 校準用(實驗 3、4 的做法)。
- `nested_subsets(train, sizes=(100, 250, 500, None), seed)`:把 train 依時間排序,子集 n 取等距索引(`round(i * N / n)`),所以 100 ⊂ 250 ⊂ 500 ⊂ 全部不保證嚴格包含;改為:先取 100 等距,250 = 100 ∪ 再從剩餘等距取 150,以此類推(巢狀且時期均勻)。
- 驗收:每個子集的「年-月」分布直方圖與全集的 JSD < 0.05;dev 與 train 無同段落樣本。

### 2.6 `samples.build(segments, target="T") -> list[Sample]` 🔧
- 對段落中每個 T 的回合 t_i(非符號類、非系統訊息):`Sample(ctx = 前 12 個回合(同段落內;符號類回合以 `[程式碼/算式]` 取代文字,保留說話者), target = t_i, meta = {date, n_msgs, has_sticker, len_chars, segment_id, block_id})`。
- 段落第一個回合若是 T 的(沒有前文)→ 保留,ctx 為空(模型學「開場」)。
- 預期數量(待 `samples.stats()` 確認):train 約 700、test 約 126(實驗 4 的 706 / 126 是回合數,扣掉少數系統訊息)。
- 驗收:`stats()` 輸出:樣本數、前文回合數分布、目標長度分布、貼圖目標比例、每月樣本數;**這張表進論文的資料節**。

### 2.7 `prompt.render(sample) -> (prompt_text, target_text)` / `prompt.encode(tokenizer, sample) -> (input_ids, loss_mask)` 🔧
- 格式(純文字,Base 與 Instruct 相同):
  ```
  對方: 這題我不會
  對方: [貼圖]
  我: 哪一題
  我: 拍給我看
  對方: ...
  我: <目標回合第一則>
  我: <目標回合第二則>
  <eos>
  ```
- `prompt_text` 以 `我:` 結尾(含空格);`target_text` = 目標回合第一則的文字 + 之後每則前綴 `\n我: ` + 最後 `<eos>`。
- loss 遮罩:只有 `target_text` 的 token 為 1;**驗收測試:目標第一個內容 token 的 mask = 1,prompt 最後一個 token 的 mask = 0**(IMPersona 的 off-by-one)。
- 訓練與推論**只能**經過這個函式;`generate` 用 `prompt_text` 直接續寫。
- 截斷:從最舊的前文回合開始丟,直到 ≤ 512 token(目標不截)。

---

## 3. 尺(③)

### 3.1 `rulers.author_lm.NB` ✅
- 字元 1–3-gram(含 `\n` 與佔位字元),add-1 平滑(α = 1.0),含類別先驗;`margin(text)` = log P(S|x) − log P(T|x)。**pipeline 用 `llr_T(text) = −margin`**(越大越像 T)。
- 擬合資料:**train 的非符號類回合**(S 與 T 都要;S 是對照作者)。
- 單回合效能(過濾後 test,實驗 4):0.77 / 調門檻 0.80 / AUC 0.885 → **單一候選的分數是吵雜的**,只做排序與邊際用(§5.2)。
- 袋:`bag_llr(texts) = Σ llr_T`;每 k 的門檻由 `calibrate.per_k_threshold` 給。

### 3.2 `rulers.ncd` ✅ / `rulers.delta` ✅
- NCD:zlib-9,`ncd(x, ref)` 對每人 10 個約 3,000 字的訓練參考袋取平均;分數 = ncd_S − ncd_T(越大越像 T)。用途:驗收與尺驗證(k ≥ 25),不用於訓練。
- Cosine Delta:詞表 300、z 參照 400 個 25 回合訓練袋(seed 0)。暫列;只用於 E4 驗證報告,Tier 2 有多作者參照時再升格。

### 3.3 `rulers.features` 🔧(R1,算在**未過濾**的回合上,除非註明)
每個特徵定義為「每則訊息 / 每回合的比率」,袋層級取平均:

| # | 特徵 | 定義 | 已知 T vs S |
|---|---|---|---|
| f1 | 語氣詞率 | 訊息以 喔/啊/呀/啦/耶/欸/吧/嘛/齁/嗎 結尾的比例(分兩組:T 組 喔啊呀啦耶欸、S 組 嗎) | 喔/啊 vs 嗎 |
| f2 | 全形標點占比 | 全形 ，。！？：；、 / (全形 + 半形 ,.!?:;) | 非符號回合 T 0.32 / S 0.17(標點極少,鑑別度 0.68;10/03 的 0.18 / 0.61 無法重現,見 survey 實驗 4 更正) |
| f3 | 句尾標點率 | 訊息以標點結尾的比例 | 不顯著(實驗 1) |
| f4 | 貼圖率 | 貼圖/圖片訊息比例 | 0.11 vs 0.33 |
| f5 | 連發則數 | 每回合訊息數(中位數、>1 的比例) | 2 vs 1 |
| f6 | 訊息長度 | 字元數的中位數與 IQR | 不顯著 |
| f7 | 「～」率 | 含 ～ 或 ~ 的訊息比例 | 2.3% |
| f8 | 口頭禪率 | 命中 T 清單(好啊、沒錯、讚喔、…,由 Zeta 清單定)的比例 | — |
| f9 | 問句率 | 含 嗎/? /? 的比例 | S 高 |
| f10 | 英文字母占比 | ASCII 字母 / 全部字元(未過濾) | — |
| f11 | 數字占比 | 數字 / 全部字元 | — |
| f12 | 笑聲率 | 含 哈{2,}/XD/笑死 的比例 | — |
| f13 | 第一人稱率 | 含 我 的比例 | — |
| f14 | 顏文字/表情率 | `` `` 比例 | — |

- **實作(persona/rulers/features.py)**:16 個特徵 = particle_T、particle_S、fullwidth_punct、end_punct、sticker、burst、msg_len、tilde、catch_T、catch_S、question、letters、digits、laugh、first_person、emoticon;算在尺的母體(非符號回合)上。stage-1 鑑別度(k=50 袋,T vs S):first_person 1.00、particle_T 0.99、sticker 0.99、catch_S 0.99、question 0.99、particle_S 0.98、catch_T 0.97、digits 0.91、burst 0.87、emoticon 0.84、end_punct 0.78、tilde 0.77、msg_len 0.75、fullwidth_punct 0.68、laugh 0.63、letters 0.60(全部 ≥ 0.60,都進加權)。
- `band(train_T_bags, k=50, q=(0.05, 0.95))`:每個特徵在 1,000 個 T 訓練袋上的 5–95 百分位 = **容忍帶**(S28)。候選集的特徵落在帶內不加不扣,帶外扣。
- `discrimination(f)`:每個特徵在 k=50 的 within-T vs cross 的 AUC(IRT 式鑑別度);AUC < 0.6 的特徵不進加權距離,只報告。
- 加權特徵距離 `d_feat(bag, T) = Σ_f w_f · min(|z_f|, 5)`,w_f ∝ (鑑別度_f − 0.5)(只含 weighted 特徵,正規化);z 用 T 訓練袋的平均與標準差。(**2026-10-13 修改**:原本的逆變異數加權被 laugh、emoticon 這種幾乎恆為 0 的稀有特徵吃掉 89% 權重,改成以鑑別度加權並把 |z| 封頂。)
- 驗收:所有特徵在 T 的 5 個日期區塊之間的管制圖(S14)列出;超出 3σ 的區塊標記。

### 3.4 `rulers.calibrate` ✅ / 🔧
- `oof_scores(model_factory, train, blocks)`:5 個日期區塊 OOF,每 fold 重擬合。
- `per_k_threshold(k)`:OOF 訓練袋(每人每區塊 40 袋)上最大化平衡準確率(實驗 3、4 的做法)。
- `isotonic(llr_oof, y)`:PAV 把 LLR 映成校準機率;報 Cllr(S10)。**訓練用的分數 = 校準後的 log-odds**,這樣 δ 邊際有可解讀的意義(例如 δ = log 3)。
- 驗收:校準後在 dev 的 Brier 分數低於未校準;Cllr < 1(比無資訊好)。

### 3.5 `rulers.validate` ✅ → 🔧 包成測試(每次改前處理都要重跑)
- `e1`:單回合 acc / AUC(報告用)。
- `e3`:k ∈ {5, 10, 25} 校準準確率 + AUC;**門檻:NB k=10 ≥ 0.95**。
- `e4`:k=50 隨機袋與同時期袋,P(cross > max within);**門檻:Cosine Delta、NCD ≥ 0.95,Min-max ≥ 0.90**。
- `e5`:應答子集的 OOF 單回合 AUC;**門檻:NB ≥ 0.90**(實驗 3:0.944)。
- 全部通過才允許進 §4–§6;結果寫進 `reports/ruler_validation.json`。

---

## 4. 基準線與對照組(④)

| 代號 | 系統 | 說明 |
|---|---|---|
| B0 | zero-shot(Instruct) | 系統提示只說「用台灣口語繁中回 LINE 訊息」;同一 `prompt.render` 的前文 |
| B1-n | many-shot(Instruct),n ∈ {10, 50, 200, all} | 系統提示 + n 個「對方前一回合 → 我的回合」範例(從 train 等距抽)+ 前文;記錄 token 數(all ≈ 700 對 ≈ 2–3 萬 token,KV 快取 < 1 GB,可行;S39) |
| B2 | 檢索式 many-shot(Instruct),k=20 | 用 bge-m3 以當前前文檢索最相近的 20 個訓練前文,放它們的「前文尾 → 我的回合」 |
| B3 | best-of-8 on SFT | 每前文取 8 個候選、用訓練尺挑最高(= RAFT 的推論版);看尺本身能推多遠 |
| C-real | 真實回覆 | 上界;所有指標的參考點 |
| C-shuf | 打亂的真實回覆 | 把 T 的真實測試回覆隨機配到別的前文:**風格完美、內容錯**;用來檢查哪些指標只看風格、哪些看內容 |
| C-S | 對方(S)的真實回覆 | 靈敏度對照:尺必須把它和 T 分開 |
| C-base | 底模(Base)直接續寫 | 未訓練的純文字續寫 |

- 所有系統共用同一組測試前文、同一個解碼設定、同 3 個種子。
- 驗收:B0 與 C-base 的指標先出來,**三條驗收線的門檻在看到它們之後只能調嚴不能放鬆**(預先登記紀律)。

---

## 5. 訓練(⑤ ⑥)

### 5.1 `train_sft` 🔧
- 底模:`Qwen/Qwen3.5-4B-Base` 與 `Qwen/Qwen3.5-4B`(Instruct);bf16;Windows 下 `sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False`。
- LoRA(peft 0.21):r 8、α 16、dropout 0.05、target = q,k,v,o,gate,up,down;消融:只 q,k,v,o。
- 最佳化:AdamW、lr 1e-4、cosine、warmup 3%、weight decay 0;micro-batch 2 × grad-accum 4 = 8;seq 768;梯度檢查點;epoch 5 / 4 / 3 / 3(依子集);dev loss 每 epoch 評估,取最低者。
- 記憶體估計(待實測):4B bf16 權重約 8.4 GB + LoRA/optimizer < 0.3 GB + 啟動(seq 768、micro 2、檢查點)約 2–3 GB → 約 12 GB,16 GB 可行;不行再退 QLoRA(要裝 bitsandbytes,先問)。
- 資料:§2.6 樣本 × `prompt.encode`;只算目標 token 的 loss。
- 金絲雀(S33):train 加 3 條虛構句子(各出現 1 次);訓練後量它們的續寫機率與 NLL,當記憶稽核的基線。
- 輸出:adapter 權重(私有)、train/dev loss 曲線、每 epoch 的 dev NLL。
- 驗收:`tests/test_mask.py`(§2.7);dev NLL 低於底模的 dev NLL;金絲雀 NLL 沒有降到明顯低於同長度真實句。

### 5.2 `generate` 🔧
- `sample(model, prompts, n, seed)`:批次 16、temperature 0.8、top-p 0.95、max 64、停在 `\n對方:` / `<eos>` / 第 6 則;輸出去掉標記後的回合文字(多則以 `\n` 接)。(**2026-11-02**:改成依 token 預算分批,一批最多 16 個 prompt;另記候選在 64 個新 token 內是否出現 eos(`finished`),只認 eos,比這裡的停止條件嚴;前文和 `prompt.encode` 一樣截到 512 token。見 §10。)
- 後處理:OpenCC `s2twp` 比對前後差異 → `leak_chars`;LCS 與訓練目標 ≥ 12 字 → `copy=True`。
- 用途:評估(n=1 × 3 種子)、B3(n=8)、DPO 候選(n=8,train 前文)。

### 5.3 `pairs.build(sft_model, train_samples, ruler) -> list[Pair]` 🔧(本方法的核心)
1. 對每個 train 前文取 8 個候選(§5.2)。
2. 分數 `s(c)` = 校準後 NB log-odds(§3.4),**上限** = T 真實回覆分數的 95 百分位(典型帶上界,S26):`s ← min(s, cap)`。(**2026-11-02 實作後修改**:等張校準在原始 LLR 37.81 就到天花板 6.907,校準後分數的 95 百分位就是天花板,封頂沒有作用(`CODE_REVIEW_2026-10-17.md` #2)。改成在原始 LLR 上封頂:cap_raw = T 訓練回合 OOF 原始 LLR 的 95 百分位(`reports/`:51.28),`s = 校準(min(raw, cap_raw))`;挑 chosen / rejected 時用 min(raw, cap_raw) 排序、同分隨機。見 §10。)
3. 約束(不滿足的候選不能當 chosen):
   - 離題:bge-m3 cos(候選, 真實回覆) 必須 > 「無關真實回覆 vs 真實回覆」的 cos 分布的 90 百分位(用 train 的隨機配對估;S19)。
   - 抄襲:`copy=True` 者排除(S30)。
   - 外漏:`leak_chars > 0` 者排除(S37)。
   - **2026-11-02 新增**:符號類候選(`flag_symbols.is_symbol_turn`)與沒結束的候選(64 個新 token 內沒有 eos)不當 chosen 也不當 rejected(§10)。
4. 行為格(MAP-Elites,S25):`cell(reply) = (長度級距 ∈ {≤5, 6–15, 16–40, >40 字}, 有無貼圖, 連發則數 ∈ {1, 2, ≥3}, 有無語氣詞)`;以**真實回覆**的格為目標格。
5. chosen = 通過約束且 `s > +δ` 的候選中,**先找目標格內**的最高分者;目標格沒有就退到鄰格(貼圖旗標相同、長度級距 ±1),再沒有才任一格;所用層級記錄在 pair 裡並報統計(**2026-10-13 實作後修改**:冒煙測試 8 個候選只有 3/12 前文命中精確格,原規則產生 0 對)。rejected = 任一格內 `s < −δ` 的最低分候選;δ = log 3(校準後 log-odds;待第一次正式跑完視配對數調整,只能調大)。
6. 沒有合格 chosen → chosen = 真實回覆(DITTO 式),標記 `ditto=True`;全部配對中 ditto 比例上限 50%,超過就隨機丟棄 ditto 配對到 50%。
7. 沒有合格 rejected → 該前文不產生配對。
8. 輸出統計(進論文附錄):每格候選數、通過約束比例、ditto 比例、chosen/rejected 的分數分布。
- 驗收:chosen 集合的 R1 特徵分布與真實回覆的 JSD < 0.05(S27);rejected 的分數平均明顯低於 chosen(差 > 2δ)。(**2026-11-02**:兩項都只算非 DITTO 配對,DITTO 的 chosen 就是真實回覆;見 §10。)

### 5.4 `train_dpo` 🔧(自寫)
- 參考模型:SFT(LoRA 合併後的權重,凍結);策略:在合併後權重上掛新 LoRA(同 §5.1 設定)。
- 預先算一次參考模型對 chosen / rejected 的 logp(只算目標 token),存檔。
- 損失:`−log σ(β[(logp_θ(c) − logp_ref(c)) − (logp_θ(r) − logp_ref(r))])`,β 0.1;lr 5e-6;1 epoch;micro 1 × accum 8。
- 提早停止:每 100 步在 dev 前文取樣 64 則,算 **驗收尺**(§6)的 D-AUC 與 Band;D-AUC 不再下降就停(不用 KL 懲罰,S29;Gao 2023)。
  **2026-11-02 實作與此不同**(§10):第 0 步、步數為 10 的倍數時(`C.DPO["eval_every"]`)與最後一步之後,在 60 個 dev 前文(`C.DPO["dev_eval_n"]`)上各生成 1 則;訓練跑完 1 epoch,再從這些評估中挑一個。這不是這裡登記的「不再下降就停」;正式 DPO 前要由使用者確認或改掉。
- 驗收:`tests/test_dpo_loss.py` 用一個小例子驗證損失與 trl 公式一致(手算);訓練後 chosen 的 logp 相對 rejected 上升。

---

## 6. 評估(⑦)

### 6.1 自動指標(`evaluate.run(system) -> dict`)

| 指標 | 定義 | 方向 | 看什麼 |
|---|---|---|---|
| **NLL** | 真實 T 測試回覆在該模型下的每 token 負對數似然(teacher forcing、同 prompt) | ↓ | 最直接、最便宜的擬合度;所有 SFT/DPO 條件都報 |
| **D-AUC₁** | 獨立分辨器(測試期:真實 T vs 生成,字元 n-gram NB,5-fold OOF;**fold 按字串分配**,同一字串不會同時在訓練與測試 fold)的單回合 AUC | → 0.5 | 主尺;同時報底線 real-vs-real(≈ 0.5)與靈敏度 real-T vs real-S(過濾後 0.885) |
| **D-AUC₁₀** | 同上,k=10 袋 | → 0.5 | 放大系統性差異 |
| **NCD-Δ** | 生成回覆袋(k=25)的 ncd_S − ncd_T,與真實 T 袋的差 | → 0 | 無模型交叉檢查 |
| **Band** | 16 個 R1 特徵中落在 T 容忍帶內的個數(k=50 袋) | ↑(≥ 12) | 可解釋;抓平均化與灌水 |
| **JSD-feat** | 生成與真實 T 的特徵分布 JSD(每個特徵的直方圖,平均) | ↓ | 分布層級(S27) |
| **OnTopic** | cos(生成, 真實) 高於無關對 90 百分位的比例 | ↑ | 內容;C-shuf 應該低 |
| **MMI** | log p(回覆∣前文) − log p(回覆),底模計算 | ↑ | 銜接(S40) |
| **Copy** | LCS ≥ 12 字命中訓練目標的比例 | ↓ | 記憶 |
| **Leak** | OpenCC s2twp 改動字元數 / 100 字 | ↓ | 大陸用語 |
| **Div** | distinct-2;Hill q=1 的有效回覆種類數(稀疏化到相同 n) | ≈ 真實 | 平均化 |
| **Canary** | 3 條金絲雀的續寫命中率 | 0 | 記憶稽核 |

- 全部指標對 C-real、C-shuf、C-S、C-base 都算一次,作為表格第一區(定義指標的座標)。
- 統計:每個系統 3 個種子;系統間差異用**以測試前文為單位的配對 bootstrap**(2,000 次)報效果量與 95% CI;多重比較不校正但全部列出。
- **2026-11-02**:符號類的生成回覆(`is_symbol_turn(reply_text(msgs))`)不進分辨器、NCD、Band、JSD、OnTopic、MMI、Copy、Leak、Div,比例另報 `symbol_share`(真實目標在資料層已排除符號類回合);Div 預設稀疏化到 100 則,回覆不到 100 則的系統另報和真實回覆在同一個 n 的比值;配對 bootstrap 在 `report.py`。見 §10。

### 6.2 本人盲判(`human_eval`)
- 題型 A(真假辨識):同一前文,真實回覆 vs 系統回覆,隨機左右;本人選「哪個是我寫的」。SPRT 式停止:逐題累積,當正確率 95% CI 落在 0.65 的一側就停,最多 100 題(S17)。
  **更正 2026-11-02**:上一句的規則逐題看一次 95% 區間,不是 SPRT;對 (n, k) 精確計算,真實正確率 0.65 時判「低於」的機率 0.120(名目 0.025;`CODE_REVIEW_2026-10-17.md` #5)。它留在程式裡當紀錄(`human_eval.ci_stop`),不再使用。現在的停止規則是 **Wald SPRT**(`human_eval.sprt_stop`,參數在 `config.HUMAN`):
  - 題型 A:兩個簡單假設 p = 0.55 對 p = 0.75(0.65 兩側的無差異區),α = β = 0.05。n 題中選對 k 題時,對數概似比 Λ = k·ln(0.75/0.55) + (n − k)·ln(0.25/0.45);Λ ≥ ln((1 − β)/α) = ln 19 ≈ 2.94 停在「高於」,Λ ≤ ln(β/(1 − α)) ≈ −2.94 停在「低於」,否則繼續;到 100 題還沒碰到界線就停在「未定」。
  - 題型 B:0.50 對 0.70,α = β = 0.05,最多 100 題。
  - 系統各自一條序列,不合併;線 3 用停止時的點估計判(§6.3)。一個系統的題目用完時,如果還沒碰到界線、也還沒到 100 題,線 3 記為未定(`report.lines_from_human` 回 None,標 `more_items_needed`)。
  - 操作特性(題型 A、最多 100 題,對 (n, k) 精確計算):真實正確率 0.70 的系統通過線 3 的機率 0.201(舊規則 0.173),平均 44 題停(舊規則 78 題);0.75 時 0.042(舊 0.025);0.60 時 0.821(舊 0.859);真實 0.65 時判「低於」0.463,落在無差異區內,這是設計允許的。題型 B 對稱:真實偏好 0.55 時通過的機率 0.201。
  - **待使用者決定**(只能調嚴):要不要縮小無差異區。例:0.60 / 0.70 時,真實 0.70 的通過機率 0.163(平均約 82 題);再把上限放到 200 題是 0.081(約 113 題)。決定後寫回這裡與 `METHOD_NEURIPS.md` §3.6(該節還寫著舊規則)。
- 題型 B(偏好):同一前文,系統 X vs 系統 Y(最佳 SFT/DPO vs 最佳 many-shot),本人選「哪個比較像我」。
- 取樣:優先取三把尺(D-AUC、Band、NCD)意見分歧的前文(S18),其餘隨機補足。(**2026-11-02 實作**:分歧前文的清單用 `--priority` 傳入,程式不自己算;priority 只決定選哪些前文。一個前文只給一個系統(題型 A、B 都是,用種子打亂的區塊分配),同一前文與它的回覆最多出一題;出題依時間順序;題型 A 的前文裡,其他題的真實回覆顯示成 `我: [略]`;系統回覆缺的、空的或符號類的不出題,數目記在 `stats.json`。146 個測試前文分給 2 個系統時,一個系統最多 73 題,到不了 100 題,怎麼安排待決定。)
- 題目檔只含去識別化文字;答案與時間本機記錄。(**2026-11-02**:題組只能寫在 `runs/` 底下;`score` 只印數字與系統名稱。)

### 6.3 三條驗收線(預先登記;看到 B0 / C-base 之後只能調嚴)
1. **線 1 自動**:D-AUC₁ ≤ 0.75 且 D-AUC₁₀ ≤ 0.90(比 S 更難與 T 分開),且顯著優於最佳 many-shot(配對 bootstrap CI 不含 0)。
2. **線 2 特徵**:Band(k=50 袋內落在 T 容忍帶的特徵數)**不低於 C-real 在同一組測試前文的 Band − 1**(C-real 本身只有 12.0 / 15.3,因為測試期相對訓練期有漂移;絕對門檻 12/14 會比真人還嚴),Div 不低於真實的 80%,Copy ≤ 2%,Leak 不高於 C-real。
3. **線 3 本人**:題型 A 正確率 ≤ 0.65(停止時的點估計),題型 B 對最佳 many-shot 偏好 ≥ 0.60。
- **2026-11-02 實作**(`report.acceptance`、`report.lines_from_human`,門檻在 `config.LINES`)。下面三點是實作時的解讀,**待使用者確認**:線 1 的「顯著優於」= D-AUC₁(系統)− D-AUC₁(最佳 prompting)的配對 bootstrap 95% CI 整個低於 0(只看 D-AUC₁);線 2 的 Div = distinct-2 與 Hill-1 都 ≥ 真實回覆在同一個回覆數下的 0.80 倍;線 3 在 100 題還沒碰到 SPRT 界線時用點估計判,題目用完時記為未定(§6.2)。`symbol_share` 沒有登記門檻。
- **待確認 2026-11-02**:線 2 理由中的 12.0 / 15.3 是用只含文字回合的特徵帶算的;#4 修正後特徵帶含純貼圖回合,要在本機重跑 stage1 與 `run_eval --system controls` 再看這個理由(`CODE_REVIEW_2026-10-17.md` #4)。
- 三條都過 = 成功;只過 1、2 = 「尺上像、人覺得不像」,要回頭檢查尺(Baumler 2026 的警訊);只過 3 = 尺太嚴或抓到取樣痕跡。

---

## 7. 執行順序與預算(Tier 1)

| 步 | 內容 | 估計時間 | 產出 |
|---|---|---|---|
| 1 | §2 全部 + `samples.stats()` + §2.4 白名單驗收(重跑 E4) | 半天 | 資料統計表、尺驗證報告 |
| 2 | §3.3 特徵 + 容忍帶 + 管制圖;§3.4 校準 | 半天 | ruler_validation.json |
| 3 | §4 基準線 B0、B1、B2、C-* 在 test 上生成與評估 | 1–2 小時 GPU | 基準線表;**鎖定驗收線門檻** |
| 4 | SFT:2 底模 × 4 子集 × 3 種子 = 24 次(每次約 10–20 分鐘,待實測) | 約 6–8 小時 GPU | 資料量曲線、Base vs Instruct |
| 5 | B3 best-of-8 + §5.3 配對(8 候選 × ~700 前文 × 2 底模) | 1–2 小時 GPU | 配對統計 |
| 6 | DPO:最佳底模 × 全部資料 × 3 種子(+ 消融:無格、無上限、無 ditto) | 2–3 小時 GPU | 主結果 |
| 7 | 本人盲判(題型 A、B) | 本人 1–2 小時 | 線 3 |
| 8 | 報告 | — | 表格、圖、附錄 |

消融(都在最佳底模、全部資料、3 種子):LoRA attention-only;前文 3 小時切段;近期加權(2026 年樣本權重 ×2,S13);配對無行為格;分數無上限;無 ditto 回退。

---

## 8. 測試清單(`tests/`)— 已實作 30 個測試,2026-10-13 全部通過;對應關係見 §10(2026-11-02:54 個,見 §10 最後)
1. `test_parse`:訊息數、回合數、段落數與既有數字一致。
2. `test_placeholders`:往返可逆;白名單命中數。
3. `test_symbol_rule`:E4 門檻(§2.4)。
4. `test_split`:dev/test 無同段落;子集時期 JSD < 0.05。
5. `test_mask`:目標第一 token mask=1、prompt 末 token mask=0;Base 與 Instruct 的 tokenizer 都測。
6. `test_prompt_same`:訓練與推論走同一函式(比對字串)。
7. `test_rulers`:E3 NB k=10 ≥ 0.95;E5 ≥ 0.90;校準後 Brier 改善。
8. `test_pairs`:chosen 特徵 JSD < 0.05;分數差 > 2δ;ditto ≤ 50%。
9. `test_dpo_loss`:小例子手算一致。
10. `test_eval_controls`:C-real 的 D-AUC ≈ 0.5(CI 含 0.5);C-S 的 D-AUC ≥ 0.85;C-shuf 的 OnTopic 明顯低於 C-real。

---

## 9. Tier 2 需求(規劃中,不在 v1)
- 多位同意參與者(目標 5–10 人)的 LINE 匯出;每人 ≥ 500 回合;對象代號進前文(S15)。
- 多作者 z 參照(Cosine Delta 升格)、G-study 變異數分解(S6)、合成對照(S8)。
- 先在多人資料上適應、再做個人 LoRA(S34);熟人盲測取代本人盲判。
- 公開語料(PTT 推文 / Dcard 留言)做尺的跨作者驗證,但那不是私訊,只能當輔助。
- 貼圖編號(需另一種匯出方式)→ 學「哪一張」。

---

## 10. 實作狀態(2026-10-13)

**套件** `persona/`(純本機;不印訊息原文):

| 模組 | 內容 | 測試 / 冒煙 |
|---|---|---|
| `config.py` | §0 全部常數 | — |
| `parse_line.py` `turns.py` `normalize.py` `flag_symbols.py` `split.py` `samples.py` `prompt.py` | 資料層 §2 | 15 個單元測試全過;回合數與實驗 1–4 一致(4,967 則 / 23 系統行 / 785·862·134·149) |
| `rulers/` (`ngrams` `author_lm` `profiles` `ncd` `stats` `calibrate` `features` `validate`) | 尺 §3 | **逐格重現實驗 4**(E1/E3/E4 每個數字相同);v1 白名單規則 **PASS** 全部預登記門檻;等張校準 Cllr 1.77 → 0.60、Brier 0.176 → 0.130 |
| `stage1.py` | 資料統計、尺驗證、校準、特徵帶 → `reports/` | 102 秒 |
| `modeling.py` `data_sft.py` `train_sft.py` | LoRA SFT §5.1(自寫迴圈、cosine、梯度檢查點、金絲雀稽核) | 0.8B 冒煙 4 步 OK;**4B:10.1 GB VRAM、約 1 s/step(短前文)**,估全量 3 epoch 約 5–15 分鐘(**待確認 2026-11-02**,見本節最後的「計時」) |
| `generate.py` `embed.py` | 取樣、抄襲檢查、OpenCC 外漏(未裝 → None)、bge-m3 離題門檻 | 離題門檻 0.590(null 90 百分位) |
| `pairs.py` `make_pairs.py` | §5.3 配對(行為格階梯) | 0.8B 冒煙:12 前文 → 12 對(ditto 4),chosen 2.65 / rejected −2.91,chosen-vs-real 特徵 JSD 0.045(兩者都含 DITTO 配對,`CODE_REVIEW_2026-10-17.md` #10) |
| `train_dpo.py` | 自寫 DPO(參考 logp 預算、單模型) | 0.8B 冒煙 OK(step 0 loss = log 2,如預期) |
| `evaluate.py` `run_eval.py` | 驗收尺 §6 + 對照組 | **對照組(146 測試前文)**:C-real D-AUC₁ 0.50、底線 0.46 ± 、靈敏度 0.76(k=10:1.00);C-shuf 風格指標 = C-real、離題率 0.10;C-S D-AUC₁ 0.76、Band 6.6 vs 12.0(**2026-11-02**:#4、#11、#12、#13 改了 Band 的母體、C-S、離題門檻、多樣性,這些數字要重跑 `run_eval --system controls`) |
| `baselines.py` | B0/B1/B2(chat template、thinking 關) | 解析器單元測試;尚未在 4B 上跑 |
| `human_eval.py` | 盲判題組 + SPRT 停止(**更正 2026-11-02**:當時的規則逐題看 95% 區間,不是 SPRT,`CODE_REVIEW_2026-10-17.md` #5;現在是 Wald SPRT,§6.2) | 停止規則單元測試 |
| `tests/` | 30 個測試(`python tests/run_tests.py`,`--fast` 跳過慢的) | 全過 |

**實作時改掉的設計**(都已寫回 §5.3 / §3.3 / §6):行為格階梯;特徵距離改鑑別度加權 + |z| 封頂;分辨器 fold 按字串分配(C-shuf 對照組抓到了原本的字串記憶漏洞:D-AUC 0.08 → 0.50)。

**還沒做 / 要決定的**:
- `Qwen/Qwen3.5-4B-Base` **不在 HF 快取**(只有 Instruct 的 `Qwen/Qwen3.5-4B`、0.8B、2B、9B);要跑 Base vs Instruct 得先下載(~8 GB)。
- OpenCC 未安裝 → Leak 指標目前回 None;要裝 `opencc-python-reimplemented`。
- Qwen3.5 是混合線性注意力模型,沒裝 `flash-linear-attention` / `causal_conv1d` 時用 PyTorch 參考實作(正確但慢);Windows 上這兩個套件不一定裝得起來,先用實測速度判斷是否需要。
- 正式實驗(§7 步 3–7)尚未跑:B0 / C-base 跑完才鎖驗收線門檻;DPO 的提早停止(用驗收尺)只在 epoch 結尾評估,還沒接上 dev 生成。**更正 2026-11-02**:當時 DPO 完全沒有評估(`eval_every` 沒被使用,`CODE_REVIEW_2026-10-17.md` #8);2026-11-02 已接上 dev 評估,見下。
- 實驗 4 與 stage-1 的數字都還是單次執行,未獨立重算。

### 2026-11-02:程式審查後的修改

依 `CODE_REVIEW_2026-10-17.md`(#1–#18 與文末的複查)。行號是 2026-11-02 工作樹的行號。需要模型或 GPU 的部分沒有在雲端執行(清單與本機重跑順序在 `CODE_REVIEW_2026-10-17.md` 文末);2026-11-02 之前在本機產生的 `runs/` 輸出與 `reports/eval_table.json` 的列是舊程式的結果。

**資料層**
- `text.py` / `data.py` 拆分(commit 2df8fa7):回覆文字、抄襲檢查、外漏計數移到不 import torch 的 `text.py`(外漏改用 difflib 對齊);`load_all` 移到 `data.py`(`stage1.load_all` 舊名保留)。evaluate、pairs、human_eval、report 因此能在沒有 torch 的機器上測。
- 資料清單:stage1 最後寫 `reports/data_manifest.json`(切分常數,以及它們產生的回合、樣本、子集計數;只有數字,`stage1.py:101`)。`data.load_all()` 發現資料和清單不同就停,訊息指出最早不同的鍵(`data.py:51-93`)。真資料的 `reports/` 還沒有這個檔。
- 特徵帶的母體(#4):特徵帶、鑑別度、管制圖改用含純貼圖回合的非符號訓練回合(`stage1.py:85-89`),和驗收的袋一樣;純貼圖回合歸入同日期的日期區塊(`data.py:27-36`),文字回合的區塊不變,所以 NB 的 OOF 不受影響。NB、profile、NCD 尺仍只用文字回合。合成資料上 C-real 的 Band 13.7 → 14.4(16 個特徵);真資料要重跑 stage1。

**SFT(#9、#17;§2.5)**
- 階段 1 在 train − dev 上訓練,dev NLL 最低的 epoch 存成 `adapter_devsel`;階段 2 重新載入底模、掛新的 LoRA,在 train + dev 上訓練同樣的 epoch 數,存成 `adapter_best`(`train_sft.py:96-166, :199-219`;`C.SFT["retrain_full"]`;`--no-retrain` 跳過,`--quick`、`--measure` 也不做)。全部資料時,階段 2 的訓練時間最多約為階段 1 的 1.17 倍(797 / 679 個樣本)。
- 子集 100 / 250 / 500 的階段 2 用 train + dev 上以同樣巢狀、時間等距方式抽的同大小子集(`train_sft.py:50-65`),資料量曲線上的樣本數維持 100 / 250 / 500。**待確認**:另一種讀法是「子集 + 整個 dev」,會多出 118 個晚期樣本。
- 金絲雀稽核改在存下來的 `adapter_best` 上做(`train_sft.py:78-94, :220-222`);階段 2 之後 dev 不再是保留資料,另記 `canary_devsel`(階段 1 的 adapter)與 `dev_items_trained`。

**生成(#15、#16;複查 R2-5、R2-6)**
- 依 token 預算分批:prompt 由長到短排;一批在 (prompt 數 + 1) × (最長 prompt + 64) × n ≤ 12,000 時才再加一個,最多 16 個(`generate.py:11-57`)。n = 8、prompt 約 300 token 時一批 4 個(原本 16),`make_pairs` 會變慢;VRAM 夠的話可以調大 `token_budget`。基準線在生成前記下 prompt token 數的最小 / 中位 / 最大(`baselines.py:87-112`)。
- `finished`:續寫在 64 個新 token 內出現 eos(或 pad)才算結束(`generate.py:27-57, :68-77`)。這比 §5.2 嚴:用 `\n對方:` 結束回合但沒有 eos 的續寫也算沒結束。真資料跑過 make_pairs 後要看 `stats.json` 的 `cand_unfinished`。
- 生成的 prompt 和 `prompt.encode` 一樣,從最舊的前文回合開始丟,直到 ≤ 512 token(`generate.generation_prompt`,`generate.py:60-65`;基準線 `baselines.py:43-46`)。之前生成用的是不截斷的前文,所以前文超過 512 token 的樣本,舊的 make_pairs / run_eval 輸出與現在不同。

**配對(#2、#3、#10、#16;§5.3)**
- 封頂改在原始 LLR 尺度:cap_raw = T 訓練回合 OOF 原始 LLR 的 95 百分位(`reports/`:51.28,5.1% 的 T 回合高於它),分數 = 校準(min(raw, cap_raw))(`pairs.py:72-89`)。等張校準在 raw 37.81 就到天花板 6.907,只看分數仍分不出封頂與不封頂,所以挑 chosen / rejected 改用 min(raw, cap_raw) 排序,差距 1e-9 以內的同分用種子隨機挑(`pairs.py:92-97, :144-147`)。**待確認**:這個排序超出審查的修法;要照 §5.3 的原公式就改回 `pairs.py:147`,那樣「分數無上限」消融會和主實驗相同。
- OOF 陣列和訓練回合要逐位對齊(個數與說話者),對不齊就停(`pairs.py:76-79`):重跑 stage1 時 `ruler_validation.json` 要來自同一份資料。
- 符號類候選與沒結束的候選不當 chosen 也不當 rejected(`pairs.py:114-128`)。新的統計:`cand_symbol`、`symbol_share`、`cand_unfinished`、`cand_capped`、`cand_capped_share`、`cap_raw`、`cap_T_oof_above`。
- §5.3 的兩個驗收(chosen vs 真實的特徵 JSD、chosen 平均分數)只算非 DITTO 配對;含 DITTO 的值另存為 `*_all`(`pairs.py:177-202`、`make_pairs.py:75-76`)。

**DPO(#8;複查 R2-1 到 R2-4;與 §5.4 不同)**
- 評估:第 0 步(SFT 起點)、步數為 `--eval-every`(`C.DPO["eval_every"]` = 10)的倍數時、最後一步之後,在一組固定的 dev 前文(`C.DPO["dev_eval_n"]` = 60,用 run 的種子抽)上各生成 1 則,算 D-AUC₁₀、Band、特徵 JSD、`symbol_share`(`train_dpo.py:105-121, :226-292`)。生成的回覆只寫進 `runs/<dpo>/dev_outputs.jsonl`。
- 選法(`SELECTION`,`train_dpo.py:47-52, :123-131`):`symbol_share` 比第 0 步高出 0.05 以上(60 則中 3 則)的排在後面;其餘取 D-AUC₁₀ 最低,同分取 Band 高,再同分取較早的步。第 0 步也能被選中,這時 `adapter_best` 是不起作用的 LoRA,`run_eval --dpo` 等於 SFT 模型。
- **與 §5.4 不同**:§5.4 是每 100 步看 64 則、「D-AUC 不再下降就停」;實作是訓練完 1 epoch,再從這些評估中取最好的一個。在 60 個前文上反覆取 D-AUC₁₀ 的最小值是偏樂觀的估計。這條規則是實作時定的(不是預先登記的),寫進這裡的時間早於任何正式 DPO;要不要把它當成登記的規則,由使用者決定。
- dev 保留:SFT 的 `adapter_best` 如果在 dev 上訓練過(`runs/<sft>/metrics.json` 的 `phase2.n_dev_in_train` > 0),DPO 跑兩次:先以 SFT 的 `adapter_devsel` 為參考、用 dev 選步數(選定的 → `adapter_devsel`),再以 SFT 的 `adapter_best` 為參考跑同樣的步數(選定的那一步 → `adapter_best`,最後一步 → `adapter_last`)(`train_dpo.py:155-176, :208-303`)。訓練時間約加倍。配對的前文落在 dev 就拒絕(`train_dpo.py:189-191`)。`metrics.json` 記 `dev_heldout`、`selection_sft_adapter`、`sft_n_dev_in_train`、`retrain`。

**驗收尺(§6.1;#1、#3、#11、#12、#13;複查 R1-3、R1-6)**
- NLL 改算真實測試回覆(`evaluate.py:275`;規格原本就這樣寫,錯在程式)。
- 符號類生成回覆(`flag_symbols.is_symbol_turn(text.reply_text(msgs))`)不進分辨器、NCD、抄襲、外漏、離題、Band、JSD、多樣性、MMI,比例記為 `symbol_share`(`evaluate.py:244-277`、`run_eval.py:155-158`);pairs、human_eval、report、train_dpo 用同一個判定。`symbol_share` 本身沒有登記門檻,**待決定**。
- C-S 從 `t.msgs` 建(連發、長度、語氣詞、顏文字照原樣),純貼圖回合保留、符號類排除(`evaluate.py:229-234`、`run_eval.py:130`)。
- 離題門檻在訓練期(含 dev)的真實回覆上配(`embed.fit_ontopic_threshold`,`embed.py:58-66`),make_pairs 與 run_eval 共用;之前 run_eval 是在測試回覆上配的。
- 多樣性預設稀疏化到 100 則(`C.EVAL["rarefy_n"]`),先排序再抽,所以 C-shuf 與 C-real 相同(`evaluate.py:181-188`);另報和真實回覆在同一個回覆數下的比值 `div_distinct2_ratio`、`div_hill1_ratio`(`evaluate.py:262-268`),線 2 用這兩個比值。
- 分辨器拆成 `discriminator_scores`(OOF 分數)與 `discriminator`(`evaluate.py:54-95`),bootstrap 重用 OOF 分數。

**報告與驗收線(#7;`report.py`、`config.LINES`)**
- `paired_bootstrap`:以測試前文為單位重抽 2,000 次(`C.EVAL["boot_n"]`)。`d_auc_bootstrap`:每個系統的 OOF 分數只算一次,所以 CI 不含分辨器重新擬合的變異。另有 `compare`(CLI `--compare`)、`table`(CLI `--table`)、`acceptance`(線 1–2)、`lines_from_human`(線 3)。
- 門檻集中在 `config.LINES`:D-AUC₁ 0.75、D-AUC₁₀ 0.90、Band 差 1、多樣性 0.80、Copy 0.02;線 3 指向 `C.HUMAN["A"]["p_line"]`(0.65)與 `C.HUMAN["B"]["p_line"]`(0.60)。三個實作解讀待確認,見 §6.3。

**本人盲判(#5、#6;複查 R1-1、R1-2、R1-4、R1-5)**:Wald SPRT 與出題方式見 §6.2;參數在 `config.HUMAN`(`max_n` 100;題型 A 0.55 / 0.75;題型 B 0.50 / 0.70;α = β = 0.05)。CLI 是 `python -m persona.human_eval make` 與 `score`,資料夾必須在 `runs/` 底下。

**輸出目錄(#14)**:`train_sft` 的目錄名在非預設設定時加 `_attn`(`--attn-only`)、`_lora-…`(其他 LoRA 目標)、`_ep<N>`、`_noretrain`;`make_pairs` 預設 `pairs_<sft>_s<seed>`;`train_dpo` 預設 `dpo_<sft>_<pairs 目錄>_s<seed>`;`run_eval` 的 C-base 改名 `untrained_<model>_s<seed>`、基準線 `B-<kind>[-<shots>]_<model>_s<seed>`。目錄已存在且非空時,沒有 `--overwrite` 就拒絕,在載入模型之前檢查。`train_sft --quick`(`sft_quick_all_s0`)與 `make_pairs --quick`(`pairs_quick`)的目錄名沒變,重跑要 `--overwrite`;`train_dpo --quick` 的目錄改為 `dpo_sft_quick_all_s0_pairs_quick_s0`。

**常數(`config.py`)**:`C.DPO["eval_every"]` 10、`C.DPO["dev_eval_n"]` 60、`C.SFT["retrain_full"]` True、`C.EVAL["rarefy_n"]` 100、`C.EVAL["boot_n"]` 2,000、`C.HUMAN`、`C.LINES`。

**測試**:`python tests/run_tests.py --synth`(2026-11-02):54 個測試,47 過、0 失敗、7 SKIP(`test_mask` 2 個缺 transformers、`test_train` 5 個缺 torch)。`generate.token_batches`、`train_dpo.selection_key` 等放在 import torch 的模組裡的純函式有單元測試(`tests/test_train.py` 末三個),本機有 torch 時才跑(複查 R2-8)。

**計時(待確認)**:上表的「約 1 s/step」來自 `step_seconds`,它只量每步 4 個 micro-batch 中的最後一個加上參數更新(`train_sft.py:130, :142`),所以一步的實際時間更長,5–15 分鐘的估計也可能偏低;`metrics.json` 新增的 `seconds`(整段訓練的牆鐘時間)可以拿來重估。
