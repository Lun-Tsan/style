# 程式審查:`persona/`(2026-10-17)

範圍:還沒有真正跑過的部分(`make_pairs` `pairs` `train_dpo` `run_eval` `evaluate` `generate` `baselines` `human_eval` `train_sft` `data_sft` `prompt`),`rulers/` 與資料層較輕。方法:雲端 session 沒有資料、torch、GPU,所以是**讀程式**,純 Python 的部分用合成輸入與 `reports/` 裡的數字驗算。審查由一個 agent 做;標「已核對」的項目,另一個 session 回到程式碼逐行確認過,其他項目只有審查者一人讀過。

**沒有問題的部分**(審查者的結論,未另外核對):DPO loss 的符號與公式;`seq_logp` 只加總目標 token + EOS,位移一格正確;參考模型的 logp 與 policy 用同一個 [chosen, rejected] 右側補齊的批次;`lm_loss`(`data_sft.py:58-59`)有算到第一個目標 token,沒有 IMPersona 的少算一格(但 `test_mask` 的邊界斷言依構造必然成立,真正擋住錯誤的是這個位移);校準只用訓練期 OOF;獨立分辨器在測試期重訓、fold 按字串分;行為格階梯、±log 3、DITTO ≤ 50% 與 §5.3 相符;種子有傳下去;`reports/*.json` 沒有 ≥ 9 個中文字的連續字串,n-gram 最長 3 字。

## 會影響已經寫進文件的數字

| # | 位置 | 問題 | 受影響的數字 | 核對 |
|---|---|---|---|---|
| 1 | `evaluate.py:237-238` | NLL 算的是**系統自己生成的回覆**,不是真實測試回覆(規格 §6.1 表:「真實 T 測試回覆在該模型下的 NLL」)。一個只會回「好啊」的模型 NLL 最好。修法:`model_nll(model, tok, self.samples, self.real)` | `eval_table.json` 的 `base_s0_quick` NLL 1.851 不是規格定義的量 | 已核對 |
| 4 | `stage1.py:91-94` vs `evaluate.py:110` | 特徵帶只用 T 的**文字回合**配(純貼圖回合沒有 `ruler_text`),驗收時的袋卻包含純貼圖回覆(測試期 13.7%)。貼圖帶 [0.027, 0.094],C-real 的袋至少 0.137,所以 C-real 的貼圖特徵永遠在帶外;不回純貼圖的系統反而得分 | handover §4 第 9 點「真人 T 在測試期只有 12.0 / 15.3 個特徵在帶內(時期漂移)」至少有一部分是這個不一致造成的;驗收線 2「≥ C-real − 1」的理由要重看 | 已核對邏輯;0.137 與 13.7% 是審查者算的 |
| 11 | `run_eval.py:101`(`evaluate.py:213`) | C-S 用 `ruler_text` 當回覆:整個回合變成一則、顏文字已換成佔位字元。連發永遠是 1,長度、語氣詞、口頭禪、顏文字都算錯。修法:用 `t.msgs` | C-S 的 Band 6.6 部分是假象 | 已核對 |

## 訓練訊號與驗收的正確性

| # | 位置 | 問題 | 核對 |
|---|---|---|---|
| 2 | `pairs.py:73-74`、`:121` | **分數封頂沒有作用**。等張校準的最高一階是 p = 0.999 → log-odds 6.907(`reports/nb_calibration.json` 的 `iso_p` 最大值就是 0.999);OOF 時 T 已有 9.6% 回合在這個天花板,訓練內分數更高,95 百分位 = 天花板,`min(s, cap)` 不會生效 → 「不封頂」消融與主實驗相同。很多候選並列 6.907 時 `max` 取第一個生成的。修法:在不飽和的尺度上封頂(例如每字元的原始 LLR、OOF 的 T 95 百分位),並列時隨機 | 已核對天花板;9.6% 是審查者算的 |
| 3 | `evaluate.py:45/225`、`pairs.py:93-108` | 生成的回覆**從不套符號類規則**,真實目標則一定套過。前文裡有 `我: [程式碼/算式]`,模型生出含 `( / =` 的回覆是合理的。審查者用 repo 的分辨器模擬:10% 的生成帶「(笑)」→ D-AUC₁₀ 0.42 → 0.84;20% → 0.98(驗收線 1 會失敗)。DPO 時只有離題檢查擋在它與 chosen 之間。修法:對生成也套 `is_symbol_turn`,報比率、同樣排除、不能當 chosen | 未核對(模擬) |
| 5 | `human_eval.py:65-74` | 不是 SPRT:每題都看一次 Clopper–Pearson 95% 區間。真實正確率 0.65 時「判定低於」的機率 12.3%(名目 2.5%);真實 0.70 時停下來的點估計 ≤ 0.65 的機率 17%。修法:真正的 Wald SPRT(0.65 對一個選定的對立值,界線 ln((1−β)/α)、ln(β/(1−α))),或固定題數。引用查證那邊也得到同樣結論(`PROBLEMS_AND_SOLUTIONS.md` S2 的補充) | 已核對程式;機率是審查者模擬的 |
| 6 | `human_eval.py:32, 36-53, 91` | 盲判設計:多個系統時,同一前文的 A 型題目連續出現、顯示同一則真實回覆,重複本身就洩題;停止規則把所有系統合併算,不是每個系統各自算;真實回覆沒有去識別化(§6.2 要求) | 未核對 |
| 10 | `pairs.py:140, 148-153` | 「chosen vs 真實」的特徵 JSD 與 chosen 平均分數包含 DITTO 配對(chosen 就是真實回覆、分數是訓練內的),讓 §5.3 的兩個驗收(JSD < 0.05、差距 > 2δ)偏向通過。修法:只算非 DITTO | 未核對 |
| 12 | `run_eval.py:40-41` | 離題門檻在**測試期**真實回覆上配,違反「門檻只在訓練期校準」;應沿用 `make_pairs` 的訓練期門檻 | 已核對 |
| 13 | `evaluate.py:234` | 多樣性沒有稀疏化(`n_rarefy` 從沒傳入),不同 n 的 Hill-1 直接比 | 未核對 |

## 與規格不一致(§10 沒有記錄)

| # | 位置 | 問題 | 核對 |
|---|---|---|---|
| 7 | — | 配對 bootstrap(`report.py`,§0 表格標 🔧)不存在 → 驗收線 1 的「顯著優於最佳 prompting」目前算不出來 | 已核對(找不到) |
| 8 | `train_dpo.py:77, 107-139` | DPO **完全沒有評估**(`eval_every` 沒被使用),把最後一步存成 `adapter_best`。handover §6 第 5 點「只在 epoch 結尾評估」與 l.9 的 docstring 都不對 | 已核對 |
| 9 | `train_sft.py:36-48` | 沒有「dev 選 epoch 後用全訓練期重訓」(§0、§2.5) | 已核對 |
| 18a | `PIPELINE_SPEC.md` §0 | 寫「訓練用的尺含 R1 特徵帶」,但配對程式沒用(`feature_distance` 沒被呼叫);handover §5 與程式一致,§0 要改 | 未核對 |

## 長時間 GPU 跑到後面才會出事的

| # | 位置 | 問題 |
|---|---|---|
| 14 | `train_sft.py:59`、`make_pairs.py:239`、`train_dpo.py:80` | 輸出目錄名不含 LoRA 目標、epoch 或來源 SFT:`--attn-only` 會覆蓋 `runs/sft_instruct_all_s0/adapter_best`;不同 SFT / 消融的配對與 DPO 互相覆蓋。修法:目錄名帶設定,或拒絕寫入已存在的目錄 |
| 15 | `baselines.py:92, 99` | B1-200 與 B1-all 的 prompt(約 6K 與 20–30K token)用 batch 16 生成,大概會 OOM;`--kind many` 不給 `--shots` 就是全部。修法:按 token 預算分批 |
| 16 | `prompt.py:130`(經 `train_dpo`) | 在 64 token 被截斷的候選也補上 EOS,DPO 會學到被截斷的結尾。修法:記錄生成是否自然結束,丟掉沒結束的 |
| 17 | `train_sft.py:128` | 金絲雀稽核用最後一個 epoch 的權重,不是 `adapter_best` |

其他小問題:`Isotonic` 遇到同分時依輸入順序切(`stats.py:107`);`leak_chars` 在轉換改變長度後字元對不齊(`generate.py:81`);`parse_line` 的 ValueError 會印出兩人的顯示名稱(會進終端機紀錄,不進檔案);`--system base` 預設載入的是 Instruct;`train_dpo` docstring 的 `--sft runs/...` 路徑寫錯。

## 建議的處理順序(還沒動手,等決定)

1. 正式實驗前一定要修:#1、#4、#11(會改已寫下的對照組數字,修完重跑 `run_eval --system controls`,把 handover §4 第 9 點與驗收線 2 的理由重看)、#2、#3、#12、#14。
2. 盲判前要修:#5、#6。
3. DPO 前要修或在 §10 記錄:#8、#10、#16;#9 決定要不要做,不做就寫進 §10。
4. #7 在寫論文表格前補。
