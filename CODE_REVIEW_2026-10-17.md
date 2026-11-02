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

**2026-11-02**:已經動手,逐項狀態見文末「處理狀態 2026-11-02」。

1. 正式實驗前一定要修:#1、#4、#11(會改已寫下的對照組數字,修完重跑 `run_eval --system controls`,把 handover §4 第 9 點與驗收線 2 的理由重看)、#2、#3、#12、#14。
2. 盲判前要修:#5、#6。
3. DPO 前要修或在 §10 記錄:#8、#10、#16;#9 決定要不要做,不做就寫進 §10。
4. #7 在寫論文表格前補。

## 處理狀態 2026-11-02

修正分成六個工作包(評估與資料層、配對、盲判與報告、生成、SFT、DPO),修完由兩個 agent 複查,複查找到的問題再修一輪(下一節)。行號是 2026-11-02 工作樹的行號。狀態:**已修** = 改了程式,改到的部分在雲端用合成資料執行過;**已修,GPU 路徑未執行** = 改了程式,需要模型或 GPU 的部分只做了 `ast.parse`、讀程式、用替身 torch 追流程;**未修** = 附原因。`python tests/run_tests.py --synth`(2026-11-02):54 個測試,47 過、0 失敗、7 SKIP(`test_mask` 2 個缺 transformers、`test_train` 5 個缺 torch)。

| # | 狀態 | 改了哪裡 | 驗證 / 還要做的 |
|---|---|---|---|
| 1 | 已修,GPU 路徑未執行 | `evaluate.py:275`:`model_nll(model, tok, self.samples, self.real)` | `test_nll_is_taken_on_the_real_replies` 用替身確認傳進去的是真實回覆;真模型的 NLL 沒算過。`base_s0_quick` 的 1.851 是舊定義,要重跑 |
| 2 | 已修;排序方式待確認 | `pairs.py:72-89`(封頂改在原始 LLR:T 訓練回合 OOF LLR 的 95 百分位;OOF 陣列與訓練回合對不齊就停)、`:92-97`(同分用種子隨機挑)、`:144-147`(用 min(raw, cap_raw) 排序) | 由 `reports/ruler_validation.json` 的 OOF 陣列重算:cap_raw 51.28,5.1% 的 T 回合高於它;等張校準在 raw 37.81 已到 6.907,所以 cal(cap_raw) 仍是天花板,只照原修法改分數,封頂還是沒作用。實作改用封頂後的原始 LLR 排序,超出原修法;要照原公式就改回 `:147`。`test_fit_scorer_raw_cap`、`test_build_pairs_raw_cap_ties_and_exclusions`;候選生成沒執行 |
| 3 | 已修 | `evaluate.py:244-277`、`run_eval.py:155-158`(`--mmi`)、`pairs.py:121-128`、`human_eval.py:45-52`、`report.py:124-185`、`train_dpo.py:111-131`:符號類的生成回覆不進尺、不當 chosen / rejected、不出題,比例記為 `symbol_share` | `test_symbol_like_replies_leave_the_rulers`(外漏用替身計數器,OpenCC 未裝;MMI 用替身)。`symbol_share` 沒有登記門檻,待決定 |
| 4 | 已修;真資料未重算 | `stage1.py:85-89`(特徵帶、鑑別度、管制圖含純貼圖回合)、`data.py:27-36`(純貼圖回合歸入同日期的日期區塊;文字回合的區塊不變) | 合成資料(`PERSONA_SYNTH=1`、200 袋的特徵帶):C-real Band 13.7 → 14.4(16 個特徵)。真資料的 `reports/feature_bands.json` 還是舊母體 |
| 5 | 已修;無差異區待決定 | `human_eval.py:155-176`(Wald SPRT,參數在 `C.HUMAN`)、`:179-192`(舊規則留作 `ci_stop`,不使用) | 對 (n, k) 精確計算、最多 100 題:舊規則在真實 0.65 時判「低於」0.120(上表的 12.3% 是模擬值),真實 0.70 時通過線 3 的機率 0.173;新規則 0.201,平均 44 題停(舊 78 題)。數字與選項在 `PIPELINE_SPEC.md` §6.2 |
| 6 | 已修 | `human_eval.py:55-152`:一個前文只給一個系統(題型 A、B 都是,`:88-94`)、依時間出題(`:110`)、題型 A 前文裡其他題的真實回覆顯示成 `我: [略]`(`:37-42, :112`)、顯示的回覆逐則去識別化;`score` 對系統各自做 SPRT,合併的數字不下結論(`:195-236`) | `tests/test_human.py`;CLI 只在假輸出上跑過。待決定:146 個測試前文分給 2 個系統時,一個系統最多 73 題,到不了 100 題 |
| 7 | 已修 | 新檔 `report.py`:`table`(`:60`)、`paired_bootstrap`(`:80`)、`d_auc_bootstrap`(`:124`)、`compare`(`:195`)、`acceptance`(`:253`)、`lines_from_human`(`:288`)、CLI(`:310`);門檻在 `config.LINES` | `--table` 在現有 `reports/eval_table.json` 上跑過,`--compare` 只在假輸出上。D-AUC 的 bootstrap 沒有重新擬合分辨器(docstring 寫明)。三個實作解讀待確認(`PIPELINE_SPEC.md` §6.3) |
| 8 | 已修,GPU 路徑未執行 | `train_dpo.py`:第 0 步、`--eval-every`(10)的倍數步、最後一步之後,在 60 個 dev 前文上評估(`:226-292`);選法(`:47-52, :123-131`);SFT 的 `adapter_best` 在 dev 上訓練過時分兩階段(`:155-176, :208-303`) | 只在替身 torch 下跑過 `run()`。選法與 §5.4 不同(`PIPELINE_SPEC.md` §10),正式 DPO 前要確認 |
| 9 | 已修,GPU 路徑未執行 | `train_sft.py:96-166`(`train_epochs`)、`:199-219`(階段 2:重新載入底模、新 LoRA、train+dev、選定的 epoch 數)、`--no-retrain`(`:249`) | 只在替身 torch / peft 下追過。子集 100 / 250 / 500 的階段 2 用 train+dev 上同大小的巢狀子集(`:50-65`),待確認 |
| 10 | 已修 | `pairs.py:177-202`、`make_pairs.py:75-76`:§5.3 的兩個驗收只算非 DITTO 配對,含 DITTO 的另存 `*_all` | `build_pairs` 在合成資料上端到端跑過 |
| 11 | 已修 | `evaluate.py:229-234`、`run_eval.py:130`:C-S 從 `t.msgs` 建 | `test_controls_define_axes`;真資料 C-S 的 Band 6.6 要重跑 |
| 12 | 已修,bge-m3 未執行 | `embed.py:58-66`(`fit_ontopic_threshold`,只用訓練期回覆)、`run_eval.py:35-53`、`make_pairs.py:69` | `test_ontopic_threshold_from_training_replies` 用替身編碼器;C-shuf 的離題率 0.10 是用測試期門檻算的,要重跑 |
| 13 | 已修 | `evaluate.py:181-188`(排序後再抽)、`:244`(預設 100 則)、`:262-268`(和 C-real 在同一個 n 的比值) | `test_controls_define_axes`、`test_diversity_ratio_at_the_same_n`;`eval_table.json` 現有的列沒有稀疏化(div_n 146) |
| 14 | 已修 | `train_sft.py:68-75, :182-183`、`make_pairs.py:24-33`、`train_dpo.py:98-102, :169-171`、`run_eval.py:68-73, :108-118`、`human_eval.py` 的 `make`:目錄名帶設定;目錄非空時沒有 `--overwrite` 就拒絕,在載入模型之前 | `test_make_pairs_out_dir`、`test_clis_stay_under_runs`;`run_eval --system controls` 用替身 torch 跑過兩次(第二次被拒);train_sft、train_dpo 只在替身下跑過 |
| 15 | 已修,GPU 路徑未執行 | `generate.py:11-57`(依 token 預算分批,預設 12,000 token、一批最多 16 個 prompt)、`baselines.py:87-112`(生成前記 prompt token 數) | 真的 `model.generate` 與 VRAM 沒測 |
| 16 | 已修,GPU 路徑未執行 | `generate.py:27-57, :68-77`(`finished`)、`pairs.py:114-128`(沒結束的候選不參加)、`make_pairs.py:61, :72` | 只認 eos,比 §5.2 嚴(`PIPELINE_SPEC.md` §10);真資料跑 make_pairs 後看 `stats.json` 的 `cand_unfinished` |
| 17 | 已修,GPU 路徑未執行 | `train_sft.py:78-94`(`load_lora`、`canary_audit`)、`:220-222`(稽核存下來的 `adapter_best`) | 階段 2 後 dev 已進訓練資料,另記 `canary_devsel` 與 `dev_items_trained` |
| 18a | 已修(文件) | `PIPELINE_SPEC.md` §0 | 寫明配對程式實際用的訓練尺 |

其他小問題:
- Isotonic 同分依輸入順序切(`rulers/stats.py:107`):未修(這一輪沒有人改 `rulers/`)。
- `leak_chars` 在轉換改變長度後對不齊:已修(commit 2df8fa7,`text.py:40-61` 改用 difflib 對齊);OpenCC 未裝,沒執行過。
- `parse_line` 的 ValueError 印顯示名稱:已修(commit 2df8fa7,`parse_line.py:93-95` 只印代號)。
- `--system base` 預設載入 Instruct:預設沒改;列名改成 `untrained_<model>_s<seed>`(`run_eval.py:112`),看得出用的是哪個底模。
- `train_dpo` docstring 的 `--sft` 路徑:已修(`train_dpo.py:3, :6`)。`run_eval.py:8` 的用法範例還是舊名 `--dpo dpo_instruct_s0`。

## 修正後的複查(2026-11-02)

R1 複查評估、配對、盲判與報告的修正,R2 複查 SFT、DPO、生成與基準線的修正。

| 編號 | 問題 | 狀態 |
|---|---|---|
| R1-1 | 題型 A 的前文會顯示其他題的真實回覆;用 `--priority` 時,一題的真實回覆可能在它被問之前就出現在別題的前文裡 | 已修:題目依時間排,priority 只決定選哪些前文(`human_eval.py:110`);前文裡其他題的真實回覆顯示成 `我: [略]`(`:37-42, :112`);`stats.json` 的兩個計數由構造為 0(`tests/test_human.py:124` 檢查) |
| R1-2 | 某系統的題目用完時,線 3 用那段短序列的點估計判 | 已修:沒碰到界線、也沒到 100 題時線 3 記為未定,標 `more_items_needed`(`report.py:301-305`、`human_eval.py:292-295`) |
| R1-3 | 外漏與 MMI 還含符號類回覆 | 已修:`evaluate.py:259, :277`、`run_eval.py:155-158` |
| R1-4 | 用登記的無差異區(0.55 / 0.75),真實 0.70 的系統通過線 3 的機率 0.201,比舊規則(0.173)高 | 未修:要不要調嚴由使用者決定(只能調嚴);選項在 `PIPELINE_SPEC.md` §6.2 |
| R1-5 | 題型 B 仍讓多個系統對同一前文出題,重複出現的基準回覆洩漏哪一邊是基準 | 已修:題型 B 也一個前文只給一個系統(`human_eval.py:88-94`) |
| R1-6 | 系統留下的回覆不到 100 則時,多樣性和 C-real 不在同一個 n 比 | 已修:`evaluate.py:262-268` 另報同一個 n 的比值,`report.py:243-251` 用它 |
| R2-1 | SFT 重訓後的 `adapter_best` 看過 dev,DPO 在 dev 上選檢查點就不是保留資料 | 已修,GPU 路徑未執行:DPO 分兩階段(`train_dpo.py:155-176, :208-303`) |
| R2-2 | 第 0 步(SFT 起點)沒有評估 | 已修:`train_dpo.py:247-248` |
| R2-3 | 選法不看 `symbol_share`,dev 回覆變成符號類的檢查點反而排前面 | 已修:比第 0 步高出 0.05 以上的排在後面(`train_dpo.py:47, :123-131`) |
| R2-4 | 程式把選法標成「預先登記」,但 #8 沒有規則、§5.4 登記的規則不同 | 程式的標示已拿掉(`train_dpo.py:48-52`);與 §5.4 的差異寫進 `PIPELINE_SPEC.md` §10;要不要登記由使用者決定 |
| R2-5 | 生成用不截斷的前文,訓練與 NLL 用截到 512 token 的前文 | 已修,GPU 路徑未執行:`generate.py:60-65, :74`、`baselines.py:43-46, :104` |
| R2-6 | `finished` 只認 eos,§5.2 也把 `\n對方:` 當停止 | 未改程式(介面約定以 eos 定義);寫在 `generate.py:68-73` 與 `PIPELINE_SPEC.md` §10 |
| R2-7 | `persona/README.md` 的 DPO 目錄名過時 | 已修(2026-11-02 文件更新) |
| R2-8 | `generate.token_batches`、`train_dpo.selection_key` / `dev_subset` / `default_out`、`train_sft.default_out` / `size_key` 沒有單元測試 | 已修:`tests/test_train.py` 末三個測試(`test_token_batches`、`test_dpo_selection_key_and_dev_subset`、`test_sft_run_names`);這些模組最上層 import torch,所以雲端記 SKIP、本機才跑;雲端用替身 torch 執行過一次,全過 |

## 尚未在 GPU 上驗證

下列檔案的改動有一部分要模型、真 tokenizer 或 GPU;這些部分只做了 `ast.parse`、讀過 transformers 5.17 / peft 0.21 的原始碼、用替身 torch 追過,沒有真的執行:
- `persona/train_sft.py`:階段 2 重訓、重新載入底模、`load_lora`、金絲雀稽核、VRAM 釋放。
- `persona/train_dpo.py`:訓練中生成 dev 回覆、兩階段 DPO、存 `adapter_best` / `adapter_devsel` / `adapter_last`、訓練中生成時的 VRAM。
- `persona/generate.py`:`model.generate` 的分批與手動左側補齊、在 eos / pad 截斷、Qwen 實際的 eos / pad id、`generation_prompt` 用真 tokenizer 的截斷。
- `persona/baselines.py`:B1-200、B1-all 的 prompt 長度與 VRAM。
- `persona/embed.py`:bge-m3 編碼(`fit_ontopic_threshold` 只用替身編碼器測過)。
- `persona/evaluate.py`、`persona/run_eval.py`:NLL、MMI(只用替身確認傳入的資料)、外漏(OpenCC 未裝);`run_eval.main` 只在替身 torch 下跑過。
- `persona/make_pairs.py`:整個 `run()`(候選生成 + bge-m3)。
- 要真資料、不要 GPU 的:`human_eval make` / `score` 與 `report --compare` 只在假輸出上跑過。

## 需要在本機重跑

依序(已存在的輸出目錄要加 `--overwrite`):
1. `python tests/run_tests.py`(真資料,含要 torch / transformers 的測試)。
2. `python -m persona.stage1`:重產 `reports/feature_bands.json`(#4 的母體)並寫 `reports/data_manifest.json`;之後的指令會比對這份清單。
3. `python -m persona.run_eval --system controls --overwrite`:Band 的特徵帶、C-S、離題門檻、多樣性的算法改了,`symbol_share` 與 `div_*_ratio` 是新欄位;重看 handover §4 第 9 點與線 2「≥ C-real − 1」的理由。
4. 冒煙:`python -m persona.train_sft --quick --overwrite`(不含階段 2;最便宜的重訓測試是 `--size 100`)→ `python -m persona.make_pairs --quick --overwrite` → `python -m persona.train_dpo --quick` → `python -m persona.run_eval --system sft --run sft_quick_all_s0 --dpo dpo_sft_quick_all_s0_pairs_quick_s0 --quick`。
5. `run_eval --system base` 與基準線 B0 / B1 / B2:`base_s0_quick` 的 NLL 1.851 不是 §6.1 的量;prompt 截斷、離題門檻、稀疏化都改了。B1-200、B1-all 先跑,看記下的 prompt token 數。
6. 2026-11-02 之前在本機產生的 SFT、配對、DPO、評估輸出與 `reports/eval_table.json` 的舊列都是舊程式的結果:`adapter_best` 的意思變了(重訓)、舊的稽核用最後一個 epoch、舊配對的封頂沒作用、生成的 prompt 不同。
