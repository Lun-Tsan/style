# Pipeline 圖與依賴審視(2026-11-02)

整個預想的 pipeline 畫成一張總覽,每個階段一張;另有一張由原始碼掃出來的 import 關係圖。圖是用程式畫的(`make_figures.py` + `svgfig.py`),每個方塊的位置與文字都寫在程式裡,之後要調整細節就改那幾行,重跑一次全部重畫:

```bash
python docs/pipeline/make_figures.py      # 重畫 00–09;09 依當下的 persona/ import 關係產生
python docs/pipeline/depgraph.py          # 只印出 import 的分層與邊
```

圖裡的顏色:綠 = 已在真資料上跑過、數字在 `reports/`;黃 = 程式寫好、測過,但正式實驗(或 GPU 路徑)還沒跑;灰色文件形 = 檔案產物;虛線框「私有」= 含訊息原文,不進 git。數字全部來自 `handover.md`、`PIPELINE_SPEC.md` 與 `CODE_REVIEW_2026-10-17.md`,沒有新算的。

| 圖 | 內容 | 對應程式 |
|---|---|---|
| `00_overview.svg` | 資料 → 尺 → SFT → 配對 → DPO → 驗收 → 盲判;對照組與基準線在下方定座標、鎖門檻;訓練尺與驗收尺是兩組 | 全部 |
| `01_data_layer.svg` | LINE 匯出 → 訊息 → 回合(三種文字檢視)→ 符號類規則 → 段落 → 切分 → 樣本 → prompt;資料清單 | `parse_line` `turns` `normalize` `flag_symbols` `split` `samples` `prompt` `data` `synth` |
| `02_rulers.svg` | 三族尺(NB 作者 LM、距離 / NCD、16 個特徵)各自的驗證、校準、容忍帶,以及它們流向 ⑤ 與 ⑦ | `rulers/*` `stage1` |
| `03_controls_baselines.svg` | C-real / C-shuf / C-S / 未訓練模型與 B0–B3,經同一個 Evaluator 進 `eval_table.json`;鎖門檻 | `run_eval` `baselines` `evaluate` |
| `04_sft.svg` | 階段 1(dev 選 epoch → `adapter_devsel`)、階段 2(train+dev 重訓 → `adapter_best`)、金絲雀稽核 | `train_sft` `data_sft` `modeling` |
| `05_pairs.svg` | 8 候選 → 約束 → 行為格 → chosen / rejected / DITTO;訓練尺的封頂在原始 LLR 尺度 | `make_pairs` `pairs` `generate` `embed` |
| `06_dpo.svg` | 參考 = 合併後 SFT(log-prob 先算)、policy LoRA、第 0 步起的 dev 評估與選檢查點、SFT 有重訓時的兩階段 | `train_dpo` |
| `07_acceptance.svg` | 符號類過濾 → 六組尺 → `eval_table.json` → 配對 bootstrap → 三條線的判定 | `evaluate` `run_eval` `report` |
| `08_human_eval.svg` | 出題(一個前文一個系統、遮住其他題的真實回覆)、作答、每個系統一條 Wald SPRT、線 3 | `human_eval` `report` |
| `09_dependencies.svg` | `persona/` 的 import 分層;紫色 = import 時就要 torch / transformers / peft | — |

## 依賴關係看到的事

從 `09_dependencies.svg` 與這次改程式的過程整理;標「已處理」的是 2026-10-17 到 11-02 之間改掉的,其餘是規劃方向。

**已處理**
- **torch 的邊界**。原本 `evaluate` → `generate` → torch,所以對照組、配對邏輯、盲判這些純 Python 的部分在沒有 GPU 的機器上連 import 都不行。把 `reply_text` / `CopyChecker` / `leak_chars` 搬到 `text.py`、`load_all` 搬到 `data.py`,並讓 `embed` 與 `make_pairs` 在函式內才 import torch 之後,圖裡的白色模組(`evaluate` `pairs` `report` `human_eval` `data` 與整個 `rulers/`)都能在雲端用合成資料測。現在 import 時就要 torch 的只剩 `modeling` `data_sft` `generate` `baselines` 與三個驅動程式。
- **階段之間的契約是 `reports/` 裡的檔案**:`ruler_validation.json` 的 OOF 分數給 ⑤ 的校準與封頂、`feature_bands.json` 給 ⑥ 與 ⑦、`data_manifest.json` 給所有驅動程式核對。`make_pairs` 現在會檢查 OOF 陣列和訓練回合逐位對齊,重跑 `stage1` 之後 `ruler_validation.json` 必須來自同一份資料。
- **所有驅動程式只經由 `data.load_all` 拿資料**,切分與子集每次重算;切分常數一改,`data_manifest.json` 會擋下後面的步驟,不會默默換資料。
- **輸出目錄互相覆蓋**(#14):目錄名帶設定(SFT 的 `_attn` / `_epN` / `_noretrain`,配對與 DPO 的來源 run,評估的底模與種子),非空目錄要 `--overwrite`。

**規劃方向(還沒動)**
1. `stage1` 同時是指令與函式庫:`stage1.load_all` 的舊名還在被 `tests/` 以外的地方引用嗎?目前沒有(都改成 `data.load_all`),可以拿掉別名。
2. `nb_calibration.json` 只是報告:`make_pairs` 從 `ruler_validation.json` 的 OOF 陣列重新配等張校準,兩邊結果相同但是兩個來源。可以讓 `fit_scorer` 直接讀 `nb_calibration.json` 的斷點,或把它拿掉。
3. 日期區塊有兩份計算:`rulers.validate` 自己算 `date_blocks`(OOF 摺),`data.load_all` 另算一份給管制圖。現在兩者一致(文字母體相同),但沒有測試保證;可以讓 `validate` 用 `turn.block`。
4. `generate.token_batches`、`train_dpo.selection_key` / `dev_subset`、`train_sft.default_out` 是純函式,放在 import torch 的模組裡,測試只能在本機跑(`tests/test_train.py` 末三個)。搬到不含 torch 的模組就能在雲端測。
5. `human_eval` 在 `main()` 裡 import `report`,`report` import `evaluate`:依賴方向是 盲判 → 報表 → 評估。合理,但圖上它因此被排到最上層;若之後要讓 `report` 獨立於 `evaluate`(只讀 `eval_table.json`),把 `d_auc_bootstrap` 的分辨器部分留在 `evaluate` 即可。
6. `run_eval --system base` 的 `base` 指「未訓練」,`--model base` 才是 Base 底模;列名已改成 `untrained_<model>`,但旗標名還是容易混。

## 待使用者決定(圖上標「待決定」的)

這些在 `handover.md` §6 第 10 點也有,這裡對照圖的位置列一次:
- ④ 子集 100 / 250 / 500 的階段 2 用同大小重抽(現行)還是「子集 + 整個 dev」。
- ⑤ 排序改用封頂後的原始 LLR(超出審查的修法);`finished` 只認 EOS,比 §5.2 嚴。
- ⑥ 選檢查點的規則與 §5.4 不同(跑完 1 epoch 取最好的,而不是「D-AUC 不再下降就停」);要不要預先登記。
- ⑦ `symbol_share` 要不要門檻;線 1 的「顯著」用 D-AUC₁;線 2 要 distinct-2 與 Hill-1 都過;12.0 / 15.3 的理由重跑後重看。
- ⑧ SPRT 無差異區 0.55 / 0.75 要不要調嚴(真實 0.70 的系統通過線 3 的機率 0.20);146 個前文分給多個系統時到不了 100 題。

## 本機要重跑的順序

依 `CODE_REVIEW_2026-10-17.md` 文末:`python tests/run_tests.py` → `python -m persona.stage1`(重產 `feature_bands.json`、新寫 `data_manifest.json`)→ `run_eval --system controls --overwrite` → `train_sft --quick --overwrite`、`make_pairs --quick --overwrite`、`train_dpo --quick`、`run_eval --system sft --run sft_quick_all_s0 --dpo dpo_sft_quick_all_s0_pairs_quick_s0 --quick` → `run_eval --system base` 與基準線。圖裡黃色的部分跑過之後,把對應方塊改成綠色(`status="done"`)重畫即可。
