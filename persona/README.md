# persona — pipeline v1 程式碼(2026-10-13;2026-11-02 依程式審查修改)

規格:`../PIPELINE_SPEC.md`(§10 是實作狀態,2026-11-02 的修改在 §10 最後);審查與修正狀態:`../CODE_REVIEW_2026-10-17.md`。所有常數在 `config.py`。全程本機;程式不印訊息原文。

## 執行順序

```bash
# 0. 測試(慢的要跑尺驗證 ~2 分鐘與 tokenizer;--fast 跳過)
python tests/run_tests.py

# 1. 資料層 + 尺驗證 + 校準 + 特徵帶 + 資料清單  → reports/{data_stats,ruler_validation,nb_calibration,feature_bands,data_manifest}.json
python -m persona.stage1                 # v1 規則(顏文字白名單)
python -m persona.stage1 --reproduce     # 用實驗 4 的原規則重現數字(逐格相同)
#    之後的指令載入資料時比對 data_manifest.json:切分常數或資料變了就停下來,要先重跑 stage1

# 2. 對照組與基準線(驗收尺)            → runs/eval_<name>/ 與 reports/eval_table.json
python -m persona.run_eval --system controls                        # C-real、C-shuf、C-S
python -m persona.run_eval --system base                            # → untrained_instruct_s0(C-base;--model base 換 Base 底模)
python -m persona.run_eval --system baseline --kind zero            # → B-zero_instruct_s0
python -m persona.run_eval --system baseline --kind many --shots 50 # → B-many-50_instruct_s0(不給 --shots 就是全部範例)
python -m persona.run_eval --system baseline --kind retrieval --shots 20

# 3. SFT(每個子集 × 底模 × 種子)        → runs/sft_<model>_<size>_s<seed>/{adapter_best,adapter_devsel}/
python -m persona.train_sft --model instruct --size 100 --seed 0
python -m persona.train_sft --model instruct --size all --seed 0
python -m persona.run_eval --system sft --run sft_instruct_all_s0    # → sft_instruct_all_s0_s0
#    階段 1:train − dev 上訓練,dev NLL 選 epoch → adapter_devsel/;階段 2:train + dev 上用同樣 epoch 數重訓 → adapter_best/
#    --no-retrain 跳過階段 2(目錄加 _noretrain);--attn-only → _attn;非預設的 --epochs N → _epN

# 4. 候選 + 配對 + DPO
python -m persona.make_pairs --sft sft_instruct_all_s0              # → runs/pairs_sft_instruct_all_s0_s0/
python -m persona.train_dpo --sft sft_instruct_all_s0 --pairs runs/pairs_sft_instruct_all_s0_s0/pairs.jsonl
#    → runs/dpo_sft_instruct_all_s0_pairs_sft_instruct_all_s0_s0_s0/:adapter_best(dev 上選定的步)、adapter_last(最後一步)
#    dev 評估在第 0 步、--eval-every N(預設 10)的倍數步、最後一步;SFT 有重訓時先從它的 adapter_devsel 選步數,再從 adapter_best 重跑
python -m persona.run_eval --system sft --run sft_instruct_all_s0 --dpo dpo_sft_instruct_all_s0_pairs_sft_instruct_all_s0_s0_s0

# 5. 本人盲判(題組只能放在 runs/ 底下;一個前文只給一個系統;Wald SPRT,系統各自停)
python -m persona.human_eval make --kind A --systems sft_instruct_all_s0_s0 --out runs/human_A --n 100
python -m persona.human_eval make --kind B --systems sft_instruct_all_s0_s0,B-many-50_instruct_s0 --out runs/human_B   # 最後一個是基準
python -m persona.human_eval score --dir runs/human_A --answers answers.json    # answers.json:{"<題號>": "left" 或 "right"}
#    --priority sids.json 優先選這些測試前文(例如三把尺意見分歧的;清單自己準備)
#    146 個測試前文分給 2 個系統時,一個系統最多 73 題,到不了 100 題(怎麼安排待決定)

# 6. 論文表格與比較(只輸出數字)
python -m persona.report --table                                     # reports/eval_table.json → markdown,同一系統的種子取平均 ± sd
python -m persona.report --compare sft_instruct_all_s0_s0 B-many-50_instruct_s0   # 以測試前文為單位的配對 bootstrap(2,000 次)
```

輸出目錄(`runs/sft_*`、`runs/pairs_*`、`runs/dpo_*`、`runs/eval_*`、盲判資料夾)已存在且非空時,指令會在載入模型之前拒絕寫入;要覆寫就加 `--overwrite`。

### 沒有真資料時(雲端、CI):合成匯出

`synth.py` 產生假的兩人 LINE 匯出(片語是這個檔案自己寫的,不含任何真實訊息;風格差異的方向照 stage-1 報告),並回傳解析器必須重現的計數(訊息、系統行、回合、段落、各期文字回合)。`PERSONA_SYNTH=1` 時 config 換成假名字,所有輸出改寫到 `runs/synth/`,不會蓋掉真的 `reports/`。

```bash
python tests/run_tests.py --synth          # 54 個測試跑在合成匯出上,缺 torch / transformers 的記為 SKIP(2026-11-02:47 過、0 失敗、7 SKIP)
PERSONA_SYNTH=1 python -m persona.stage1   # 整個 stage 1 → runs/synth/reports/
python -m persona.synth --seed 0           # 只產生匯出 → runs/synth/chathistory_synth.txt(+ .truth.json)
```

合成資料只驗證程式(計數、規則、尺的管線能跑、方向對),不能拿來支持任何關於真人的數字。

每個訓練類指令都有 `--quick`(0.8B、十幾個樣本)可先冒煙。`sft_quick_all_s0`、`pairs_quick` 已存在時要加 `--overwrite`;`train_sft --quick` 不做 train+dev 重訓,要測重訓用 `--size 100`。

## 模組

| 模組 | 做什麼 |
|---|---|
| `parse_line` `turns` `normalize` `flag_symbols` `split` `samples` `prompt` | LINE 匯出 → 回合 → 段落 → 樣本;符號類回合規則;唯一的 prompt 組裝器 + loss 遮罩 |
| `data` `text` | 不 import torch 的共用部分:`data.load_all`(回合、段落、日期區塊;比對 `reports/data_manifest.json`)與資料清單;`text`(回覆文字、抄襲檢查、OpenCC 外漏) |
| `synth` | 合成兩人匯出(測試與冒煙用,不含真實訊息) |
| `rulers/author_lm` `profiles` `ncd` `features` `calibrate` `validate` | NB 作者 LM、距離、NCD、16 個可解釋特徵與容忍帶、等張校準、E1/E3/E4/E5 驗證 |
| `modeling` `data_sft` `train_sft` | 模型載入(Windows GQA patch)、LoRA、SFT 迴圈(dev 選 epoch + train+dev 重訓)、金絲雀稽核 |
| `generate` `embed` `pairs` `make_pairs` `train_dpo` | 取樣(依 token 預算分批、記錄是否出現 eos)、bge-m3 離題門檻(訓練期配)、行為格配對、自寫 DPO(dev 評估選檢查點) |
| `evaluate` `run_eval` `baselines` `human_eval` | 驗收尺(獨立分辨器、NCD、特徵帶、JSD、離題、抄襲、外漏、多樣性、NLL、MMI;符號類的生成回覆不進尺,另報 `symbol_share`)、對照組、prompting 基準線、盲判題組與 Wald SPRT |
| `report` | 論文表格、以測試前文為單位的配對 bootstrap(含 D-AUC 的快速版)、驗收線 1–3 的判定(門檻在 `config.LINES`) |

## 已知限制
- `Qwen/Qwen3.5-4B-Base` 不在 HF 快取;OpenCC 未安裝(Leak 指標回 None);Qwen3.5 的線性注意力核心未安裝(用參考實作,較慢)。
- 2026-11-02 改過的程式中,要模型或 GPU 的部分(生成、NLL、MMI、bge-m3、SFT 重訓、DPO 評估)還沒在 GPU 上跑過;清單與本機重跑順序在 `../CODE_REVIEW_2026-10-17.md` 文末。
- 權重(`runs/*/adapter_best`、`adapter_devsel`、`adapter_last`)與 `runs/*/outputs.jsonl`、`runs/*/dev_outputs.jsonl`、`human_eval` 題組含本人訊息,視同聊天紀錄,不外流。
