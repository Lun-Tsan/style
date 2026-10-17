# persona — pipeline v1 程式碼(2026-10-13)

規格:`../PIPELINE_SPEC.md`(§10 是實作狀態)。所有常數在 `config.py`。全程本機;程式不印訊息原文。

## 執行順序

```bash
# 0. 測試(慢的要跑尺驗證 ~2 分鐘與 tokenizer;--fast 跳過)
python tests/run_tests.py

# 1. 資料層 + 尺驗證 + 校準 + 特徵帶  → reports/{data_stats,ruler_validation,nb_calibration,feature_bands}.json
python -m persona.stage1                 # v1 規則(顏文字白名單)
python -m persona.stage1 --reproduce     # 用實驗 4 的原規則重現數字(逐格相同)

# 2. 對照組與基準線(驗收尺)            → runs/eval_*/ 與 reports/eval_table.json
python -m persona.run_eval --system controls
python -m persona.run_eval --system base
python -m persona.run_eval --system baseline --kind zero
python -m persona.run_eval --system baseline --kind many --shots 50
python -m persona.run_eval --system baseline --kind retrieval --shots 20

# 3. SFT(每個子集 × 底模 × 種子)        → runs/sft_<model>_<size>_s<seed>/adapter_best
python -m persona.train_sft --model instruct --size 100 --seed 0
python -m persona.train_sft --model instruct --size all --seed 0
python -m persona.run_eval --system sft --run sft_instruct_all_s0

# 4. 候選 + 配對 + DPO                     → runs/pairs_*/pairs.jsonl, runs/dpo_*/adapter_best
python -m persona.make_pairs --sft sft_instruct_all_s0 --out pairs_instruct_all_s0
python -m persona.train_dpo --sft sft_instruct_all_s0 --pairs runs/pairs_instruct_all_s0/pairs.jsonl
python -m persona.run_eval --system sft --run sft_instruct_all_s0 --dpo dpo_instruct_s0

# 5. 本人盲判題組(persona.human_eval.make_items / score)
```

每個訓練類指令都有 `--quick`(0.8B、十幾個樣本)可先冒煙。

## 模組

| 模組 | 做什麼 |
|---|---|
| `parse_line` `turns` `normalize` `flag_symbols` `split` `samples` `prompt` | LINE 匯出 → 回合 → 段落 → 樣本;符號類回合規則;唯一的 prompt 組裝器 + loss 遮罩 |
| `rulers/author_lm` `profiles` `ncd` `features` `calibrate` `validate` | NB 作者 LM、距離、NCD、16 個可解釋特徵與容忍帶、等張校準、E1/E3/E4/E5 驗證 |
| `modeling` `data_sft` `train_sft` | 模型載入(Windows GQA patch)、LoRA、SFT 迴圈、金絲雀稽核 |
| `generate` `embed` `pairs` `make_pairs` `train_dpo` | 取樣、抄襲/外漏檢查、bge-m3 離題門檻、行為格配對、自寫 DPO |
| `evaluate` `run_eval` `baselines` `human_eval` | 驗收尺(獨立分辨器、NCD、特徵帶、JSD、離題、抄襲、外漏、多樣性、NLL、MMI)、對照組、prompting 基準線、盲判 |

## 已知限制
- `Qwen/Qwen3.5-4B-Base` 不在 HF 快取;OpenCC 未安裝(Leak 指標回 None);Qwen3.5 的線性注意力核心未安裝(用參考實作,較慢)。
- 權重(`runs/*/adapter_best`)與 `runs/*/outputs.jsonl`、`human_eval` 題組含本人訊息,視同聊天紀錄,不外流。
