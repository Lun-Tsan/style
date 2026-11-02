"""Draw the pipeline figures into docs/pipeline/*.svg.   python docs/pipeline/make_figures.py

Each fig_* function is one figure; positions are explicit (20 px grid) so a label or box can be moved by editing
one number. 07_dependencies.svg is computed from the imports in persona/ (depgraph.py), the rest are drawn by hand
from PIPELINE_SPEC.md and the code as of the date in the caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import depgraph  # noqa: E402
from svgfig import BLUE, BORDER, INK, MUTED, RED, Fig, est_width  # noqa: E402

OUT = HERE
DATE = "2026-11-02"


# ------------------------------------------------------------------------------------------------ 07 dependencies
def fig_dependencies():
    mods = depgraph.modules()
    lay = depgraph.layers(mods)
    torch = depgraph.torch_closure(mods)
    drivers = {"stage1", "train_sft", "make_pairs", "train_dpo", "run_eval", "human_eval", "report", "synth"}
    # order inside a layer: rulers first, then data layer, then model side, so related modules sit together
    def order_key(m):
        groups = ["rulers.", "config", "normalize", "flag_symbols", "synth", "parse_line", "turns", "split", "samples",
                  "prompt", "text", "data", "embed", "modeling", "data_sft", "generate", "evaluate", "pairs",
                  "human_eval", "report", "baselines", "stage1", "train_sft", "make_pairs", "train_dpo", "run_eval"]
        for i, g in enumerate(groups):
            if m.startswith(g):
                return (i, m)
        return (len(groups), m)

    L = sorted(set(lay.values()), reverse=True)        # top layer = drivers (largest depth) drawn first (top)
    rows = {l: sorted([m for m in mods if lay[m] == l], key=order_key) for l in L}
    bw, bh, gap, x0, y0, rowgap = 112, 34, 14, 40, 70, 96
    W = max(len(r) for r in rows.values()) * (bw + gap) + 2 * x0 + 260
    H = y0 + len(L) * rowgap + 100
    f = Fig(W, H, "persona/ 模組 import 關係(由原始碼掃出,%s)" % DATE,
            "實線 = 模組頂層 import;虛線 = 函式內才 import(執行時才需要);往上的弧線 = 延遲 import 造成的互相依賴。"
            "紫色 = import 時就需要 torch / transformers / peft 的模組;白色模組在沒有 GPU 套件的機器上也能 import。")
    pos = {}
    for ri, l in enumerate(L):
        row = rows[l]
        total = len(row) * (bw + gap) - gap
        sx = x0 + (W - 2 * x0 - 260 - total) / 2
        y = y0 + ri * rowgap
        f.text(x0 - 20, y + bh / 2 + 4, f"L{l}", size=10, color=MUTED, anchor="start")
        for ci, m in enumerate(row):
            x = sx + ci * (bw + gap)
            pos[m] = (x, y)
            fill = "#ede9fe" if m in torch else "#ffffff"
            stroke = "#7c3aed" if m in torch else BORDER
            f.rect(x, y, bw, bh, fill, stroke, rx=6, sw=1.4 if m in drivers else 1.0)
            label = m.replace("rulers.", "r.")
            size = 11 if est_width(label, 11) < bw - 10 else 9.5
            f.text(x + bw / 2, y + bh / 2 + 4, label, size=size, anchor="middle", mono=True,
                   weight="bold" if m in drivers else "normal")
            f.boxes[m] = (x, y, bw, bh)
    # edges: from importer (upper) to imported (lower); straight lines between box centres, clipped to box edges
    n_config = 0
    for m, d in mods.items():
        for t, kind in d["edges"].items():
            if t not in pos:
                continue
            if t == "config":                             # every module imports config; drawing it says nothing
                n_config += 1
                continue
            x1, y1 = pos[m][0] + bw / 2, pos[m][1] + bh
            x2, y2 = pos[t][0] + bw / 2, pos[t][1]
            col = "#c4b5fd" if (m in torch and t in torch) else "#cbd5e1"
            if kind == "lazy":
                col = "#9ca3af"
            if lay[t] >= lay[m]:                          # same row (a cycle cut by the layering): arc over the row
                x1, y1 = pos[m][0] + bw / 2, pos[m][1]
                x2, y2 = pos[t][0] + bw / 2, pos[t][1]
                f.parts.append(f'<path d="M {x1:.0f} {y1:.0f} Q {(x1 + x2) / 2:.0f} {y1 - 40:.0f} {x2:.0f} {y2:.0f}" fill="none" '
                               f'stroke="{col}" stroke-width="0.9" stroke-dasharray="4,3" marker-end="url(#arrowmuted)"/>')
                continue
            f.line([(x1, y1), (x2, y2)], stroke=col, sw=0.9, dash="4,3" if kind == "lazy" else None, arrow=True,
                   color_id="muted")
    f.text(pos["config"][0] + bw / 2, pos["config"][1] + bh + 14, f"← {n_config} 個模組 import(邊省略)", size=9.5,
           anchor="middle", color=MUTED)
    # external deps on the right
    ex, ey = W - 240, y0
    f.text(ex, ey - 12, "外部套件(import 時)", size=11, weight="bold", color=MUTED)
    extmods = {}
    for m, d in mods.items():
        for e in d["ext_top"]:
            extmods.setdefault(e, []).append(m)
    for i, (e, ms) in enumerate(sorted(extmods.items())):
        yy = ey + i * 40
        f.rect(ex, yy, 190, 30, "#faf5ff", "#7c3aed", rx=6)
        f.text(ex + 12, yy + 19, e, size=11, anchor="start", mono=True)
        f.text(ex + 178, yy + 19, f"{len(ms)} 個模組直接 import", size=9, anchor="end", color=MUTED)
    f.legend(x0, H - 48, [("model", "import 時需要 torch / transformers / peft"), ("plan", "不需要 GPU 套件"),
                          ("dash", "函式內的延遲 import")], cols=3)
    return "09_dependencies.svg", f


# ------------------------------------------------------------------------------------------------ 00 overview
def fig_overview():
    f = Fig(1700, 800, "整個 pipeline:資料 → 尺 → 訓練 → 驗收(規劃與現況,%s)" % DATE,
            "綠 = 已在真資料上跑過、數字在 reports/;黃 = 程式寫好、測過,正式實驗未跑。訓練用的尺(② → ⑤)與驗收用的尺(② → ⑦,加上 ⑦ 內在測試期重訓的分辨器)是不同的兩組。"
            "驗收線門檻在看過 ③ 的 B0 與未訓練模型之後鎖定,之後只能調嚴。標「私有」的產物含訊息原文,不進 git。")
    Y = 330                                   # the spine
    f.box("chat", 40, Y, 150, 96, "LINE 匯出", ["chathistory.txt", "·4,967 則、兩人", "·2025-06 ~ 2026-09"],
          kind="artifact", private=True)
    f.box("s1", 250, Y, 200, 96, "① 資料層", ["parse_line → turns", "flag_symbols · split", "samples · prompt"],
          status="done", tag="stage1")
    f.box("smp", 510, Y + 8, 150, 80, "樣本", ["·前 12 回合 → T 回合", "·train 679 / dev 118", "·test 146"], kind="artifact")
    f.box("s4", 740, Y, 160, 96, "④ SFT (LoRA)", ["train_sft", "·2 底模 × 4 子集", "·× 3 種子 = 24 次"], status="todo")
    f.box("s5", 990, Y, 170, 96, "⑤ 候選 → 配對", ["make_pairs · pairs", "·8 候選 / 前文", "·行為格 + 約束"], status="todo")
    f.box("s6", 1240, Y, 130, 96, "⑥ DPO", ["train_dpo", "·參考 = 合併後 SFT", "·dev 上用尺選步"], status="todo")
    f.box("s7", 1440, Y - 8, 160, 112, "⑦ 驗收", ["run_eval · evaluate", "·分辨器在測試期重訓", "·三條線", "·report:配對 bootstrap"],
          status="todo")
    f.box("s8", 1440, 150, 160, 96, "⑧ 本人盲判", ["human_eval", "·A:哪個是我寫的", "·B:哪個比較像我"], status="todo")
    f.arrow("chat", "s1")
    f.arrow("s1", "smp")
    f.arrow("smp", "s4", label="train / dev")
    f.arrow("s4", "s5", label="adapter(私有)")
    f.arrow("s5", "s6", label="pairs.jsonl(私有)")
    f.arrow("s6", "s7", label="adapter(私有)")
    f.arrow("s7", "s8", label="outputs.jsonl(私有)", label_side="l")
    # rulers above the spine
    f.box("s2", 250, 90, 410, 150, "② 尺(只用訓練期)",
          ["rulers/: author_lm · profiles · ncd · features",
           "·NB 字元 1–3-gram 作者 LM、頻率距離、NCD、16 個可解釋特徵",
           "·驗證 E1 / E3 / E4 / E5(validate),預登記門檻全過",
           "·等張校準(OOF)、按 k 校準門檻、T 的容忍帶(5–95 百分位)"],
          status="done", tag="stage1")
    f.box("rep", 740, 100, 240, 130, "reports/(只有數字)",
          ["ruler_validation.json", "nb_calibration.json", "feature_bands.json", "data_manifest.json", "data_stats.json"],
          kind="artifact")
    f.arrow("s1", "s2", label="訓練期回合(非符號類)", sides=("t", "b"))
    f.arrow("s2", "rep")
    f.arrow("rep", "s5", label="訓練尺:NB + OOF 校準 + 封頂", sides=("b", "t"), via=[(860, 290), (1075, 290)], label_at=0.6)
    f.arrow("rep", "s7", label="驗收尺:特徵帶、NCD 參照", sides=("r", "l"), via=[(1200, 165), (1200, 282), (1400, 282)],
            label_at=0.08)
    # controls and baselines below the spine
    f.box("s3", 740, 520, 380, 110, "③ 對照組 + 基準線(測試期)",
          ["run_eval --system controls / base / baseline",
           "·C-real · C-shuf · C-S · 未訓練模型(已跑:對照組)",
           "·B0 zero-shot · B1 many-shot · B2 檢索式 · B3 best-of-8"],
          status="todo", tag="對照組 ✓")
    f.box("tab", 1180, 530, 170, 90, "eval_table.json", ["·每個系統一列", "·對照組定座標"], kind="artifact")
    f.box("lines", 1400, 500, 260, 150, "三條驗收線(預先登記)",
          ["·線 1 D-AUC₁ ≤ 0.75、D-AUC₁₀ ≤ 0.90,", "  且配對 bootstrap 優於最佳 prompting",
           "·線 2 Band ≥ C-real − 1、多樣性 ≥ 80%、", "  抄襲 ≤ 2%、外漏 ≤ C-real",
           "·線 3 本人正確率 ≤ 0.65(SPRT)、偏好 ≥ 0.60"], kind="note", mono_lines=False)
    f.arrow("smp", "s3", label="test 146 前文", sides=("b", "l"), via=[(585, 575)])
    f.arrow("s3", "tab")
    f.arrow("tab", "lines", label="鎖門檻", sides=("r", "l"))
    f.arrow("s7", "tab", sides=("b", "t"), label="每個系統一列", dash="4,3", shift=-40)
    f.arrow("s7", "lines", sides=("b", "t"), label="對門檻", shift=40)
    f.arrow("s8", "lines", sides=("r", "r"), label="線 3", via=[(1680, 198), (1680, 575)], label_at=0.5, label_side="r")
    f.legend(40, 740, [("done", "已在真資料上跑過"), ("todo", "程式寫好、正式實驗未跑"), ("artifact", "檔案產物"),
                       ("private", "含原文,不進 git")], cols=4)
    return "00_overview.svg", f


# ------------------------------------------------------------------------------------------------ 01 data layer
def fig_data_layer():
    f = Fig(1560, 700, "① 資料層:LINE 匯出 → 訊息 → 回合 → 段落 → 樣本 → prompt(%s)" % DATE,
            "一則訊息只走一條路;三種文字檢視(text_a / text_b / ruler_text)在 Turn 上同時存在,尺只看 ruler_text,prompt 只看去識別化後的原文。"
            "符號類回合不當目標、不進尺,在前文裡換成佔位符。所有計數寫進 data_manifest.json,之後每個驅動程式載入時都會核對,常數一改就會被擋下。")
    Y = 80
    f.box("chat", 40, Y, 170, 110, "chathistory.txt", ["·4,967 則訊息", "·23 行系統訊息", "·兩人、2025-06 ~ 2026-09"],
          kind="artifact", private=True)
    f.box("parse", 250, Y, 230, 110, "parse_line.parse", ["日期行 YYYY.MM.DD 星期X", "HH:MM 說話者 文字 → Msg",
          "·說話者名以前綴比對(可含空白)", "·系統行丟棄並計數;續行接回上一則"], status="done")
    f.box("turns", 545, Y, 230, 110, "turns.build_turns", ["同人 · 同日 · 間隔 ≤ 30 分 → Turn",
          "text_a:佔位字元取代貼圖", "text_b:去掉純貼圖訊息", "ruler_text:顏文字 → 佔位字元"], status="done")
    f.box("flag", 840, Y, 240, 110, "flag_symbols.is_symbol_turn", ["在 ruler_text 上判斷",
          "·任一 ; { } ( ) = < > # [ ] _ \\ / * + & | % ^ $ @ `", "·網址、程式關鍵字", "·顏文字先換掉,不會被誤判"], status="done")
    f.box("seg", 1145, Y, 200, 110, "turns.assign_segments", ["間隔 ≤ 6 小時 → 段落", "·跨午夜仍同段", "·2,291 回合 / 205 段落"],
          status="done")
    f.arrow("chat", "parse"); f.arrow("parse", "turns", label="Msg"); f.arrow("turns", "flag", label="Turn"); f.arrow("flag", "seg", label="symbol")
    Y2 = 300
    f.box("split", 250, Y2, 300, 130, "split", ["時間切分:目標日期 ≥ 2026-07-20 → test",
          "dev_cut:train 最後 15%(整段落)→ dev", "date_blocks:K = 5 個日期區塊(OOF 校準用)",
          "nested_subsets:100 / 250 / 500 / 全部(巢狀、時間等距)"], status="done")
    f.box("samples", 640, Y2, 300, 130, "samples.build_samples", ["同段落內前 ≤ 12 回合 → T 的回合",
          "·目標:T、非符號類(可為純貼圖)", "·train 797(679 + dev 118)、test 146", "·stats():只輸出統計量"], status="done")
    f.box("prompt", 1020, Y2, 340, 130, "prompt.render / encode", ["純文字:一則一行,「我:」/「對方:」",
          "·前文的符號類回合 → [程式碼/算式]", "·去識別化:網址、信箱、電話、證號、地址", "·目標 = 整個回合 + EOS;loss 遮罩從第一個目標 token 起",
          "·超過 512 token 先丟最舊的前文"], status="done")
    f.arrow("seg", "split", sides=("b", "t"), via=[(1245, 260), (400, 260)], label="segments", label_at=0.5)
    f.arrow("split", "samples", label="split · block"); f.arrow("samples", "prompt", label="Sample")
    Y3 = 500
    f.box("man", 250, Y3, 300, 110, "data.manifest → data_manifest.json", ["·各期各人回合數、文字回合、符號回合",
          "·train / dev / test 樣本數、子集大小", "·切分常數(日期、30 分、6 小時、15%、12 回合)"], kind="artifact")
    f.box("stats", 640, Y3, 300, 110, "samples.stats → data_stats.json", ["·目標字數分位數、連發數分布", "·純貼圖目標占比(11–14%)",
          "·前文含佔位符占比(約一半)"], kind="artifact")
    f.box("consumers", 1020, Y3, 340, 110, "下游(都只經由 data.load_all)", ["stage1 · train_sft · make_pairs", "train_dpo · run_eval",
          "·載入時核對 manifest,不符就停"], kind="note")
    f.arrow("split", "man", sides=("b", "t")); f.arrow("samples", "stats", sides=("b", "t"))
    f.arrow("man", "consumers", label="載入時核對", dash="4,3", sides=("r", "l"), via=[(590, 555), (590, 640), (990, 640), (990, 555)], label_at=0.5)
    f.arrow("prompt", "consumers", sides=("b", "t"), label="SFT / 生成 / NLL 共用同一組裝器")
    f.box("synth", 40, Y2, 170, 130, "synth(測試用)", ["·假的兩人、假的訊息", "·計數已知,解析器必須重現", "·PERSONA_SYNTH=1 → runs/synth/"],
          kind="note", mono_lines=False)
    f.arrow("synth", "parse", sides=("t", "b"), dash="4,3", label="同格式", via=[(125, 250), (365, 250)])
    f.legend(40, 660, [("done", "已在真資料上跑過"), ("artifact", "檔案產物(只有數字)"), ("private", "含原文,不進 git")], cols=3)
    return "01_data_layer.svg", f


# ------------------------------------------------------------------------------------------------ 02 rulers
def fig_rulers():
    f = Fig(1560, 760, "② 尺:只在訓練期的非符號類回合上配適、驗證、校準(%s)" % DATE,
            "三族尺各自驗證過才拿去用:作者 LM 與距離走 validate 的 E1 / E3 / E4 / E5,特徵走鑑別度(≥ 0.60 才加權)。"
            "驗證尺用 k = 50 的袋、判定作者用 k = 10–25,兩件事分開;袋分數是加總,門檻一定按 k 在訓練期 OOF 校準。"
            "⑦ 的獨立分辨器不在這裡,它在測試期用真實對生成重訓。")
    f.box("pop", 40, 90, 220, 120, "訓練期回合", ["·非符號類、有文字:S 698 / T 707", "·(特徵帶另含純貼圖回合)",
          "·5 個日期區塊 → OOF"], kind="artifact")
    f.box("nb", 330, 70, 260, 100, "rulers.author_lm  NB", ["字元 1–3-gram、加一平滑", "margin = Σ log-likelihood ratio",
          "·M5;單回合 0.77 / 調門檻 0.81 / AUC 0.89"], status="done")
    f.box("dist", 330, 190, 260, 110, "rulers.profiles · ncd", ["M1 Burrows Δ · M2 Cosine Δ · M3 Min-max",
          "M4 JSD(300 個 n-gram)· M6 NCD(zlib)", "·z 參照 = 200 袋 × k 25"], status="done")
    f.box("feat", 330, 320, 260, 120, "rulers.features  16 個特徵", ["語氣詞 ×2 · 全形標點 · 句尾標點 · 貼圖率",
          "連發 · 長度 · ～ · 口頭禪 ×2 · 問句", "字母 · 數字 · 笑聲 · 第一人稱 · 顏文字"], status="done")
    f.arrow("pop", "nb", sides=("r", "l")); f.arrow("pop", "dist", sides=("r", "l")); f.arrow("pop", "feat", sides=("r", "l"))
    f.box("val", 680, 70, 330, 230, "rulers.validate.run → judge", [
          "E1 單回合(測試期):acc thr0 / 調門檻 / AUC",
          "E3 袋 k = 5 / 10 / 25 / 50:門檻按 k 在 OOF 校準",
          "   NB k=10 → 0.99–1.00(門檻 0 會假跌到 0.877)",
          "E4 同人兩半 < 跨人:P(cross > max within)",
          "   隨機 / 同時期配對,k = 25 / 50",
          "   k=50:Cosine Δ 0.97 · NCD 0.98 · Min-max 0.95",
          "E5 附和短句子集的 OOF AUC(≥ 0.90)",
          "·預登記門檻:E3 ≥ 0.95、E4 ≥ 0.95 / 0.90、E5 ≥ 0.90"], status="done")
    f.arrow("nb", "val", sides=("r", "l")); f.arrow("dist", "val", sides=("r", "l"))
    f.box("cal", 680, 330, 330, 110, "rulers.calibrate  Calibrator", ["等張回歸(PAV):OOF 單回合 LLR → P(T)",
          "·Cllr 1.77 → 0.60、Brier 0.176 → 0.130", "·上界被夾在 p = 0.999(log-odds 6.9)→ 封頂要用原始尺度"], status="done")
    f.arrow("val", "cal", sides=("b", "t"), label="oof_single_nb_margin")
    f.box("bands", 680, 470, 330, 120, "features.fit_bands · discrimination", ["容忍帶 = T 的 1000 個 k=50 袋的 5–95 百分位",
          "鑑別度 = T 袋 vs S 袋的 AUC(≥ 0.60 才加權)", "·最強:第一人稱 1.00、語氣詞 0.99、貼圖 0.99、問句 0.99",
          "·管制圖:各日期區塊的 |z| > 3"], status="done")
    f.arrow("feat", "bands", sides=("b", "l"), via=[(460, 530)])
    f.box("out1", 1100, 90, 220, 90, "ruler_validation.json / .md", ["·E1–E5 全部數字 + 判定", "·oof 分數(校準與配對共用)"], kind="artifact")
    f.box("out2", 1100, 330, 220, 70, "nb_calibration.json", ["·等張斷點與配適統計"], kind="artifact")
    f.box("out3", 1100, 480, 220, 90, "feature_bands.json", ["·bands · discrimination", "·weights · control_chart"], kind="artifact")
    f.arrow("val", "out1"); f.arrow("cal", "out2"); f.arrow("bands", "out3")
    f.box("use5", 1380, 90, 150, 110, "→ ⑤ 訓練尺", ["·NB(訓練期全體)", "·等張校準(OOF)", "·原始 LLR 封頂"], kind="note", mono_lines=False)
    f.box("use7", 1380, 440, 150, 150, "→ ⑦ 驗收尺", ["·特徵帶(線 2)", "·NCD 參照", "·抄襲檢查的訓練目標",
          "·不含:分辨器(⑦ 重訓)"], kind="note", mono_lines=False)
    f.arrow("out1", "use5", label="oof"); f.arrow("out2", "use5", sides=("r", "l"), via=[(1350, 365), (1350, 160)], label_at=0.9)
    f.arrow("out3", "use7")
    f.text(40, 650, "被推翻的說法(留著不刪):bge-m3 看得到否定(錯,cos(可以啊, 不行耶) = 0.77);全形標點 T 18% vs S 61%(無法重現,半形標點線索大半是程式碼)。",
           size=10.5, color=MUTED)
    f.legend(40, 700, [("done", "已在真資料上跑過(單次執行,未獨立重算)"), ("artifact", "reports/ 檔案(只有數字)")], cols=2)
    return "02_rulers.svg", f


# ------------------------------------------------------------------------------------------------ 03 controls + baselines
def fig_controls():
    f = Fig(1500, 620, "③ 對照組與基準線(測試期 146 個前文):先定座標、鎖門檻(%s)" % DATE,
            "四個對照組把每個指標的座標定出來:C-real 是地板(分辨器 AUC 應接近 0.5)、C-S 是天花板(真人 T vs 真人 S)、C-shuf 證明風格指標與內容指標分開、未訓練模型是訓練前的起點。"
            "基準線是沒有訓練的替代方案;看過 B0 與未訓練模型之後驗收線門檻鎖定,之後只能調嚴。輸出目錄非空時要 --overwrite。")
    f.box("test", 40, 90, 170, 110, "test 樣本", ["·146 個前文 → 真實回覆", "·日期 ≥ 2026-07-20", "·同一份給所有系統"], kind="artifact")
    f.box("ctrl", 280, 70, 330, 150, "對照組(不需要模型)", ["C-real   真實回覆(地板)", "C-shuf   真實回覆打亂到別的前文",
          "C-S      對方的測試期回合(從訊息建)", "·run_eval --system controls", "·已跑(10/13);#4 #11 #13 改了算法,要重跑"],
          status="done", tag="對照組 ✓")
    f.box("base", 280, 260, 330, 110, "未訓練模型(C-base)", ["run_eval --system base → untrained_<model>_s<seed>",
          "·Qwen3.5-4B Instruct,純文字 prompt", "·訓練前的起點;--model base 換 Base 底模"], status="todo")
    f.box("bl", 280, 410, 330, 150, "prompting 基準線(baselines)", ["B0 zero-shot  B1 many-shot 10/50/200/all",
          "B2 檢索式(bge-m3 找 20 個相近前文)", "B3 best-of-8(用訓練尺挑,待接)", "·chat template、thinking 關",
          "·prompt 依 token 預算分批;B1-all 可達 2–3 萬 token"], status="todo")
    for k in ("ctrl", "base", "bl"):
        f.arrow("test", k, sides=("r", "l"))
    f.box("ev", 700, 180, 300, 170, "evaluate.Evaluator", ["固定不變的部分:真實回覆、C-S 輸出、",
          "NCD 參照、特徵帶、抄襲檢查、離題門檻", "·離題門檻在訓練期回覆上配(#12)", "·地板 / 靈敏度 AUC 先算好",
          "evaluate(outputs) → 一列指標(見 ⑦)"], status="todo")
    f.arrow("ctrl", "ev", sides=("r", "l"), label="outputs"); f.arrow("base", "ev", sides=("r", "l")); f.arrow("bl", "ev", sides=("r", "l"))
    f.box("tab", 1090, 200, 180, 130, "eval_table.json", ["·一列一個系統", "·列名含底模與種子", "·runs/eval_<name>/", "  outputs.jsonl(私有)"],
          kind="artifact")
    f.arrow("ev", "tab")
    f.box("lock", 1090, 400, 360, 130, "鎖定驗收線門檻(只能調嚴)", ["·線 1 的 0.75 / 0.90 對照 C-real 地板與 C-S 天花板",
          "·線 2 的「≥ C-real − 1」等重跑後重看(12.0 / 15.3 的理由)", "·最佳 prompting = 線 1 配對 bootstrap 的對手",
          "·symbol_share 要不要門檻:待決定"], kind="note", mono_lines=False)
    f.arrow("tab", "lock", sides=("b", "t"))
    f.legend(40, 590, [("done", "已在真資料上跑過"), ("todo", "程式寫好、未跑"), ("artifact", "檔案產物")], cols=3)
    return "03_controls_baselines.svg", f


# ------------------------------------------------------------------------------------------------ 04 SFT
def fig_sft():
    f = Fig(1500, 580, "④ SFT:LoRA 微調,dev 選 epoch 後在 train+dev 重訓(%s)" % DATE,
            "資料量曲線:子集 100 / 250 / 500 / 全部(巢狀、時間等距)× 底模 Instruct / Base × 種子 0 / 1 / 2 = 24 次。"
            "階段 1 用 dev NLL 選 epoch 數;階段 2 重新載入底模、掛新的 LoRA,在 train+dev(子集則是同大小的重抽)上訓練同樣的 epoch 數,存成 adapter_best。這一段只用替身 torch 追過,本機先跑 --quick。")
    f.box("data", 40, 90, 230, 130, "data.load_all → 樣本", ["train 679 / dev 118(整段落切)", "子集 100 / 250 / 500 / all(巢狀)",
          "·核對 data_manifest.json", "·3 條金絲雀句加進訓練集"], status="done")
    f.box("enc", 330, 90, 260, 130, "prompt.encode(共用組裝器)", ["prompt ≤ 512 token(丟最舊前文)", "目標 = 整個回合 + EOS",
          "loss 遮罩:從第一個目標 token 起", "·micro 2 × accum 4 = 有效 batch 8"], status="done")
    f.arrow("data", "enc", label="Sample")
    f.box("p1", 680, 70, 330, 170, "階段 1  train_epochs(train − dev)", ["LoRA r8 α16 dropout 0.05,全部 linear 層",
          "lr 1e-4、warmup 3%、cosine;epoch 5 / 4 / 3 / 3", "每個 epoch 算 dev NLL,最低的存起來",
          "→ adapter_devsel(私有)", "·best_epoch、dev 曲線進 metrics.json"], status="todo")
    f.arrow("enc", "p1", sides=("r", "l"), label="items")
    f.box("p2", 680, 290, 330, 170, "階段 2  重訓(train + dev)", ["del + gc + empty_cache,重載底模、新 LoRA",
          "同樣 RNG 種子、同樣 schedule 形狀", "best_epoch 個 epoch,不再看 dev", "→ adapter_best(私有)",
          "·--no-retrain 跳過(目錄加 _noretrain)"], status="todo")
    f.arrow("p1", "p2", sides=("b", "t"), label="best_epoch")
    f.box("audit", 1100, 150, 300, 150, "金絲雀稽核(#17)", ["load_lora(adapter_best) 後再算", "3 條虛構句的 NLL/token",
          "vs 真實 dev 目標的 NLL/token", "·階段 2 後 dev 已是訓練資料,", "  另記 canary_devsel(階段 1)"], status="todo")
    f.arrow("p2", "audit", sides=("r", "l"))
    f.arrow("p1", "audit", sides=("r", "l"), dash="4,3", via=[(1050, 155), (1050, 225)])
    f.box("out", 1100, 360, 300, 130, "runs/sft_<model>_<size>_s<seed>[_attn][_epN]", ["adapter_best/  adapter_devsel/(私有)",
          "metrics.json:loss 曲線、dev NLL、", "  phase2、canary、seconds、vram", "·目錄非空要 --overwrite(#14)"], kind="artifact")
    f.arrow("p2", "out", sides=("r", "l"))
    f.box("open", 40, 300, 550, 160, "待決定 / 待確認", ["·子集 100 / 250 / 500 的階段 2:同大小重抽(現行)還是「子集 + 整個 dev」?",
          "·「約 1 s/step」只量了每步最後一個 micro-batch,metrics.json 的 seconds 下次重估",
          "·Qwen3.5-4B-Base 不在 HF 快取,Base vs Instruct 要先下載約 8 GB",
          "·重訓讓 adapter_best 看過 dev → DPO 選檢查點時改用 adapter_devsel(見 ⑥)"], kind="note", mono_lines=False)
    f.legend(40, 610, [("done", "已在真資料上跑過"), ("todo", "程式寫好、GPU 上未執行"), ("artifact", "檔案產物")], cols=3)
    return "04_sft.svg", f


# ------------------------------------------------------------------------------------------------ 05 pairs
def fig_pairs():
    f = Fig(1560, 760, "⑤ 候選 → 配對:尺在行為格裡挑好壞配對(make_pairs · pairs,%s)" % DATE,
            "訓練尺 = 訓練期全體上的 NB + OOF 等張校準,封頂改在原始 LLR 尺度(校準分數在 6.9 飽和,原本的封頂沒有作用)。"
            "一個前文 8 個候選;符號類、沒結束(64 token 內沒有 EOS)、抄襲、外漏、離題的候選不能當 chosen;chosen 從真實回覆所在的行為格找起。"
            "沒有合格 chosen 時用真實回覆(DITTO),最多占一半。§5.3 的兩個驗收只算非 DITTO 配對。")
    f.box("sft", 40, 90, 200, 90, "SFT 模型", ["runs/<sft>/adapter_best", "·Qwen3.5-4B + LoRA"], kind="model")
    f.box("ctx", 40, 220, 200, 90, "訓練前文", ["·train 679(不含 dev)", "·prompt 截到 512 token"], kind="artifact")
    f.box("gen", 300, 130, 280, 130, "generate.generate_replies", ["每前文 8 個候選,temperature 0.8、top-p 0.95",
          "·依 token 預算分批(12,000)", "·finished:64 token 內出現 EOS", "→ candidates.jsonl(私有)"], status="todo")
    f.arrow("sft", "gen", sides=("r", "l")); f.arrow("ctx", "gen", sides=("r", "l"))
    f.box("scorer", 300, 330, 280, 170, "pairs.Scorer(訓練尺)", ["NB 在訓練期全體(S 698 / T 707)上配",
          "校準:ruler_validation.json 的 OOF 分數", "cap_raw = T 的 OOF 原始 LLR 95 百分位", "score = 校準(min(raw, cap_raw))",
          "·排序用 min(raw, cap_raw);同分隨機", "·OOF 與訓練回合對不齊就停"], status="todo")
    f.box("rep", 40, 360, 200, 110, "reports/", ["ruler_validation.json", "(oof_single_nb_margin)", "feature_bands.json"], kind="artifact")
    f.arrow("rep", "scorer", sides=("r", "l"))
    f.box("gate", 680, 70, 300, 190, "約束(不能當 chosen)", ["符號類:is_symbol_turn(reply_text)", "沒結束:finished = False",
          "抄襲:與訓練目標有 ≥ 12 字相同子串", "外漏:OpenCC 轉換會改到的字(未裝 → 略)", "離題:bge-m3 cos < 訓練期 null 90 百分位",
          "·符號類與沒結束的連 rejected 也不當"], status="todo")
    f.arrow("gen", "gate", sides=("r", "l"), label="8 候選")
    f.box("cell", 680, 300, 300, 150, "行為格(cell)", ["長度級距 {≤5, 6–15, 16–40, >40 字}", "× 有無貼圖 × 連發 {1, 2, ≥3} × 有無語氣詞",
          "階梯:真實回覆的格 → 鄰格 → 任一格", "·記錄 chosen 的 level 0 / 1 / 2"], status="todo")
    f.arrow("scorer", "cell", sides=("r", "l"), label="score / rank")
    f.arrow("gate", "cell", sides=("b", "t"), label="ok")
    f.box("pick", 1080, 150, 300, 200, "build_pairs 的選法", ["chosen  = 合格且 score > +log 3 中", "          最低 level 裡 rank 最高者",
          "rejected = score < −log 3 中 rank 最低者", "沒有 rejected → 這個前文沒有配對", "沒有 chosen → 真實回覆(DITTO),",
          "          DITTO ≤ 50%,多的隨機丟"], status="todo")
    f.arrow("cell", "pick", sides=("r", "l"))
    f.box("out", 1080, 420, 300, 150, "runs/pairs_<sft>_s<seed>/", ["pairs.jsonl(私有):sid、chosen、rejected、",
          "  ditto、level、分數", "stats.json:配對數、DITTO 比例、cand_symbol、", "  cand_unfinished、cand_capped、封頂數值、",
          "  非 DITTO 的 chosen 分數與特徵 JSD"], kind="artifact")
    f.arrow("pick", "out", sides=("b", "t"))
    f.box("check", 680, 520, 300, 110, "§5.3 驗收(非 DITTO)", ["chosen 集合 vs 真實回覆的特徵 JSD < 0.05",
          "chosen 平均分數 − rejected 平均 > 2 log 3", "·含 DITTO 的值另存 *_all"], kind="note", mono_lines=False)
    f.arrow("out", "check", sides=("l", "r"), dash="4,3")
    f.text(40, 660, "待決定:排序改用封頂後的原始 LLR(超出審查的修法,可在 pairs.py 一行改回只用 score);finished 只認 EOS,比 §5.2 的「\\n對方: 也算停」嚴,跑過真資料後看 cand_unfinished。",
           size=10.5, color=MUTED)
    f.legend(40, 700, [("todo", "程式寫好,GPU 上未執行(配對邏輯本身在合成資料上端到端跑過)"), ("model", "模型 / 權重(私有)"), ("artifact", "檔案產物")], cols=3)
    return "05_pairs.svg", f


# ------------------------------------------------------------------------------------------------ 06 DPO
def fig_dpo():
    f = Fig(1560, 660, "⑥ DPO:自寫,參考 = 合併後的 SFT,dev 上用驗收尺選檢查點(train_dpo,%s)" % DATE,
            "loss = −log σ(β[(log πθ − log πref)(chosen) − (log πθ − log πref)(rejected)]),β 0.1、lr 5e-6、1 epoch、有效 batch 8。"
            "參考模型的 log-prob 先算好,VRAM 裡只有一個模型。第 0 步(= SFT)、每 10 步、最後一步在 60 個 dev 前文上各生成 1 則,用 D-AUC₁₀ / Band / symbol_share 選檢查點。"
            "SFT 的 adapter_best 在 dev 上訓練過時,先從 adapter_devsel 選步數,再從 adapter_best 重跑同樣步數。這一段只用替身 torch 追過。")
    f.box("pairs", 40, 90, 220, 90, "pairs.jsonl(私有)", ["·sid、chosen、rejected", "·dev 前文的配對會被拒絕"], kind="artifact")
    f.box("sft", 40, 220, 220, 110, "SFT run", ["adapter_devsel(dev 保留)", "adapter_best(train+dev)", "·metrics.json 的 phase2 決定走法"], kind="model")
    f.box("ref", 340, 90, 300, 130, "參考模型 πref", ["底模 + SFT adapter 合併、凍結", "ref_logps:每對的 chosen / rejected", "  目標 token log-prob 先算好",
          "·之後只留 policy 在 VRAM"], status="todo")
    f.arrow("pairs", "ref", sides=("r", "l"), label="encode"); f.arrow("sft", "ref", sides=("r", "l"))
    f.box("pol", 340, 270, 300, 130, "policy πθ = 合併權重 + 新 LoRA", ["seq_logp:目標 token 的 log-prob 加總",
          "dpo_loss(β 0.1),梯度裁剪 1.0", "AdamW lr 5e-6,accum 8 → ~85 步 / epoch", "·step 0 時 LoRA B = 0,πθ = πref"], status="todo")
    f.arrow("ref", "pol", sides=("b", "t"), label="refs")
    f.box("ev", 740, 150, 340, 210, "evaluate_dev(第 0 步、每 10 步、最後一步)", ["60 個 dev 前文(run 種子抽),各生成 1 則",
          "·eval 模式、use_cache 開,RNG 狀態還原", "dev_metrics vs 同前文的真實回覆:", "  D-AUC₁₀(分辨器)、Band、特徵 JSD、symbol_share",
          "→ dev_outputs.jsonl(私有)、eval_history", "SELECTION:symbol_share 超過第 0 步 + 0.05 排後;",
          "  再取 D-AUC₁₀ 最低;同分取 Band 高、較早的步"], status="todo")
    f.arrow("pol", "ev", sides=("r", "l"), label="每 10 步")
    f.box("out", 1180, 120, 330, 170, "runs/dpo_<sft>_<pairs>_s<seed>/", ["adapter_best(選定的步;可能是第 0 步)", "adapter_last(最後一步)",
          "adapter_devsel(兩階段時第一段選定的)", "metrics.json:loss / reward_acc / margin 曲線、", "  eval_history、best_step、selection"], kind="artifact")
    f.arrow("ev", "out", sides=("r", "l"), label="save_pretrained")
    f.box("two", 740, 420, 340, 150, "兩階段(SFT 有重訓時)", ["① 參考 = adapter_devsel:帶 dev 評估,選出步數 k",
          "② 參考 = adapter_best:重跑 k 步,不評估", "   第 k 步 → adapter_best,最後 → adapter_last", "·時間約加倍;dev 對選檢查點的模型保留"],
          kind="note", mono_lines=False)
    f.arrow("sft", "two", sides=("b", "l"), via=[(150, 495)], dash="4,3", label="phase2.n_dev_in_train > 0", label_at=0.35)
    f.box("open", 1180, 360, 330, 210, "與 §5.4 不同,待決定", ["·§5.4:每 100 步看 64 則,「D-AUC 不再下降就停」;",
          "  實作:跑完 1 epoch,取評估中最好的(偏樂觀)", "·在 60 個前文上反覆取最小值,要不要改成", "  預先登記的停止規則?", "·消融:無格、無上限、無 DITTO(目錄名區分)",
          "·run_eval --dpo 疊在 SFT 的 adapter_best 上"], kind="note", mono_lines=False)
    f.arrow("out", "open", sides=("b", "t"), dash="4,3")
    f.legend(40, 670, [("todo", "程式寫好,GPU 上未執行"), ("model", "模型 / 權重(私有)"), ("artifact", "檔案產物")], cols=3)
    return "06_dpo.svg", f


# ------------------------------------------------------------------------------------------------ 07 acceptance
def fig_acceptance():
    f = Fig(1560, 800, "⑦ 驗收:與訓練尺不同的一組尺、四個對照組、三條預先登記的線(evaluate · run_eval · report,%s)" % DATE,
            "每個系統對 146 個測試前文各生成 1 則;符號類的生成回覆不進任何尺(真實目標在資料層就排除了符號類),比例另報 symbol_share。"
            "分辨器在測試期用「真實 vs 生成」重訓(fold 按字串分),和訓練尺沒有共用參數;特徵帶與 NCD 參照來自 stage 1。"
            "線 1 與線 2 由 report.acceptance 判,線 3 由盲判(⑧);顯著性用以測試前文為單位的配對 bootstrap(2,000 次)。")
    f.box("sys", 40, 90, 200, 150, "系統輸出", ["[{sid, msgs}] × 146", "·SFT / SFT+DPO / 基準線", "·對照組 C-real C-shuf C-S",
          "·未訓練模型", "→ runs/eval_<name>/outputs.jsonl"], kind="artifact", private=True)
    f.box("sym", 300, 110, 200, 110, "符號類過濾", ["is_symbol_turn(reply_text)", "→ symbol_share", "·其餘進尺;空回覆 = 純貼圖"], status="todo")
    f.arrow("sys", "sym", sides=("r", "l"))
    rulers = [
        ("disc", 570, 70, "獨立分辨器 D-AUC₁ / D-AUC₁₀", ["NB 真實 vs 生成,測試期重訓", "fold 按字串(seeded hash)", "·地板 真實 vs 真實 ≈ 0.5", "·靈敏度 真實 T vs 真實 S(0.76)"]),
        ("ncd", 570, 200, "NCD-Δ", ["k=25 袋的對方相似度", "− 真實回覆的值(訓練期參照)"]),
        ("band", 570, 290, "Band(特徵帶)", ["k=50 袋,16 個特徵在 T 的 5–95 帶內的個數", "·帶含純貼圖回合(#4),要重算"]),
        ("jsd", 570, 380, "特徵 JSD", ["16 個特徵的逐則直方圖 vs 真實"]),
        ("misc", 570, 450, "OnTopic · Copy · Leak · Div", ["bge-m3 cos ≥ 訓練期門檻;≥12 字抄襲;OpenCC 外漏(未裝);", "distinct-2、Hill-1 稀疏化到 100 則,另報與 C-real 同 n 的比值"]),
        ("nll", 570, 540, "NLL · MMI(要模型)", ["真實測試回覆在該模型下的 NLL/token(#1)", "MMI:log p(y|c) − log p(y),用未訓練模型"]),
    ]
    for key, x, y, title, lines in rulers:
        h = 24 + 15 * len(lines) + 10
        f.box(key, x, y, 420, h, title, lines, status="todo")
        f.arrow("sym", key, sides=("r", "l"), via=[(535, 165), (535, y + h / 2)])
    f.box("tab", 1080, 200, 180, 170, "eval_table.json", ["·一列一個系統", "·symbol_share、d_auc1、", "  d_auc10、ncd_delta、", "  band、jsd、copy、leak、",
          "  div_*、*_ratio、nll、mmi"], kind="artifact")
    for key in ("disc", "ncd", "band", "jsd", "misc", "nll"):
        f.arrow(key, "tab", sides=("r", "l"), via=[(1040, f.boxes[key][1] + f.boxes[key][3] / 2), (1040, 285)])
    f.box("boot", 1080, 430, 400, 110, "report:配對 bootstrap(2,000 次)", ["以測試前文為單位重抽,兩個系統同一組索引",
          "D-AUC 快速版:OOF 分數算一次,不重訓(文件註明)", "--compare a b → diff、95% CI、P(diff > 0)"], status="todo")
    f.arrow("tab", "boot", sides=("b", "t"))
    f.box("lines", 1080, 590, 400, 150, "report.acceptance(config.LINES)", ["線 1  D-AUC₁ ≤ 0.75 且 D-AUC₁₀ ≤ 0.90,",
          "      且 D-AUC₁ 的 CI 低於最佳 prompting", "線 2  Band ≥ C-real − 1;distinct-2 與 Hill-1 ≥ 0.8 × C-real(同 n);",
          "      Copy ≤ 2%;Leak ≤ C-real", "線 3  由 ⑧ 盲判(report.lines_from_human)", "·三值判定:True / False / None(算不出)"],
          kind="note", mono_lines=False)
    f.arrow("boot", "lines", sides=("b", "t"))
    f.text(40, 300, "對照組定座標:", size=11, weight="bold", color=MUTED)
    for i, t in enumerate(["C-real:地板(D-AUC ≈ 0.5、Band 的參照)", "C-shuf:風格對、內容錯 → 風格指標 = C-real、", "        OnTopic 掉到 0.10",
                           "C-S:天花板(真人 S vs 真人 T:0.76)", "未訓練模型:訓練前的起點;鎖門檻後只能調嚴"]):
        f.text(40, 320 + 16 * i, t, size=10.5, color=MUTED)
    f.text(40, 440, "待決定:symbol_share 的門檻;線 1 的「顯著」用 D-AUC₁;", size=10.5, color=MUTED)
    f.text(40, 456, "線 2 要 distinct-2 與 Hill-1 都過;12.0 / 15.3 的理由重跑後重看。", size=10.5, color=MUTED)
    f.legend(40, 750, [("todo", "程式寫好;對照組 10/13 跑過,算法改了要重跑"), ("artifact", "檔案產物"), ("private", "含原文")], cols=3)
    return "07_acceptance.svg", f


# ------------------------------------------------------------------------------------------------ 08 human eval
def fig_human():
    f = Fig(1500, 580, "⑧ 本人盲判:題型 A「哪一個是你寫的?」、題型 B「哪一個比較像你?」,Wald SPRT 停止(human_eval,%s)" % DATE,
            "一個測試前文只給一個系統(題型 A、B 都是),所以同一則真實回覆或基準回覆不會在兩題出現;題型 A 裡其他題的真實回覆若出現在前文中,顯示成「我: [略]」。"
            "每個系統各自一條 SPRT 序列:p = 0.55 對 0.75(題型 A,線在 0.65)、0.50 對 0.70(題型 B,線在 0.60),α = β = 0.05,最多 100 題;碰到界線或 100 題時停,用停止點的點估計判線 3。"
            "題目用完還沒停 → 線 3 記為未定(more_items_needed)。題組只能寫在 runs/ 底下。")
    f.box("outs", 40, 100, 220, 120, "runs/eval_<name>/outputs.jsonl", ["·要比的系統(1 個或多個)", "·題型 B 的基準 = 最佳 prompting", "·test 樣本(真實回覆、前文)"],
          kind="artifact", private=True)
    f.box("make", 340, 80, 360, 170, "human_eval make", ["選前文:--priority 的優先(尺意見分歧的),其餘隨機",
          "分配:permuted blocks,一個前文一個系統", "過濾:缺、空、符號類的回覆不出題(計數)", "顯示:依時間排;去識別化;左右隨機",
          "→ items.jsonl(給本人看)、key.json、stats.json"], status="todo")
    f.arrow("outs", "make", sides=("r", "l"))
    f.box("judge", 800, 100, 220, 130, "本人作答", ["·每題選 left / right", "·answers.json(本機)", "·時間與答案本機記錄"], kind="note", mono_lines=False)
    f.arrow("make", "judge", sides=("r", "l"), label="items.jsonl(私有)")
    f.box("score", 800, 300, 360, 170, "human_eval score", ["每個系統一條序列(依題號)", "sprt_stop:Λ = k·ln(p₁/p₀) + (n−k)·ln((1−p₁)/(1−p₀))",
          "  Λ ≥ ln 19 → above;Λ ≤ −ln 19 → below", "  到 100 題 → undecided(點估計 + Clopper–Pearson)", "合併數字只報不判"], status="todo")
    f.arrow("judge", "score", sides=("b", "t"), label="answers.json")
    f.box("line3", 1200, 300, 260, 170, "線 3(report.lines_from_human)", ["題型 A:停止點正確率 ≤ 0.65", "題型 B:停止點偏好 ≥ 0.60",
          "·題目用完未停 → None", "·舊規則(逐題看 95% CI)留作紀錄:", "  真實 0.65 時誤判「低於」12%"], kind="note", mono_lines=False)
    f.arrow("score", "line3", sides=("r", "l"))
    f.box("open", 340, 320, 360, 150, "待決定", ["·無差異區要不要調嚴:現行 0.55 / 0.75 時,真實 0.70 的",
          "  系統通過線 3 的機率 0.20;0.60 / 0.70 → 0.16(約 82 題)", "·146 個前文分給 2 個以上系統時到不了 100 題:",
          "  一組題只放一個系統?或分多組?", "·偏好測試的對手(最佳 prompting)要先跑出來"], kind="note", mono_lines=False)
    f.legend(40, 590, [("todo", "程式寫好、在假輸出上測過;真資料未跑"), ("artifact", "檔案產物"), ("private", "含原文,不進 git")], cols=3)
    return "08_human_eval.svg", f


FIGS = [fig_overview, fig_data_layer, fig_rulers, fig_controls, fig_sft, fig_pairs, fig_dpo, fig_acceptance, fig_human, fig_dependencies]


def main():
    for fn in FIGS:
        name, fig = fn()
        (OUT / name).write_text(fig.render(), encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
