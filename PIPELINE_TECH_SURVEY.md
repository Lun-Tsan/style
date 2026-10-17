# 語體仿效 Pipeline:既有技術調查

**日期**:2026-10-10(2026-10-13 補實驗 4 與 ① 定案)
**研究問題**:用單一真人的少量 LINE 聊天紀錄(台灣口語繁中、短句、含貼圖),以語言學量化標準當訓練訊號與驗收標準,LoRA 微調本機模型(Qwen3.5-4B),讓回覆風格接近本人。多人泛化留到之後。
**本文件目的**:把 pipeline 每一步需要的技術對應到既有文獻;**新方法或目前查不到解法的地方留空(⬜)**。

## 圖例

| 符號 | 意思 |
|---|---|
| ✅ | 既有技術,可直接用 |
| 🟡 | 既有技術,需調整才能用 |
| ⬜ | **留空**:新方法,或目前查不到既有解法 |
| ⏳ | 尚待確定(屬於設計決定,不是文獻問題) |

## 查證說明

- 本文件引用的文獻都回到原始來源(出版社頁、arXiv、ACL Anthology、官方 repo)確認過。「部分成立」的條目已改寫成修正後的說法。
- 2026-10-10 這一輪:23 條新增文獻逐條查證(15 條成立、6 條修正細節、1 條原文無法存取,見各條附註);8 個「是否為空白」的問題經過系統搜尋。DIAL、Flipping the Dialogue 兩篇我另外讀了全文確認關鍵細節;Turing-RL、Sawant 確認了摘要。
- 很多 2026 年的條目還是預印本,發表處以 arXiv 或會議頁面標示為準。
- 「本專案實測」是我們自己對 `chathistory.txt` 做的實驗,不是文獻。

---

## 總覽

| 階段 | 既有技術 | 需調整 | 留空 |
|---|---|---|---|
| ① 前處理 | 切對話、合併連發、去重原則 | 貼圖標記、去識別化、過濾規則 | 中文短訊的品質過濾規則(文獻無;**本專案實驗 4 驗證了符號類過濾規則**) |
| ② 樣本與切分 | 「前文 → 本人回覆」格式、只算本人 loss、資料量曲線 | 按時間/整段對話切分 | — |
| ③ 風格尺 | Delta 家族、字元 n-gram、分類器兩樣本檢定、JS 散度 | 中文社群特徵集、短樣本處理、校準 | 繁中短訊風格尺 + 人判驗證;逐特徵雜訊底線正規化;全形/半形當作者特徵 |
| ④ 基準線 | zero-shot、many-shot ICL、檢索範例 | — | 單一個人:many-shot vs LoRA |
| ⑤ SFT LoRA | LoRA、QLoRA、每人一個 LoRA、超參數參考 | 原始預訓練版 vs instruct | 單一個人:原始預訓練版 vs instruct |
| ⑥ 偏好訓練 | DPO、DITTO、分辨器排序配對(DIAL)、best-of-N、過度最佳化原則 | 防平均化、防作弊 | 用可解釋文體特徵為單一真人排序配對 |
| ⑦ 評估 | 兩樣本檢定、熟人盲測、多樣性、記憶/抄襲、顯著性檢定 | 台灣用語外漏檢查 | 繁中個人聊天模仿的評測基準 |
| ⑧ 倫理 | 同意、外洩風險、揭露規範 | 台灣個資辨識 | — |
| ⑨ 之後 | 台灣底模、貼圖挑選、多人個人化 | — | 單一個人的貼圖時機 |

---

## ① 前處理

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 解析 LINE 匯出檔 | 自寫解析器(`analyze_chat.py`,已完成並驗證) | 本專案 | ✅ | — |
| 切成對話段落 | 訊息間隔超過門檻就切開:IMPersona 6 小時、impersona-env 48 小時、robo-boys 4 小時 | IMPersona 程式碼;impersona-env;robo-boys 部落格 | 🟡 | ⏳ 門檻(預設 6 小時,另測 3 小時) |
| 合併連發訊息 | IMPersona 用 `<|msg|>` 分隔;WeClone 合併 2 分鐘內的訊息、用逗號接(會抹掉連發邊界) | IMPersona 程式碼;WeClone | 🟡 | 用分隔符保留連發;⏳ 分隔符、合併時間門檻 |
| 貼圖、圖片 | WeClone 轉成 emoji 或略過動態貼圖;MOD、PEGS 在生成中插入 `[tag]`/`[IMG]` 標記;Li & Zhang 2025 把 `<picture>` 等佔位符當文體特徵 | WeClone;MOD;PEGS;Li & Zhang 2025 | 🟡 | 保留 `[貼圖]` `[圖片]` 標記。LINE 匯出沒有貼圖編號,只能學「何時發」 |
| 去識別化 | Presidio(樣式比對 + NER)。**沒有台灣或中國身分證辨識器**;研究顯示去識別化只能降低、不能杜絕外洩 | Presidio;Lukas et al. 2023 | 🟡 | 自訂台灣電話、身分證、地址規則;說話者改成「我/對方」⏳ |
| 去重 | 去重能大幅減少模型背出訓練文字;重複 10 次的序列被生成的頻率約高 1000 倍 | Lee et al. 2022;Kandpal et al. 2022 | 🟡 | 只對長訊息去重;「好啊」這類高頻短句是風格,不能去重(我們的推論) |
| 品質過濾 | IMPersona 的規則:回覆 ≤ 2 個字(以空白切)就刪、刪純 emoji、刪單則 > 50 字、英文停用詞。作者自己的資料因此刪掉 76%(63,574 → 15,044) | IMPersona 程式碼 | ❌ 不沿用 | 中文沒有空白,這套規則幾乎會刪光所有中文回覆 |
| **中文短訊的品質過濾規則** | 文獻無。本專案:符號類回合過濾(寬鬆規則 `CODE_RE`:半形符號 / 網址 / 程式關鍵字),實驗 4 驗證 | 本專案實測(實驗 4) | 🟡 | 只處理符號類訊息,不刪短回覆與貼圖;目標排除、前文改成 `[程式碼/算式]` 佔位符;顏文字白名單與半形標點的保留方式待驗證 |

## ② 樣本建立與切分

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 樣本格式 | 「這段對話到目前為止 → 本人的下一個回合」 | IMPersona;WeClone;NTT | ✅ | ⏳ 前文長度上限、要不要放時間戳記 |
| 只對本人回覆算 loss | 遮掉 prompt、只算目標說話者的回合 | CoSER;LLaMA-Factory 預設 `train_on_prompt=False` | ✅ | 寫測試確認第一個 token 有算到(IMPersona 的 `train.py` 少算一格) |
| 訓練與推論同格式 | — | IMPersona 程式碼是反例:Together 訓練資料沒有系統提示,推論卻加了長系統提示 | 🟡 | 用同一支程式組 prompt |
| 關閉思考模式 | 研究發現 CoT 可能降低角色扮演表現;WeClone 預設關閉 | Feng et al. 2025(arXiv 2502.16940);WeClone | ✅ | Qwen3.5 `enable_thinking=False` |
| 測試集切法 | IMPersona 的驗證集**不是按時間切**(可能和訓練樣本同一段對話) | IMPersona 程式碼 | 🟡 | 按時間切(2026-07-20 起),以整段對話為單位 |
| 資料量曲線 | IMPersona:B25 到 B8k,依時間間隔挑樣本,小子集複製 2–4 份;Panza:25 封信就達到約 90% 的最大 BLEU | IMPersona 程式碼;Panza | ✅ | 100 / 250 / 500 / 全部,巢狀;⏳ 抽法 |

**本專案實測(Hung LT)**:訓練期回覆樣本 833 則(758 則有文字),測試期 148 則(130 則有文字);以「同一人、同一天、間隔 ≤ 30 分鐘」為一個回合。

## ③ 風格尺

### R1 語言學特徵

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 特徵選擇的理論框架 | 多維度分析:一起出現的語言特徵構成語域變異的維度 | Biber 1988 | ✅ | 當作選特徵的依據 |
| 網路溝通的語言分析框架 | CMDA:結構、意義、互動、社會行為四個層次 | Herring 2004 | ✅ | 同上 |
| 中文社群媒體的作者特徵 | 115 位作者(含即時通訊):代名詞、感嘆詞等關鍵詞、情緒、詞彙句法特徵 | Zhang 2024 | 🟡 | 簡體;我們改成台灣口語 |
| **最接近的現成特徵集** | 微博人類 vs AI 留言:34 個可解釋特徵,含語氣詞、標點、emoji、佔位符;人類的標點分成「不用、正常、過量」三群 | Li & Zhang 2025(CCL) | 🟡 | 用途是偵測 AI,不是個人作者;簡體 |
| 中英文網路訊息作者辨識 | 詞彙、句法、結構、內容特徵,準確率 70–95% | Zheng et al. 2006 | 🟡 | 新聞群組,非即時聊天 |
| 即時通訊的回合式特徵 | 以回合計算的文體特徵,77 位使用者 | Cristani et al. 2012 | 🟡 | 語言未註明 |
| emoji 當作者標記 | emoji 種類較穩定;**emoji 頻率會配合對話對象改變** | Marko 2022 | 🟡 | 我們只有一位對象,要寫進限制 |
| 人類 vs LLM 的語言特徵差異 | 英文新聞:人類句長和詞彙變化較大 | Muñoz-Ortiz et al. 2024 | 🟡 | 群體層級、英文 |
| 分布型特徵的比較 | Jensen–Shannon 散度 | Lin 1991(書目已確認;內容未能直接存取原文) | ✅ | 用在長度、連發則數 |
| 台灣用語外漏 | 大陸系模型用繁中提問時較容易用錯地區用語;OpenCC 有 s2tw(台灣字)、s2twp(台灣詞) | SC-TC-Bench;OpenCC | 🟡 | 字典不全,要人工抽查;⏳ 需安裝 OpenCC |
| **全形/半形標點當作者特徵** | | 搜尋無結果 | ⬜ | **留空**(本專案實測:含程式碼時字元層級全形占比 T 0.075 / S 0.246;非符號回合 T 0.32 / S 0.17 但標點極少,鑑別度 0.68;10/03 的 18% / 61% 無法重現) |
| **以「真人自身變異」正規化的逐特徵距離** | 最接近:Sawant 2026 用「同作者上限 0.756、跨作者下限 0.626」校準 LUAR 分數,但只在 embedding 層級、英文部落格 | Sawant 2026 | ⬜ | **留空**:逐特徵、以本人真實回覆互比當雜訊底線,沒有找到前例 |

### R2 頻率輪廓距離

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| Delta 距離 | 最高頻詞的 z 分數向量之間的曼哈頓距離(取平均) | Burrows 2002 | ✅ | — |
| Cosine Delta | 改用 z 分數向量的夾角;後續評估發現優於原始 Delta | Smith & Aldridge 2011(JQL 18(1): 63–88;改良效果由 Evert et al. 2015 等後續研究確認) | ✅ | — |
| 為什麼 Cosine 較好 | 關鍵在向量正規化;作者訊號在於各高頻詞偏離的「輪廓」,不在偏離的大小 | Evert et al. 2017(「最穩健」的說法出自同團隊 2015 年的研究) | ✅ | — |
| 中文不斷詞 | 字元 n-gram 語言模型:8 位武俠作家,94% | Peng et al. 2003 | 🟡 | 每位作者有 50–190 萬字,我們每人只有約 2 萬字 |
| 字元 n-gram 的效果 | 綜述:字元 n-gram 與語言無關、效果很好 | Stamatatos 2009 | ✅ | — |
| 短樣本問題 | 穩定判別約需 2,500 字(拉丁散文)到 5,000 字(多數小說);隨機抽字比連續段落好 | Eder 2015 | 🟡 | **我們把 50 則合成一組也只有幾百字,遠低於門檻 → R2 可能不穩,先當次要指標** ⏳ |
| 單則推文 | 50 位候選作者時,單則推文的作者辨識準確率 49.5–69.7% | Schwartz et al. 2013 | ✅ | 佐證單則短訊訊號弱 |
| 極短文本 | 1–10 個 token 的文本最難辨識作者(檢索 Success@5:0.187 vs 長文 0.543) | AuthBench(2026 預印本) | ✅ | 同上 |

### 正式風格距離總表(2026-10-10 補充,15 條文獻逐條回原文查證)

| 類別 | 距離 | 定義 | 對我們的適用性 | 出處 |
|---|---|---|---|---|
| 高頻輪廓 | Burrows' Delta | Δ = (1/n) Σ \|z_i(A) − z_i(B)\|,即 z 分數向量的曼哈頓距離 | 需要多作者參照語料估 z 分數;我們只有兩人 → 用 bootstrap 袋代替,要註明 | Burrows 2002 |
| 高頻輪廓 | Cosine Delta | 1 − cos(z(A), z(B)) | 最穩健;關鍵是向量正規化 | Smith & Aldridge 2011;Evert et al. 2017 |
| 高頻輪廓 | Linear / Quadratic / Rotated Delta | Argamon 把 Delta 解釋成 Laplace 分布下的似然排序;Linear = 中位數與平均絕對離差的曼哈頓距離;Quadratic = z 分數上的歐氏距離(各維獨立高斯);**Rotated = 用共變異數矩陣特徵向量的版本**(Mahalanobis 類) | 樣本少時共變異數估不穩,Rotated 不適用 | Argamon 2008(摘要;細節經 Stein & Argamon 2006、Jannidis et al. 2015 確認) |
| 高頻輪廓 | Eder's Delta | Delta 乘上排名權重 w_i = (n − i + 1)/n,降低低頻特徵影響(stylo 程式實際用 (n − i + 2)/n) | 為高度屈折語言設計;**沒有正式論文定義它**,引 stylo 套件 | Eder, Rybicki & Kestemont 2016(stylo);Jannidis et al. 2015 |
| 高頻輪廓 | Min–max(Ružička) | 1 − Σ min(f_i(A), f_i(B)) / Σ max(f_i(A), f_i(B)) | 不需參照語料;在冒名者框架下「出人意料地強」,結果穩定 | Kestemont et al. 2016 |
| 高頻輪廓 | 卡方(CBDF) | 以最高頻 500 詞計算的 chi-by-degrees-of-freedom | 語料比較最佳;但相似度只能相對於語料內部同質性解讀;大語料時卡方檢定失效 | Kilgarriff 2001 |
| 高頻輪廓 | Delta 參數 | 更長的高頻詞表(最多 800)、移除人稱代名詞、刪掉單一文本佔 >70% 出現次數的詞,可大幅提高準確率 | 中文版要重做參數掃描 | Hoover 2004 |
| 資訊理論 | JSD | ½KL(P‖M) + ½KL(Q‖M),log₂ 時在 [0, 1] | 分布型特徵 | Lin 1991 |
| 資訊理論 | 作者語言模型 LLR | log P(x\|A) − log P(x\|B),字元 n-gram LM | 唯一能對單則短訊給分的正式距離(= NB) | Peng et al. 2003;Juola 2008(交叉熵、KL) |
| 資訊理論 | NCD | [C(xy) − min(C(x), C(y))] / max(C(x), C(y)) | 不用設計特徵、保留全形半形與 emoji;短文本訊號弱 | Cilibrasi & Vitányi 2005;Benedetto et al. 2002 |
| 區辨詞彙 | Zeta / Craig's Zeta | Burrows 2007:以「含該詞的段落比例」區分中頻(Zeta)與低頻(Iota)詞;Craig's Zeta(Hoover 2010 描述)= p_A(含) + p_B(不含),範圍 0–2;Schöch et al. 2018 用差值形式 p_A − p_B,範圍 −1 到 1 | 直接對應口頭禪;可解釋 | Burrows 2007;Craig & Kinney 2009(書中公式未能直接查閱);Schöch et al. 2018 |
| 豐富度 | Yule's K | K = 10⁴ (Σ i²V_i − N) / N² | 設計目標是與長度無關,但實證上並不完全(Tweedie & Baayen 1998);只能在語料層級算 | Yule 1944(經 Stamatatos 2009、quanteda 確認) |
| 豐富度 | MTLD | 維持 TTR ≥ 0.72 的連續詞串平均長度 | 受測指標中唯一不隨長度變動的 | McCarthy & Jarvis 2010(0.72 經 koRpus 確認) |
| 語域 | 正式度 F-score | F = (名詞 + 形容詞 + 介系詞 + 冠詞 − 代名詞 − 動詞 − 副詞 − 感嘆詞 + 100)/2,以詞類百分比計 | 英文詞類設計,對口語聊天意義不大 | Heylighen & Dewaele 1999 |
| 評判框架 | General Imposters | 重複抽特徵子集:目標作者是否比每個冒名者都更接近待測文本;比例 = 分數,門檻在訓練資料上調 | 需要冒名者群;可用原始模型輸出或之後的 PTT 資料 | Koppel & Winter 2014;Seidman 2013;Kestemont et al. 2016 |
| 評判框架 | STEL | 四元組測試:兩組「只差風格的改寫對」,模型要依風格配對;因為是改寫,內容大致固定 | 驗證尺量的是風格不是內容;需要改寫對 | Wegmann & Nguyen 2021(EMNLP) |

**本專案實測(2026-10-10,上表所有距離在 `chathistory.txt` 上實作並經獨立重算;完整數據見下方「實驗紀錄」實驗 3 與 `FORMAL_DISTANCE_EXPERIMENT.md`)**:
- 單一回合(中位數 16 字)沒有任何正式距離可靠:NB 作者 LM 0.73(調門檻 0.79);Min-max、mean-Z Zeta 約 0.70;配對比較 NB 0.89 vs 最佳正式距離 NCD 0.78。
- 分袋:NB 的加總分數用「0 當門檻」會嚴重偏向 T,**必須用訓練期袋校準門檻**;校準後 k=10 約 0.94–0.96(隨校準抽樣變動)、k=25 達 0.99;正式距離要 k=50 才 ≥ 0.95。NCD 是跨時期最穩的距離。
- 尺驗證「同人兩半 < 跨人」在 k=50 **六種距離全部沒過**(0.37–0.70);原因是程式碼/數學式佔訓練字元 60–70%,時期差異大於人際差異;拿掉後通過率 0.86–0.97。
- 決定:逐則與分袋主尺 = NB 作者 LM LLR(分袋門檻校準);NCD 當無特徵交叉檢查;Cosine Delta 暫列;丟 Burrows Delta、JSD、raw cosine;Zeta 只留標記詞清單。**前處理必須拿掉或單獨標記程式碼/數學式訊息。**

### R3 分辨器

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 兩樣本檢定 | 訓練分類器分兩組樣本,同分布時準確率接近 0.5 | Lopez-Paz & Oquab 2017 | ✅ | 主驗收指標 |
| 分辨器的陷阱 | 分辨器可能抓到長度、取樣方式這類機器痕跡,而不是風格 | Kannan & Vinyals 2017;Ippolito et al. 2020 | ✅ | 控制長度與解碼參數,檢查它靠哪些特徵分辨 |
| 現成作者/風格模型 | LUAR、Wegmann、StyleDistance(英文);mStyleDistance(含合成簡中);多語作者模型的中文來自維基百科 | 各原文 | 🟡 | 只能當參考 |
| 語意 embedding | 語意 embedding 在風格任務上表現差 | STEB(2026 預印本);Wegmann 2022 | ✅ 排除 | 本專案實測:bge-m3 判斷「同意思、不同人」只差 0.018 |
| LLM 風格裁判 | Claude 打 1–7 分,參考 60 組真實連發訊息,評長度、拆則、標點、俚語、口頭禪、語氣、語域,不看內容;口頭禪用太多要扣分 | impersona-env `style_judge.py`(MIT 授權) | 🟡 | 英文、雲端;只當輔助 |
| LLM 裁判的偏誤 | 位置偏誤、冗長偏誤、偏好自家模型、同家族資料外洩偏誤 | Zheng et al. 2023;Panickssery et al. 2024;Li et al. 2025(preference leakage) | ✅ | 不用 Qwen 當裁判 |

**本專案實測**:字元 n-gram Naive Bayes 分辨你和學生,配對準確率 0.88(全部)、0.85(短的泛用回覆)、0.91(同類應答);只看長度 0.46。bge-m3 在向量上加分類器可追平(0.86/0.85),直接比相似度較差(0.78/0.73)。

### 尺的驗證

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 自動指標 vs 人的判斷 | 不同 LLM 和任務之間,和人的一致度差異很大,用前要先驗證 | JUDGE-BENCH | ✅ | — |
| 作者向量 vs 本人自評 | LUAR 相似度與參與者自評的「像不像我」只有 r = 0.244 | Baumler et al. 2026(ACL) | 🟡 | 約 150 字的個人文章,不是聊天 |
| 指標相關性被高估 | 既有的指標–人判相關,被資料集的人為痕跡灌水 | Pauli et al. 2025 | ✅ | 驗證設計要避開 |
| 風格轉換指標 | 正式度指標 vs 人判,四種語言的短句 | Briakou et al. 2021 | 🟡 | 一般正式度,非個人風格 |
| **極短非正式訊息的風格指標驗證** | | 搜尋無結果 | ⬜ | **留空** |

## ④ 基準線

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| zero-shot 模仿 | Claude、GPT-4o、o1 加 15–38 則範例:熟人判為真人的比例 25% / 5.4% / 12.5% | IMPersona | ✅ | — |
| prompt 模仿非正式文體 | 給 5 則範例時,部落格的作者驗證只有 17–21%;範例從 2 加到 10 幾乎沒差 | Catch Me If You Can(2025) | ✅ | — |
| many-shot ICL | 範例增加到數百、數千則常常持續變好,**但不一定單調**(有任務在 50–250 則時就到頂或變差) | Agarwal et al. 2024(NeurIPS);Bertsch et al. 2025(NAACL) | ✅ | 測多種範例數 ⏳ |
| 檢索範例 | 檢索本人過去的文字放進 prompt;推論時才加檢索、但訓練沒加,反而會變差 | LaMP;Panza(RAFT) | ✅ | 選做 |
| RAG vs 每人微調 | LaMP 上 RAG +14.92%、每人 PEFT +1.07%、合併 +15.98%;資料越多 PEFT 越有利 | Salemi & Zamani 2024 | 🟡 | 用的是任務指標,不是風格 |
| **單一個人:many-shot vs LoRA** | IMPersona 的 prompt 基準只有 15–38 則;NTT 的基準最多 10 回合 | 搜尋無結果 | ⬜ | **留空** |

## ⑤ SFT LoRA

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| LoRA | 凍結原始權重,只訓練低秩矩陣 | Hu et al. 2022(ICLR) | ✅ | — |
| QLoRA | 4-bit 底模 + LoRA;**LoRA 要掛在所有線性層才追得上全參數微調** | Dettmers et al. 2023(NeurIPS) | ✅ | 9B 需要 4-bit ⏳;支持「全部線性層」 |
| 每人一個 LoRA | IMPersona、OPPU、NTT、Panza、WeClone | 各原文 | ✅ | — |
| 超參數 | IMPersona(論文):rank 8、alpha 16、學習率 1e-4、依資料量跑 5/4/3 輪、batch 8;WeClone:rank 8、學習率 1e-4、2 輪 | IMPersona;WeClone | ✅ | 程式碼裡 alpha 是 32,和論文不同 ⏳ |
| 資料少時先用多人資料適應 | 每人資料少時直接微調,學到的是語料風格;所以先在全體資料上適應 | NTT 2024;OPPU | 🟡 | 之後多人版再做 |
| 原始預訓練版 vs instruct | **群體層級的直接比較:從原始預訓練版訓練的使用者模型較自然(80.21 vs 56.54)**,困惑度也較低;另一研究也發現原始預訓練版較準、較多樣 | Naous et al. 2026(Flipping the Dialogue,ICLR);Kang et al. 2026;West & Potts 2025;Kirk et al. 2024 | 🟡 | **建議把 Qwen3.5-4B-Base 從「選做」改成主要比較條件** |
| **單一個人:原始預訓練版 vs instruct** | IMPersona 只用 instruct 版,沒有比較 | 搜尋無結果 | ⬜ | **留空** |

## ⑥ 用風格尺引導的偏好訓練

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| DPO | 不需要獎勵模型和強化學習,直接用配對訓練;β 控制偏離參考模型的程度 | Rafailov et al. 2023(NeurIPS) | ✅ | — |
| 真人當「好」、模型當「差」 | DITTO:不到 10 則示範;**參考模型固定**(每輪更新會從 70.1 掉到 45.8);SPIN;PAL;PsPLUG | DITTO(Shaikh et al.,ICLR 2025,原標題「Show, Don't Tell」);SPIN;PAL;PsPLUG | ✅ | 參考模型固定為 SFT 模型 |
| **用分辨器分數排序組配對** | **DIAL:抽 8 個候選,用分辨器分數排序;只保留「好 > 0、差 < 0」的配對**。分辨器準確率從 87.9% 降到 69.5%,人工判斷成功率從 67.0% 降到 55.0%。對象是約 10 萬名心理諮詢使用者組成的群體,英文 | Zhu et al.(DIAL,arXiv 2512.20773 v4) | 🟡 | **組配對的方法直接借用**;改成單一真人 + 可解釋特徵 |
| 用 LLM 裁判當獎勵 | Turing-RL:LLM 裁判評「多像真人」當 GRPO 獎勵;出現「比真人還像真人」的作弊,所以把獎勵設上限 | Wang et al. 2026 | 🟡 | 我們的分數也要設上限 |
| 用作者模型當獎勵 | ASTRAPOP:LUAR 獎勵,比較 PPO、DPO、CPO(英文改寫任務,引 arXiv 版);以作者驗證訓練的模型當 GRPO 獎勵(經典小說作家) | ASTRAPOP(arXiv 2403.08043);Liu et al. 2026(CoNLL) | 🟡 | 改寫或長文,不是聊天回覆 |
| 先篩選再訓練(best-of-N) | RAFT:抽樣、依分數排序、只用高分的再訓練;ReST:生成 + 篩選的迴圈 | Dong et al. 2023(TMLR);Gulcehre et al. 2023 | ✅ | 可以當比 DPO 更便宜的第一步 |
| 過度最佳化 | 對代理獎勵最佳化太多,真正的品質反而下降;**在他們的設定裡 KL 懲罰沒有幫助,效果像提早停止**;best-of-N 比 RL 更有效率 | Gao et al. 2023(ICML) | ✅ | 訓練用的尺 ≠ 驗收用的尺;提早停止 |
| 防平均化 | DivPO(好 = 少見但高品質、差 = 常見低品質);GEM(保留多樣性的 SFT 目標);MMI 重新排序 | Lanchantin et al. 2025;GEM(ICLR 2025);Li et al. 2016 | 🟡 | 注意不要讓模型比你本人更多變 |
| **用可解釋文體特徵,為單一真人的少量資料排序配對** | 最接近的 DIAL 用學出來的分辨器、群體層級;Turing-RL 用 LLM 裁判;沒有找到用可解釋文體特徵、針對單一真人的做法 | 搜尋結果:部分相關 | ⬜ | **留空 = 我們的主要方法貢獻** |

## ⑦ 評估

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 獨立分辨器 | 兩樣本檢定 | Lopez-Paz & Oquab 2017 | ✅ | 重新訓練,和訓練用的分數分開 |
| 人工盲測 | 熟人 3 分鐘即時聊天(IMPersona);熟人判斷信件可疑度(Panza);專家五選一(Dennett 實驗:51%,隨機 20%) | IMPersona;Panza;Schwitzgebel et al. 2023 | ✅ | 你本人盲判 + ⏳ 一位常跟你傳訊息的人 |
| 校準的上下限 | 同作者上限與跨作者下限 | Sawant 2026 | 🟡 | 可加在 R3 上 |
| 多樣性 | distinct-1/2 | Li et al. 2016 | ✅ | 加上「最常見 20 種回覆佔比」 |
| 記憶與抄襲 | 訓練資料可被逐字抽出;記憶隨模型大小、重複次數、提示長度增加 | Carlini et al. 2021;Carlini et al. 2023 | ✅ | 計算和訓練回覆的重疊率 |
| 顯著性檢定 | 配對 bootstrap;如何選擇檢定 | Berg-Kirkpatrick et al. 2012;Dror et al. 2018 | ✅ | — |
| 台灣用語外漏 | SC-TC-Bench(現象);OpenCC(工具) | 各原文 | 🟡 | ⏳ 安裝 OpenCC |
| **繁中(台灣)個人聊天模仿的評測基準** | Dcard vs 微博對照語料(沒有作者資訊);PTT 八卦版問答語料(**沒有作者 ID**,不能做個人模型);繁中統計文體學只做過正式論文摘要 | Lian & Hsieh 2018;zake7749 Gossiping 語料;Yu & Chang 2026 | ⬜ | **留空** |

## ⑧ 倫理與隱私

| 需求 | 既有技術 | 出處 | 狀態 | 我們的調整 |
|---|---|---|---|---|
| 同意 | IMPersona 取得所有對話當事人同意,處理程式由資料提供者自己執行 | IMPersona | ✅ | 先模仿你自己;前文裡有學生訊息,全程本機 |
| 外洩風險 | 去識別化只能降低外洩;句子級差分隱私仍外洩約 3%;只出現在輸入的個資,在 QLoRA 後也可能被抽出 | Lukas et al. 2023;Szep et al. 2026 | ✅ | 模型權重視同聊天紀錄,不外流 |
| 輸出過濾的極限 | 擋逐字複製的過濾器很容易被繞過 | Ippolito et al. 2022 | ✅ | 抄襲率只當稽核,不當保證 |
| 揭露 | 與 AI 互動要讓人知道(EU AI Act 第 50 條,2026-08-02 起適用) | EU AI Act | ✅(參考規範) | 不讓模型代替你回覆真人 |
| 數位分身倫理 | 退役程序、透明、僅限成人、雙方同意(原文針對模擬已故者) | Hollanek & Nowaczyk-Basińska 2024 | 🟡 | 參考 |

## ⑨ 之後再做

| 需求 | 既有技術 | 出處 | 狀態 |
|---|---|---|---|
| 台灣在地底模 | Breeze 2(8B)、TAIDE(語料偏正式) | 各原文 | 🟡 |
| 先用台灣口語資料預適應 | PTT 八卦版問答語料(論壇語域,和私人聊天不同) | zake7749 | 🟡 |
| 挑哪一張貼圖 | PESRS(依前文 + 最近用過的貼圖);PerSRV;PEARL;U-Sticker 資料集(Telegram,有使用者 ID) | 各原文 | 🟡 |
| 多人個人化 | OPPU、Per-Pcs、PPlug、TinyStyler | 各原文 | 🟡 |
| **單一個人的貼圖時機** | 現有研究是挑哪一張,或預設要回貼圖 | 搜尋無結果 | ⬜ |

---

## 留空彙整:新方法與待解問題

| # | 留空項目 | 文獻現況 | 在 pipeline 的位置 |
|---|---|---|---|
| 1 | **用可解釋文體特徵、為單一真人少量資料排序 DPO 配對** | 部分相關:DIAL(群體、學出來的分辨器)、Turing-RL(LLM 裁判) | ⑥ 主要方法貢獻 |
| 2 | **以本人真實回覆互比當雜訊底線、正規化的逐特徵距離** | 部分相關:Sawant 2026 只在 embedding 層級校準 | ③ R1 |
| 3 | **繁中台灣口語短訊的風格尺,並對人判驗證** | 無:極短非正式訊息的指標驗證沒有找到 | ③ 驗證 |
| 4 | 全形/半形標點當作者特徵 | 無 | ③ R1 |
| 5 | 中文短訊的品質過濾規則 | 無(IMPersona 規則不適用);**本專案實驗 4 驗證了符號類過濾規則**,其餘(短回覆、貼圖、顏文字的處理)仍無文獻 | ① |
| 6 | 單一個人:原始預訓練版 vs instruct | 部分相關:群體層級支持原始預訓練版 | ⑤ |
| 7 | 單一個人:many-shot vs LoRA | 無 | ④ |
| 8 | 繁中(台灣)個人聊天模仿的資料集與評測 | 無 | ⑦ |
| 9 | 單一個人的貼圖時機 | 無 | ⑨ |
| 10 | 個人 LoRA 後的台灣用語外漏量測 | 無(先前文獻調查) | ⑦ |

**定位提醒**:「沒有找到」只代表這幾輪的搜尋範圍(主要是 WebSearch;Google Scholar、部分出版社頁面無法存取)。正式寫論文前,建議再做一次完整的文獻回顧。另有幾篇看到但打不開、值得人工確認:Liu & Jin「Stylometric Assessment of LLM Stylistic Imitation: How Faithfully Do Models Reproduce Intra-Authorial Stylistic Dynamics?」(SSRN 5986968),可能和留空 #2 有關;「Gendered Use of Sentence-Final Particles in Taiwan Mandarin」(Harvard DASH)。

## 根據這次調查,建議修改 pipeline 的地方

1. **⑤ 把 Qwen3.5-4B-Base 列為主要比較條件**:群體層級的直接比較支持原始預訓練版(Flipping the Dialogue)。
2. **⑥ 直接借用 DIAL 的配對方法**:抽多個候選 → 分數排序 → 只保留「好 > 0、差 < 0」的配對。分數設上限,防止 Turing-RL 遇到的「比真人還像」作弊。
3. **⑥ 先試 best-of-N(RAFT)**:比 DPO 便宜;Gao 2023 也顯示它對偏離參考模型的「預算」較有效率。
4. **⑥ 不要依賴 KL 懲罰防過度最佳化**:Gao 2023 在他們的設定裡發現沒幫助,改用提早停止 + 獨立驗收尺。
5. **③ R1 參考 Li & Zhang 2025 的 34 個特徵**,改成台灣口語版本。
6. **③ R2 降為次要指標**:Eder 2015 的穩定門檻是 2,500–5,000 字,我們每組只有幾百字。
7. **③ 人判驗證是必要的,不是選做**:作者向量和本人自評的相關只有 0.244(Baumler 2026)。
8. **④ many-shot 要測多種範例數**:效果不一定隨範例數單調上升(Agarwal 2024)。
9. **⑧ 寫進限制**:emoji 頻率會配合對話對象改變(Marko 2022),我們只有一位對象。

---

## 實驗紀錄

四個實驗都在 `chathistory.txt`(4,967 則訊息,2025-06-20 ~ 2026-09-19)上進行;全程本機、純 Python(bge-m3 實驗用 GPU 做 embedding),資料不外流。共同設定:**回合** = 同一人、同一天、間隔 ≤ 30 分鐘的連發訊息合併;**時間切分**:2026-07-20 之前為訓練期、之後為測試期;S = 學生,T = 使用者(Hung LT)。實驗 1–3 都經另一個 agent 獨立重寫程式重算主要數字;**實驗 4 是單次執行、尚未獨立重算**。

### 實驗 1(2026-10-03):兩人分不分得開?字元 n-gram Naive Bayes 分辨器

**目的**:確認風格訊號夠不夠強,並給任何裁判一個必須超過的底線。
**方法**:字元 1–3-gram 多項式 Naive Bayes(Laplace α=1),預測一個回合是誰說的。訓練 1,954 回合(S 1,002 / T 952),測試 337 回合(S 168 / T 169)。

| 設定 | 單回合準確率 | 配對準確率(長度 ±30% 配對,200 對) |
|---|---|---|
| 全部回合,貼圖當 token | 0.751 | 0.815 |
| 貼圖移除(主設定) | 0.731 | **0.890** [0.838, 0.930];全部 3,314 對 0.880 [0.829, 0.925] |
| 短泛用回覆(≤10 字、無程式碼/數字,n=86) | 0.663 [0.553, 0.761] | **0.870** [0.815, 0.913];bootstrap [0.738, 0.951] |
| 同類應答(都是「好/了解」類,訓練期 LOO,879 對) | — | **0.907** [0.804, 0.989] |
| 只看長度(對照) | 0.615 | 0.460 |

**風格指紋(程式直接計算)**:

| 特徵 | T(Hung LT) | S(學生) | 分得開? |
|---|---|---|---|
| 語氣詞/100 則:喔 / 啊 / 嗎 | 4.8 / 3.6 / 3.9 | 0.1 / 0.6 / 13.2 | ✅ |
| 全形標點比例(10/03 數字,**無法重現**;重算見實驗 4 之後的更正) | 18% | 61% | ⚠ |
| 貼圖+圖片佔訊息比例 | 11% | 33% | ✅ |
| 一回合幾則(中位數) | 2 | 1 | ✅ |
| 「～」結尾 | 2.3% | 0% | ✅(稀疏) |
| 口頭禪 | 好啊、沒問題、讚喔、沒錯、OK | 嗯嗯、好的、我懂了、謝謝你 | ✅ |
| 每則字數(中位數) | 9 | 8 | ❌ 分不開 |
| 結尾有無標點、回覆延遲 | ~同 | ~同 | ❌ 分不開 |

**分析**:風格訊號很強,而且主要不是話題:控制內容後(同類應答)仍有 0.91。可解釋的差異集中在語氣詞、標點寬度、貼圖、連發、口頭禪;長度和回覆速度沒有身分訊息。這個結果給出裁判底線:**任何 LLM 或 embedding 裁判在配對題上至少要到 0.85–0.88**。

### 實驗 2(2026-10-03):語意 embedding(bge-m3)能不能當裁判?

**目的**:檢驗 Hugging Face 上的 sentence-similarity 模型是否看得到風格、能判斷什麼內容。
**方法**:bge-m3 dense CLS 向量(L2 正規化,cosine)。風格題用與實驗 1 相同的 200 對;內容題用測試期 216 組「前文 → 真實回覆」。

| 風格:誰說的(配對準確率) | 全部回合 | 短泛用 |
|---|---|---|
| NB(實驗 1) | **0.880** | **0.852** |
| bge-m3 cosine 對兩人中心向量 | 0.783 | 0.734 |
| bge-m3 kNN(k=25) | 0.741 | 0.585 |
| bge-m3 向量 + logistic probe(訓練期擬合) | 0.858 | 0.851 |
| 與 NB 的差(bootstrap) | 中心向量 −0.097 [−0.165, −0.033];probe −0.023 [−0.084, +0.034] | |

- 同意思的應答(扣掉一字不差的重複)同人 vs 跨人 cosine 只差 **0.018** [0.004, 0.033];同義 vs 無關句差約 0.4 → cosine 的尺度被「意思」主導。

| 內容:bge-m3 能判斷什麼 | 結果 |
|---|---|
| 真實回覆 vs 本人其他隨機回覆,哪個較貼近前文 | **0.739** [0.704, 0.775](字元重疊基準 0.59) |
| 同上,但干擾回覆取自同一天 | 0.653 [0.605, 0.705] |
| 本題回答 vs 別題回答 AUC | **0.944** |
| 意思相反的短句:cos(可以啊, 不行耶) | **0.774**,高於 cos(好啊, 嗯嗯) = 0.765 |
| 固定真實回答,「同義不同口吻」vs「意思相反」哪個較像 | 9/20、另一組獨立句子 7/12 → 丟銅板 |

**分析**:語意 embedding 直接比相似度看不到風格(訓練目標就是忽略措辭);在向量上訓練分類器可追平 NB 但不更好。它能篩離題(0.74、AUC 0.94),**抓不到意思相反**(先前「抓得到 可以啊 vs 不行耶」的說法被推翻)。結論:風格裁判用 NB;bge-m3 只當離題篩檢。

### 實驗 3(2026-10-10):正式文體學距離能不能分出一句話是誰說的?

**目的**:把 ③ 總表的正式距離實作在同一份資料上,與 NB 作者語言模型(= 對數似然比)正面比較;回答「單句能不能分、要幾句才穩、尺可不可信」。
**方法**:字元 1–3-gram,詞表 = 訓練期最高頻 300 個(100 / 1000 當消融);每人一個訓練期輪廓。距離:Burrows' Delta、Cosine Delta(z 分數參照 = 400 個訓練期 25 則隨機袋,**偏離標準做法,因只有兩位作者**)、raw cosine、Min–max、JSD(add-0.5、log₂)、NB LLR、NCD(zlib-9,對 10 個 ~3,000 字訓練參考袋平均)、Craig's Zeta(10 回合段落;加總、mean-Z、marker-share 三種計分)。測試回合 283(S 134 / T 149,中位數 16 字);短泛用 86(中位數 7 字)。流程:Opus 5.5 實作 → 獨立審查 → 修正 → 再審(verdict:minor-issues)。完整報告見 `FORMAL_DISTANCE_EXPERIMENT.md`。

**E1 單回合判定(n=283)**

| 方法 | 門檻 0 / 最近輪廓 | 訓練期 OOF 調門檻 | 調門檻後與 NB(調)的配對差 |
|---|---|---|---|
| **NB 作者 LM LLR** | **0.731** [0.676, 0.782] | **0.788** [0.736, 0.834] | — |
| Zeta mean-Z | 0.565 | 0.707 [0.650, 0.759] | −0.081 [−0.134, −0.028] |
| Min–max | 0.604 | 0.696 [0.639, 0.749] | −0.092 [−0.145, −0.042] |
| Zeta marker share | 0.668 | 0.689 | −0.099 |
| raw cosine | 0.541 | 0.671 | −0.117 |
| Cosine Delta | 0.664 [0.606, 0.719] | 0.647 | −0.141 |
| NCD(zlib) | 0.629 | 0.590 | −0.198 |
| Burrows' Delta | 0.604 | 0.548 | −0.240 |
| JSD | 0.530 | 0.548 | −0.240 |
| 只看長度 | 0.615 | — | — |

短泛用子集(n=86,中位數 7 字):所有方法 0.59–0.72,**包括 NB(0.66 / 調門檻 0.72)**。

**E2 配對(同實驗 1 的 200 對)**:NB **0.890**;最佳正式距離 NCD 0.775 [0.711, 0.831],差 −0.115 [−0.221, −0.030];其他距離 0.63–0.75,與 NB 的差全部 CI 不含 0。

**E3 分袋:要幾則才分得穩**(每人 300 袋,取自 134 S / 149 T 測試回合,袋之間重疊;門檻在訓練期 OOF 袋上按 k 校準,不用測試資料)

| 方法 | k=1 | k=3 | k=5 | k=10 | k=25 | k=50 | AUC ≥ 0.95 起點 |
|---|---|---|---|---|---|---|---|
| **NB LLR** | 0.800 | 0.887 | 0.942 | **0.955**(重抽校準袋:0.937、0.947) | **0.992**(重抽 0.995) | 1.000 | k=5 |
| Min–max | 0.713 | 0.767 | 0.830 | 0.865 | 0.925 | 0.970 | k=25 |
| Cosine Delta | 0.623 | 0.768 | 0.803 | 0.868 | 0.913 | 0.987 | k=25 |
| NCD | 0.587 | 0.693 | 0.732 | 0.835–0.903 | 0.918 | 0.993 | k=10 |
| Zeta 加總 | 0.635 | 0.748 | 0.787 | 0.770 | 0.883 | 0.968 | k=25 |
| Burrows' Delta | 0.537 | 0.672 | 0.757 | 0.740 | 0.855 | 0.810 | k=50 |
| JSD | 0.540 | 0.690 | 0.787 | 0.875 | 0.840 | 0.853 | 未達 |

- **門檻 0 的假象**(第一版的錯誤):NB 分數是整袋加總,偏 T 的量隨 k 累積,門檻 0 時 k=50 只有 0.877(錯的全是 S 袋被判成 T),但 AUC 已是 1.0。審查者抓到後改為校準門檻。
- **跨時期穩定度**(訓練期 5 個日期區塊的 OOF AUC,k=25,最差區塊):NCD 0.948–0.985、NB 0.815–0.914、Zeta marker share 0.921、Cosine Delta 0.792。NB 在測試期特別好的部分原因是 T 的程式碼比例剛好低(0.38 vs 訓練期 0.60)。
- 分袋可分性不是長度假象:只看長度的 AUC k=10/25/50 = 0.66 / 0.53 / 0.43;長度配對的袋 NB AUC 仍 0.998。

**E4 尺驗證:同一人兩半 vs 兩人互比**(訓練期,k=50,2,000 次)。通過標準:跨人距離 > 兩個同人距離的比例 ≥ 95%。

| 距離 | 同人 T | 同人 S | 跨人 | P(跨人 > max 同人) |
|---|---|---|---|---|
| Burrows' Delta | 0.73 | 0.86 | 0.88 | 0.44 |
| Cosine Delta | 0.93 | 0.95 | 1.09 | 0.53 |
| raw cosine | 0.27 | 0.37 | 0.37 | 0.37 |
| Min–max | 0.54 | 0.62 | 0.63 | 0.40 |
| JSD | 0.17 | 0.22 | 0.23 | 0.42 |
| NCD | 0.87 | 0.87 | 0.91 | 0.70 |

**全部沒過。** 事後診斷:符號類訊息(程式碼、數學式、顏文字;`CODE_RE`)佔訓練字元 S 0.68 / T 0.60,**不同時期的話題差異大於兩人差異**。把這些回合拿掉重擬合(訓練 S 679 / T 706,測試 241):E4 通過率升到 0.86–0.97;Min–max 單回合 0.772(= NB)、k=5 達 0.948、k=10 0.985;NCD k=10 0.978。(事後分析,只當線索。)

**E5 內容控制**(同類應答子集,訓練期):排除一字不差的重複後,距離的通過率只有 0.74–0.85;但 **NB 的 OOF 單回合 AUC 0.944 [0.896, 0.983],連訓練時沒見過的字串也有 0.965 [0.875, 1.000](n=25)**;Zeta 變體 0.85–0.91;Min–max 在未見字串上只有 0.30。

**Zeta 標記 n-gram**(Z = p_T − p_S):T:喔 +0.65、喔⏎ +0.55、啊 +0.46、就是 +0.43、所以 +0.41、問題 +0.41、這個 +0.37;S:嗎⏎我 −0.39、我不 −0.32、怎麼 −0.31、不知道 −0.29、可是 −0.26、我想 −0.25、嗯嗯 −0.22。加總計分的平均 Z 為 +0.062,所以分數隨長度漂向 T。

**消融**:詞表 100/300/1000 對 Cosine Delta、JSD 略有幫助;lzma 比 zlib 慢約 70 倍、無明顯增益;Cosine/Burrows Delta 對 z 參照袋的抽樣敏感(10 個種子:Burrows 校準後分袋準確率可差 0.10;種子 0 對 Cosine Delta 偏有利)。

**審查者的其他發現**:k=10 的 NB 結果是臨界值(三次校準抽樣兩次低於 0.95),**規劃用 k=25**;評估單位是回合不是單則訊息;「最佳正式距離」是從約 20 個設定裡挑的事後變體,公平比較(都調門檻)時每種正式距離都顯著輸 NB。

**分析與決定**

1. **單一回合分不出來**:最好 0.79;正式距離更低。7 字短句全部不可靠。
2. **十幾到二十五則才分得穩**:NB 校準後 k=25 達 0.99;正式距離要 k=50。
3. **尺的驗證沒過,原因是符號類訊息**:這直接決定 pipeline ①:**程式碼/數學式必須拿掉或單獨標記**。
4. **訓練用的排序分數,門檻一定要在訓練期校準**(pipeline ⑥ 的教訓)。
5. **採用**:NB 作者 LM LLR 當逐則與分袋主尺;NCD 當無特徵交叉檢查(k ≥ 25、報 AUC);Cosine Delta 暫列;丟 Burrows' Delta、JSD、raw cosine;Zeta 只留標記詞清單。
6. **下一步**(預先登記):拿掉符號類訊息後重跑 E1–E5,確認尺能否通過。→ **2026-10-13 完成,見實驗 4。**

### 實驗 4(2026-10-13):拿掉符號類訊息後,尺能不能通過驗證?

**目的**:實驗 3 預先登記的下一步。回答三件事:(a) 符號類訊息拿掉後,距離的「同人 < 跨人」驗證過不過;(b) 干擾是時間漂移還是符號類內容;(c) 哪一種過濾規則該進 pipeline ①。
**方法**:純 Python、無 GPU、無 subagent(每種規則 83 秒;`recheck_nocode.py` / `recheck_nocode.json` 在 session scratchpad);**單次執行,尚未獨立重算**。其他設定同實驗 3(詞表 300、z 參照 400 袋、5 個日期區塊 OOF 校準)。兩種規則:**寬鬆** = `CODE_RE`(任一半形符號 `; { } ( ) = < > # [ ] _ \ / * + & | % ^ $ @` ` `` `、網址、程式關鍵字);**嚴格** = 關鍵字 / 網址 / `; { }`,或「≥ 20 個非空白字元且符號占比 ≥ 30%」。E4 多加一種「同時期袋」模式(同一次重抽的所有袋都來自同一個日期區塊),把時間漂移和符號內容分開。完整表格見 `FORMAL_DISTANCE_EXPERIMENT.md` §E。

**過濾後的資料**

| 規則 | 訓練 S / T(拿掉) | 測試 S / T(拿掉) | 拿掉的字元占比 訓練 S / T | 測試 S / T |
|---|---|---|---|---|
| 不過濾 | 785 / 862 | 134 / 149 | — | — |
| 寬鬆 | 679 / 706(106 / 156) | 115 / 126(19 / 23) | 0.68 / 0.60 | 0.70 / 0.38 |
| 嚴格 | 710 / 741(75 / 121) | 118 / 128(16 / 21) | 0.54 / 0.56 | 0.64 / 0.31 |

符號類字元占比在五個訓練區塊之間擺盪(S 0.18–0.79、T 0.32–0.67),這就是壓垮 E4 的同人變異。

**E1 單回合**(各規則的測試集不同,不是配對比較):NB 0.731 → 寬鬆 0.772(調門檻 0.797、AUC 0.885)/ 嚴格 0.752(0.813、AUC 0.906);Min–max 0.604 → 0.772 / 0.728(AUC 0.744 → 0.854 / 0.844);NCD 0.629 → 0.718 / 0.683(AUC 0.713 → 0.814);Cosine Delta 0.664 → 0.710 / 0.687。**單回合仍 ≤ 0.81**。

**E3 分袋(校準門檻;括號 AUC)**

| 規則 | 方法 | k=5 | k=10 | k=25 |
|---|---|---|---|---|
| 不過濾 | NB / Min–max / Cosine Δ / NCD | 0.947 / 0.820 / 0.817 / 0.823 | 0.937 / 0.857 / 0.843 / 0.833 | 0.993 / 0.917 / 0.930 / 0.930 |
| 寬鬆 | NB / Min–max / Cosine Δ / NCD | 0.983 / 0.967 / 0.923 / 0.940 | **1.000** [0.99, 1.00] / 0.970 / 0.963 / 0.983 | 全部 1.000 |
| 嚴格 | NB / Min–max / Cosine Δ / NCD | 0.967 / 0.957 / 0.903 / 0.797 | 1.000 / 0.980 / 0.967 / 0.907 | 1.000 / 1.000 / 1.000 / 0.980 |

**E4 尺驗證:P(跨人 > max 同人),k=50,1,000 次**

| 規則 | 袋的抽法 | Burrows Δ | Cosine Δ | Min–max | JSD | NCD |
|---|---|---|---|---|---|---|
| 不過濾 | 隨機 | 0.44 | 0.54 | 0.40 | 0.43 | 0.69 |
| 不過濾 | 同時期 | 0.52 | 0.59 | 0.46 | 0.46 | 0.74 |
| 寬鬆 | 隨機 | 0.91 | **0.96** | 0.94 | 0.89 | **0.96** |
| 寬鬆 | 同時期 | 0.95 | **0.97** | 0.94 | 0.92 | **0.97** |
| 嚴格 | 隨機 | 0.83 | 0.96 | 0.85 | 0.84 | 0.84 |
| 嚴格 | 同時期 | 0.89 | 0.95 | 0.86 | 0.88 | 0.83 |

k=25 時所有規則都只有 0.73–0.89(不過)。

**分析與決定**

1. **干擾是符號類內容,不是時間**:不過濾、只配對同時期,通過率只從 0.40–0.69 升到 0.46–0.74;寬鬆過濾後升到 0.91–0.97。
2. **寬鬆規則比嚴格規則好**:嚴格規則留下數學式、顏文字、括號片段,五種距離有三種停在 0.83–0.86。
3. **尺驗證要 k=50,判定作者 k=10–25 就夠**:兩個需求分開寫進 pipeline。過濾後 NB k=10 無錯(之前 0.94–0.96 臨界)。
4. **單回合還是分不出來(≤ 0.81)** → 訓練時逐候選的分數是「吵雜分數」,只能用排序、邊際與袋層級統計,不能當點判斷(pipeline ⑥)。
5. **Pipeline ① 定案**:v1 採寬鬆規則;符號類回合不當目標、不進尺;前文中換成 `[程式碼/算式]` 佔位符。兩個修正待驗證(重跑本節當單元測試;門檻:Cosine Δ、NCD ≥ 0.95、Min–max ≥ 0.90,k=50):顏文字 / LINE 表情代碼先換成佔位符;半形標點改由 R1 特徵在未過濾文本上計算。
6. **尺的組合(過濾後)**:NB LLR(判定、校準門檻)、NCD(無模型交叉檢查,兩種模式都過)、Cosine Delta(過,但依賴兩作者 z 參照,暫列)、Min–max(0.94,候選)。
7. **注意**:寬鬆規則拿掉 T 的回合比例(18%)比 S(13.5%)高,其中包含只因半形括號或斜線而被抓到的一般聊天(兩人各約 31–32 則,對稱)。**2026-10-13 更正**:10/03 的「全形標點 18% vs 61%」無法重現(analyze_chat.py 已遺失,定義不明)。重算:(a) 含程式碼的全部訓練訊息,字元層級全形占比 T 0.075 / S 0.246(方向同 10/03,數字不同);(b) 尺用的非符號回合,T 0.32 / S 0.17,但標點極少(T 81 個、S 159 個標點,約 1,100–1,500 則訊息),k=50 袋的鑑別度只有 0.68。→ 半形標點線索大半來自程式碼,不是聊天風格;寬鬆規則拿掉它不算損失。

---

## 參考文獻

### 文體學與作者鑑定
- Burrows, J. (2002). 'Delta': A measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing*, 17(3), 267–287. https://academic.oup.com/dsh/article-lookup/doi/10.1093/llc/17.3.267
- Smith, P. W. H., & Aldridge, W. (2011). Improving authorship attribution: Optimizing Burrows' Delta method. *Journal of Quantitative Linguistics*, 18(1), 63–88. https://doi.org/10.1080/09296174.2011.533591 (出版社頁無法存取;cosine 做法經 Evert et al. 2015 確認:https://aclanthology.org/W15-0709.pdf)
- Evert, S., Proisl, T., Jannidis, F., Reger, I., Pielström, S., Schöch, C., & Vitt, T. (2017). Understanding and explaining Delta measures for authorship attribution. *Digital Scholarship in the Humanities*, 32(suppl. 2), ii4–ii16. https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676
- Eder, M. (2015). Does size matter? Authorship attribution, small samples, big problem. *Digital Scholarship in the Humanities*, 30(2), 167–182. https://academic.oup.com/dsh/article/30/2/167/390738
- Stamatatos, E. (2009). A survey of modern authorship attribution methods. *JASIST*, 60(3), 538–556. https://doi.org/10.1002/asi.21001
- Argamon, S. (2008). Interpreting Burrows's Delta: Geometric and probabilistic foundations. *Literary and Linguistic Computing*, 23(2), 131–147. https://academic.oup.com/dsh/article/23/2/131/1005964
- Hoover, D. L. (2004). Testing Burrows's Delta. *Literary and Linguistic Computing*, 19(4), 453–475. https://academic.oup.com/dsh/article/19/4/453/1004797
- Eder, M., Rybicki, J., & Kestemont, M. (2016). Stylometry with R: A package for computational text analysis. *The R Journal*, 8(1), 107–121. https://journal.r-project.org/articles/RJ-2016-007/
- Jannidis, Pielström, Schöch, & Vitt (2015). Improving Burrows' Delta(DH2015 摘要;含 Eder 排名因子公式). https://dh-abstracts.library.virginia.edu/works/2275
- Kestemont, M., Stover, J., Koppel, M., Karsdorp, F., & Daelemans, W. (2016). Authenticating the writings of Julius Caesar. *Expert Systems with Applications*, 63, 86–96. https://doi.org/10.1016/j.eswa.2016.06.029
- Kilgarriff, A. (2001). Comparing corpora. *International Journal of Corpus Linguistics*, 6(1), 97–133. https://www.kilgarriff.co.uk/Publications/2001-K-CompCorpIJCL.pdf
- Cilibrasi, R., & Vitányi, P. M. B. (2005). Clustering by compression. *IEEE Transactions on Information Theory*, 51(4), 1523–1545. https://arxiv.org/abs/cs/0312044
- Benedetto, D., Caglioti, E., & Loreto, V. (2002). Language trees and zipping. *Physical Review Letters*, 88(4), 048702. https://arxiv.org/abs/cond-mat/0108530
- Burrows, J. (2007). All the way through: Testing for authorship in different frequency strata. *Literary and Linguistic Computing*, 22(1), 27–47. https://academic.oup.com/dsh/article/22/1/27/1043372
- Craig, H., & Kinney, A. F. (Eds.). (2009). *Shakespeare, Computers, and the Mystery of Authorship*. Cambridge University Press. https://www.cambridge.org/core/books/shakespeare-computers-and-the-mystery-of-authorship/89F8AEF655C5FFD3D73E500870C71972
- Schöch, Schlör, Zehe, Gebhard, Becker, & Hotho (2018). Burrows' Zeta: Exploring and evaluating variants and parameters. *DH2018*. https://dh2018.adho.org/en/burrows-zeta-exploring-and-evaluating-variants-and-parameters/
- Yule, G. U. (1944). *The Statistical Study of Literary Vocabulary*. Cambridge University Press.(公式經 quanteda、koRpus 文件確認)
- McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392. https://doi.org/10.3758/BRM.42.2.381
- Heylighen, F., & Dewaele, J.-M. (1999). Formality of language: Definition, measurement and behavioral determinants. Internal report, Free University of Brussels. http://pespmc1.vub.ac.be/Papers/Formality.pdf
- Koppel, M., & Winter, Y. (2014). Determining if two documents are written by the same author. *JASIST*, 65(1), 178–187. https://doi.org/10.1002/asi.22954
- Seidman, S. (2013). Authorship verification using the impostors method. *PAN @ CLEF 2013*. https://ceur-ws.org/Vol-1179/CLEF2013wn-PAN-Seidman2013.pdf
- Wegmann, A., & Nguyen, D. (2021). Does it capture STEL? A modular, similarity-based linguistic style evaluation framework. *EMNLP 2021*, 7109–7130. https://aclanthology.org/2021.emnlp-main.569/
- Juola, P. (2008). Authorship attribution. *Foundations and Trends in Information Retrieval*, 1(3), 233–334. https://doi.org/10.1561/1500000005
- Peng, F., Schuurmans, D., Wang, S., & Keselj, V. (2003). Language independent authorship attribution using character level language models. *EACL*. https://aclanthology.org/E03-1053.pdf
- Schwartz, R., Tsur, O., Rappoport, A., & Koppel, M. (2013). Authorship attribution of micro-messages. *EMNLP*. https://aclanthology.org/D13-1193.pdf
- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press. https://www.cambridge.org/core/books/variation-across-speech-and-writing/A546CF5ED8F8E62F1432CB2F369CF356
- Herring, S. C. (2004). Computer-mediated discourse analysis: An approach to researching online behavior. In *Designing for Virtual Communities in the Service of Learning* (pp. 338–376). Cambridge University Press. https://homes.luddy.indiana.edu/herring/cmda.pdf
- Zhang (2024). *Authorship Analysis in Chinese Social Media Texts*. Cambridge Elements. https://www.cambridge.org/core/elements/authorship-analysis-in-chinese-social-media-texts/CDFBDEE28085274DE8E3FEA3921AFD06
- Li & Zhang (2025). Linguistic differences between AI and human comments in Weibo. *CCL 2025*. https://aclanthology.org/2025.ccl-1.64/
- Zheng, R., Li, J., Chen, H., & Huang, Z. (2006). A framework for authorship identification of online messages. *JASIST*, 57(3). https://experts.arizona.edu/en/publications/a-framework-for-authorship-identification-of-online-messages-writ/
- Cristani, M., et al. (2012). Conversationally-inspired stylometric features for authorship attribution in instant messaging. *ACM Multimedia*. https://eprints.gla.ac.uk/149770
- Marko (2022). "Depends on who I'm writing to"(完整標題見連結). *Frontiers in Communication*. https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2022.840646/full
- Muñoz-Ortiz, A., Gómez-Rodríguez, C., & Vilares, D. (2024). Contrasting linguistic patterns in human and LLM-generated news text. *Artificial Intelligence Review*, 57(10), 265. https://arxiv.org/abs/2308.09067
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145–151. https://doi.org/10.1109/18.61115
- Sawant, Y. G. (2026). Theory-grounded evaluation exposes the authorship gap in LLM personalization. *CTB Workshop @ ICML 2026*. https://arxiv.org/abs/2604.26460
- Mikros (2025). "Beyond the surface … GPT-4o's capacity for literary style imitation"(標題節錄,完整標題見連結). *Digital Scholarship in the Humanities*, 40(2). https://academic.oup.com/dsh/article/40/2/587/8118784
- Zeng & Nini (2026). Authorship impersonation via LLM prompting does not evade authorship verification methods. arXiv. https://arxiv.org/abs/2603.29454

### 風格表示與指標
- Rivera-Soto, R. A., et al. (2021). Learning universal authorship representations (LUAR). *EMNLP*. https://aclanthology.org/2021.emnlp-main.70.pdf
- Wegmann, A., Schraagen, M., & Nguyen, D. (2022). Same author or just same topic? Towards content-independent style representations. *RepL4NLP*. https://aclanthology.org/2022.repl4nlp-1.26.pdf
- StyleDistance (NAACL 2025). https://arxiv.org/html/2410.12757v1 ;mStyleDistance. https://arxiv.org/html/2502.15168
- STEB (2026 預印本). https://arxiv.org/abs/2606.31741
- AuthBench (2026 預印本). https://arxiv.org/html/2609.06771
- Lopez-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*. https://arxiv.org/abs/1610.06545
- Kannan, A., & Vinyals, O. (2017). Adversarial evaluation of dialogue models. https://arxiv.org/pdf/1701.08198
- Ippolito, D., et al. (2020). Automatic detection of generated text is easiest when humans are fooled. *ACL*. https://arxiv.org/abs/1911.00650
- Baumler et al. (2026). "Can you make it sound like you?" *ACL 2026*. https://arxiv.org/html/2604.24444
- Briakou, E., et al. (2021). Evaluating the evaluation metrics for style transfer. *EMNLP*. https://aclanthology.org/2021.emnlp-main.100
- Pauli, Augenstein, & Assent (2025). "Mind the style gap." *Findings of EMNLP*. https://arxiv.org/abs/2502.15022
- Bavaresco, A., et al. (2024). JUDGE-BENCH: LLMs instead of human judges? https://arxiv.org/abs/2406.18403
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685
- Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM evaluators recognize and favor their own generations. https://arxiv.org/abs/2404.13076
- Li, D., et al. (2025). Preference leakage. https://arxiv.org/abs/2502.01534

### 個人模仿與個人化
- Shi, Q., et al. (2025). IMPersona: Evaluating individual level LM impersonation. https://arxiv.org/html/2504.04332 ;程式碼(無授權條款):https://github.com/princeton-nlp/impersona
- Shi, Q. (2026). Impersona-Env(MIT 授權). https://github.com/benshi34/impersona-env
- Nicolicioiu, A., et al. Panza. https://arxiv.org/html/2407.10994v4
- NTT 使用者專屬對話模型(2024). https://arxiv.org/abs/2409.00887
- Tan, Z., et al. (2024). OPPU: One PEFT Per User. *EMNLP*. https://arxiv.org/html/2402.04401
- Salemi, A., & Zamani, H. (2024). Comparing retrieval-augmentation and parameter-efficient fine-tuning for privacy-preserving personalization. https://arxiv.org/abs/2409.09510
- Salemi, A., et al. (2023). LaMP. https://arxiv.org/html/2304.11406v3
- Catch Me If You Can? Not Yet (2025). https://arxiv.org/abs/2509.14543
- PersonalBench (2026). https://arxiv.org/abs/2608.19746
- TwinVoice (2025). https://arxiv.org/html/2510.25536
- CoSER (2025). https://arxiv.org/html/2502.09082
- WeClone. https://github.com/xming521/WeClone
- Li, J., et al. (2016). A persona-based neural conversation model. *ACL*. https://aclanthology.org/P16-1094.pdf
- Schwitzgebel, E., et al. (2023). Creating a large language model of a philosopher(Dennett 實驗). https://arxiv.org/abs/2302.01339
- Naous, T., Laban, P., Xu, W., & Neville, J. (2026). Flipping the dialogue: Training and evaluating user language models. *ICLR 2026*. https://arxiv.org/html/2510.06552
- Kang et al. (2026). Pretrained persona mixture models and tandem models for human simulation. https://arxiv.org/html/2609.22607
- West, P., & Potts, C. (2025). Base models beat aligned models at randomness and creativity. https://arxiv.org/abs/2505.00047
- Kirk, R., et al. (2024). Understanding the effects of RLHF on LLM generalisation and diversity. *ICLR*. https://arxiv.org/abs/2310.06452
- Feng et al. (2025). 推理能力不一定提升角色扮演(6 個 benchmark、24 個 LLM;完整標題見連結). https://arxiv.org/abs/2502.16940

### 訓練方法
- Hu, E. J., et al. (2022). LoRA. *ICLR 2022*. https://arxiv.org/abs/2106.09685
- Dettmers, T., et al. (2023). QLoRA. *NeurIPS 2023*. https://arxiv.org/abs/2305.14314
- Rafailov, R., et al. (2023). Direct preference optimization. *NeurIPS 2023*. https://arxiv.org/abs/2305.18290
- Shaikh, O., et al. (2025). Aligning language models with demonstrated feedback(DITTO;arXiv v1 標題為「Show, Don't Tell」). *ICLR 2025*. https://arxiv.org/abs/2406.00888
- Chen, Z., et al. (2024). SPIN: Self-play fine-tuning. https://arxiv.org/abs/2401.01335
- PAL (2025). https://arxiv.org/html/2511.10215v1
- PsPLUG (2026). https://arxiv.org/html/2601.06362
- Liu, S., Agarwal, S., & May, J. (2024). Authorship style transfer with policy optimization (ASTRAPOP). arXiv. https://arxiv.org/abs/2403.08043
- Zhu, Z., et al. DIAL: Direct iterative adversarial learning for realistic multi-turn dialogue simulation. arXiv 2512.20773 v4. https://arxiv.org/html/2512.20773
- Wang, Y. S., et al. (2026). Learning user simulators with Turing rewards. https://arxiv.org/abs/2606.19336
- Liu, Bahja, Kovatchev, & Lee (2026). Capturing classic authorial style in long-form story generation with GRPO fine-tuning. *CoNLL 2026*. https://arxiv.org/abs/2512.05747
- Dong, H., et al. (2023). RAFT: Reward rAnked FineTuning. *TMLR*. https://arxiv.org/abs/2304.06767
- Gulcehre, C., et al. (2023). Reinforced self-training (ReST). arXiv. https://arxiv.org/abs/2308.08998
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML 2023*. https://proceedings.mlr.press/v202/gao23h.html
- Lanchantin, J., et al. (2025). Diverse preference optimization (DivPO). https://arxiv.org/abs/2501.18101
- Li, Z., et al. (2025). GEM: Preserving diversity in supervised fine-tuning. *ICLR 2025*. https://proceedings.iclr.cc/paper_files/paper/2025/file/a548ef984f30bca3abdc09f43743827f-Paper-Conference.pdf
- Li, J., et al. (2016). A diversity-promoting objective function for neural conversation models(MMI、distinct-n). *NAACL*. https://aclanthology.org/N16-1014/
- Agarwal, R., et al. (2024). Many-shot in-context learning. *NeurIPS 2024*. https://arxiv.org/abs/2404.11018
- Bertsch, A., et al. (2025). In-context learning with long-context models. *NAACL 2025*. https://arxiv.org/abs/2405.00200

### 統計
- Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012). An empirical investigation of statistical significance in NLP. *EMNLP-CoNLL*. https://aclanthology.org/D12-1091/
- Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in NLP. *ACL*. https://aclanthology.org/P18-1128/

### 資料、隱私與倫理
- Presidio(已移至 data-privacy-stack). https://github.com/microsoft/presidio ;支援的實體:https://presidio.dataprivacystack.org/supported_entities/
- OpenCC. https://github.com/BYVoid/OpenCC
- Lee, K., et al. (2022). Deduplicating training data makes language models better. *ACL*. https://aclanthology.org/2022.acl-long.577/
- Kandpal, N., Wallace, E., & Raffel, C. (2022). Deduplicating training data mitigates privacy risks. https://arxiv.org/abs/2202.06539
- Carlini, N., et al. (2021). Extracting training data from large language models. https://arxiv.org/abs/2012.07805
- Carlini, N., et al. (2023). Quantifying memorization across neural language models. https://arxiv.org/abs/2202.07646
- Lukas, N., et al. (2023). Analyzing leakage of personally identifiable information in language models. https://arxiv.org/abs/2302.00539
- Szep et al. (2026). 只出現在輸入的個資,在 QLoRA 微調後也可能被抽出(完整標題見連結). https://arxiv.org/html/2601.17480
- Mireshghallah, F., et al. (2022). Memorization in NLP fine-tuning methods. https://arxiv.org/abs/2205.12506
- Ippolito, D., et al. (2022). Preventing verbatim memorization gives a false sense of privacy. https://arxiv.org/abs/2210.17546
- EU AI Act, Article 50. https://artificialintelligenceact.eu/article/50/
- Hollanek, T., & Nowaczyk-Basińska, K. (2024). Griefbots, deadbots, postmortem avatars. *Philosophy & Technology*, 37, 63. https://link.springer.com/article/10.1007/s13347-024-00744-w

### 台灣與中文資源
- Lyu, Luo, Kang, & Koenecke (2025). SC-TC-Bench. *FAccT*. https://arxiv.org/abs/2505.22645
- Breeze 2 (2025). https://arxiv.org/abs/2501.13921
- TAIDE. https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat ;https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602
- Lian & Hsieh (2018). A topic-aware comparable corpus of Chinese variations(Dcard vs 微博). https://arxiv.org/abs/2411.10955
- PTT 八卦版問答語料(無作者 ID). https://github.com/zake7749/Gossiping-Chinese-Corpus
- Yu & Chang (2026). 以統計方法辨識 ChatGPT 生成之論文摘要. *中國統計學報* 64. https://www.stat.org.tw/jcsa/data/vol64/V64N2-2.pdf

### 貼圖
- Fei, Li, et al. MOD(DSTC10). https://arxiv.org/html/2109.01839
- STICKERCONV / PEGS (ACL 2024). https://arxiv.org/abs/2402.01679
- Gao, S., et al. (2021). Learning to respond with your favorite stickers (PESRS). *ACM TOIS*, 39(2). https://shengaopku.github.io/files/2021-tois-sticker.pdf
- PerSRV (WWW 2025). https://arxiv.org/abs/2410.21801
- PEARL (Findings of EMNLP 2025). https://arxiv.org/abs/2509.17749
- U-Sticker (SIGIR 2025). https://arxiv.org/abs/2502.19108
