# qasc_plus — 獨立 benchmark + 多方法比較框架

針對 [QASC](https://github.com/Arthur031221/quantum-allosteric-scanner)(CTQW 異位殘基排名)
建立的**獨立驗證環境**:自建 family-declustered benchmark、實作文獻中的競爭方法、
在同一套判準下比較,並附平庸對照組以檢驗任何結果是否只是假象。

所有方法的輸入簽章與 QASC 完全相同:**Cβ 座標 (N,3) + 活性位點殘基索引**。

---

## 快速開始

```bash
python3 scripts/build_dataset.py 1666 80     # 建 tier-A benchmark(需連網)
python3 scripts/evaluate.py --targets data/targets
python3 scripts/evaluate.py --targets data/qasc_targets   # QASC 原本的 3 個目標
```

---

## 1. 資料集怎麼來的

ASD / ASBench / CASBench 的官方伺服器目前**皆已失效**(`mdl.shsmu.edu.cn` 無回應、
`casbench.org` 無法連線),下載連結為 JS 驅動無法自動取得。因此改為**從 RCSB PDB
直接自建**,規則完全機械化、可重現、無任何手動輸入。

### 位點定義(與 ASBench/CASBench 同一慣例:以晶體複合物中接觸配體的殘基界定位點)

| 項目 | 規則 |
|---|---|
| `anchor`(正位/活性位點) | 重原子距**輔因子**配體 ≤ 4.5 Å 的殘基。輔因子清單為核苷酸、NAD/FAD/SAM/CoA/PLP、血基質等 |
| `y`(異位位點) | 重原子距**類藥物**配體 ≤ 4.5 Å 的殘基。類藥物 = 重原子 ≥ 12、含 N/S/鹵素(排除只有 C/O 的 PEG、甘油)、非輔因子、非結晶添加物 |
| 保留條件 | 兩位點不重疊,且異位位點距 anchor **≥ 8 Å**(distal) |

### 兩層資料集

- **tier-A**(`data/targets/`):候選來自 RCSB **全文檢索 "allosteric"** 的條目
  (1666 筆),即由沉積者自己標註為異位相關的結構。信心較高。
- **tier-B**(`data/targets_b/`):候選來自一般的「含輔因子 + ≥2 配體」查詢,
  標註純為幾何代理。信心較低,用於增加統計檢定力。

兩層皆以 **UniProt accession 去重**(PDBe SIFTS),確保 family-declustered:
同一個蛋白質家族只取一個代表。

### ⚠️ 標註是代理,不是專家策展

`y` 的真正意義是「**一個類藥物分子結晶在此,且遠離催化位點**」。這是該領域對候選
異位位點的操作型定義,但**不等同於實驗證實的異位調控位點**。任何基於此 benchmark
的結論都必須帶著這個限制陳述。

---

## 2. 實作的方法

全部在 `methods/`,每個方法只吃 Cβ 座標 + anchor。

### 量子通道(`methods/quantum.py`)

| 名稱 | 說明 |
|---|---|
| `qasc_baseline` | QASC 原版:Laplacian-CTQW 無限時間平均可通達性 + 鄰接矩陣 IPR 共振轉移,noisy-or 融合 |
| `qasc_degseed` | **修正版**:IPR 通道改用**度數加權**初始態。均勻疊加態是 Laplacian 的零本徵向量,但**不是鄰接矩陣的本徵向量**,故在 A-walk 下會漂移;度數加權才是自然初始態 |
| `qasc_normlap` | 改用對稱正規化 Laplacian,消除度數異質性(CTQW 在 hub 主導的圖上表現不佳) |
| `enaqt` | Lindblad 純去相干傳輸,去相干率**以最大耦合 J_max 為單位校準**,效率定義為有限時間窗內佔據機率的時間積分 |

### 古典對照(`methods/btb.py`, `methods/enm.py`)

| 名稱 | 說明 |
|---|---|
| `btb` / `btb_raw` | Bond-to-bond propensity 的殘基級移植:`M = ½ G Bᵀ L† B`(Laplacian 偽逆的 Green 函數),自 anchor 播種;`btb` 額外做距離條件分位數迴歸。**只計算 source 欄位**,不展開 m×m 矩陣 |
| `apop` | APOP 式:加勁局部鄰域的彈簧以模擬配體結合,依最慢全域模態的頻率位移排名。**不使用 anchor** |
| `corrsite` | CorrSite2.0 式:GNM 快/慢模態各自對 anchor 的運動相關性,取兩者 Z-score 最大值 |
| `prs` | 擾動響應掃描:GNM 共變異數矩陣即線性響應算子 |

### ALPS —— 本研究提出的新方法(`methods/alps.py`)

```
score(i) = z_d[ Σ_{k≤K} |λ_k(H_i) − λ_k(H_0)| / λ_k(H_0) ]
```

`H_0` 是接觸圖的 Kirchhoff 矩陣;`H_i` 是把殘基 i 鄰域(10 Å)內所有邊加勁
(×2)後的同一矩陣,模擬配體結合;取**最低 K=3 個非零本徵值**的相對位移;
`z_d[·]` 是對「距活性位點相同距離的其他殘基」做的條件 z-score。

**每個設計決策都對應本研究量到的一件事,不是猜的:**

| 設計 | 依據 |
|---|---|
| 用擾動響應而非傳播振幅 | QASC 的分數與距離相關 −0.60~−0.71,本質是近度排序器,在獨立目標上 AUC 掉到 0.5 以下。差值可讓基準距離結構抵消 |
| 讀**頻譜**而非傳輸 | 在完全相同的擾動框架下比較三種讀數(無限時間相干傳輸=QASC 的觀測量、有限窗相干傳輸、古典擴散),全部輸給頻譜讀數(9–27% vs 91%)。局部加勁幾乎不動主導長時間相干傳輸的本徵值**簡併**結構,故該觀測量雜訊大;但它乾淨地移動低階本徵值本身 |
| 只取最低 3 個模態 | K=3 勝過 K=5、K=10。異位槓桿存在於全域集體運動,高階模態只加局部雜訊 |
| 距離條件 z-score | 把本方法從 82% 提升到 91%;同一修正也把 QASC 自己從 9% 提升到 27% |

**算子的雙重讀法**:Kirchhoff 矩陣同時是 QASC 的 CTQW Hamiltonian 與 GNM 算子。
其低階本徵值既是量子漫步最慢的相干頻率,也是彈性網路最慢的振動模態——**同一組數字**。
ALPS 量的是「局部結合事件如何重新調諧這個共用頻譜」。

> ⚠️ **本方法不宣稱量子優勢**:此量的古典與量子讀法完全相同,而且在本研究中,
> 明確相干的觀測量表現**較差**。超參數在 tier-A 上選定,tier-B 為留出集。

### 平庸對照組(關鍵)

| 名稱 | 說明 |
|---|---|
| `ctrl_burial` | 接觸度數(埋藏程度)—— 純幾何,無任何異位模型 |
| `ctrl_dist` | 距 anchor 的距離 —— 連圖都不用 |
| `ctrl_random` | 亂數 —— 檢驗判準本身的假陽性率 |

**任何方法若贏不過這三個,就沒有證據價值。**

---

## 3. 評估判準

| 指標 | 說明 |
|---|---|
| `perm_p` | QASC 自己的判準:單尾置換檢定,真異位殘基平均分是否高於 distal 非 anchor 背景 |
| `auc` | 在模型實際選擇的候選池(distal 非 anchor)內的 ROC-AUC |
| `hit5` | top-5 是否命中任一真異位殘基 |
| `dcc` | top-5 質心到真異位位點質心的距離。**STINGAllo 的成功判準是 DCC ≤ 4 Å** —— 這是「有沒有指對地方」,置換 p 值不測這件事 |

所有方法共用完全相同的後處理(rank percentile → QASC 的圖平滑)與候選池,確保公平。

---

## 4. 結果

見 [`RESULTS.md`](RESULTS.md)。
