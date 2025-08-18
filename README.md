# Thermal Displacement Prediction – Multi-Strategy Evaluation

本專案針對車床熱變位資料，系統性比較三種「檔案層級 (file-level)」資料切分策略 (Time / Environment / Few-Shot)，並在每種策略下評估三個模型 (Linear / RandomForest / GradientBoosting)。同時遵守 6H 檔案僅取前 5 小時 (Time ≤ 300) 的規則，產生兩組結果：

1. 原始截斷資料 (`truncated_data`)
2. 5H 強制版本 (`truncated_data_5h`)

> 早期 <7 的 RMSE 係來自「逐列 (row-level) 隨機拆分」造成資料洩漏；改為檔案層級切分後 RMSE 合理上升且更貼近真實泛化難度。Few-Shot 線性模型在嚴格條件下仍為最佳策略，後續優化將以其為基線。

## 📌 三種資料切分策略概要
| 策略 | 定義 | 目的 / 模擬情境 | 風險控制 | 訓練資料覆蓋度 |
|------|------|----------------|-----------|----------------|
| Time-based | 依日期排序前 70% 訓練 / 後 30% 測試 | 未來時序預測 (temporal drift) | 避免時間洩漏 | 中 |
| Environment-based | 恆溫 → 訓練；非恆溫 → 測試 (不足則 fallback 時序) | Domain shift (穩定→變動) | 檔案嚴格分離 | 低~中 |
| Few-Shot (K=1) | 每 (控溫+溫度) 組合抽 1 檔訓練，其餘測試 | 極低標訓資料跨條件泛化 | 最嚴格 | 極低 |

## 🧪 Time vs Environment 策略 – 三模型 RMSE 比較
指標：RMSE_X / RMSE_Z / AVG = (RMSE_X + RMSE_Z)/2 (單位 μm)。

### 原始截斷資料
| Strategy | Model | RMSE_X | RMSE_Z | AVG |
|----------|-------|--------|--------|------|
| time_split | linear | 17.8671 | 6.2093 | 12.0382 |
| time_split | random_forest | 23.1642 | 10.5158 | 16.8400 |
| time_split | gradient_boosting | 23.4623 | 9.6576 | 16.5599 |
| environment_split | linear | 21.6738 | 5.4725 | 13.5732 |
| environment_split | random_forest | 17.9254 | 11.5829 | 14.7542 |
| environment_split | gradient_boosting | 16.0042 | 10.0335 | 13.0188 |

### 5H 強制版本
| Strategy | Model | RMSE_X | RMSE_Z | AVG |
|----------|-------|--------|--------|------|
| time_split | linear | 17.5986 | 5.7030 | 11.6508 |
| time_split | random_forest | 23.0064 | 9.9311 | 16.4688 |
| time_split | gradient_boosting | 24.0387 | 8.4701 | 16.2544 |
| environment_split | linear | 20.9801 | 5.2881 | 13.1341 |
| environment_split | random_forest | 18.1799 | 10.3497 | 14.2648 |
| environment_split | gradient_boosting | 16.3975 | 9.5256 | 12.9615 |

### 觀察重點
1. 線性模型在兩策略均優於樹模型：資料量與 domain shift 下樹模型較易過擬合雜訊。
2. Environment split：Z 軸 RMSE 低但 X 軸高 → domain shift 對 X 軸更敏感。
3. 5H 截斷：Time+Linear 平均 RMSE 改善 (12.04 → 11.65)；顯示前段訊號更線性化。
4. Gradient Boosting 在 Environment 略優 RF，但仍不敵 Linear → 需後續調參 & 特徵增強。
5. Few-Shot (未列於此表) 線性最佳 AVG 約 9.90 → 9.36 (5H)；即使訓練資料極少仍領先，為推薦主線。

## ✅ 為何選擇 Few-Shot + Linear 作為主線基線
1. 嚴格一般化評估：最少訓練檔案的真實風險視圖。
2. 解釋性：係數直接對應溫度/時間，利於物理討論。
3. 穩健性：在不利資料情境仍維持最低 AVG RMSE。
4. 擴展性：可延伸 K>1 曲線、條件自適應、meta-learning。
5. 與樹模型對比：凸顯資料與特徵尚未支援複雜模型優勢 → 指引後續特徵工程方向。

## 🔍 為何 RMSE 高於早期 <7
| 因素 | 說明 | 影響 |
|------|------|------|
| Row-level 隨機拆分 | 同檔案序列被打散 (洩漏鄰近結構) | 人工偏低 |
| File-level 切分 | 序列整檔分離 | 泛化難度回歸真實 |
| Few-Shot | 訓練覆蓋極低 | 難度↑ |
| 6H→5H 截斷 | 保留較早期非穩態段 | 非線性↑ |
| 未調參 | RF / GBDT 預設參數 | 尚未最佳化 |
| X 軸本質較難 | 熱變位受多重耦合 | 平均被拉高 |

## 🧱 核心程式元件 (概述)
| 檔案 | 角色 | 重點 |
|------|------|------|
| `thermal_displacement_prediction.py` | 建模管線 | 特徵擷取、插值補值、X/Z 分開訓練、RMSE+R²、特徵重要性 |
| `experiment_splitting_comparison.ipynb` | 實驗主控 | 實作三切分策略、重新訓練多模型、結果彙整、5H 重跑、比較表 |
| `results_summary*.csv` | 結果匯出 | 各策略×模型×資料版本 RMSE / R² |
| `best_per_strategy*.csv` | 精簡摘要 | 每策略最佳模型 (依 AVG RMSE) |
| `overall_best*.json` | 全域最佳 | Few-Shot + Linear 基線記錄 |
| `five_hour_truncation_report.csv` | 截斷稽核 | 確認所有 6H 檔案裁為 5H |

## 🚩 後續 Roadmap (優先序)
1. Few-Shot K 擴展：K=1→2→3 （觀察 learning curve）
2. 特徵工程：溫度梯度 / 移動統計 / 穩態偵測標籤 / 動態交互項
3. 樹模型調參：RF (max_features, min_samples_leaf), GBDT (learning_rate, subsample)
4. 正規化實驗：Ridge / ElasticNet 對比 + 統一早期 baseline 導入
5. Anomaly handling：序列尖峰平滑 / sensor drift 檢測
6. Segmented RMSE 深化：早期加熱 vs 穩態 vs 降溫 分段化特徵
7. 不確定性：預測區間 (Bootstrap / Quantile Regression)
8. 模型解釋：Permutation importance + SHAP (抽樣) / 物理對照
9. 自動化：CLI + config + 批次紀錄 (MLflow / Weights & Biases 可選)
10. 部署前檢查：資料漂移監控原型 (KS / PSI)

## 📂 目錄結構（精簡）
```
AI_Race/
  ├─ experiment_splitting_comparison.ipynb   # 多策略實驗主 Notebook
  ├─ thermal_displacement_prediction.py      # 建模與評估管線
  ├─ results_summary.csv / _5h               # 原始 & 5H 結果
  ├─ best_per_strategy.csv / _5h             # 每策略最佳紀錄
  ├─ overall_best.json / _5h                 # 全域最佳策略模型
  ├─ five_hour_truncation_report.csv         # 截斷稽核
  ├─ 2025_dataset_0806 3/                    # 原始資料 (git ignore)
  └─ truncated_data_5h/                      # 5H 截斷後資料 (git ignore)
```

## 🛠️ 快速再現 (建議流程)
1. 建立/啟動虛擬環境 + 安裝 `requirements.txt`
2. 生成 (或下載) 截斷資料 / 5H 資料
3. 開啟 `experiment_splitting_comparison.ipynb` 全部執行
4. 查看 `results_summary_5h.csv` 與 `best_per_strategy_5h.csv`
5. 若新增特徵 / 模型 → 更新管線腳本並重跑 Notebook

## 🔐 資料與版本控管原則
- 原始大量 CSV 不入庫；僅保留程式 + 文檔 + 結果摘要。
- 匿名化 / 移除敏感欄位後方可外部分享。
- 可提供「小型裁剪示例」協助外部說明。

## 📎 需求延伸建議
- 若需發表 / 內部簡報：建議將 Few-Shot vs Time vs Environment 置於主圖表；洩漏說明附錄。
- 若需自動化：轉換 splitting + training 為參數化 CLI (`python run_experiment.py --split few_shot --dataset 5h`).
- 若需研究報告：補齊 learning curve、特徵歸因、domain shift 分析 (溫控條件差異統計)。

---

# (Legacy) AI_Race Baseline – ElasticNet vs BayesianRidge

以下為最初雙線性基線說明，僅保留歷史背景，不再更新。

## 舊基線結果摘要
- BayesianRidge: RMSE = 8.057
- ElasticNet: RMSE = 8.089 (差距 0.032)

> 該結果因採行 row-level 隨機拆分存在資訊洩漏，與現行 file-level 策略不可直接比較。

## 原始內容 (節錄)

（下列章節保留原指令、增強建議與操作指南）

## 目錄結構
```
AI_Race/
  ├─ learn.ipynb                    # 主要 Notebook：雙模型訓練與比較
  ├─ requirements.txt               # Python 套件依賴
  ├─ README.md                      # 專案說明文件
  ├─ .gitignore                     # 忽略大型資料與輸出
  ├─ 2025_dataset_0806 3/           # (已被忽略) 原始訓練資料
  │   └─ train/                     # 43 個訓練 CSV 檔案
  ├─ preds_out_elasticnet/          # (已被忽略) ElasticNet 預測結果
  ├─ preds_out_bayesian/            # (已被忽略) BayesianRidge 預測結果
  └─ preds_out/                     # (已被忽略) 舊版預測輸出 (已棄用)
```

## 環境安裝
```zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Notebook 功能 (learn.ipynb)

### 📊 核心功能
1. 資料處理 / 多檔合併
2. 特徵工程：自動篩選可數值化欄位
3. 雙模型訓練：ElasticNetCV / BayesianRidge
4. 客製 RMSE：僅第 101 列以後
5. 分離輸出與檔案級 RMSE 分解

### 📈 評估與診斷
- 每檔案 RMSE、勝負統計、係數與特徵重要性、視覺化、異常檢查

## 🚀 執行步驟 (Legacy)
略（與現行流程重疊）。

## 🔬 增強方向 (原始列舉)
- 時間序列特徵、溫度梯度、階段識別、Gradient Boosting / RF、深度學習、交叉驗證、漂移偵測、解釋性工具等。

## 💻 常用指令 / Git / 環境管理
仍適用現行專案，詳見上文或保留段落。

## 📋 實驗記錄 (Legacy)
- Version 1.0 (2025-01-17): BayesianRidge 8.057 / ElasticNet 8.089

---

End of README

## 目錄結構
```
AI_Race/
  ├─ learn.ipynb                    # 主要 Notebook：雙模型訓練與比較
  ├─ requirements.txt               # Python 套件依賴
  ├─ README.md                      # 專案說明文件
  ├─ .gitignore                     # 忽略大型資料與輸出
  ├─ 2025_dataset_0806 3/           # (已被忽略) 原始訓練資料
  │   └─ train/                     # 43 個訓練 CSV 檔案
  ├─ preds_out_elasticnet/          # (已被忽略) ElasticNet 預測結果
  ├─ preds_out_bayesian/            # (已被忽略) BayesianRidge 預測結果
  └─ preds_out/                     # (已被忽略) 舊版預測輸出 (已棄用)
```

> 原始資料與大量輸出檔案不進版本控制，避免 repo 過大與權限問題；若需共用，請用雲端/壓縮檔管道。

## 環境安裝
```zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Notebook 功能 (learn.ipynb)

### 📊 核心功能
1. **資料處理**: 合併多個訓練 CSV 建立單一 DataFrame
2. **特徵工程**: 自動篩選高有效數值特徵 (避開無法轉數值欄)
3. **雙模型訓練**: 
   - 🔵 **ElasticNetCV**: 交叉驗證調整 `alpha` 與 `l1_ratio`
   - 🟣 **BayesianRidge**: 不確定性友善的線性回歸
4. **競賽評估**: 自訂 RMSE - 僅計算每檔案第 101 列以後的預測誤差
5. **分離輸出**: 兩個演算法獨立儲存預測結果便於比較
6. **品質檢驗**: 預測值範圍檢查、異常值偵測、統計分析

### 📈 評估與診斷
- **每檔案 RMSE 分解**: 43 個檔案個別精準度分析
- **勝負統計**: 逐檔比較兩演算法表現
- **特徵重要性**: 係數排序與影響力分析  
- **視覺化圖表**: RMSE 分布、準確度分級、預測差異比較
- **預測品質檢查**: 自動偵測異常預測值

## 🚀 執行步驟

### 1️⃣ 環境設定
```zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ 資料準備
- 將 43 個訓練 CSV 檔案放入 `2025_dataset_0806 3/train/`
- 確認檔案命名格式符合預期 (如: `_20200615_GV1-1203_....csv`)

### 3️⃣ 執行 Notebook (learn.ipynb)
**按順序執行以下 Cell 群組:**

#### 🔵 ElasticNet 訓練 (Cell 3-4)
- Cell 3: 載入資料 → 特徵選擇 → 訓練 ElasticNet → 顯示 RMSE
- Cell 4: 產生預測檔案並輸出到 `preds_out_elasticnet/`

#### 🟣 BayesianRidge 訓練 (Cell 8)  
- Cell 8: 訓練 BayesianRidge → 產生預測檔案到 `preds_out_bayesian/`

#### 📊 模型比較分析 (Cell 13-15)
- Cell 13: 雙模型 RMSE 比較 → 勝負統計 → 檔案級別分析
- Cell 14: 綜合準確度評估 → 視覺化圖表 → 改進建議  
- Cell 15: 預測品質檢查 → 異常值偵測 → 預測差異比較

### 4️⃣ 結果選擇
根據評估結果，選擇表現較佳的演算法進行最終提交：
- **建議**: 使用 `preds_out_bayesian/` (RMSE 較低)
- **備選**: 使用 `preds_out_elasticnet/` (差距微小)

## 📂 資料管理策略

### 忽略原則
- **大型資料**: `2025_dataset_0806 3/` 整個資料夾已被 `.gitignore` 忽略
- **預測輸出**: `preds_out_*` 所有預測結果資料夾均被忽略
- **權限考量**: 避免敏感資料進入版本控制

### 分享策略
若需分享 **小型示例資料**：
```zsh
mkdir data_samples
# 複製 1-2 個裁剪後的小檔案
echo "!data_samples/" >> .gitignore
echo "!data_samples/*.csv" >> .gitignore
```

### 雲端備份
建議使用雲端服務備份完整資料：
- Google Drive / OneDrive (小型專案)
- AWS S3 / Azure Blob (大型專案)
- 內部檔案系統 (企業環境)

## 🔬 增強方向 (後續可做)

### 🛠️ 特徵工程
- **時間序列特徵**: 移動平均、差分、滾動統計、週期性特徵
- **溫度關係**: 相對溫度、溫度梯度、熱擴散模式
- **階段識別**: 加工階段 one-hot 編碼、轉速變化偵測
- **加速度特徵**: 位移變化率、振動頻率分析

### 🤖 進階模型
- **Gradient Boosting**: XGBoost、LightGBM、CatBoost
- **樹模型**: Random Forest、Extra Trees
- **深度學習**: LSTM、Transformer、CNN-LSTM
- **模型融合**: Stacking、Blending、Voting

### 📊 評估改進
- **交叉驗證**: GroupKFold (以檔名分層避免資料洩漏)
- **時間序列驗證**: TimeSeriesSplit、滑動視窗驗證
- **多評估指標**: MAE、MAPE、R²、方向準確度

### 🔧 工程化
- **管線自動化**: sklearn Pipeline + Custom Transformers
- **配置管理**: YAML/JSON 參數檔案
- **命令列介面**: Click/argparse CLI 工具
- **容器化**: Docker + 可複現環境

### 📈 監控與診斷
- **模型解釋性**: SHAP、LIME、Permutation Importance
- **預測穩定性**: 預測區間、不確定性量化
- **資料漂移**: 分布變化監控、特徵重要性變化

## 💻 常用指令

### Git 操作
```zsh
# 檢查狀態
git status

# 提交變更 (僅程式碼和文件)
git add learn.ipynb README.md requirements.txt .gitignore
git commit -m "feat: implement dual-algorithm comparison"

# 推送到遠端
git push -u origin main
```

### 環境管理
```zsh
# 啟動虛擬環境
source .venv/bin/activate

# 更新套件
pip install --upgrade -r requirements.txt

# 匯出環境 (如有新增套件)
pip freeze > requirements.txt
```

### 快速驗證
```zsh
# 檢查預測檔案數量
ls preds_out_elasticnet/*.csv | wc -l    # 應為 43
ls preds_out_bayesian/*.csv | wc -l      # 應為 43

# 檢查檔案大小 (確認非空)
du -sh preds_out_*
```

## 🔒 權限與安全

### 資料安全
- ❌ **禁止**: 將完整原始資料推入 Git
- ❌ **禁止**: 包含敏感機台資訊的檔案
- ✅ **允許**: 程式碼、README、小型示例資料

### 分享原則
- **內部**: 可分享程式碼與結果摘要
- **外部**: 需經過資料脫敏與授權確認
- **雲端**: 使用加密傳輸與存取控制

## 📋 實驗記錄

### Version 1.0 (當前)
- **日期**: 2025-01-17
- **ElasticNet RMSE**: 8.089
- **BayesianRidge RMSE**: 8.057 ⭐
- **勝負**: BayesianRidge 23檔 vs ElasticNet 20檔
- **特徵數**: 25/25 (完整特徵集)

### 預期改進目標
- **短期**: RMSE < 7.5 (透過特徵工程)
- **中期**: RMSE < 7.0 (透過進階模型)
- **長期**: RMSE < 6.5 (透過模型融合)
