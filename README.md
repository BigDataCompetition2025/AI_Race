# AI_Race Baseline

競賽機器學習基線版本：提供資料夾結構、環境安裝、以及兩個線性基線模型 (ElasticNet / BayesianRidge) 的完整比較方案，便於快速驗證流程與精準度評估。

## 🏆 演算法比較結果

根據當前實驗結果：
- **🟣 BayesianRidge**: RMSE = 8.057 (勝出)
- **🔵 ElasticNet**: RMSE = 8.089 
- **差距**: 0.032 (BayesianRidge 略佳)

兩個演算法預測結果相近且都在良好範圍內 (< 10)，可依競賽策略選擇提交。

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

## 📞 支援與貢獻

### 問題回報
1. 檢查 Cell 執行順序是否正確
2. 確認資料路徑與檔案完整性
3. 查看 Notebook 輸出中的錯誤訊息
4. 提供完整的錯誤堆疊資訊

### 貢獻方式
1. Fork 專案到個人帳號
2. 創建功能分支 (`git checkout -b feature/new-model`)
3. 提交變更並推送
4. 發起 Pull Request

---

## 📄 授權
內部競賽/研究使用。如需對外開源請補充授權條款。

**🎯 祝競賽順利！如有問題歡迎討論。**
