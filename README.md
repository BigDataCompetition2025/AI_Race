# AI_Race Baseline

簡單競賽起始版本：提供資料夾結構、環境安裝、以及兩個線性基線模型 (ElasticNet / Bayesian Ridge) 的使用方式，便於快速驗證流程與後續擴充。

## 目錄結構 (精簡)
```
AI_Race/
  ├─ learn.ipynb              # Notebook：資料讀取 + ElasticNet + BayesianRidge + 診斷
  ├─ requirements.txt         # 主要 Python 套件
  ├─ README.md
  ├─ .gitignore               # 忽略大型原始資料與輸出
  ├─ 2025_dataset_0806 3/     # (已被忽略) 原始訓練 CSV 大量檔案資料夾
  └─ preds_out/               # (已被忽略) 預測輸出資料夾
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
1. 合併多個訓練 CSV 建立單一 DataFrame。
2. 自動篩選高有效數值特徵 (避開無法轉數值欄)。
3. ElasticNetCV：交叉驗證調整 `alpha` 與 `l1_ratio`，多輸出 (Disp. X, Disp. Z)。
4. 自訂競賽 RMSE：僅計算每檔案第 101 列以後的預測誤差。
5. 產出預測檔：覆寫第 101 列以後的目標欄位。
6. Diagnostic：每檔案 RMSE、重要係數排序、訓練耗時計時。
7. BayesianRidge 範例：展示不確定性友善的線性模型替代方案。

## 執行步驟 (ElasticNet baseline)
在 `learn.ipynb` 內依序執行：
- 前幾個 cell：載入資料 → 建立特徵/目標 → 訓練 ElasticNet → 顯示 pseudo-eval RMSE。
- Debug/提交 cell：產生預測並輸出到 `preds_out/` (被忽略)。
- Diagnostic cell：檢視各檔案 RMSE 與特徵重要性。

## 資料放置與忽略策略
- 將所有原始 CSV 放入 `2025_dataset_0806 3/train/` (Notebook 內的 `TRAIN_DIR`)。
- `.gitignore` 已忽略整個資料夾與 `preds_out/`。
- 若需分享 **小型示例**：可新建 `data_samples/` 放 1~2 個裁剪後的小檔，並在 `.gitignore` 末尾加：
  ```
  !data_samples/
  !data_samples/*.csv
  ```

## 增強方向 (後續可做)
- 特徵工程：時間窗口移動平均、差分、溫度相對值、階段 one-hot、加速度 / 斜率。
- 模型：Gradient Boosting (XGBoost / LightGBM / CatBoost)、Random Forest、Neural Networks。
- Cross-file Validation：以檔名分層的 GroupKFold，避免洩漏同檔案時間序列。
- 資料品質：缺失值統一填補策略 (例如以歷史中位數)、異常值剪裁 (winsorize)。
- 管線化：將特徵生成 → 標準化 → 模型 → 評估寫成 Python 模組與 CLI。

## 指令摘要 (常用)
```zsh
# 重新產出預測檔 (在 notebook 執行後，如需命令列形式可改裝為腳本)
# 目前流程主要在 learn.ipynb

# Git 推送
git add README.md
git commit -m "docs: add README"
# 若已改分支為 main：
git push -u origin main
# 若仍是 master：
# git push -u origin master
```

## 權限 / 安全
- 請勿將帶有客戶 / 機台敏感資訊的完整原始資料推入 Git。
- 若需共享大型資料，可用：雲端硬碟、S3、或壓縮後透過公司內部檔案系統。

## 授權
內部競賽 / 研究使用。若需對外開源再補授權條款。

---
歡迎依需求擴充；可先在 README 新開 Features / Changelog 區段跟進進度。
