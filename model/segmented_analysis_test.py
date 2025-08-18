"""
分區RMSE分析測試程式
專門測試三種模型的分區性能，識別極端情況
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """載入並準備資料"""
    data_path = r"d:\Github\BigDataCompetition\2025_dataset_0806_x893\2025_dataset_0806\train"
    
    # 載入資料
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    all_data = []
    
    for file_path in csv_files[:10]:  # 只用前10個檔案快速測試
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except:
            pass
    
    data = pd.concat(all_data, ignore_index=True)
    
    # 準備特徵
    time_features = ['Time']
    pt_features = [f'PT{i:02d}' for i in range(1, 14)]
    tc_features = [f'TC{i:02d}' for i in range(1, 9)]
    motor_temp_features = ['Spindle Motor', 'X Motor', 'Z Motor']
    
    feature_columns = time_features + pt_features + tc_features + motor_temp_features
    feature_columns = [col for col in feature_columns if col in data.columns]
    target_columns = ['Disp. X', 'Disp. Z']
    
    # 處理缺失值
    for col in feature_columns + target_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    return data, feature_columns, target_columns

def segmented_rmse_analysis(X_test, y_test, y_pred_X, y_pred_Z, model_name):
    """分區RMSE分析"""
    print(f"\n=== {model_name} 分區RMSE分析 ===")
    
    results = {}
    
    # 1. 按變位大小分區
    print("\n1. 按變位大小分區:")
    for axis, actual, pred in [('X', y_test['Disp. X'], y_pred_X), 
                               ('Z', y_test['Disp. Z'], y_pred_Z)]:
        percentiles = [0, 25, 50, 75, 100]
        range_boundaries = np.percentile(actual, percentiles)
        
        for i in range(len(percentiles)-1):
            mask = (actual >= range_boundaries[i]) & (actual < range_boundaries[i+1])
            if i == len(percentiles)-2:
                mask = actual >= range_boundaries[i]
            
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(actual[mask], pred[mask]))
                count = mask.sum()
                range_name = f"P{percentiles[i]}-P{percentiles[i+1]}"
                print(f"  {axis}軸 {range_name}: RMSE={rmse:.4f} μm, 樣本數={count}")
    
    # 2. 按溫度分區
    print("\n2. 按主軸馬達溫度分區:")
    spindle_temp = X_test['Spindle Motor']
    temp_boundaries = np.percentile(spindle_temp, [0, 33, 67, 100])
    temp_labels = ['低溫', '中溫', '高溫']
    
    for axis, actual, pred in [('X', y_test['Disp. X'], y_pred_X), 
                               ('Z', y_test['Disp. Z'], y_pred_Z)]:
        for i, label in enumerate(temp_labels):
            if i < len(temp_labels) - 1:
                mask = (spindle_temp >= temp_boundaries[i]) & (spindle_temp < temp_boundaries[i+1])
            else:
                mask = spindle_temp >= temp_boundaries[i]
            
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(actual[mask], pred[mask]))
                count = mask.sum()
                print(f"  {axis}軸 {label}({temp_boundaries[i]:.1f}°C): RMSE={rmse:.4f} μm, 樣本數={count}")
    
    # 3. 極端值分析
    print("\n3. 極端值分析 (前5%誤差):")
    for axis, actual, pred in [('X', y_test['Disp. X'], y_pred_X), 
                               ('Z', y_test['Disp. Z'], y_pred_Z)]:
        abs_errors = np.abs(actual - pred)
        error_threshold = np.percentile(abs_errors, 95)
        extreme_mask = abs_errors >= error_threshold
        
        if extreme_mask.sum() > 0:
            extreme_rmse = np.sqrt(mean_squared_error(actual[extreme_mask], pred[extreme_mask]))
            max_error = abs_errors.max()
            extreme_count = extreme_mask.sum()
            
            print(f"  {axis}軸極端情況: RMSE={extreme_rmse:.4f} μm, 最大誤差={max_error:.4f} μm, 樣本數={extreme_count}")
    
    return results

def compare_models_segmented():
    """比較三種模型的分區性能"""
    print("🔬 載入資料...")
    data, feature_columns, target_columns = load_and_prepare_data()
    
    print(f"資料大小: {data.shape}")
    print(f"特徵數量: {len(feature_columns)}")
    
    # 分割資料
    X = data[feature_columns]
    y = data[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        '線性回歸': {
            'X': LinearRegression(),
            'Z': LinearRegression()
        },
        '隨機森林': {
            'X': RandomForestRegressor(n_estimators=50, random_state=42),
            'Z': RandomForestRegressor(n_estimators=50, random_state=42)
        },
        '梯度提升': {
            'X': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Z': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
    }
    
    all_results = {}
    
    for model_name, model_dict in models.items():
        print(f"\n{'='*50}")
        print(f"訓練 {model_name} 模型")
        print(f"{'='*50}")
        
        # 訓練模型
        model_dict['X'].fit(X_train_scaled, y_train['Disp. X'])
        model_dict['Z'].fit(X_train_scaled, y_train['Disp. Z'])
        
        # 預測
        y_pred_X = model_dict['X'].predict(X_test_scaled)
        y_pred_Z = model_dict['Z'].predict(X_test_scaled)
        
        # 整體性能
        rmse_X = np.sqrt(mean_squared_error(y_test['Disp. X'], y_pred_X))
        rmse_Z = np.sqrt(mean_squared_error(y_test['Disp. Z'], y_pred_Z))
        r2_X = r2_score(y_test['Disp. X'], y_pred_X)
        r2_Z = r2_score(y_test['Disp. Z'], y_pred_Z)
        
        print(f"\n整體性能:")
        print(f"X軸 RMSE: {rmse_X:.4f} μm, R²: {r2_X:.4f}")
        print(f"Z軸 RMSE: {rmse_Z:.4f} μm, R²: {r2_Z:.4f}")
        print(f"平均 RMSE: {(rmse_X + rmse_Z)/2:.4f} μm")
        
        # 分區分析
        segmented_results = segmented_rmse_analysis(X_test, y_test, y_pred_X, y_pred_Z, model_name)
        
        all_results[model_name] = {
            'rmse_X': rmse_X,
            'rmse_Z': rmse_Z,
            'avg_rmse': (rmse_X + rmse_Z)/2,
            'r2_X': r2_X,
            'r2_Z': r2_Z,
            'segmented': segmented_results
        }
    
    # 總結比較
    print(f"\n{'='*50}")
    print("模型總結比較")
    print(f"{'='*50}")
    
    print(f"{'模型':<10} {'平均RMSE(μm)':<15} {'X軸RMSE(μm)':<15} {'Z軸RMSE(μm)':<15}")
    print("-" * 60)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<10} {results['avg_rmse']:<15.4f} {results['rmse_X']:<15.4f} {results['rmse_Z']:<15.4f}")
    
    return all_results

if __name__ == "__main__":
    results = compare_models_segmented()
