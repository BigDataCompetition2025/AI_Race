"""
車床熱變位預測模型
目標：基於溫度及變位資訊建立車床熱變位預測模型

輸入特徵 (25 dim):
- 時間 (1 dim): Time
- 車床位置溫度 (21 dim): PT01-PT13 + TC01-TC08
- 控制器擷取溫度 (3 dim): Spindle Motor, X Motor, Z Motor

輸出:
- X軸變位量: Disp. X
- Z軸變位量: Disp. Z

評估方式: RMSE
訓練集:測試集 = 77.03:22.97
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# (Removed hardcoded Windows font to stop findfont warnings on macOS/Linux)
plt.rcParams['axes.unicode_minus'] = False

class ThermalDisplacementPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model_X = None
        self.model_Z = None
        self.feature_columns = None
        self.target_columns = ['Disp. X', 'Disp. Z']
        
    def load_data(self):
        """載入所有CSV檔案並合併"""
        print("開始載入資料...")
        
        # 取得所有CSV檔案路徑
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        print(f"找到 {len(csv_files)} 個CSV檔案")
        
        all_data = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                print(f"載入檔案: {os.path.basename(file_path)}, 資料筆數: {len(df)}")
                all_data.append(df)
            except Exception as e:
                print(f"載入檔案 {file_path} 時發生錯誤: {e}")
        
        # 合併所有資料
        self.data = pd.concat(all_data, ignore_index=True)
        print(f"總資料筆數: {len(self.data)}")
        
        return self.data
    
    def explore_data(self):
        """資料探索分析"""
        print("\n=== 資料探索分析 ===")
        print(f"資料形狀: {self.data.shape}")
        print(f"\n欄位名稱:")
        print(self.data.columns.tolist())
        
        # 檢查缺失值
        print(f"\n缺失值統計:")
        missing_data = self.data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        # 基本統計資訊
        print(f"\n目標變數統計:")
        print(self.data[self.target_columns].describe())
        
        # 繪製目標變數分布圖
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(self.data['Disp. X'], bins=50, alpha=0.7)
        axes[0].set_title('Disp. X Distribution')
        axes[0].set_xlabel('Disp. X')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(self.data['Disp. Z'], bins=50, alpha=0.7)
        axes[1].set_title('Disp. Z Distribution')
        axes[1].set_xlabel('Disp. Z')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('displacement_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.data.describe()
    
    def prepare_features(self):
        """準備特徵和目標變數"""
        print("\n=== 準備特徵資料 ===")
        
        # 定義特徵欄位
        # 時間特徵 (1 dim)
        time_features = ['Time']
        
        # 車床位置溫度 (21 dim)
        pt_features = [f'PT{i:02d}' for i in range(1, 14)]  # PT01-PT13 (13個)
        tc_features = [f'TC{i:02d}' for i in range(1, 9)]   # TC01-TC08 (8個)
        position_temp_features = pt_features + tc_features
        
        # 控制器擷取溫度 (3 dim)
        motor_temp_features = ['Spindle Motor', 'X Motor', 'Z Motor']
        
        # 合併所有輸入特徵
        self.feature_columns = time_features + position_temp_features + motor_temp_features
        
        print(f"輸入特徵數量: {len(self.feature_columns)}")
        print(f"時間特徵 (1): {time_features}")
        print(f"位置溫度特徵 (21): {position_temp_features}")
        print(f"馬達溫度特徵 (3): {motor_temp_features}")
        
        # 檢查特徵是否存在
        missing_features = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_features:
            print(f"警告: 以下特徵在資料中不存在: {missing_features}")
            self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
            print(f"實際可用特徵數量: {len(self.feature_columns)}")
        
        return self.feature_columns
    
    def handle_missing_data(self):
        """處理缺失資料"""
        print("\n=== 處理缺失資料 ===")
        
        # 檢查缺失值
        missing_counts = self.data[self.feature_columns + self.target_columns].isnull().sum()
        print("各欄位缺失值數量:")
        print(missing_counts[missing_counts > 0])
        
        # 使用線性插值填補缺失值
        for col in self.feature_columns + self.target_columns:
            if self.data[col].isnull().sum() > 0:
                print(f"對 {col} 進行線性插值")
                self.data[col] = self.data[col].interpolate(method='linear')
                
                # 如果首尾仍有缺失值，使用前後填補
                self.data[col] = self.data[col].fillna(method='bfill').fillna(method='ffill')
        
        # 再次檢查缺失值
        final_missing = self.data[self.feature_columns + self.target_columns].isnull().sum().sum()
        print(f"處理後剩餘缺失值數量: {final_missing}")
        
        return self.data
    
    def split_data(self, test_size=0.2297):  # 77.03:22.97
        """分割訓練集和測試集"""
        print(f"\n=== 資料分割 ===")
        
        X = self.data[self.feature_columns]
        y = self.data[self.target_columns]
        
        # 分割資料
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"訓練集大小: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.2f}%)")
        print(f"測試集大小: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.2f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type='linear'):
        """訓練模型"""
        print(f"\n=== 訓練模型 ({model_type}) ===")
        
        # 標準化特徵
        X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        
        if model_type == 'linear':
            # 線性回歸模型
            self.model_X = LinearRegression()
            self.model_Z = LinearRegression()
        elif model_type == 'random_forest':
            # 隨機森林模型
            self.model_X = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_Z = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            # 梯度提升模型
            self.model_X = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6,
                random_state=42
            )
            self.model_Z = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6,
                random_state=42
            )
        
        # 分別訓練X軸和Z軸的模型
        print("訓練 X軸變位預測模型...")
        self.model_X.fit(X_train_scaled, self.y_train['Disp. X'])
        
        print("訓練 Z軸變位預測模型...")
        self.model_Z.fit(X_train_scaled, self.y_train['Disp. Z'])
        
        print("模型訓練完成！")
        
        return self.model_X, self.model_Z
    
    def evaluate_model(self):
        """評估模型"""
        print("\n=== 模型評估 ===")
        
        # 標準化測試集
        X_test_scaled = self.scaler_X.transform(self.X_test)
        
        # 預測
        y_pred_X = self.model_X.predict(X_test_scaled)
        y_pred_Z = self.model_Z.predict(X_test_scaled)
        
        # 計算整體RMSE
        rmse_X = np.sqrt(mean_squared_error(self.y_test['Disp. X'], y_pred_X))
        rmse_Z = np.sqrt(mean_squared_error(self.y_test['Disp. Z'], y_pred_Z))
        
        print(f"整體 X軸變位 RMSE: {rmse_X:.6f}")
        print(f"整體 Z軸變位 RMSE: {rmse_Z:.6f}")
        print(f"整體平均 RMSE: {(rmse_X + rmse_Z) / 2:.6f}")
        
        # 計算R²分數
        from sklearn.metrics import r2_score
        r2_X = r2_score(self.y_test['Disp. X'], y_pred_X)
        r2_Z = r2_score(self.y_test['Disp. Z'], y_pred_Z)
        
        print(f"X軸變位 R²: {r2_X:.6f}")
        print(f"Z軸變位 R²: {r2_Z:.6f}")
        
        # 分區RMSE分析
        segmented_results = self.segmented_rmse_analysis(y_pred_X, y_pred_Z)
        
        # 繪製預測結果對比圖
        self.plot_predictions(y_pred_X, y_pred_Z)
        
        return {
            'rmse_X': rmse_X,
            'rmse_Z': rmse_Z,
            'avg_rmse': (rmse_X + rmse_Z) / 2,
            'r2_X': r2_X,
            'r2_Z': r2_Z,
            'segmented_results': segmented_results
        }
    
    def segmented_rmse_analysis(self, y_pred_X, y_pred_Z):
        """分區RMSE分析 - 識別極端情況和不同條件下的模型表現"""
        print("\n=== 分區RMSE分析 ===")
        
        # 創建結果字典
        segmented_results = {}
        
        # 1. 按變位大小分區分析
        print("\n1. 按變位大小分區:")
        segmented_results['displacement_ranges'] = self._analyze_by_displacement_range(y_pred_X, y_pred_Z)
        
        # 2. 按溫度範圍分區分析
        print("\n2. 按主軸馬達溫度分區:")
        segmented_results['temperature_ranges'] = self._analyze_by_temperature_range(y_pred_X, y_pred_Z)
        
        # 3. 按時間分區分析
        print("\n3. 按時間分區:")
        segmented_results['time_ranges'] = self._analyze_by_time_range(y_pred_X, y_pred_Z)
        
        # 4. 極端值分析
        print("\n4. 極端值分析:")
        segmented_results['extreme_cases'] = self._analyze_extreme_cases(y_pred_X, y_pred_Z)
        
        return segmented_results
    
    def _analyze_by_displacement_range(self, y_pred_X, y_pred_Z):
        """按變位大小分區分析"""
        ranges_results = {}
        
        for axis, actual, pred in [('X', self.y_test['Disp. X'], y_pred_X), 
                                   ('Z', self.y_test['Disp. Z'], y_pred_Z)]:
            # 定義變位大小區間
            percentiles = [0, 25, 50, 75, 100]
            range_boundaries = np.percentile(actual, percentiles)
            
            axis_results = []
            for i in range(len(percentiles)-1):
                mask = (actual >= range_boundaries[i]) & (actual < range_boundaries[i+1])
                if i == len(percentiles)-2:  # 最後一個區間包含最大值
                    mask = actual >= range_boundaries[i]
                
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(actual[mask], pred[mask]))
                    count = mask.sum()
                    range_name = f"P{percentiles[i]}-P{percentiles[i+1]}"
                    
                    axis_results.append({
                        'range': range_name,
                        'boundaries': (range_boundaries[i], range_boundaries[i+1]),
                        'count': count,
                        'rmse': rmse
                    })
                    
                    print(f"  {axis}軸 {range_name}: RMSE={rmse:.4f}, 樣本數={count}")
            
            ranges_results[f'{axis}_axis'] = axis_results
        
        return ranges_results
    
    def _analyze_by_temperature_range(self, y_pred_X, y_pred_Z):
        """按溫度範圍分區分析"""
        temp_results = {}
        
        spindle_temp = self.X_test['Spindle Motor']
        
        # 定義溫度區間 (基於四分位數)
        temp_boundaries = np.percentile(spindle_temp, [0, 33, 67, 100])
        temp_labels = ['低溫', '中溫', '高溫']
        
        for axis, actual, pred in [('X', self.y_test['Disp. X'], y_pred_X), 
                                   ('Z', self.y_test['Disp. Z'], y_pred_Z)]:
            axis_results = []
            for i, label in enumerate(temp_labels):
                if i < len(temp_labels) - 1:
                    mask = (spindle_temp >= temp_boundaries[i]) & (spindle_temp < temp_boundaries[i+1])
                else:
                    mask = spindle_temp >= temp_boundaries[i]
                
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(actual[mask], pred[mask]))
                    count = mask.sum()
                    
                    axis_results.append({
                        'temp_range': label,
                        'boundaries': (temp_boundaries[i], temp_boundaries[i+1] if i < len(temp_boundaries)-1 else temp_boundaries[i]),
                        'count': count,
                        'rmse': rmse
                    })
                    
                    print(f"  {axis}軸 {label}區間: RMSE={rmse:.4f}, 樣本數={count}")
            
            temp_results[f'{axis}_axis'] = axis_results
        
        return temp_results
    
    def _analyze_by_time_range(self, y_pred_X, y_pred_Z):
        """按時間分區分析"""
        time_results = {}
        
        time_data = self.X_test['Time']
        
        # 定義時間區間
        time_boundaries = np.percentile(time_data, [0, 25, 50, 75, 100])
        time_labels = ['初期', '前期', '中期', '後期']
        
        for axis, actual, pred in [('X', self.y_test['Disp. X'], y_pred_X), 
                                   ('Z', self.y_test['Disp. Z'], y_pred_Z)]:
            axis_results = []
            for i, label in enumerate(time_labels):
                if i < len(time_labels) - 1:
                    mask = (time_data >= time_boundaries[i]) & (time_data < time_boundaries[i+1])
                else:
                    mask = time_data >= time_boundaries[i]
                
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(actual[mask], pred[mask]))
                    count = mask.sum()
                    
                    axis_results.append({
                        'time_period': label,
                        'boundaries': (time_boundaries[i], time_boundaries[i+1] if i < len(time_boundaries)-1 else time_boundaries[i]),
                        'count': count,
                        'rmse': rmse
                    })
                    
                    print(f"  {axis}軸 {label}: RMSE={rmse:.4f}, 樣本數={count}")
            
            time_results[f'{axis}_axis'] = axis_results
        
        return time_results
    
    def _analyze_extreme_cases(self, y_pred_X, y_pred_Z):
        """極端值分析"""
        extreme_results = {}
        
        for axis, actual, pred in [('X', self.y_test['Disp. X'], y_pred_X), 
                                   ('Z', self.y_test['Disp. Z'], y_pred_Z)]:
            
            # 計算絕對誤差
            abs_errors = np.abs(actual - pred)
            
            # 找出最大誤差的前5%樣本
            error_threshold = np.percentile(abs_errors, 95)
            extreme_mask = abs_errors >= error_threshold
            
            if extreme_mask.sum() > 0:
                extreme_rmse = np.sqrt(mean_squared_error(actual[extreme_mask], pred[extreme_mask]))
                max_error = abs_errors.max()
                extreme_count = extreme_mask.sum()
                
                # 找出極端值的特徵
                extreme_indices = np.where(extreme_mask)[0]
                extreme_temp = self.X_test['Spindle Motor'].iloc[extreme_indices]
                extreme_time = self.X_test['Time'].iloc[extreme_indices]
                
                extreme_results[f'{axis}_axis'] = {
                    'count': extreme_count,
                    'rmse': extreme_rmse,
                    'max_error': max_error,
                    'avg_temp': extreme_temp.mean(),
                    'avg_time': extreme_time.mean(),
                    'error_threshold': error_threshold
                }
                
                print(f"  {axis}軸極端情況 (誤差>P95): RMSE={extreme_rmse:.4f}, 最大誤差={max_error:.4f}, 樣本數={extreme_count}")
                print(f"    極端樣本平均溫度: {extreme_temp.mean():.2f}, 平均時間: {extreme_time.mean():.2f}")
        
        return extreme_results
    
    def plot_predictions(self, y_pred_X, y_pred_Z):
        """繪製預測結果對比圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # X-axis prediction vs actual
        axes[0, 0].scatter(self.y_test['Disp. X'], y_pred_X, alpha=0.5)
        axes[0, 0].plot([self.y_test['Disp. X'].min(), self.y_test['Disp. X'].max()], 
                        [self.y_test['Disp. X'].min(), self.y_test['Disp. X'].max()], 'r--')
        axes[0, 0].set_xlabel('Actual Disp. X')
        axes[0, 0].set_ylabel('Predicted Disp. X')
        axes[0, 0].set_title('Disp. X Prediction')
        
        # Z-axis prediction vs actual
        axes[0, 1].scatter(self.y_test['Disp. Z'], y_pred_Z, alpha=0.5)
        axes[0, 1].plot([self.y_test['Disp. Z'].min(), self.y_test['Disp. Z'].max()], 
                        [self.y_test['Disp. Z'].min(), self.y_test['Disp. Z'].max()], 'r--')
        axes[0, 1].set_xlabel('Actual Disp. Z')
        axes[0, 1].set_ylabel('Predicted Disp. Z')
        axes[0, 1].set_title('Disp. Z Prediction')
        
        # X residuals
        residuals_X = self.y_test['Disp. X'] - y_pred_X
        axes[1, 0].scatter(y_pred_X, residuals_X, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Disp. X')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_title('Disp. X Residuals')
        
        # Z residuals
        residuals_Z = self.y_test['Disp. Z'] - y_pred_Z
        axes[1, 1].scatter(y_pred_Z, residuals_Z, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Disp. Z')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Disp. Z Residuals')
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self):
        """特徵重要性分析"""
        if hasattr(self.model_X, 'feature_importances_'):
            print("\n=== 特徵重要性分析 ===")
            importance_X = self.model_X.feature_importances_
            importance_Z = self.model_Z.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance_X': importance_X,
                'Importance_Z': importance_Z
            })
            feature_importance_df['Avg_Importance'] = (feature_importance_df['Importance_X'] + feature_importance_df['Importance_Z']) / 2
            feature_importance_df = feature_importance_df.sort_values('Avg_Importance', ascending=False)
            print("前15個重要特徵:")
            print(feature_importance_df.head(15))
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)
            x_pos = np.arange(len(top_features))
            plt.barh(x_pos, top_features['Avg_Importance'])
            plt.yticks(x_pos, top_features['Feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            return feature_importance_df

    def plot_segmented_analysis(self, segmented_results):
        """繪製分區分析結果圖表"""
        print("\n=== 生成分區分析圖表 ===")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Segmented RMSE Analysis', fontsize=16)
        self._plot_displacement_ranges(axes[0, 0], segmented_results['displacement_ranges'])
        self._plot_temperature_ranges(axes[0, 1], segmented_results['temperature_ranges'])
        self._plot_time_ranges(axes[0, 2], segmented_results['time_ranges'])
        self._plot_extreme_analysis(axes[1, 0], segmented_results['extreme_cases'])
        self._plot_rmse_heatmap(axes[1, 1], segmented_results)
        self._plot_error_distribution(axes[1, 2])
        plt.tight_layout()
        plt.savefig('segmented_rmse_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_displacement_ranges(self, ax, data):
        """繪製變位大小分區結果"""
        x_ranges = [item['range'] for item in data['X_axis']]
        x_rmse = [item['rmse'] for item in data['X_axis']]
        z_rmse = [item['rmse'] for item in data['Z_axis']]
        
        x = np.arange(len(x_ranges))
        width = 0.35
        
        ax.bar(x - width/2, x_rmse, width, label='X Axis', alpha=0.8)
        ax.bar(x + width/2, z_rmse, width, label='Z Axis', alpha=0.8)
        
        ax.set_xlabel('Displacement Range')
        ax.set_ylabel('RMSE (μm)')
        ax.set_title('RMSE by Displacement Range')
        ax.set_xticks(x)
        ax.set_xticklabels(x_ranges, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_temperature_ranges(self, ax, data):
        """繪製溫度分區結果"""
        temp_ranges = [item['temp_range'] for item in data['X_axis']]
        x_rmse = [item['rmse'] for item in data['X_axis']]
        z_rmse = [item['rmse'] for item in data['Z_axis']]
        
        x = np.arange(len(temp_ranges))
        width = 0.35
        
        ax.bar(x - width/2, x_rmse, width, label='X Axis', alpha=0.8)
        ax.bar(x + width/2, z_rmse, width, label='Z Axis', alpha=0.8)
        
        ax.set_xlabel('Temperature Range')
        ax.set_ylabel('RMSE (μm)')
        ax.set_title('RMSE by Temperature Range')
        ax.set_xticks(x)
        ax.set_xticklabels(temp_ranges)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_time_ranges(self, ax, data):
        """繪製時間分區結果"""
        time_ranges = [item['time_period'] for item in data['X_axis']]
        x_rmse = [item['rmse'] for item in data['X_axis']]
        z_rmse = [item['rmse'] for item in data['Z_axis']]
        
        x = np.arange(len(time_ranges))
        width = 0.35
        
        ax.bar(x - width/2, x_rmse, width, label='X Axis', alpha=0.8)
        ax.bar(x + width/2, z_rmse, width, label='Z Axis', alpha=0.8)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('RMSE (μm)')
        ax.set_title('RMSE by Time Period')
        ax.set_xticks(x)
        ax.set_xticklabels(time_ranges)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_extreme_analysis(self, ax, data):
        """繪製極端值分析"""
        axes_names = ['X Axis', 'Z Axis']
        extreme_rmse = [data['X_axis']['rmse'], data['Z_axis']['rmse']]
        max_errors = [data['X_axis']['max_error'], data['Z_axis']['max_error']]
        
        x = np.arange(len(axes_names))
        width = 0.35
        
        ax.bar(x - width/2, extreme_rmse, width, label='Extreme RMSE', alpha=0.8, color='red')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, max_errors, width, label='Max Error', alpha=0.8, color='orange')
        
        ax.set_xlabel('Axis')
        ax.set_ylabel('Extreme RMSE (μm)', color='red')
        ax2.set_ylabel('Max Error (μm)', color='orange')
        ax.set_title('Extreme Error Analysis (Top 5%)')
        ax.set_xticks(x)
        ax.set_xticklabels(axes_names)
        ax.grid(True, alpha=0.3)
    
    def _plot_rmse_heatmap(self, ax, data):
        """繪製RMSE熱力圖"""
        # 創建熱力圖數據
        conditions = ['P0-P25', 'P25-P50', 'P50-P75', 'P75-P100']
        axes_names = ['X Axis', 'Z Axis']
        
        rmse_matrix = []
        for axis_key in ['X_axis', 'Z_axis']:
            axis_rmse = [item['rmse'] for item in data['displacement_ranges'][axis_key]]
            rmse_matrix.append(axis_rmse)
        
        im = ax.imshow(rmse_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_yticks(np.arange(len(axes_names)))
        ax.set_xticklabels(conditions)
        ax.set_yticklabels(axes_names)
        ax.set_title('RMSE Heatmap')
        
        # 添加數值標籤
        for i in range(len(axes_names)):
            for j in range(len(conditions)):
                ax.text(j, i, f'{rmse_matrix[i][j]:.2f}',
                        ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax, label='RMSE (μm)')
    
    def _plot_error_distribution(self, ax):
        """繪製誤差分布圖"""
        # 這個方法需要預測結果，我們在主要評估中調用
        ax.text(0.5, 0.5, 'Error Distribution\n(Prediction Data Needed)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)

def main():
    """主程式"""
    # 資料路徑
    data_path = r"d:\Github\BigDataCompetition\2025_dataset_0806_x893\2025_dataset_0806\train"
    
    # 建立預測器
    predictor = ThermalDisplacementPredictor(data_path)
    
    # 載入資料
    data = predictor.load_data()
    
    # 資料探索
    predictor.explore_data()
    
    # 準備特徵
    predictor.prepare_features()
    
    # 處理缺失資料
    predictor.handle_missing_data()
    
    # 分割資料
    predictor.split_data()
    
    # 訓練線性回歸模型
    print("\n" + "="*50)
    print("訓練線性回歸模型")
    print("="*50)
    predictor.train_model('linear')
    linear_results = predictor.evaluate_model()
    
    # 訓練隨機森林模型
    print("\n" + "="*50)
    print("訓練隨機森林模型")
    print("="*50)
    predictor.train_model('random_forest')
    rf_results = predictor.evaluate_model()
    
    # 訓練梯度提升模型
    print("\n" + "="*50)
    print("訓練梯度提升模型")
    print("="*50)
    predictor.train_model('gradient_boosting')
    gb_results = predictor.evaluate_model()
    
    # 特徵重要性分析
    predictor.feature_importance_analysis()
    
    # 分區RMSE分析比較
    print("\n" + "="*50)
    print("分區RMSE分析比較")
    print("="*50)
    
    # 重新載入模型進行分區分析
    print("\n🔍 隨機森林模型分區分析:")
    predictor.train_model('random_forest')
    rf_results_detailed = predictor.evaluate_model()
    if 'segmented_results' in rf_results_detailed:
        predictor.plot_segmented_analysis(rf_results_detailed['segmented_results'])
    
    print("\n🔍 梯度提升模型分區分析:")
    predictor.train_model('gradient_boosting')
    gb_results_detailed = predictor.evaluate_model()
    
    print("\n🔍 線性回歸模型分區分析:")
    predictor.train_model('linear')
    linear_results_detailed = predictor.evaluate_model()
    
    # 比較模型結果
    print("\n" + "="*50)
    print("模型比較")
    print("="*50)
    print(f"線性回歸 - 平均RMSE: {linear_results['avg_rmse']:.6f}")
    print(f"隨機森林 - 平均RMSE: {rf_results['avg_rmse']:.6f}")
    print(f"梯度提升 - 平均RMSE: {gb_results['avg_rmse']:.6f}")
    
    # 找出最佳模型
    best_rmse = min(linear_results['avg_rmse'], rf_results['avg_rmse'], gb_results['avg_rmse'])
    if best_rmse == linear_results['avg_rmse']:
        print("線性回歸模型表現最好！")
    elif best_rmse == rf_results['avg_rmse']:
        print("隨機森林模型表現最好！")
    else:
        print("梯度提升模型表現最好！")

if __name__ == "__main__":
    main()
