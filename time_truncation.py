# === 競賽資料時間截斷處理 ===
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def truncate_overtime_data(csv_file_path, condition_map):
    """
    根據加工條件截斷超時資料
    
    Parameters:
    - csv_file_path: CSV 檔案路徑
    - condition_map: 加工條件對照表
    
    Returns:
    - truncated_df: 截斷後的 DataFrame
    - truncation_info: 截斷資訊
    """
    
    # 載入原始資料
    df = pd.read_csv(csv_file_path, low_memory=False)
    filename = os.path.basename(csv_file_path)
    
    try:
        # 從檔名提取日期
        date_str = filename.split('_')[1]
        condition = condition_map.get(date_str, {})
        
        if not condition:
            print(f"⚠️ 找不到 {filename} 的加工條件，跳過截斷")
            return df, {'truncated': False, 'reason': 'no_condition_info'}
        
        # 計算總加工時間
        total_hours = 0
        stage_times = []
        
        # 第一段時間
        if 'time1' in condition or pd.notna(condition.get('time1', np.nan)):
            stage1_time = condition.get('time1', 0)
            stage_times.append(stage1_time)
            total_hours += stage1_time
        
        # 第二段時間  
        if 'time2' in condition or pd.notna(condition.get('time2', np.nan)):
            stage2_time = condition.get('time2', 0)
            stage_times.append(stage2_time)
            total_hours += stage2_time
            
        # 第三段時間
        if 'time3' in condition or pd.notna(condition.get('time3', np.nan)):
            stage3_time = condition.get('time3', 0)
            stage_times.append(stage3_time)
            total_hours += stage3_time
        
        # 如果無法取得時間資訊，使用預設值
        if total_hours == 0:
            # 從加工條件表手動查詢
            total_hours = estimate_total_time_from_filename(filename)
        
        print(f"📋 {filename}: 計劃加工時間 = {total_hours} 小時")
        
        # 假設取樣頻率 (需要根據實際資料調整)
        if 'Time' in df.columns:
            # 使用 Time 欄位計算
            time_values = pd.to_numeric(df['Time'], errors='coerce')
            max_time_seconds = total_hours * 3600  # 轉換為秒
            
            # 找到截斷點
            valid_mask = time_values <= max_time_seconds
            truncate_index = valid_mask.sum()
            
        else:
            # 使用行數估算 (假設固定取樣頻率)
            sampling_rate_per_hour = estimate_sampling_rate(df, total_hours)
            expected_rows = int(total_hours * sampling_rate_per_hour)
            truncate_index = min(expected_rows, len(df))
        
        # 執行截斷
        if truncate_index < len(df):
            truncated_df = df.iloc[:truncate_index].copy()
            truncation_info = {
                'truncated': True,
                'original_rows': len(df),
                'truncated_rows': truncate_index,
                'removed_rows': len(df) - truncate_index,
                'planned_hours': total_hours,
                'truncate_reason': 'overtime_data_removal'
            }
            print(f"✂️  截斷 {filename}: {len(df)} → {truncate_index} 行 (移除 {len(df) - truncate_index} 行)")
        else:
            truncated_df = df.copy()
            truncation_info = {
                'truncated': False,
                'original_rows': len(df),
                'planned_hours': total_hours,
                'truncate_reason': 'no_overtime_data'
            }
            print(f"✅ {filename}: 無需截斷 ({len(df)} 行)")
        
        return truncated_df, truncation_info
        
    except Exception as e:
        print(f"❌ 處理 {filename} 時發生錯誤: {e}")
        return df, {'truncated': False, 'reason': f'error: {e}'}

def estimate_total_time_from_filename(filename):
    """從檔名推估總加工時間"""
    
    # 手動對照表 (根據檔案環境設定總表)
    time_mapping = {
        '20200615': 5.0,   # 2000rpm, 5H
        '20200616': 5.0,   # 1000rpm, 5H  
        '20200617': 5.0,   # 1000rpm 2.5H + 2000rpm 2.5H
        '20200618': 5.0,   # 2000rpm 2.5H + 1000rpm 2.5H
        '20200701': 5.0,   # 2000rpm, 5H
        '20200702': 5.0,   # 1000rpm, 5H
        '20200703': 5.0,   # 1000rpm 2.5H + 2000rpm 2.5H
        '20200706': 5.0,   # 2000rpm 2.5H + 1000rpm 2.5H
        '20200708': 6.0,   # 2000rpm, 6H (變溫)
        '20200709': 6.0,   # 1000rpm, 6H (變溫)
        '20200710': 6.0,   # 1000rpm 3H + 2000rpm 3H (變溫)
        '20200713': 6.0,   # 2000rpm 3H + 1000rpm 3H (變溫)
        '20200715': 6.0,   # 1800rpm 2.5H + 停機 1H + 1200rpm 2.5H
        # ... 其他日期
    }
    
    date_part = filename.split('_')[1] if '_' in filename else ''
    return time_mapping.get(date_part, 5.0)  # 預設 5 小時

def estimate_sampling_rate(df, total_hours):
    """估算取樣頻率"""
    
    # 方法1: 使用 Time 欄位
    if 'Time' in df.columns:
        time_values = pd.to_numeric(df['Time'], errors='coerce').dropna()
        if len(time_values) > 1:
            max_time_seconds = time_values.max()
            sampling_rate = len(time_values) / (max_time_seconds / 3600)  # 每小時樣本數
            return sampling_rate
    
    # 方法2: 假設固定取樣頻率
    # 根據現有資料推估 (例如: 每分鐘1筆)
    estimated_rate_per_hour = 60  # 每小時60筆
    return estimated_rate_per_hour

def batch_truncate_files(train_dir, condition_map, output_dir=None):
    """批次處理所有檔案的時間截斷"""
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    truncation_summary = []
    
    print(f"=== 📂 批次截斷處理: {len(train_files)} 個檔案 ===")
    
    for file_path in train_files:
        filename = os.path.basename(file_path)
        
        # 執行截斷
        truncated_df, info = truncate_overtime_data(file_path, condition_map)
        
        # 儲存處理後的檔案
        if output_dir:
            output_path = os.path.join(output_dir, filename)
            truncated_df.to_csv(output_path, index=False)
        
        # 記錄處理結果
        info['filename'] = filename
        truncation_summary.append(info)
    
    # 顯示處理摘要
    summary_df = pd.DataFrame(truncation_summary)
    
    print(f"\n=== 📊 截斷處理摘要 ===")
    truncated_count = summary_df['truncated'].sum()
    total_removed = summary_df['removed_rows'].sum() if 'removed_rows' in summary_df.columns else 0
    
    print(f"📋 處理檔案總數: {len(summary_df)}")
    print(f"✂️  截斷檔案數: {truncated_count}")
    print(f"📉 總移除行數: {total_removed:,}")
    print(f"💾 平均移除比例: {(total_removed / summary_df['original_rows'].sum() * 100):.1f}%")
    
    # 顯示需要截斷的檔案
    if truncated_count > 0:
        truncated_files = summary_df[summary_df['truncated'] == True]
        print(f"\n⚠️  需要截斷的檔案:")
        for _, row in truncated_files.iterrows():
            removal_pct = (row['removed_rows'] / row['original_rows'] * 100)
            print(f"   {row['filename']}: 移除 {row['removed_rows']} 行 ({removal_pct:.1f}%)")
    
    return summary_df

# === 使用範例 ===
if __name__ == "__main__":
    
    # 載入條件對照表
    from smart_data_split import load_condition_mapping  # 使用之前的函數
    condition_map = load_condition_mapping()
    
    # 設定路徑
    TRAIN_DIR = "/Users/benjamin/1132/11325/AI_Race/2025_dataset_0806 3/train"
    OUTPUT_DIR = "/Users/benjamin/1132/11325/AI_Race/truncated_data"
    
    # 執行批次截斷
    summary = batch_truncate_files(TRAIN_DIR, condition_map, OUTPUT_DIR)
    
    # 儲存處理報告
    report_file = f"{OUTPUT_DIR}/truncation_report.csv"
    summary.to_csv(report_file, index=False)
    print(f"\n📄 處理報告已儲存: {report_file}")
