import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd

# Paths
BASE_DIR = r"C:\Users\Mehedi\adavtive_model\adavtive_dataset"
RAW_TOUCH_FEAT = os.path.join(BASE_DIR, "raw_data", "touchalytics", "featMat.csv")
RAW_HMOG_ACT   = os.path.join(BASE_DIR, "raw_data", "hmog", "hmog_activity_features.csv")
OUT_DIR        = os.path.join(BASE_DIR, "processed")

# Touchalytics column mapping
TOUCH_COLS = [
    "user_id", "doc_id", "inter_stroke_time", "stroke_duration",
    "start_x", "start_y", "stop_x", "stop_y",
    "direct_distance", "mean_resultant_length",
    "up_down_left_right_flag", "direction_end_to_end",
    "phone_id",
    "velocity_20pct", "velocity_50pct", "velocity_80pct",
    "acceleration_20pct", "acceleration_50pct", "acceleration_80pct",
    "median_velocity_last3",
    "largest_deviation",
    "deviation_20pct", "deviation_50pct", "deviation_80pct",
    "average_direction", "trajectory_length",
    "ratio_end_to_end_trajectory", "average_velocity",
    "median_acceleration_first5",
    "mid_pressure", "mid_area", "mid_finger_orientation",
    "change_finger_orientation", "phone_orientation"
]

print("==================================================")
print("  Processing Raw Data into Unified ML/RL Ready Set")
print("==================================================")

# 1. Process HMOG Raw Data (100 users, 12,419 activities)
print("\n[1/3] Reading HMOG Raw Activities...")
df_hmog = pd.read_csv(RAW_HMOG_ACT)
print(f"      HMOG loaded: {df_hmog.shape[0]} rows, {df_hmog['user_id'].nunique()} users")

# Filter writing & map tasks where interaction happens
hmog_clean = df_hmog[df_hmog['task_type'].isin(['writing', 'map'])].copy()

# 2. Process Touchalytics Raw Strokes (41 users, 21,158 strokes)
print("\n[2/3] Reading Touchalytics Raw Strokes...")
df_touch = pd.read_csv(RAW_TOUCH_FEAT, header=None, names=TOUCH_COLS)
print(f"      Touchalytics loaded: {df_touch.shape[0]} strokes, {df_touch['user_id'].nunique()} users")

# Aggregate Touchalytics into session/document chunks (Doc_id based)
touch_grouped = df_touch.groupby(['user_id', 'doc_id']).agg(
    touch_duration_mean_ms = ('stroke_duration', lambda x: x.median() * 1000), # convert sec to ms
    swipe_speed_mean_pxps  = ('velocity_50pct', 'median'),
    swipe_distance_mean_px = ('direct_distance', 'median'),
    contact_size_mean      = ('mid_area', 'median'),
    touch_x_mean           = ('start_x', 'median'),
    touch_y_mean           = ('start_y', 'median')
).reset_index()

# 3. Build Unified Ready Dataset for Training
print("\n[3/3] Transforming & Generating Final Model Columns...")
final_rows = []

# Process HMOG instances
for _, row in hmog_clean.iterrows():
    # Typing error proxy as miss_rate
    err_rate = row['backspace_rate'] if pd.notna(row['backspace_rate']) else 0.08
    miss_rate = float(np.clip(err_rate, 0.02, 0.40))
    tap_acc = 1.0 - miss_rate
    
    # Reaction time proxy based on typing speed or touch intervals
    typing_spd = row['typing_speed_kpm'] if pd.notna(row['typing_speed_kpm']) else 180.0
    reaction_ms = float(np.clip(600 - (typing_spd * 1.5), 180, 800))
    reaction_norm = (reaction_ms - 180) / (800 - 180)
    
    # Target Optimal Button Size (Heuristic based on Fitts's Law: higher error/slower -> larger button)
    if miss_rate > 0.16 or reaction_norm > 0.65:
        target_btn = 3 # Extra Large (64dp)
    elif miss_rate > 0.11 or reaction_norm > 0.45:
        target_btn = 2 # Large (56dp)
    elif miss_rate > 0.06:
        target_btn = 1 # Medium (48dp)
    else:
        target_btn = 0 # Small (40dp)

    final_rows.append({
        'user_id': f"hmog_{row['user_id']}",
        'session_id': f"s_{row['activity_id']}",
        'source_dataset': 'HMOG',
        'context_is_walking': 1 if row['context'] == 'walking' else 0,
        'touch_duration_mean_ms': round(float(row['touch_duration_mean_ms']) if pd.notna(row['touch_duration_mean_ms']) else 100.0, 2),
        'swipe_speed_mean_pxps': round(float(row['swipe_speed_mean_pxps']) if pd.notna(row['swipe_speed_mean_pxps']) else 800.0, 2),
        'swipe_distance_mean_px': round(float(row['swipe_distance_mean_px']) if pd.notna(row['swipe_distance_mean_px']) else 250.0, 2),
        'typing_speed_kpm': round(float(typing_spd), 2),
        'backspace_rate': round(float(err_rate), 4),
        'contact_size_mean': round(float(row['contact_size_mean']) if pd.notna(row['contact_size_mean']) else 0.025, 4),
        'touch_x_mean': round(float(row['touch_x_mean']) if pd.notna(row['touch_x_mean']) else 500.0, 2),
        'touch_y_mean': round(float(row['touch_y_mean']) if pd.notna(row['touch_y_mean']) else 1000.0, 2),
        'tap_accuracy': round(tap_acc, 4),
        'miss_rate': round(miss_rate, 4),
        'reaction_time_mean_ms': round(reaction_ms, 2),
        'reaction_time_norm': round(reaction_norm, 4),
        'target_button_size': target_btn # 0=Small, 1=Default, 2=Large, 3=XL
    })

# Process Touchalytics instances
for _, row in touch_grouped.iterrows():
    # Touchalytics has scrolling/swiping primarily
    duration = row['touch_duration_mean_ms']
    speed = row['swipe_speed_mean_pxps']
    
    # Fast swiping & short duration = confident user; slow/unsteady = needs larger targets
    err_proxy = 0.05 if speed > 500 else 0.14
    miss_rate = float(err_proxy)
    tap_acc = 1.0 - miss_rate
    reaction_ms = float(np.clip(duration * 2.5, 200, 750))
    reaction_norm = (reaction_ms - 200) / (750 - 200)
    
    if miss_rate > 0.12 or reaction_norm > 0.60:
        target_btn = 2 # Large
    else:
        target_btn = 1 # Medium
        
    final_rows.append({
        'user_id': f"touch_{row['user_id']}",
        'session_id': f"doc_{row['doc_id']}",
        'source_dataset': 'Touchalytics',
        'context_is_walking': 0,
        'touch_duration_mean_ms': round(float(duration), 2),
        'swipe_speed_mean_pxps': round(float(speed), 2),
        'swipe_distance_mean_px': round(float(row['swipe_distance_mean_px']), 2),
        'typing_speed_kpm': np.nan,
        'backspace_rate': np.nan,
        'contact_size_mean': round(float(row['contact_size_mean']), 4),
        'touch_x_mean': round(float(row['touch_x_mean']), 2),
        'touch_y_mean': round(float(row['touch_y_mean']), 2),
        'tap_accuracy': round(tap_acc, 4),
        'miss_rate': round(miss_rate, 4),
        'reaction_time_mean_ms': round(reaction_ms, 2),
        'reaction_time_norm': round(reaction_norm, 4),
        'target_button_size': target_btn
    })

ready_df = pd.DataFrame(final_rows)

# Save Final Unified Ready Dataset
out_csv = os.path.join(OUT_DIR, "final_ready_dataset.csv")
ready_df.to_csv(out_csv, index=False)
print(f"\n✅ Final Ready Dataset Saved: {out_csv}")
print(f"   Shape: {ready_df.shape[0]} rows × {ready_df.shape[1]} columns")
print(f"   Unique Users: {ready_df['user_id'].nunique()}")
print(f"   Class Distribution (target_button_size):")
print(ready_df['target_button_size'].value_counts().sort_index().to_dict())
