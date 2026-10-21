"""
=============================================================
Adaptive UI Dataset Builder
=============================================================
এই script ৩টা raw source থেকে ডেটা নিয়ে ২টা ready dataset বানাবে:

SOURCE 1: তোমার নিজের collected data (adaptive_button_dataset_final.csv)
SOURCE 2: HMOG dataset (100 users - swipe, typing, touch behavior)
SOURCE 3: Touchalytics dataset (41 users - featMat.csv)

OUTPUT:
  ├── cold_start_dataset.csv     → Cold-Start ML model training
  ├── rl_dataset.csv             → Reinforcement Learning training
  └── README_datasets.md         → Column-by-column explanation

Run:
    python build_datasets.py
=============================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd

# ===================== PATHS =====================
BASE_DIR = r"C:\Users\Mehedi\adavtive_model"

# Source files
YOUR_DATA_PATH   = os.path.join(BASE_DIR, "adaptive_button_dataset_final.csv")
HMOG_ACT_PATH    = r"C:\Users\Mehedi\.gemini\antigravity\scratch\hmog_work\hmog_activity_features.csv"
HMOG_USER_PATH   = r"C:\Users\Mehedi\.gemini\antigravity\scratch\hmog_work\hmog_user_profiles.csv"
TOUCHALYTICS_RAW = r"C:\Users\Mehedi\Downloads\data\data\data.csv"
TOUCHALYTICS_FEAT= r"C:\Users\Mehedi\Downloads\features\features\featMat.csv"

# Output files
COLD_START_OUT = os.path.join(BASE_DIR, "cold_start_dataset.csv")
RL_OUT         = os.path.join(BASE_DIR, "rl_dataset.csv")
README_OUT     = os.path.join(BASE_DIR, "README_datasets.md")

# ===================== LABEL ENCODING =====================
BUTTON_LABEL_MAP = {
    "small_button_ui":       0,   # 40dp
    "default_button_ui":     1,   # 48dp
    "large_button_ui":       2,   # 56dp
    "extra_large_button_ui": 3,   # 64dp
}
BUTTON_SIZE_DP = {0: 40, 1: 48, 2: 56, 3: 64}

# RL action definitions
RL_ACTIONS = {
    0: "increase_button_size",
    1: "decrease_button_size",
    2: "increase_font_size",
    3: "decrease_font_size",
    4: "increase_spacing",
    5: "decrease_spacing",
    6: "no_change",
}

print("=" * 60)
print("  Adaptive UI Dataset Builder")
print("=" * 60)


# ================================================================
# PART 1: তোমার নিজের collected data load ও process
# ================================================================
print("\n[1/5] Loading your collected data...")
your_df = pd.read_csv(YOUR_DATA_PATH)
print(f"      ✓ {len(your_df)} rows, {your_df['user_id'].nunique()} users")

# button_label কে numeric করো
your_df["button_label_encoded"] = your_df["button_label"].map(BUTTON_LABEL_MAP)
# final_processed_dataset.csv-এ ইতিমধ্যে int থাকতে পারে — সেটাও handle করো
already_int = pd.to_numeric(your_df["button_label_encoded"], errors="coerce")
your_df["button_label_encoded"] = already_int.fillna(
    pd.to_numeric(your_df["button_label"], errors="coerce")
)
your_df = your_df.dropna(subset=["button_label_encoded"])
your_df["button_label_encoded"] = your_df["button_label_encoded"].astype(int)

# miss_rate validate (0-1 এর মধ্যে থাকতে হবে)
your_df["miss_rate"]    = your_df["miss_rate"].clip(0, 1)
your_df["tap_accuracy"] = your_df["tap_accuracy"].clip(0, 1)

# reaction_time normalize: min-max দিয়ে (0 = দ্রুত, 1 = ধীর)
rt_min, rt_max = your_df["avg_reaction_time_ms"].min(), your_df["avg_reaction_time_ms"].max()
your_df["reaction_time_norm"] = (
    (your_df["avg_reaction_time_ms"] - rt_min) / (rt_max - rt_min + 1e-8)
)

# context: তোমার data-তে context নেই → 0 ধরা হচ্ছে (বসে ব্যবহার করেছে)
your_df["context_is_walking"] = 0

print(f"      Labels: {dict(your_df['button_label_encoded'].value_counts().sort_index())}")


# ================================================================
# PART 2: HMOG user profiles load (swipe, typing, touch)
# ================================================================
print("\n[2/5] Loading HMOG user profiles...")
hmog_profiles = None
if os.path.exists(HMOG_USER_PATH):
    hmog_raw = pd.read_csv(HMOG_USER_PATH)
    hmog_raw = hmog_raw.reset_index()  # user_id index → column
    hmog_profiles = hmog_raw.rename(columns={"index": "user_id"})
    print(f"      ✓ {len(hmog_profiles)} HMOG users loaded")

    # HMOG থেকে আমরা কী নেব
    HMOG_COLS = {
        "swipe_speed_mean_pxps"   : "swipe_speed_mean_pxps",
        "swipe_distance_mean_px"  : "swipe_distance_mean_px",
        "touch_duration_mean_ms"  : "touch_duration_mean_ms",
        "contact_size_mean"       : "contact_size_mean",
        "touch_x_mean"            : "touch_x_mean",
        "touch_y_mean"            : "touch_y_mean",
        "typing_speed_kpm"        : "typing_speed_kpm",
        "backspace_rate"          : "backspace_rate",
        "inter_touch_interval_median_ms": "inter_touch_interval_median_ms",
    }
    # যে column গুলো আছে সেগুলোই নাও
    available = {k: v for k, v in HMOG_COLS.items() if k in hmog_profiles.columns}
    hmog_subset = hmog_profiles[["user_id"] + list(available.keys())].copy()

    # Statistics বের করো (simulator calibration-এর জন্য)
    print("\n      --- HMOG Statistics (Sitting, Writing task) ---")
    hmog_act = pd.read_csv(HMOG_ACT_PATH) if os.path.exists(HMOG_ACT_PATH) else None
    if hmog_act is not None:
        writing_sit = hmog_act[(hmog_act["task_type"] == "writing") & (hmog_act["context"] == "sitting")]
        writing_walk= hmog_act[(hmog_act["task_type"] == "writing") & (hmog_act["context"] == "walking")]
        for col in ["typing_speed_kpm", "backspace_rate", "touch_duration_mean_ms", "swipe_speed_mean_pxps"]:
            if col in writing_sit.columns:
                vs = writing_sit[col].median()
                vw = writing_walk[col].median()
                print(f"      {col:40s} sit={vs:.3f}  walk={vw:.3f}")
else:
    print("      ⚠ HMOG profiles not found — run hmog_extract.py first")


# ================================================================
# PART 3: Touchalytics featMat.csv load
# ================================================================
print("\n[3/5] Loading Touchalytics features...")

# featureString.txt থেকে column names (34টি)
TOUCHALYTICS_COLS = [
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
    "change_finger_orientation", "phone_orientation",
]

touch_stats = None
if os.path.exists(TOUCHALYTICS_FEAT):
    touch_raw = pd.read_csv(TOUCHALYTICS_FEAT, header=None, names=TOUCHALYTICS_COLS)
    n_users = touch_raw["user_id"].nunique()
    print(f"      ✓ {len(touch_raw):,} strokes, {n_users} users")

    # Per-user aggregate statistics বের করো
    touch_stats = touch_raw.groupby("user_id").agg(
        n_strokes             = ("stroke_duration", "count"),
        stroke_duration_mean  = ("stroke_duration", "median"),
        swipe_velocity_median = ("velocity_50pct", "median"),
        swipe_velocity_80pct  = ("velocity_80pct", "median"),
        trajectory_len_median = ("trajectory_length", "median"),
        direct_distance_median= ("direct_distance", "median"),
        straightness_ratio    = ("ratio_end_to_end_trajectory", "median"),  # 1 = সরাসরি, <1 = বাঁকা
        touch_pressure_median = ("mid_pressure", "median"),
        contact_area_median   = ("mid_area", "median"),
        inter_stroke_time_med = ("inter_stroke_time", "median"),
    ).reset_index()

    # Source column
    touch_stats["data_source"] = "touchalytics"
    print(f"      ✓ Per-user stats: {touch_stats.shape}")
else:
    print("      ⚠ featMat.csv not found at:", TOUCHALYTICS_FEAT)


# ================================================================
# PART 4: COLD-START DATASET তৈরি
# ================================================================
print("\n[4/5] Building Cold-Start ML Dataset...")

"""
Cold-Start Dataset: প্রতিটি row = একজন user-এর একটি session
Model input (X): user-এর touch behaviour
Model output (Y): কোন button size তার জন্য best

Source: primarily তোমার collected data
HMOG/Touchalytics: behavior statistics যোগ করে rows augment করা হবে
"""

# ---- তোমার data থেকে cold-start rows বানাও ----
cold_rows = []
for _, row in your_df.iterrows():
    cold_rows.append({
        # === Identifiers ===
        "user_id"              : row["user_id"],
        "session_id"           : row["session_id"],
        "data_source"          : "your_collected_data",

        # === Input Features (X) ===
        "tap_accuracy"         : row["tap_accuracy"],
        "miss_rate"            : row["miss_rate"],
        "reaction_time_mean_ms": row["avg_reaction_time_ms"],
        "reaction_time_norm"   : row["reaction_time_norm"],
        "age"                  : row.get("age", np.nan),
        "screen_width"         : row.get("screen_width", np.nan),
        "screen_height"        : row.get("screen_height", np.nan),
        "context_is_walking"   : row["context_is_walking"],

        # HMOG থেকে পাওয়া features (তোমার data-তে নেই, তাই NaN রাখা হচ্ছে)
        # — পরে তুমি app-এ এগুলো collect করবে
        "touch_duration_mean_ms": np.nan,
        "swipe_speed_mean_pxps" : np.nan,
        "swipe_distance_mean_px": np.nan,
        "typing_speed_kpm"      : np.nan,
        "backspace_rate"        : np.nan,
        "contact_size_mean"     : np.nan,
        "touch_x_mean"          : np.nan,
        "touch_y_mean"          : np.nan,

        # === Target Label (Y) ===
        "target_button_size"    : row["button_label_encoded"],
        "target_button_size_dp" : BUTTON_SIZE_DP.get(
                                      int(row["button_label_encoded"])
                                      if (not pd.isna(row["button_label_encoded"])
                                          and str(row["button_label_encoded"]).lstrip("-").isdigit()) else 1, 48),
        "stage"                 : row.get("stage", "unknown"),
    })

# ---- HMOG augmented rows বানাও (synthetic label = Fitts Law দিয়ে) ----
if hmog_act is not None:
    print("      Adding HMOG-augmented rows with synthetic labels...")

    # প্রতি user-এর writing session এর median behavior নাও
    hmog_user_behav = hmog_act[hmog_act["task_type"] == "writing"].groupby(
        ["user_id", "context"]
    ).agg(
        touch_duration_mean_ms = ("touch_duration_mean_ms", "median"),
        swipe_speed_mean_pxps  = ("swipe_speed_mean_pxps", "median"),
        swipe_distance_mean_px = ("swipe_distance_mean_px", "median"),
        typing_speed_kpm       = ("typing_speed_kpm", "median"),
        backspace_rate         = ("backspace_rate", "median"),
        contact_size_mean      = ("contact_size_mean", "median"),
        touch_x_mean           = ("touch_x_mean", "median"),
        touch_y_mean           = ("touch_y_mean", "median"),
    ).reset_index()

    for _, row in hmog_user_behav.iterrows():
        # Synthetic label: backspace_rate দিয়ে button size ঠিক করো (Fitts' Law proxy)
        # যে ইউজার বেশি ভুল করে → তার বড় button দরকার
        br = row.get("backspace_rate", 0.1)
        if br > 0.15:
            synth_label = 3   # XL (64dp)
        elif br > 0.10:
            synth_label = 2   # Large (56dp)
        elif br > 0.06:
            synth_label = 1   # Default (48dp)
        else:
            synth_label = 0   # Small (40dp)

        # reaction_time proxy: typing speed থেকে
        tspeed = row.get("typing_speed_kpm", 150)
        rt_proxy = max(200, 600 - tspeed * 1.5)

        cold_rows.append({
            "user_id"              : f"hmog_{int(row['user_id'])}",
            "session_id"           : f"hmog_s{int(row['user_id'])}",
            "data_source"          : "hmog_augmented",
            "tap_accuracy"         : 1.0 - float(row.get("backspace_rate", 0.1)),
            "miss_rate"            : float(row.get("backspace_rate", 0.1)),
            "reaction_time_mean_ms": rt_proxy,
            "reaction_time_norm"   : max(0, min(1, (rt_proxy - 200) / 400)),
            "age"                  : np.nan,
            "screen_width"         : np.nan,
            "screen_height"        : np.nan,
            "context_is_walking"   : 1 if row["context"] == "walking" else 0,
            "touch_duration_mean_ms": float(row.get("touch_duration_mean_ms", np.nan)),
            "swipe_speed_mean_pxps" : float(row.get("swipe_speed_mean_pxps", np.nan)),
            "swipe_distance_mean_px": float(row.get("swipe_distance_mean_px", np.nan)),
            "typing_speed_kpm"      : float(row.get("typing_speed_kpm", np.nan)),
            "backspace_rate"        : float(row.get("backspace_rate", np.nan)),
            "contact_size_mean"     : float(row.get("contact_size_mean", np.nan)),
            "touch_x_mean"          : float(row.get("touch_x_mean", np.nan)),
            "touch_y_mean"          : float(row.get("touch_y_mean", np.nan)),
            "target_button_size"    : synth_label,
            "target_button_size_dp" : BUTTON_SIZE_DP[synth_label],
            "stage"                 : "augmented",
        })

cold_df = pd.DataFrame(cold_rows)
# target_button_size-কে numeric বানাও (কিছু rows-এ string ছিল)
cold_df["target_button_size"] = pd.to_numeric(cold_df["target_button_size"], errors="coerce")
cold_df = cold_df.dropna(subset=["target_button_size"])
cold_df["target_button_size"] = cold_df["target_button_size"].astype(int)

# Label distribution দেখো
print(f"      Total Cold-Start rows: {len(cold_df)}")
print(f"      Label distribution:")
lb = cold_df["target_button_size"].value_counts().sort_index()
for k, v in lb.items():
    lname = [n for n, i in BUTTON_LABEL_MAP.items() if i == k]
    print(f"        [{int(k)}] {lname[0] if lname else '?':28s}: {v} rows")

cold_df.to_csv(COLD_START_OUT, index=False)
print(f"      ✅ Saved → {COLD_START_OUT}")


# ================================================================
# PART 5: RL DATASET তৈরি
# ================================================================
print("\n[5/5] Building RL Transition Dataset...")

"""
RL Dataset: Markov Decision Process (MDP) transitions
প্রতিটি row = একটি state-action-reward-nextstate tuple

State: user-এর behavior (miss_rate, tap_accuracy, reaction_time) + current UI config
Action: UI তে কী change করা হলো
Reward: পরিবর্তনের পর user-এর performance কতটা ভালো হলো

Source: তোমার data থেকে এবং HMOG দিয়ে simulation
"""

def compute_reward(tap_acc, miss_rate, reaction_norm, ui_changed, task_success=1.0, user_rating=np.nan):
    """
    Reward formula (paper-এ justify করা যাবে):
    R = 0.40 * tap_accuracy
      - 0.30 * miss_rate
      - 0.15 * reaction_time_norm
      + 0.10 * task_success
      - 0.05 * ui_change_penalty  (ঘনঘন পরিবর্তন ভালো না)
    """
    r = (
        0.40 * tap_acc
      - 0.30 * miss_rate
      - 0.15 * reaction_norm
      + 0.10 * task_success
      - 0.05 * int(ui_changed)
    )
    return round(float(r), 6)


rl_rows = []

# ---- তোমার collected data থেকে RL transitions বানাও ----
# প্রতিটি user-এর sessions time-ordered, consecutive pair = transition
for uid, user_sessions in your_df.groupby("user_id"):
    user_sessions = user_sessions.sort_values("session_id").reset_index(drop=True)

    for i in range(len(user_sessions) - 1):
        s_row  = user_sessions.iloc[i]     # current state
        ns_row = user_sessions.iloc[i + 1] # next state

        cur_btn  = int(s_row["button_label_encoded"])  if (not pd.isna(s_row["button_label_encoded"]) and str(s_row["button_label_encoded"]).lstrip("-").isdigit())  else 1
        next_btn = int(ns_row["button_label_encoded"]) if (not pd.isna(ns_row["button_label_encoded"]) and str(ns_row["button_label_encoded"]).lstrip("-").isdigit()) else 1

        # Action ঠিক করো: button size কীভাবে বদলেছে
        btn_diff = next_btn - cur_btn
        if btn_diff > 0:
            action = 0   # increase_button_size
        elif btn_diff < 0:
            action = 1   # decrease_button_size
        else:
            action = 6   # no_change

        ui_changed = (btn_diff != 0)

        rl_rows.append({
            # === State (S_t) ===
            "s_miss_rate"           : s_row["miss_rate"],
            "s_tap_accuracy"        : s_row["tap_accuracy"],
            "s_reaction_time_norm"  : s_row["reaction_time_norm"],
            "s_current_btn_size"    : cur_btn,
            "s_current_font_size"   : 1,   # তোমার data-তে নেই → default
            "s_current_spacing"     : 1,   # তোমার data-তে নেই → default
            "s_context_walking"     : 0,

            # === Action (A_t) ===
            "action_id"             : action,
            "action_name"           : RL_ACTIONS[action],
            "ui_changed"            : int(ui_changed),

            # === Reward Components (R_t) ===
            "r_tap_accuracy"        : ns_row["tap_accuracy"],
            "r_miss_penalty"        : -ns_row["miss_rate"],
            "r_reaction_time"       : -ns_row["reaction_time_norm"],
            "r_task_success"        : 1.0,
            "r_ui_change_penalty"   : -0.05 * int(ui_changed),
            "final_reward"          : compute_reward(
                                          ns_row["tap_accuracy"],
                                          ns_row["miss_rate"],
                                          ns_row["reaction_time_norm"],
                                          ui_changed
                                      ),

            # === Next State (S_t+1) ===
            "next_miss_rate"        : ns_row["miss_rate"],
            "next_tap_accuracy"     : ns_row["tap_accuracy"],
            "next_reaction_norm"    : ns_row["reaction_time_norm"],
            "next_btn_size"         : next_btn,
            "next_font_size"        : 1,
            "next_spacing"          : 1,
            "is_done"               : int(i == len(user_sessions) - 2),

            # === Meta ===
            "user_id"               : uid,
            "data_source"           : "your_collected_data",
        })

# ---- HMOG থেকে simulated RL transitions বানাও ----
if hmog_act is not None:
    print("      Adding simulated RL transitions from HMOG...")

    # Sitting vs Walking → context change দিয়ে action simulate করো
    for uid, udata in hmog_act[hmog_act["task_type"] == "writing"].groupby("user_id"):
        if len(udata) < 2:
            continue
        udata = udata.sort_values("session_number").reset_index(drop=True)

        for i in range(len(udata) - 1):
            s = udata.iloc[i]
            ns= udata.iloc[i + 1]

            # miss_rate proxy: backspace_rate
            s_miss  = float(s.get("backspace_rate", 0.1))
            ns_miss = float(ns.get("backspace_rate", 0.1))
            s_acc   = 1.0 - s_miss
            ns_acc  = 1.0 - ns_miss

            # reaction_time proxy: typing speed
            s_rt  = max(0, min(1, (600 - float(s.get("typing_speed_kpm", 150))) / 400))
            ns_rt = max(0, min(1, (600 - float(ns.get("typing_speed_kpm", 150))) / 400))

            # Button size: miss_rate দিয়ে ঠিক করো
            def miss_to_btn(m):
                if m > 0.15: return 3
                elif m > 0.10: return 2
                elif m > 0.06: return 1
                else: return 0

            cur_btn  = miss_to_btn(s_miss)
            next_btn = miss_to_btn(ns_miss)
            btn_diff = next_btn - cur_btn
            action   = 0 if btn_diff > 0 else (1 if btn_diff < 0 else 6)
            ui_changed = (btn_diff != 0)

            rl_rows.append({
                "s_miss_rate"           : s_miss,
                "s_tap_accuracy"        : s_acc,
                "s_reaction_time_norm"  : s_rt,
                "s_current_btn_size"    : cur_btn,
                "s_current_font_size"   : 1,
                "s_current_spacing"     : 1,
                "s_context_walking"     : 1 if s["context"] == "walking" else 0,
                "action_id"             : action,
                "action_name"           : RL_ACTIONS[action],
                "ui_changed"            : int(ui_changed),
                "r_tap_accuracy"        : ns_acc,
                "r_miss_penalty"        : -ns_miss,
                "r_reaction_time"       : -ns_rt,
                "r_task_success"        : 1.0,
                "r_ui_change_penalty"   : -0.05 * int(ui_changed),
                "final_reward"          : compute_reward(ns_acc, ns_miss, ns_rt, ui_changed),
                "next_miss_rate"        : ns_miss,
                "next_tap_accuracy"     : ns_acc,
                "next_reaction_norm"    : ns_rt,
                "next_btn_size"         : next_btn,
                "next_font_size"        : 1,
                "next_spacing"          : 1,
                "is_done"               : int(i == len(udata) - 2),
                "user_id"               : f"hmog_{uid}",
                "data_source"           : "hmog_simulated",
            })

rl_df = pd.DataFrame(rl_rows)

# Reward normalize করো (paper-এ দেখাতে হবে)
rl_df["reward_normalized"] = (
    (rl_df["final_reward"] - rl_df["final_reward"].min()) /
    (rl_df["final_reward"].max() - rl_df["final_reward"].min() + 1e-8)
)

print(f"      Total RL transitions: {len(rl_df)}")
print(f"      Action distribution:")
print(rl_df["action_name"].value_counts().to_string())
print(f"\n      Reward stats:")
print(rl_df["final_reward"].describe().round(4).to_string())

rl_df.to_csv(RL_OUT, index=False)
print(f"      ✅ Saved → {RL_OUT}")


# ================================================================
# README লেখো
# ================================================================
readme = """# Adaptive UI Research — Dataset Documentation

> Auto-generated by `build_datasets.py`

---

## Source Datasets

| Source | File | Users | What We Used |
|--------|------|-------|--------------|
| Your Collected Data | `adaptive_button_dataset_final.csv` | Real users | tap_accuracy, miss_rate, reaction_time, button_label |
| HMOG Dataset | `hmog_activity_features.csv` | 100 users | touch_duration, swipe_speed, typing_speed, backspace_rate, context |
| Touchalytics | `featMat.csv` | 41 users | stroke_duration, velocity, pressure, area |

---

## File 1: `cold_start_dataset.csv`

**উদ্দেশ্য:** নতুন ইউজারের প্রথম ২ মিনিটের আচরণ দেখে কোন UI config দেওয়া উচিত সেটা predict করা।

**Model:** XGBoost / Random Forest / MLP

### Columns

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `user_id` | string | — | ইউজার identifier (train/test split করতে লাগবে) |
| `session_id` | string | — | Session identifier |
| `data_source` | string | your_collected_data / hmog_augmented | Data কোথা থেকে এসেছে |
| `tap_accuracy` | float | 0.0–1.0 | সঠিক button-এ ট্যাপ পড়ার হার (1.0 = সবসময় সঠিক) |
| `miss_rate` | float | 0.0–1.0 | ভুল ট্যাপের হার (= 1 - tap_accuracy) |
| `reaction_time_mean_ms` | float | ms | Target দেখার পর প্রথম ট্যাপ করতে কতক্ষণ নিয়েছে |
| `reaction_time_norm` | float | 0.0–1.0 | Normalized reaction time (0 = দ্রুত, 1 = ধীর) |
| `age` | int | years | ইউজারের বয়স |
| `screen_width` | int | pixels | ফোনের স্ক্রিন প্রস্থ |
| `screen_height` | int | pixels | ফোনের স্ক্রিন উচ্চতা |
| `context_is_walking` | int | 0 or 1 | 0 = বসে ব্যবহার, 1 = হাঁটতে হাঁটতে ব্যবহার |
| `touch_duration_mean_ms` | float | ms | একটি ট্যাপে আঙুল কতক্ষণ চেপে রাখে (HMOG) |
| `swipe_speed_mean_pxps` | float | px/sec | গড় সোয়াইপ গতি (HMOG) |
| `swipe_distance_mean_px` | float | pixels | এক সোয়াইপে গড় দূরত্ব (HMOG) |
| `typing_speed_kpm` | float | keys/min | টাইপিং গতি (HMOG) |
| `backspace_rate` | float | 0.0–1.0 | মোট কী-এর মধ্যে backspace-এর হার — typing error proxy (HMOG) |
| `contact_size_mean` | float | normalized | আঙুলের স্ক্রিন contact area (HMOG) |
| `touch_x_mean` | float | pixels | গড় ট্যাপের X অবস্থান (হাত বোঝার জন্য) |
| `touch_y_mean` | float | pixels | গড় ট্যাপের Y অবস্থান (বুড়ো আঙুলের reach বোঝার জন্য) |
| **`target_button_size`** | int | **0/1/2/3** | **🎯 MODEL TARGET: 0=40dp, 1=48dp, 2=56dp, 3=64dp** |
| `target_button_size_dp` | int | 40/48/56/64 | Button size in density-independent pixels |
| `stage` | string | easy/hard | Task difficulty |

### Model Training

```python
from sklearn.model_selection import GroupShuffleSplit

X_COLS = [
    "tap_accuracy", "miss_rate", "reaction_time_norm",
    "context_is_walking", "touch_duration_mean_ms",
    "swipe_speed_mean_pxps", "typing_speed_kpm",
    "backspace_rate", "contact_size_mean",
    "touch_x_mean", "touch_y_mean",
]
Y_COL = "target_button_size"
GROUP_COL = "user_id"  # এই column দিয়ে split করতে হবে
```

---

## File 2: `rl_dataset.csv`

**উদ্দেশ্য:** RL agent-কে শেখানো — কোন state-এ কোন action নিলে reward বাড়ে।

**Model:** Contextual Bandit / DQN / PPO

### Columns

#### Current State (S_t)
| Column | Type | Description |
|--------|------|-------------|
| `s_miss_rate` | float 0–1 | এই মুহূর্তে ইউজারের miss rate |
| `s_tap_accuracy` | float 0–1 | এই মুহূর্তে ইউজারের tap accuracy |
| `s_reaction_time_norm` | float 0–1 | Normalized reaction time |
| `s_current_btn_size` | int 0–3 | বর্তমান button size level |
| `s_current_font_size` | int 0–2 | বর্তমান font size level |
| `s_current_spacing` | int 0–2 | বর্তমান spacing level |
| `s_context_walking` | int 0/1 | ইউজার কি হাঁটছে? |

#### Action (A_t)
| Column | Type | Description |
|--------|------|-------------|
| `action_id` | int 0–6 | 0=btn+, 1=btn-, 2=font+, 3=font-, 4=space+, 5=space-, 6=no_change |
| `action_name` | string | action-এর নাম (পড়ার জন্য) |
| `ui_changed` | int 0/1 | কোনো পরিবর্তন হয়েছে কি |

#### Reward Components (R_t)
| Column | Type | Description |
|--------|------|-------------|
| `r_tap_accuracy` | float | Reward: tap accuracy বেশি হলে বেশি |
| `r_miss_penalty` | float | Penalty: miss বেশি হলে কমে |
| `r_reaction_time` | float | Penalty: দেরি হলে কমে |
| `r_task_success` | float | Reward: task সম্পন্ন হলে |
| `r_ui_change_penalty` | float | Penalty: অপ্রয়োজনীয় পরিবর্তনের জন্য |
| `final_reward` | float | মোট reward (weighted sum) |
| `reward_normalized` | float 0–1 | Normalized final reward |

#### Reward Formula
```
R = 0.40 × tap_accuracy
  - 0.30 × miss_rate
  - 0.15 × reaction_time_norm
  + 0.10 × task_success
  - 0.05 × ui_changed
```

#### Next State (S_t+1)
| Column | Type | Description |
|--------|------|-------------|
| `next_miss_rate` | float | Action-এর পরের miss rate |
| `next_tap_accuracy` | float | Action-এর পরের accuracy |
| `next_reaction_norm` | float | Action-এর পরের reaction time |
| `next_btn_size` | int | পরের button size |
| `next_font_size` | int | পরের font size |
| `next_spacing` | int | পরের spacing |
| `is_done` | int 0/1 | Episode শেষ কি না |

### RL Training

```python
STATE_COLS = [
    "s_miss_rate", "s_tap_accuracy", "s_reaction_time_norm",
    "s_current_btn_size", "s_current_font_size",
    "s_current_spacing", "s_context_walking",
]
ACTION_COL = "action_id"
REWARD_COL = "final_reward"
NEXT_STATE_COLS = [
    "next_miss_rate", "next_tap_accuracy", "next_reaction_norm",
    "next_btn_size", "next_font_size", "next_spacing",
]
```

---

## Important Notes

> **⚠️ Data Leakage Warning:** Cold-start model-এর train/test split সবসময়
> `user_id` দিয়ে করতে হবে — কখনো random split করবে না।
> একই ইউজারের data দুই পাশে গেলে artificially ভালো result আসবে।

> **ℹ️ HMOG Label:** HMOG থেকে আসা rows-এর `target_button_size` label
> Fitts' Law-based হিউরিস্টিক দিয়ে তৈরি (backspace_rate থেকে)।
> এই rows `data_source == "hmog_augmented"` দিয়ে আলাদা করা আছে।
> Ablation study-তে এদের বাদ দিয়ে দেখাও যে augmentation কতটা সাহায্য করেছে।

> **ℹ️ Missing Values:** HMOG augmented rows-এ কিছু column NaN থাকবে।
> Model-এ imputation বা missing indicator feature যোগ করতে হবে।
"""

with open(README_OUT, "w", encoding="utf-8") as f:
    f.write(readme)

print(f"\n      ✅ Saved → {README_OUT}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("  ✅ ALL DONE!")
print("=" * 60)
print(f"\n  📁 Output folder: {BASE_DIR}")
print(f"\n  1️⃣  cold_start_dataset.csv")
print(f"      → {len(cold_df)} rows × {len(cold_df.columns)} columns")
print(f"      → {cold_df['user_id'].nunique()} unique users")
src_count = cold_df['data_source'].value_counts()
for src, cnt in src_count.items():
    print(f"         [{src}]: {cnt} rows")

print(f"\n  2️⃣  rl_dataset.csv")
print(f"      → {len(rl_df)} rows × {len(rl_df.columns)} columns")
print(f"      → {rl_df['action_name'].value_counts().to_dict()}")

print(f"\n  3️⃣  README_datasets.md")
print(f"      → Column-by-column documentation")

print(f"""
  ⏭  NEXT STEPS:
     1. VS Code-এ cold_start_dataset.csv খোলো
     2. NaN values handle করার strategy ঠিক করো
     3. Model training শুরু করো (train_aui_model.ipynb)
""")
