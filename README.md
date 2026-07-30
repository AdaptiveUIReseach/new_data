# Adaptive UI (AUI) Button Size Machine Learning Pipeline

This repository contains the end-to-end Machine Learning pipeline for predicting and adapting UI button sizes (`Small`, `Medium`, `Large`, `Extra_Large`) based on user interaction metrics (Human-Computer Interaction / HCI behavior).

## 📌 Project Overview
Adaptive User Interfaces (AUI) dynamically adjust interface geometry to accommodate individual user capabilities, reaction times, and motor control accuracy.

This project trains and compares multiple Machine Learning models:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **MLP Classifier (Neural Network)**
- **XGBoost Classifier (Base & Tuned via GridSearchCV)**

The tuned **XGBoost Classifier** achieves **95.24% Test Accuracy** on HCI behavioral datasets.

---

## 📊 Features & Behavioral Metrics Used
- `age`: User age
- `screen_width`: Device screen width in pixels
- `screen_height`: Device screen height in pixels
- `total_taps`: Total tap attempts
- `successful_taps`: Successful button hits
- `miss_taps`: Missed tap attempts
- `tap_accuracy`: Ratio of successful taps (`successful_taps / total_taps`)
- `avg_reaction_time_ms`: Average reaction time in milliseconds
- `miss_rate`: Ratio of missed taps (`miss_taps / total_taps`)
- `IFI`: Inter-Finger Interval (seconds between consecutive taps)

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Jupyter Notebook:**
   Open and execute `test.ipynb` in VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook test.ipynb
   ```

---

## 📦 Exported Pipeline Artifacts
- `xgboost_aui_model.pkl`: Best trained XGBoost model binary
- `feature_scaler.pkl`: Fitted `StandardScaler` object
- `label_encoder.pkl`: Fitted `LabelEncoder` object
- `features_list.txt`: Feature column names ordering
