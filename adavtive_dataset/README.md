# Adaptive UI Dataset - Data Dictionary & Guide

এই ফোল্ডারে (`adavtive_dataset`) HMOG এবং Touchalytics এর মূল র ডেটাসেটসমূহ এবং মডেল ট্রেইনের জন্য প্রস্তুতকৃত ফাইনাল ডেটাসেট সাজানো হয়েছে।

---

## 📁 ফোল্ডার স্ট্রাকচার

```text
adavtive_dataset/
├── raw_data/
│   ├── hmog/
│   │   ├── hmog_activity_features.csv  (১০০ জন ইউজারের ১২,৪১৯ টি র সেশন/অ্যাক্টিভিটি)
│   │   └── hmog_user_profiles.csv     (১০০ জন ইউজারের ওভারঅল প্রোফাইল)
│   └── touchalytics/
│       ├── data.csv                    (৫০ MB র টাচ লগ - ২,১১,৫৮০+ টাচ ইভেন্ট)
│       ├── featMat.csv                 (২১,১৫৮ টি প্রি-প্রসেসড স্ট্রোক ডেটা)
│       └── featureString.txt           (৩৪টি মূল ফিচারের নাম ও বিবরণ)
├── processed/
│   └── final_ready_dataset.csv         (মডেল ট্রেইনিংয়ের জন্য প্রস্তুত ৭,৪২৬ টি রো)
└── generate_ready_dataset.py           (র ডেটা থেকে প্রস্তুতকৃত ডেটাসেট তৈরির স্ক্রিপ্ট)
```

---

## 📊 ফাইনাল ডেটাসেট কলাম ও বিবরণ (`processed/final_ready_dataset.csv`)

| কলামের নাম | ডেটা টাইপ | মান বা রেঞ্জ | কাজ / অর্থ |
| :--- | :--- | :--- | :--- |
| `user_id` | String | `hmog_xxx`, `touch_xxx` | ইউজারের ইউনিক আইডি (GroupKFold স্প্লিটে দরকার) |
| `session_id` | String | সেশন আইডি | নির্দিষ্ট ইন্টারঅ্যাকশন বা সেশনের ট্র্যাকিং |
| `source_dataset` | String | HMOG / Touchalytics | ডেটা কোন সোর্স থেকে এসেছে |
| `context_is_walking` | Binary | `0` (বসা), `1` (হাঁটা) | ইউজার কি হাঁটছিল নাকি স্থির ছিল |
| `touch_duration_mean_ms` | Float | ms (যেমন: 80.5) | স্ক্রিনে আঙুল কতক্ষণ স্পর্শ করেছিল |
| `swipe_speed_mean_pxps` | Float | px/sec (যেমন: 650.2) | আঙুল সোয়াইপ বা স্ক্রোল করার গতি |
| `swipe_distance_mean_px` | Float | পিক্সেল | এক সোয়াইপে কতদূর ড্র্যাগ করেছে |
| `typing_speed_kpm` | Float | কী/মিনিট | কিবোর্ডে লেখার গতি (HMOG-এ আছে) |
| `backspace_rate` | Float | 0.0 - 1.0 | ব্যাকস্পেস ব্যবহারের অনুপাত (টাইপিং ভুলের পরিমাপ) |
| `contact_size_mean` | Float | 0.0 - 1.0 | স্ক্রিনে আঙুলের ক্ষেত্রফল/সাইজ |
| `touch_x_mean` | Float | পিক্সেল | ট্যাপের X অবস্থান (হাতের পজিশন বুঝতে) |
| `touch_y_mean` | Float | পিক্সেল | ট্যাপের Y অবস্থান (বুড়ো আঙুলের নাগাল বুঝতে) |
| `tap_accuracy` | Float | 0.0 - 1.0 | ট্যাপের সঠিকতা |
| `miss_rate` | Float | 0.0 - 1.0 | ভুল জায়গায় ট্যাপ পড়ার অনুপাত |
| `reaction_time_mean_ms` | Float | ms (যেমন: 350.0) | টাস্ক দেখে ট্যাপ করতে কত সময় নিল |
| `reaction_time_norm` | Float | 0.0 - 1.0 | নরমালাইজড রিঅ্যাকশন টাইম |
| **`target_button_size`** | **Integer** | **0, 1, 2, 3** | **🎯 মডেলের Target Label** <br>• `0`: Small (40dp)<br>• `1`: Default (48dp)<br>• `2`: Large (56dp)<br>• `3`: Extra Large (64dp) |

---

## 🚀 পরবর্তী ধাপ:
1. এই ডেটাসেট ব্যবহার করে Cold-Start Classifier (XGBoost / Random Forest) ট্রেন করা।
2. বাটন সাইজের অ্যাডাপ্টেশন যাচাই করার জন্য RL Agent (Contextual Bandit বা Q-learning) চালানো।
