# q_event_logger.py
# 🔍 模組化智慧事件記錄器：根據 modularity Q 值追蹤 ∇ / Ω 結構轉變（支援自動尋找最新資料夾）

import pandas as pd
import os
import json

# === 可調參數 ===
Q_THRESHOLD = 0.05  # 當 Q 值變化量超過此值，視為轉變事件

# === 自動找出最新資料夾 ===
def find_latest_test_dir(base_dir="runs"):
    all_dirs = []
    for category in ["test", "test_many", "fast"]:
        cat_path = os.path.join(base_dir, category)
        if not os.path.exists(cat_path):
            continue
        for name in os.listdir(cat_path):
            sub_path = os.path.join(cat_path, name)
            if os.path.isdir(sub_path) and name.startswith(category):
                all_dirs.append(sub_path)
    if not all_dirs:
        raise FileNotFoundError("❌ 找不到任何 test 類別的資料夾")
    return sorted(all_dirs)[-1]


# === 主邏輯：偵測 Q 值轉變事件 ===
def detect_q_events(metrics_path, save_path=None):
    df = pd.read_csv(metrics_path)

    if "modularity" not in df.columns:
        raise ValueError("❌ 缺少 modularity 欄位，無法記錄 Q 事件。")

    events = []
    prev_q = df.loc[0, "modularity"]

    for i, row in df.iterrows():
        current_q = row["modularity"]
        delta_q = abs(current_q - prev_q)

        if delta_q > Q_THRESHOLD:
            event = {
                "episode": int(row["episode"]),
                "delta_q": round(delta_q, 5),
                "q": round(current_q, 5),
                "event": "∇" if current_q < prev_q else "Ω"
            }
            events.append(event)

        prev_q = current_q

    # 預設儲存位置
    if save_path is None:
        save_path = metrics_path.replace("metrics.csv", "q_events.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    print(f"✅ Q 事件記錄完成，共 {len(events)} 筆，儲存於：{save_path}")
    return events


# === 執行區塊 ===
if __name__ == "__main__":
    latest_dir = find_latest_test_dir()
    metrics_path = os.path.join(latest_dir, "metrics.csv")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"❌ 找不到 metrics.csv：{metrics_path}")

    detect_q_events(metrics_path)
