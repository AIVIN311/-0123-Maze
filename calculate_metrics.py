# calculate_metrics.py
# ✅ 自動尋找最新 test_log.csv，計算 entropy 與智慧密度並輸出 metrics.csv

import pandas as pd 
import numpy as np
import os

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

# === 計算行動熵（H） ===
def calculate_entropy(actions):
    counts = np.bincount(actions)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# === 主流程 ===
if __name__ == "__main__":
    test_dir = find_latest_test_dir()
    test_csv_path = os.path.join(test_dir, "test_log.csv")
    save_csv_path = os.path.join(test_dir, "metrics.csv")

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"❌ 找不到 test_log.csv：{test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    actions = test_df['action'].values

    entropy = calculate_entropy(actions)
    total_steps = len(test_df)
    success_steps = test_df['reward'].gt(0.5).sum()
    params = 1e5
    wisdom_density = success_steps / (total_steps * params)

    metrics_df = pd.DataFrame({
        "episode": np.arange(1, total_steps + 1),
        "entropy": [entropy] * total_steps,
        "wisdom_density": [wisdom_density] * total_steps
    })

    metrics_df.to_csv(save_csv_path, index=False)
    print(f"✅ Metrics 儲存到：{save_csv_path}")
