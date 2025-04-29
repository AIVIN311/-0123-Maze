import pandas as pd
import numpy as np
import os

# === 設定路徑（⚡請改成你最新的test資料夾） ===
test_dir = "runs/test_2025-04-29_21-45-38"  # 記得改
test_csv_path = os.path.join(test_dir, "test_log.csv")
save_csv_path = os.path.join(test_dir, "metrics.csv")

# === 計算行動熵（H） ===
def calculate_entropy(actions):
    counts = np.bincount(actions)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# === 讀取測試紀錄 ===
test_df = pd.read_csv(test_csv_path)

# === 提取行動序列 ===
actions = test_df['action'].values

# === 計算 H 與勝率 S ===
entropy = calculate_entropy(actions)
total_steps = len(test_df)
success_steps = test_df['reward'].gt(0.5).sum()  # 獎勵大於0.5視為成功
# 假設參數量大約是100000個
params = 1e5
wisdom_density = success_steps / (total_steps * params)

# === 整理結果 ===
metrics_df = pd.DataFrame({
    "episode": np.arange(1, total_steps + 1),
    "entropy": [entropy] * total_steps,
    "wisdom_density": [wisdom_density] * total_steps
})

# === 儲存 ===
metrics_df.to_csv(save_csv_path, index=False)
print(f"✅ Metrics 儲存到 {save_csv_path}")
