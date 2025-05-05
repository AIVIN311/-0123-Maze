# 🧠 cycle_marker_utils.py
# 進階智慧週期標記器：依據 H / S / terminated 自動切段加入 cycle 編號

import pandas as pd
import os

# === 可調參數 ===
H_THRESHOLD = 0.002       # 若 H 的變化幅度大於此值，視為新 cycle 起點
S_THRESHOLD = 1e-6        # 若 S 有跳動，也可能視為切段依據
TERMINATE_SPLIT = True    # 若 terminated=True，自動換 cycle

# === 主邏輯 ===
def add_cycle_markers(metrics_path):
    df = pd.read_csv(metrics_path)

    if 'cycle' in df.columns:
        print("⚠️ 已有 cycle 欄位，將覆寫舊值")

    cycle_id = 1
    cycle_col = []
    last_H = df.loc[0, 'entropy']
    last_S = df.loc[0, 'wisdom_density']

    for idx, row in df.iterrows():
        H_diff = abs(row['entropy'] - last_H)
        S_diff = abs(row['wisdom_density'] - last_S)

        split = False
        if H_diff > H_THRESHOLD:
            split = True
        if S_diff > S_THRESHOLD:
            split = True
        if TERMINATE_SPLIT and row.get('terminated', False):
            split = True

        if split and idx != 0:
            cycle_id += 1

        cycle_col.append(cycle_id)
        last_H = row['entropy']
        last_S = row['wisdom_density']

    df['cycle'] = cycle_col
    df.to_csv(metrics_path, index=False)
    print(f"✅ 已加入 cycle 欄位，總共有 {cycle_id} 段。")


# === 測試執行（選擇性，可獨立執行） ===
if __name__ == "__main__":
    from auto_analyze_latest_run import find_latest_test_dir

    latest_dir = find_latest_test_dir()
    metrics_path = os.path.join(latest_dir, "metrics.csv")

    if not os.path.exists(metrics_path):
        print(f"❌ 找不到 metrics.csv：{metrics_path}")
    else:
        print(f"📂 載入最新測試資料夾：{latest_dir}")
        add_cycle_markers(metrics_path)
