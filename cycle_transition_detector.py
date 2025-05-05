# cycle_transition_detector.py
# ✅ 偵測智慧節奏中 cycle 切換的關鍵事件點（自動抓最新資料夾並儲存轉變點）

import pandas as pd
import os

# === 可調參數 ===
H_THRESHOLD = 0.002         # entropy 變動門檻
S_THRESHOLD = 1e-6          # wisdom density 跳動門檻
TERMINATE_SPLIT = True      # 是否依照 terminated 判斷切段

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


# === 主邏輯：標記 cycle 切換事件 ===
def detect_cycle_transitions(metrics_path):
    df = pd.read_csv(metrics_path)
    transitions = []

    last_H = df.loc[0, 'entropy']
    last_S = df.loc[0, 'wisdom_density']

    for idx, row in df.iterrows():
        H_diff = abs(row['entropy'] - last_H)
        S_diff = abs(row['wisdom_density'] - last_S)
        terminated = bool(row.get('terminated', False))

        split = False
        if H_diff > H_THRESHOLD:
            split = True
        if S_diff > S_THRESHOLD:
            split = True
        if TERMINATE_SPLIT and terminated:
            split = True

        if split and idx != 0:
            transitions.append({
                "episode": row['episode'],
                "H_diff": round(H_diff, 5),
                "S_diff": round(S_diff, 8),
                "terminated": terminated
            })

        last_H = row['entropy']
        last_S = row['wisdom_density']

    return transitions


# === 測試執行與儲存 ===
if __name__ == "__main__":
    latest_dir = find_latest_test_dir()
    test_metrics = os.path.join(latest_dir, "metrics.csv")

    if not os.path.exists(test_metrics):
        raise FileNotFoundError(f"❌ 找不到 metrics.csv：{test_metrics}")

    result = detect_cycle_transitions(test_metrics)
    print(f"✅ 偵測到 {len(result)} 個 cycle transition：")

    for r in result:
        print(r)

    # ⬇️ 儲存轉變點 CSV
    if result:
        save_path = os.path.join(latest_dir, "cycle_transition_points.csv")
        pd.DataFrame(result).to_csv(save_path, index=False)
        print(f"📄 結果已儲存：{save_path}")
